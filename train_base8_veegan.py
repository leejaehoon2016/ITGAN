
import datetime, random, copy, argparse, os
import numpy as np
import pandas as pd
import torch
from torch.nn import Dropout, Linear, Module, ReLU, Sequential
from torch.nn.functional import mse_loss, softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from util.data import load_dataset
from util.model_test import mkdir, fix_random_seed, model_save_dict, model_score_save

from util.constants import CATEGORICAL
from util.base import BaseSynthesizer
from util.transformer_origin import GeneralTransformer
from util.evaluate import compute_scores, _compute_for_distribution
from util.benchmark import benchmark
from util.evaluate_cluster import compute_cluster_scores
from tensorboardX import SummaryWriter

class Reconstructor(Module):

    def __init__(self, data_dim, reconstructor_dim, embedding_dim):
        super(Reconstructor, self).__init__()
        dim = data_dim
        seq = []
        for item in list(reconstructor_dim):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq += [Linear(dim, embedding_dim)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim):
        super(Discriminator, self).__init__()
        dim = input_dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), ReLU(), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input, output_info):
        data = self.seq(input)
        data_t = []
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed

            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(softmax(data[:, st:ed], dim=1))
                st = ed

            else:
                assert 0

        return torch.cat(data_t, dim=1)


class VEEGANSynthesizer(BaseSynthesizer):

    def __init__(self, save_arg, data_name, save_loc, test_name, random_num, GPU_NUM,
                 embedding_dim=32, gen_dim=(128, 128), dis_dim=(128, ), rec_dim=(128, 128),
                l2scale=1e-6, batch_size=500, epochs=300, train=True):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.rec_dim = rec_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs


        self.save_arg = save_arg
        self.data_name = data_name
        self.transformer = None


        self.save_loc, self.test_name, self.random_num, self.GPU_NUM = (save_loc, test_name, random_num, GPU_NUM)
        self.device = torch.device(f'cuda:{self.GPU_NUM}' if torch.cuda.is_available() else 'cpu')

        if train:
            self.save_arg["excute_time"] = str(datetime.datetime.now())
            with open(self.save_loc + "/param/" + self.data_name + "/" + self.test_name + ".txt","a") as f:
                f.write("excute_time: " + self.save_arg["excute_time"] + "\n")
            self.writer = SummaryWriter(self.save_loc + "/runs/" + self.data_name + "/" + self.test_name)

        torch.cuda.set_device(self.device)

        # for random
        fix_random_seed(self.random_num)

    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns=tuple(), ordinal_columns=tuple()):
        data = train_data
        self.train = train_data.copy()
        self.test = test_data.copy()
        self.meta = meta_data
        
        self.transformer = GeneralTransformer(act='tanh')
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dim
        self.generator = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        discriminator = Discriminator(self.embedding_dim + data_dim, self.dis_dim).to(self.device)
        reconstructor = Reconstructor(data_dim, self.rec_dim, self.embedding_dim).to(self.device)

        optimizer_params = dict(lr=1e-3, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)
        optimizerR = Adam(reconstructor.parameters(), **optimizer_params)

        
        best_model_dict = model_save_dict(self.meta["problem_type"])
        every_model_dict = {"name": "VEEGANSynthesizer", "arg" : self.save_arg, "model": [0]}
        track_score_dict, save_score_dict = {}, {}
        iter = 0

        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1
        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                iter += 1
                real = data[0].to(self.device)
                realz = reconstructor(real)
                y_real = discriminator(torch.cat([real, realz], dim=1))

                fakez = torch.normal(mean=mean, std=std)
                fake = self.generator(fakez, self.transformer.output_info)
                fakezrec = reconstructor(fake)
                y_fake = discriminator(torch.cat([fake, fakez], dim=1))

                loss_d = (
                    -(torch.log(torch.sigmoid(y_real) + 1e-4).mean())
                    - (torch.log(1. - torch.sigmoid(y_fake) + 1e-4).mean())
                )

                optimizerD.zero_grad()
                loss_d.backward()
                optimizerD.step()

                real = data[0].to(self.device)
                realz = reconstructor(real)
                y_real = discriminator(torch.cat([real, realz], dim=1))

                fakez = torch.normal(mean=mean, std=std)
                fake = self.generator(fakez, self.transformer.output_info)
                fakezrec = reconstructor(fake)
                y_fake = discriminator(torch.cat([fake, fakez], dim=1))

                numerator = -y_fake.mean() + mse_loss(fakezrec, fakez, reduction='mean')
                loss_g = numerator / self.embedding_dim
                
                optimizerG.zero_grad()
                loss_g.backward(retain_graph=True)
                optimizerG.step()
                

                real = data[0].to(self.device)
                realz = reconstructor(real)
                y_real = discriminator(torch.cat([real, realz], dim=1))

                fakez = torch.normal(mean=mean, std=std)
                fake = self.generator(fakez, self.transformer.output_info)
                fakezrec = reconstructor(fake)
                y_fake = discriminator(torch.cat([fake, fakez], dim=1))

                numerator = -y_fake.mean() + mse_loss(fakezrec, fakez, reduction='mean')
                loss_r = numerator / self.embedding_dim                
                optimizerR.zero_grad()
                loss_r.backward()
                optimizerR.step()
                self.writer.add_scalar('losses/G_loss', loss_g, iter)
                self.writer.add_scalar('losses/D_loss', loss_d, iter)
                self.writer.add_scalar('losses/loss_r', loss_r, iter)
                
                
            if True: # compute scores every epochs (# if i >= 150 and i % 2 == 0:)
                self.save_result_type2(i, track_score_dict, save_score_dict, every_model_dict)

    def sample(self, n):
        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            fake = self.generator(noise, output_info)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        self.generator.train()
        return self.transformer.inverse_transform(data)

    def model_load(self, checkpoint, choosed_model):
        dataset_name = checkpoint["arg"]["data_name"]
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        data = train_data
        self.train = train_data.copy()
        self.test = test_data.copy()
        self.meta = meta_data
        
        self.transformer = GeneralTransformer(act='tanh')
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        
        data_dim = self.transformer.output_dim
        self.generator = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        
        self.generator.load_state_dict(checkpoint["model"][choosed_model]['generator'])
        self.generator.eval()
        


    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])




if __name__ == "__main__":
    parser = argparse.ArgumentParser('VeeGAN')
    parser.add_argument('--data', type=str, default = 'adult')
    parser.add_argument('--random_num', type=int, default = 777)
    parser.add_argument('--test_name', type=str, default = 'veegan')
    parser.add_argument('--GPU_NUM', type = int, default = 0)
    arg_of_parser = parser.parse_args()

    data = arg_of_parser.data ; random_num = arg_of_parser.random_num 
    test_name = arg_of_parser.test_name ; GPU_NUM = arg_of_parser.GPU_NUM
    save_loc = "last_result" 

    arg = { 'data_name':data,
            "save_loc": save_loc,
            "test_name": test_name,
            "random_num" : random_num,
            "GPU_NUM": GPU_NUM}
    
    arg["save_arg"] = arg.copy()

                     
    mkdir(save_loc, data)
    if not os.path.isdir(os.path.join(save_loc,"save_model",data,test_name)):
        os.mkdir(os.path.join(save_loc,"save_model",data,test_name))
    
    with open(save_loc + "/param/"+ data + "/" + test_name + '.txt',"a") as f:
        f.write(data + " VEEGANSynthesizer" + "\n")
        f.write(str(arg) + "\n")

    a,b = benchmark(VEEGANSynthesizer, arg, data)
