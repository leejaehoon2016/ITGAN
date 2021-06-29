
import datetime, random, copy, argparse, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Module, Sequential
from torch.nn.functional import cross_entropy, mse_loss, sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from util.data import load_dataset
from util.model_test import mkdir, fix_random_seed, model_save_dict, model_score_save

from util.base import BaseSynthesizer
from util.transformer_origin import GeneralTransformer
from util.evaluate import compute_scores, _compute_for_distribution
from util.benchmark import benchmark
from util.evaluate_cluster import compute_cluster_scores
from tensorboardX import SummaryWriter

class ResidualFC(Module):
    def __init__(self, input_dim, output_dim, activate, bn_decay):
        super(ResidualFC, self).__init__()
        self.seq = Sequential(
            Linear(input_dim, output_dim),
            BatchNorm1d(output_dim, momentum=bn_decay),
            activate()
        )

    def forward(self, input):
        residual = self.seq(input)
        return input + residual


class Generator(Module):
    def __init__(self, random_dim, hidden_dim, bn_decay):
        super(Generator, self).__init__()

        dim = random_dim
        seq = []
        for item in list(hidden_dim)[:-1]:
            assert item == dim
            seq += [ResidualFC(dim, dim, nn.ReLU, bn_decay)]
        assert hidden_dim[-1] == dim
        seq += [
            Linear(dim, dim),
            BatchNorm1d(dim, momentum=bn_decay),
            nn.ReLU()
        ]
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Discriminator(Module):
    def __init__(self, data_dim, hidden_dim):
        super(Discriminator, self).__init__()
        dim = data_dim * 2
        seq = []
        for item in list(hidden_dim):
            seq += [
                Linear(dim, item),
                nn.ReLU() if item > 1 else nn.Sigmoid()
            ]
            dim = item
        self.seq = Sequential(*seq)

    def forward(self, input):
        mean = input.mean(dim=0, keepdim=True)
        mean = mean.expand_as(input)
        inp = torch.cat((input, mean), dim=1)
        return self.seq(inp)


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims) + [embedding_dim]:
            seq += [
                Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [
                Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input, output_info):
        return self.seq(input)


def aeloss(fake, real, output_info):
    st = 0
    loss = []
    for item in output_info:
        if item[1] == 'sigmoid':
            ed = st + item[0]
            loss.append(mse_loss(sigmoid(fake[:, st:ed]), real[:, st:ed], reduction='sum'))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(cross_entropy(
                fake[:, st:ed], torch.argmax(real[:, st:ed], dim=-1), reduction='sum'))
            st = ed
        else:
            assert 0
    return sum(loss) / fake.size()[0]


class MedganSynthesizer(BaseSynthesizer):

    def __init__(self, save_arg, data_name, save_loc, test_name, random_num, GPU_NUM,
                 embedding_dim=128, random_dim=128, generator_dims=(128, 128), discriminator_dims=(256, 128, 1),  
                 compress_dims=(), decompress_dims=(), bn_decay=0.99, l2scale=0.001,
                 pretrain_epoch=200, batch_size=1000, epochs=500, train=True):

        self.embedding_dim = embedding_dim
        self.random_dim = random_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims

        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.bn_decay = bn_decay
        self.l2scale = l2scale

        self.pretrain_epoch = pretrain_epoch
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
        
        self.transformer = GeneralTransformer()
        self.transformer.fit(data, categorical_columns, ordinal_columns)
        data = self.transformer.transform(data)
        dataset = TensorDataset(torch.from_numpy(data.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dim
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale
        )

        for i in range(self.pretrain_epoch):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                emb = encoder(real)
                rec = self.decoder(emb, self.transformer.output_info)
                loss = aeloss(rec, real, self.transformer.output_info)
                loss.backward()
                optimizerAE.step()

        self.generator = Generator(
            self.random_dim, self.generator_dims, self.bn_decay).to(self.device)
        discriminator = Discriminator(data_dim, self.discriminator_dims).to(self.device)
        optimizerG = Adam(
            list(self.generator.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale
        )
        optimizerD = Adam(discriminator.parameters(), weight_decay=self.l2scale)

        best_model_dict = model_save_dict(self.meta["problem_type"])
        every_model_dict = {"name": "MedganSynthesizer", "arg" : self.save_arg, "model": [0]}
        track_score_dict, save_score_dict = {}, {}
        iter = 0
        mean = torch.zeros(self.batch_size, self.random_dim, device=self.device)
        std = mean + 1
        for i in range(self.epochs):
            n_d = 2
            n_g = 1
            for id_, data in enumerate(loader):
                iter += 1
                real = data[0].to(self.device)
                noise = torch.normal(mean=mean, std=std)
                emb = self.generator(noise)
                fake = self.decoder(emb, self.transformer.output_info)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                real_loss = -(torch.log(y_real + 1e-4).mean())
                fake_loss = (torch.log(1.0 - y_fake + 1e-4).mean())
                loss_d = real_loss - fake_loss
                loss_d.backward()
                optimizerD.step()

                if i % n_d == 0:
                    for _ in range(n_g):
                        noise = torch.normal(mean=mean, std=std)
                        emb = self.generator(noise)
                        fake = self.decoder(emb, self.transformer.output_info)
                        optimizerG.zero_grad()
                        y_fake = discriminator(fake)
                        loss_g = -(torch.log(y_fake + 1e-4).mean())
                        loss_g.backward()
                        optimizerG.step()
                self.writer.add_scalar('losses/G_loss', loss_g, iter)
                self.writer.add_scalar('losses/D_loss', loss_d, iter)
            
            if True: # compute scores every epochs (# if i >= 150 and i % 2 == 0:)
                self.save_result_type2(i, track_score_dict, save_score_dict, every_model_dict)
                

    def sample(self, n):
        self.generator.eval()
        self.decoder.eval()

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.random_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            emb = self.generator(noise)
            fake = self.decoder(emb, self.transformer.output_info)
            fake = torch.sigmoid(fake)
            data.append(fake.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        data = data[:n]
        self.generator.train()
        self.decoder.train()
        return self.transformer.inverse_transform(data)

    def model_load(self, checkpoint, choosed_model):
        dataset_name = checkpoint["arg"]["data_name"]
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        self.train = train_data.copy()
        self.test = test_data
        data = train_data
        
        self.transformer = GeneralTransformer()
        self.transformer.fit(data, categorical_columns, ordinal_columns)
        
        data_dim = self.transformer.output_dim
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        
        self.generator = Generator(
            self.random_dim, self.generator_dims, self.bn_decay).to(self.device)

        self.generator.load_state_dict(checkpoint["model"][choosed_model]['generator'])
        self.generator.eval()
        self.decoder.load_state_dict(checkpoint["model"][choosed_model]['decoder'])
        self.decoder.eval()
        


    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser('MedGAN')
    parser.add_argument('--data', type=str, default = 'adult')
    parser.add_argument('--random_num', type=int, default = 777)
    parser.add_argument('--test_name', type=str, default = 'medgan')
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
        f.write(data + " Medgansynthesizer" + "\n")
        f.write(str(arg) + "\n")

    a,b = benchmark(MedganSynthesizer, arg, data)
