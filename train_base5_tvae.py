
import datetime, random, copy, argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, Parameter
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from util.data import load_dataset
from util.model_test import mkdir, fix_random_seed, model_save_dict, model_score_save, save_index
from torch.utils.data import DataLoader, TensorDataset

from util.base import BaseSynthesizer
from util.transformer_new import BGMTransformer
from util.evaluate import compute_scores, _compute_for_distribution
from util.evaluate_cluster import compute_cluster_scores
from util.benchmark import benchmark


from tensorboardX import SummaryWriter


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma


def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for item in output_info:
        # import pdb;pdb.set_trace()
        if item[1] == 'tanh':
            ed = st + item[0]
            std = sigmas[st]
            loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2)).sum())
            loss.append(torch.log(std) * x.size()[0])
            st = ed

        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(cross_entropy(
                recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
            st = ed

        else:
            assert 0
    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

class TVAESynthesizer(BaseSynthesizer):
    """TVAESynthesizer."""

    def __init__(self, embedding_dim, compress_dims, decompress_dims, l2scale, batch_size,epochs, save_arg, data_name,
                save_loc, test_name, random_num, GPU_NUM, train=True):

        self.embedding_dim = embedding_dim ; self.compress_dims = compress_dims ; self.decompress_dims = decompress_dims

        self.l2scale = l2scale ; self.batch_size = batch_size ; self.loss_factor = 2 ; self.epochs = epochs

        self.save_arg = save_arg
        self.data_name = data_name

        self.save_loc, self.test_name, self.random_num, self.GPU_NUM = (save_loc, test_name, random_num, GPU_NUM)
        self.device = torch.device(f'cuda:{self.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if train:
            self.save_arg["excute_time"] = str(datetime.datetime.now())
            with open(self.save_loc + "/param/" + self.data_name + "/" + self.test_name + ".txt","a") as f:
                f.write("excute_time: " + self.save_arg["excute_time"] + "\n")
            self.writer = SummaryWriter(self.save_loc + "/runs/" + self.data_name + "/" + self.test_name)

        torch.cuda.set_device(self.device)

        # for random
        fix_random_seed(self.random_num)


    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = meta_data
        self.test = test_data
        self.train = train_data.copy()
        self.transformer = BGMTransformer(self.meta, random_seed=self.random_num)
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        # print(self.batch_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dim
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        optimizerAE = optim.Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        best_model_dict = model_save_dict(self.meta["problem_type"])
        every_model_dict = {"name": "TVAESynthesizer", "arg" : self.save_arg, "model": {}}
        track_score_dict = {}
        save_score_dict = {}
        
        iter = 0
        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                iter += 1
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
                self.writer.add_scalar('losses/AE_loss', loss, iter)

            if True: # compute scores every epochs (# if i >= 150 and i % 2 == 0:)
                self.save_result_type2(i, track_score_dict, save_score_dict, every_model_dict)
                
    def sample(self, samples):
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        self.decoder.train()
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())


    def model_load(self, checkpoint, choosed_model):
        dataset_name = checkpoint["arg"]["data_name"]
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        
        self.train = train_data.copy()
        self.transformer = BGMTransformer(meta_data, random_seed=self.random_num)
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        
        self.test = test_data
        self.meta = meta_data
        
        data_dim = self.transformer.output_dim

        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        if "name" in checkpoint:
            self.decoder.load_state_dict(checkpoint["model"][choosed_model]['decoder'])
            self.decoder.eval()
        else:
            self.decoder.load_state_dict(checkpoint["info"][choosed_model]['decoder'])
            self.decoder.eval()

    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])



if __name__ == "__main__":

    parser = argparse.ArgumentParser('TVAE')
    parser.add_argument('--data', type=str, default = 'adult')
    parser.add_argument('--random_num', type=int, default = 777)
    parser.add_argument('--test_name', type=str, default = 'tvae')
    parser.add_argument('--GPU_NUM', type = int, default = 0)
    arg_of_parser = parser.parse_args()

    data = arg_of_parser.data ; random_num = arg_of_parser.random_num 
    test_name = arg_of_parser.test_name ; GPU_NUM = arg_of_parser.GPU_NUM

    embedding_dim=128 ; compress_dims = (128, 128) ; decompress_dims = (128, 128)
    l2scale=1e-5 ; batch_size=500 ; epochs=300 ; save_loc = "last_result"
    arg = {'embedding_dim':embedding_dim,
            'compress_dims':compress_dims,
            'decompress_dims':decompress_dims,
            'l2scale':l2scale,
            'batch_size':batch_size,
            'epochs':epochs,
            'data_name':data,
            "save_loc": save_loc,
            "test_name": test_name,
            "random_num" : random_num,
            "GPU_NUM": GPU_NUM}
    
    arg["save_arg"] = arg.copy()

    mkdir(save_loc, data)
    if not os.path.isdir(os.path.join(save_loc,"save_model",data,test_name)):
        os.mkdir(os.path.join(save_loc,"save_model",data,test_name))
    
    with open(save_loc + "/param/"+ data + "/" + test_name + '.txt',"a") as f:
        f.write(data + " TVAESynthesizer" + "\n")
        f.write(str(arg) + "\n")
    
    a,b = benchmark(TVAESynthesizer, arg, data)
