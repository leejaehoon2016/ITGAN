import datetime, random, copy, json, argparse, os
import numpy as np
import pandas as pd
import torch

from util.data import load_dataset
from util.model_test import mkdir, fix_random_seed, model_save_dict, model_score_save
from util.base import BaseSynthesizer
from util.evaluate import compute_scores, _compute_for_distribution
from util.evaluate_cluster import compute_cluster_scores
from util.benchmark import benchmark
from tensorboardX import SummaryWriter

from util.constants import CONTINUOUS
from pomegranate import BayesianNetwork, ConditionalProbabilityTable, DiscreteDistribution
from util.transformer_origin import Transformer
from sklearn.mixture import GaussianMixture

class IndependentSynthesizer(BaseSynthesizer):

    def __init__(self, save_arg, data_name, save_loc, test_name, random_num, train=True):
        self.gmm_n = 5
        self.save_arg = save_arg
        self.data_name = data_name
        self.save_loc, self.test_name, self.random_num = (save_loc, test_name, random_num)

        if train:
            self.save_arg["excute_time"] = str(datetime.datetime.now())
            with open(self.save_loc + "/param/" + self.data_name + "/" + self.test_name + ".txt","a") as f:
                f.write("excute_time: " + self.save_arg["excute_time"] + "\n")

        # for random
        fix_random_seed(self.random_num)

    
    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.dtype = train_data.dtype
        self.meta = Transformer.get_metadata(train_data, categorical_columns, ordinal_columns)
        self.meta2 = meta_data
        self.models = []
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                model = GaussianMixture(self.gmm_n)
                model.fit(train_data[:, [id_]])
                self.models.append(model)
            else:
                nomial = np.bincount(train_data[:, id_].astype('int'), minlength=info['size'])
                nomial = nomial / np.sum(nomial)
                self.models.append(nomial)

        self.save_result_type1(train_data, test_data, self.meta2, "IndependentSynthesizer")

      
    def model_load(self, checkpoint, choosed_model):
        dataset_name = checkpoint["arg"]["data_name"]
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        self.train = train_data.copy()
        self.test = test_data
        self.dtype = train_data.dtype
        self.meta = Transformer.get_metadata(train_data, categorical_columns, ordinal_columns)
        self.meta2 = meta_data
        self.models = []
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                model = GaussianMixture(self.gmm_n)
                model.fit(data[:, [id_]])
                self.models.append(model)
            else:
                nomial = np.bincount(data[:, id_].astype('int'), minlength=info['size'])
                nomial = nomial / np.sum(nomial)
                self.models.append(nomial)

        
    def sample(self, samples):
        data = np.zeros([samples, len(self.meta)], self.dtype)

        for i, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                x, _ = self.models[i].sample(samples)
                np.random.shuffle(x)
                data[:, i] = x.reshape([samples])
                data[:, i] = data[:, i].clip(info['min'], info['max'])
            else:
                size = len(self.models[i])
                data[:, i] = np.random.choice(np.arange(size), samples, p=self.models[i])

        return data

    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Independent')
    parser.add_argument('--data', type=str, default = 'adult')
    parser.add_argument('--random_num', type=int, default = 777)
    parser.add_argument('--test_name', type=str, default = 'ind')
    arg_of_parser = parser.parse_args()


    data = arg_of_parser.data ; random_num = arg_of_parser.random_num 
    test_name = arg_of_parser.test_name ; save_loc = "last_result" 
    arg = { 'data_name':data,
            "save_loc": save_loc,
            "test_name": test_name,
            "random_num" : random_num}
    
    arg["save_arg"] = arg.copy()

                    
    mkdir(save_loc, data)
    with open(save_loc + "/param/"+ data + "/" + test_name + '.txt',"a") as f:
        f.write(data + " IndependentSynthesizer" + "\n")
        f.write(str(arg) + "\n")

    a,b = benchmark(IndependentSynthesizer, arg, data)
    print(a.mean(),b)
