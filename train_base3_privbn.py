import shutil, subprocess, argparse
import datetime, random, copy, json, os
import numpy as np
import pandas as pd
from datetime import datetime as dt
import torch

from util.data import load_dataset
from util.model_test import mkdir, fix_random_seed, model_save_dict, model_score_save
from util.base import BaseSynthesizer
from util.evaluate import compute_scores, _compute_for_distribution
from util.evaluate_cluster import compute_cluster_scores
from util.benchmark import benchmark
from tensorboardX import SummaryWriter

from util.constants import CATEGORICAL, ORDINAL
from pomegranate import BayesianNetwork, ConditionalProbabilityTable, DiscreteDistribution
from util.transformer_origin import Transformer

def try_mkdirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


class PrivBNSynthesizer(BaseSynthesizer):

    def __init__(self, save_arg, data_name, save_loc, test_name, random_num,
                theta=20, max_samples=25000, train=True):
        assert os.path.exists("privbayes/privBayes.bin")
        self.theta = theta
        self.max_samples = max_samples

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
        self.data = train_data.copy()
        self.meta = Transformer.get_metadata(train_data, categorical_columns, ordinal_columns)
        self.meta2 = meta_data

        self.save_result_type1(train_data, test_data, self.meta2, "PrivBNSynthesizer")

    def model_load(self, checkpoint, choosed_model):
        dataset_name = checkpoint["arg"]["data_name"]
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        self.train = train_data.copy()
        self.test = test_data

        self.data = train_data.copy()
        self.meta = Transformer.get_metadata(train_data, categorical_columns, ordinal_columns)
        self.meta2 = meta_data
        
    def sample(self, n):
        try_mkdirs("__privbn_tmp/data")
        try_mkdirs("__privbn_tmp/log")
        try_mkdirs("__privbn_tmp/output")
        shutil.copy("privbayes/privBayes.bin", "__privbn_tmp/privBayes.bin")
        d_cols = []
        with open("__privbn_tmp/data/real.domain", "w") as f:
            for id_, info in enumerate(self.meta):
                if info['type'] in [CATEGORICAL, ORDINAL]:
                    print("D", end='', file=f)
                    counter = 0
                    for i in range(info['size']):
                        if i > 0 and i % 4 == 0:
                            counter += 1
                            print(" {", end='', file=f)
                        print("", i, end='', file=f)
                    print(" }" * counter, file=f)
                    d_cols.append(id_)
                else:
                    minn = info['min']
                    maxx = info['max']
                    d = (maxx - minn) * 0.03
                    minn = minn - d
                    maxx = maxx + d
                    print("C", minn, maxx, file=f)

        with open("__privbn_tmp/data/real.dat", "w") as f:
            n = len(self.data)
            np.random.shuffle(self.data)
            n = min(n, self.max_samples)
            for i in range(n):
                row = self.data[i]
                for id_, col in enumerate(row):
                    if id_ in d_cols:
                        print(int(col), end=' ', file=f)
                    else:
                        print(col, end=' ', file=f)

                print(file=f)

        privbayes = os.path.realpath("__privbn_tmp/privBayes.bin")
        arguments = [privbayes, "real", str(n), "1", str(self.theta)]
        start = dt.utcnow()
        subprocess.call(arguments, cwd="__privbn_tmp")

        return np.loadtxt(
            "__privbn_tmp/output/syn_real_eps10_theta{}_iter0.dat".format(self.theta))

    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])
    




if __name__ == "__main__":

    parser = argparse.ArgumentParser('PrivBN')
    parser.add_argument('--data', type=str, default = 'adult')
    parser.add_argument('--random_num', type=int, default = 777)
    parser.add_argument('--test_name', type=str, default = 'privbn')
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
        f.write(data + " PrivBNSynthesizer" + "\n")
        f.write(str(arg) + "\n")
    a,b = benchmark(PrivBNSynthesizer, arg, data)
    print(a.mean(),b)
    
