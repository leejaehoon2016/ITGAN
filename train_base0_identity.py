import pandas as pd
import numpy as np
import torch
from util.base import BaseSynthesizer
from util.evaluate import compute_scores
from util.evaluate_cluster import compute_cluster_scores
from util.model_test import mkdir, fix_random_seed, model_save_dict, model_score_save
from util.benchmark import benchmark
import argparse, os

class IdentitySynthesizer(BaseSynthesizer):
    """Trivial synthesizer.
    Returns the same exact data that is used to fit it.
    """
    def __init__(self, data_name, save_loc, test_name, random_num):
        self.data_name = data_name
        self.save_loc = save_loc
        self.test_name = test_name
        self.random_num = random_num
        fix_random_seed(self.random_num)

    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.data = pd.DataFrame(train_data)
        self.meta = meta_data
        
        self.save_result_type1(train_data, test_data, self.meta, "IdentitySynthesizer")

    def sample(self, samples):
        return self.data.sample(samples, replace=False).values

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Identity')
    parser.add_argument('--data', type=str, default = 'adult')
    parser.add_argument('--random_num', type=int, default = 777)
    parser.add_argument('--test_name', type=str, default = 'identity')
    arg_of_parser = parser.parse_args()
    
    data = arg_of_parser.data ; random_num = arg_of_parser.random_num 
    test_name = arg_of_parser.test_name ; save_loc = "last_result" 
    arg = { 'data_name':data,
            "save_loc": save_loc,
            "test_name": test_name,
            "random_num" : random_num}
                            
    mkdir(save_loc, data)
    
    a,b = benchmark(IdentitySynthesizer, arg, data)
    print(a.mean(),b)
