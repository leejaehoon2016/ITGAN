import datetime, json
import numpy as np
import pandas as pd
import torch
import argparse

from util.data import load_dataset
from util.model_test import mkdir, fix_random_seed
from util.base import BaseSynthesizer
from util.evaluate import compute_scores
from util.evaluate_cluster import compute_cluster_scores
from util.benchmark import benchmark

from pomegranate import BayesianNetwork, ConditionalProbabilityTable, DiscreteDistribution
from util.transformer_origin import DiscretizeTransformer


class CLBNSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, save_arg, data_name, save_loc, test_name, random_num, train=True):
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
        self.discretizer = DiscretizeTransformer(n_bins=15)
        self.discretizer.fit(train_data, categorical_columns, ordinal_columns)
        discretized_data = self.discretizer.transform(train_data)
        self.model = BayesianNetwork.from_samples(discretized_data, algorithm='chow-liu')
        self.meta = meta_data
        
        self.save_result_type1(train_data, test_data, self.meta, "CLBNSynthesizer")
    
    def model_load(self, checkpoint, choosed_model):
        dataset_name = checkpoint["arg"]["data_name"]
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        self.train = train_data.copy()
        self.test = test_data
        
        self.discretizer = DiscretizeTransformer(n_bins=15)
        self.discretizer.fit(train_data, categorical_columns, ordinal_columns)
        discretized_data = self.discretizer.transform(train_data)
        self.model = BayesianNetwork.from_samples(discretized_data, algorithm='chow-liu')
        self.meta = meta_data
        
    def bn_sample(self, num_samples):
        """Sample from the bayesian network.

        Args:
            num_samples(int): Number of samples to generate.
        """
        nodes_parents = self.model.structure
        processing_order = []

        while len(processing_order) != len(nodes_parents):
            update = False

            for id_, parents in enumerate(nodes_parents):
                if id_ in processing_order:
                    continue

                flag = True
                for parent in parents:
                    if parent not in processing_order:
                        flag = False

                if flag:
                    processing_order.append(id_)
                    update = True

            assert update

        data = np.zeros((num_samples, len(nodes_parents)), dtype='int32')
        for current in processing_order:
            distribution = self.model.states[current].distribution
            if isinstance(distribution, DiscreteDistribution):
                data[:, current] = distribution.sample(num_samples)
            else:
                assert isinstance(distribution, ConditionalProbabilityTable)
                output_size = list(distribution.keys())
                output_size = max([int(x) for x in output_size]) + 1

                distribution = json.loads(distribution.to_json())
                distribution = distribution['table']

                distribution_dict = {}

                for row in distribution:
                    key = tuple(np.asarray(row[:-2], dtype='int'))
                    output = int(row[-2])
                    p = float(row[-1])

                    if key not in distribution_dict:
                        distribution_dict[key] = np.zeros(output_size)

                    distribution_dict[key][int(output)] = p

                parents = nodes_parents[current]
                conds = data[:, parents]
                for _id, cond in enumerate(conds):
                    data[_id, current] = np.random.choice(
                        np.arange(output_size),
                        p=distribution_dict[tuple(cond)]
                    )

        return data

    def sample(self, samples):
        data = self.bn_sample(samples)
        return self.discretizer.inverse_transform(data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('CLBN')
    parser.add_argument('--data', type=str, default = 'adult')
    parser.add_argument('--random_num', type=int, default = 777)
    parser.add_argument('--test_name', type=str, default = 'clbn')
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
        f.write(data + " CLBNSynthesizer" + "\n")
        f.write(str(arg) + "\n")

    benchmark(CLBNSynthesizer, arg, data)
