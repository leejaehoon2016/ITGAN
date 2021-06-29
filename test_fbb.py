from util.model_test import fix_random_seed
import pandas as pd
import numpy as np
import os, argparse
import torch
from util.model_load import model_load
from util.full_black_box import fbb, _compute_distance
from util.data import load_dataset
from tensorboard.backend.event_processing import event_accumulator
arg_dic = { event_accumulator.COMPRESSED_HISTOGRAMS: 500, event_accumulator.IMAGES: 4, event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0, event_accumulator.HISTOGRAMS: 1,}
func = event_accumulator.EventAccumulator



dist = ['average/distance', 'average/EMD']


test_number = 1000
parser = argparse.ArgumentParser('Check FBB Score')
parser.add_argument('--data', type=str, default="adult")
parser.add_argument('--GPU_NUM', type=int, default = 0)
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--subopt', type=str, default = 0)
arg_of_parser = parser.parse_args()

data_name = arg_of_parser.data
GPU_NUM = arg_of_parser.GPU_NUM
file_name = arg_of_parser.file
subopt = arg_of_parser.subopt
print(data_name)

train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(data_name, benchmark=True)
syn_data_lst=[]

path = f"last_result/save_model/{data_name}/{file_name}.pth"
model = torch.load(path)
keys = model["model"].keys()
scores = sorted([i for i in keys if type(i) is int],reverse=True)[1:4]
model = model_load(path,GPU_NUM,"best")
syndata = [file_name, model.sample(len(train_data))]
syn_data_lst.append(syndata)
if subopt:
    for i in range(3):
        model = model_load(path,GPU_NUM,scores[i])
        fix_random_seed(777)
        syn_data_lst.append([f"sub_opt{i}", model.sample(len(train_data))])

neg_data_index = pd.Series(_compute_distance(train_data,test_data,meta_data,use_std=False))
neg_data_index.to_csv("dist_info/{}.csv".format(data_name))
# neg_data_index = pd.read_csv("dist_info/{}.csv".format(data_name),index_col=0).iloc[:,0]
neg_data = test_data[neg_data_index.argsort()][-test_number:]
pos_data_index = list(range(test_number))
pos_data = train_data[pos_data_index]

tmp_df = []

for name, syn_data in syn_data_lst:
    tmp_score = round(fbb(syn_data, pos_data, neg_data, meta_data),3)
    print(name, ":", tmp_score)
    # tmp_df.append([name,tmp_score])

# tmp_df = pd.DataFrame(tmp_df,columns=["name","s"])
# print(tmp_df.groupby("name").mean().round(3).astype(str) + " + " + tmp_df.groupby("name").std().round(3).astype(str))