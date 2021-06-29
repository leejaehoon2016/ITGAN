from token import MINUS
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
measure = {
    # "adult" : ['average/accuracy', 'average/f1', 'average/roc_auc', 'average/distance', 'average/EMD', 'average/silhouette'],
    "king" : ['average/r2', 'average/explained_variance', 'average/mean_squared_error', 'average/mean_absolute_error', 'KMeans2/silhouette'],
    "airbnb" : ['average/r2', 'average/explained_variance', 'average/mean_squared_error', 'average/mean_absolute_error', 'KMeans4/silhouette'],
    # "census" : ['average/accuracy', 'average/f1', 'average/roc_auc', 'average/distance', 'average/EMD', 'average/silhouette'],
    # "covtype" : ['average/accuracy', 'average/macro_f1', 'average/micro_f1', 'average/roc_auc', 'average/distance', 'average/EMD', 'average/silhouette'],
    "cabs" : ['average/accuracy', 'average/macro_f1', 'average/micro_f1', 'average/roc_auc', 'KMeans3/silhouette'],
    # "intrusion" : ['average/accuracy', 'average/macro_f1', 'average/micro_f1', 'average/roc_auc', 'average/distance', 'average/EMD', 'average/silhouette'],
    # "news" : ['average/r2', 'average/explained_variance', 'average/mean_squared_error', 'average/mean_absolute_error', 'average/distance', 'average/EMD', 'average/silhouette'],
    "merchandise" : ['average/accuracy', 'average/macro_f1', 'average/micro_f1', 'average/roc_auc', 'KMeans4/silhouette'],
    "credit2": ['average/accuracy', 'average/macro_f1', 'average/micro_f1', 'average/roc_auc', 'KMeans3/silhouette'],

}

def choose_best_base(data, stan = "stan"):
    score = pd.read_csv("score_info_csv_before/" + data + ".csv")[measure[data] + dist + ["model","epoch"]]
    score["stan"] = score[measure[data]].sum(axis = 1)
    if type(stan) is list:
        score["stan"] = score[stan].sum(axis = 1)
        stan = "stan"
    score = score.groupby("model").apply(lambda x: x.sort_values(by=stan, ascending=False)[:1])
    score["model"] = score["model"].apply(lambda x : "_".join(x.split("_")))
    score = score.reset_index(drop=True)
    score = score[score["epoch"] >= 10]
    mean = score.groupby("model").mean().round(2)
    score = mean.sort_values(stan, ascending = False)
    score = score.loc[[i for i in score.index if i.split("_")[0] in ["tvae", "ctgan","veegan","medgan","tablegan"]]]
    return list(score.index),list(score["epoch"])


###########################################################################
from util.model_test import fix_random_seed

test_number = 1000
parser = argparse.ArgumentParser('Check FBB Score')
parser.add_argument('--data', type=str)
# parser.add_argument('--GPU_NUM', type=int, default = 0)
arg_of_parser = parser.parse_args()

data_name = arg_of_parser.data
print(data_name)
GPU_NUM = 0
# score_name_dict = 
# score_name = "stan" if data_name =="king" else ""

train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(data_name, benchmark=True)
syn_data_lst=[]
################# base
# for name, epoch in choose_best_base("merchandise","average/macro_f1"):
#     model = model_load("last_result/save_model/{}/{}.pth".format(data_name,name),GPU_NUM,int(epoch))
#     syndata = (name, model.sample(len(train_data)))
#     syn_data_lst.append(syndata)


for name in os.listdir(f"last_result/save_model/{data_name}"):
    if name.split("_")[0] in ["privbn","simblenddiv1"]:
        continue
    tmp = float(name.split("_")[-2])
    if tmp != 0.0:
        continue
        
    print(name)
    path = f"last_result/save_model/{data_name}/{name}"
    model = torch.load(path)
    keys = model["model"].keys()
    scores = sorted([i for i in keys if type(i) is int],reverse=True)[1:4]
    print(scores)

    # model = model_load(path,GPU_NUM,"best_0")

    # tmp = float(name.split("_")[-2])
    if tmp == 0.0:
        name = "iGAN"
    elif tmp > 0:
        name = "iGAN(Q)"
    else:
        name = "iGAN(L)"

    # syndata = [name, model.sample(len(train_data))]
    # syn_data_lst.append(syndata)

    if name == "iGAN":
        for i in [1]:#range(3):
            model = model_load(path,GPU_NUM,scores[i])
            fix_random_seed(777)
            syn_data_lst.append([f"sub_opt{i}", model.sample(len(train_data))])

neg_data_index = pd.read_csv("dist_info/{}.csv".format(data_name),index_col=0).iloc[:,0]
neg_data = test_data[neg_data_index.argsort()][-test_number:]
pos_data_index = list(range(test_number))
pos_data = train_data[pos_data_index]

tmp_df = []

for name, syn_data in syn_data_lst:
    tmp_score = round(fbb(syn_data, pos_data, neg_data, meta_data),3)
    print(name, ":", tmp_score)
    tmp_df.append([name,tmp_score])

tmp_df = pd.DataFrame(tmp_df,columns=["name","s"])
print(tmp_df.groupby("name").mean().round(3).astype(str) + " + " + tmp_df.groupby("name").std().round(3).astype(str))



############3
    

# for 
# train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(data_name, benchmark=True)
# syn_data_lst=[]

# df = choose_best(data_name)
# for name, epoch in zip(list(df.index), list(df["epoch"])):
#     model = model_load("last_result/save_model/{}/{}.pth".format(data_name,name),GPU_NUM,int(epoch))
#     print(name)
#     tmp = float(name.split("_")[-3])
#     if tmp == 0.0:
#         name = "iGAN"
#     elif tmp > 0:
#         name = "iGAN(Q)"
#     else:
#         name = "iGAN(L)"
    
#     syndata = (name, model.sample(len(train_data)))
#     syn_data_lst.append(syndata)

# print("end")

# for name, score in zip(list(df.index), list(df["average/" + score_name])):
#     tmp = float(name.split("_")[-3])
#     if tmp != 0.0:
#         continue
#     check_num = 0
#     minus = 0
#     score = int(score * 100)
#     # print(score)
#     while check_num < 3:
#     # for minus in [0.01,0.02,0.03]:
#         try:
#             minus += 1
#             target_score = score - minus
#             print(name, target_score)
#             score_df = pd.read_csv("score_info_csv/" + data_name + ".csv")[measure[data_name] + dist + ["model","epoch"]]
#             score_df = score_df[score_df["model"] == name]
#             print(len(score_df))
#             score_df = score_df[(score_df["average/mean_squared_error"] * 100).astype(int) == target_score]
#             name , epoch = score_df.loc[score_df["average/distance"].idxmax(),["model","epoch"]]
#             model = model_load("last_result/save_model/{}/{}.pth".format(data_name,name),GPU_NUM,int(epoch))
#             name_each = f"subopt_{check_num}"
#             syndata = (name_each, model.sample(len(train_data)))
#             check_num += 1
#             syn_data_lst.append(syndata)
#         except:
#             pass
    
    
# # neg_data_index = pd.Series(_compute_distance(train_data,test_data,meta_data,use_std=False))
# # neg_data_index.to_csv("dist_info/{}.csv".format(data_name))
# neg_data_index = pd.read_csv("dist_info/{}.csv".format(data_name),index_col=0).iloc[:,0]
# neg_data = test_data[neg_data_index.argsort()][-test_number:]
# pos_data_index = list(range(test_number))
# pos_data = train_data[pos_data_index]

# tmp_df = []

# for name, syn_data in syn_data_lst:
#     tmp_score = round(fbb(syn_data, pos_data, neg_data, meta_data),3)
#     print(name, ":", tmp_score)
#     tmp_df.append([name,tmp_score])

# tmp_df = pd.DataFrame(tmp_df,columns=["name","s"])
# print(tmp_df.groupby("name").mean().round(3).astype(str) + " + " + tmp_df.groupby("name").std().round(3).astype(str))




# # if "igan.json" in model_lst:
# #     best_score = {
# #         "adult" :  64,
# #         "news" :  -74,
# #         "king" : 
# #     }

# #     loc = "last_result/runs/{}/igan/".format(data_name)
# #     ea = func(loc + os.listdir(loc)[-1],arg_dic)
# #     ea.Reload()
# #     score = [int(i.value * 100) for i in ea.Scalars("average/" + score_dic[data_name] )]
    
# #     dist = [i.value for i in ea.Scalars("average/distance")]
# #     df = pd.DataFrame(list(zip(score,dist)),columns = ["score","dist"])
# #     df["epoch"] = list(range(len(df)))
# #     df = df.loc[df.groupby("score")["dist"].idxmax()]
# #     df = df[(df["score"] >= best_score[data_name] -3) & (df["score"] < best_score[data_name])][["score","epoch"]].values[::-1]
# #     syn_data_lst = []
# #     for s, epoch in df:
# #         model = model_load("last_result/save_model/{}/igan.pth".format(data_name),GPU_NUM,int(epoch))
# #         syndata = ("igan_subopt_{}".format(int(s) - best_score[data_name]), model.sample(len(train_data)))
# #         syn_data_lst.append(syndata)
    
# #     for name, syn_data in syn_data_lst:
# #         tmp_score = round(fbb(syn_data, pos_data, neg_data, meta_data),3)
# #         print(name, ":", tmp_score)
