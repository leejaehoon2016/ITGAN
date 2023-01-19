import os,sys
import torch
import random
import numpy as np

func_dict = {
    "mean" : lambda x,y : (x+y)/2,
    "geo"  : lambda x,y : (x * y) ** 0.5,
    "harmony": lambda x,y : (2*x*y)/(x+y),
}

def mkdir(save_loc,data_name):
    if not os.path.isdir(save_loc+"/param/" + data_name):
        os.mkdir(save_loc+"/param/" + data_name)
    if not os.path.isdir(save_loc+"/runs/" + data_name):
        os.mkdir(save_loc+"/runs/" + data_name)
    if not os.path.isdir(save_loc+"/save_model/" + data_name):
        os.mkdir(save_loc+"/save_model/" + data_name)
    if not os.path.isdir(save_loc+"/score_info/" + data_name):
        os.mkdir(save_loc+"/score_info/" + data_name)

def fix_random_seed(random_num):
    torch.manual_seed(random_num)
    torch.cuda.manual_seed(random_num)
    torch.cuda.manual_seed_all(random_num) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_num)
    random.seed(random_num)

def model_save_dict(problem_type):
    best_model = {}
    if 'multiclass_classification' in problem_type:
        best_model["macro_f1"] = {"macro_f1":-float('inf')}
        best_model["roc_auc"]  = {"roc_auc":-float('inf')}
    elif 'classification' in problem_type:
        best_model["f1"] = {"f1":-float('inf')}
        best_model["roc_auc"]  = {"roc_auc":-float('inf')}
    elif 'likelihood' in problem_type:
        best_model["syn_likelihood"] = {"syn_likelihood":-float('inf')}
        best_model["test_likelihood"]  = {"test_likelihood":-float('inf')}
    else:
        best_model["mean_squared_error"] = {"mean_squared_error":-float('inf')}
        best_model["r2"] = {"r2" : -float('inf')}
    if 'likelihood' not in problem_type:
        best_model['silhouette'] = {'silhouette' : -float('inf')}
    best_model["geomean_dist_score"] = {"geomean":-float('inf')}
    best_model["mean"] = {"mean" : -float('inf')}
    best_model["geo"] = { "geo" : -float('inf')}
    best_model["harmony"] = { "harmony" : -float('inf')}
    return best_model

def save_index(data, score_dict, track_score, problem_type):
    score_stan = {
        "king" : ("mean_squared_error", -0.20),
        "credit2" : ("macro_f1", 0.31),
        "cabs" : ("macro_f1", 0.57)
        }
    if "binary_classification" in problem_type:
        score_stan[data] = ("f1", 0)
    elif "multiclass_classification" in problem_type:
        score_stan[data] = ("macro_f1", 0)
    else:
        score_stan[data] = ("mean_squared_error", -1000)

    result = []
    # do not satisfy standard score
    if score_dict[score_stan[data][0]] < score_stan[data][1]:
        return result

    # choose distance high
    distance_index = int(score_dict[score_stan[data][0]] * 100)
    if distance_index not in track_score or track_score[distance_index] < score_dict["distance"]:
        track_score[distance_index] = score_dict["distance"]
        result.append(distance_index)
    
    # Best Model
    stan = score_stan[data][0]
    my_score = score_dict[stan]
    name = "best"
    if name not in track_score or track_score[name] < my_score:
        track_score[name] = my_score
        result.append(name)
    return result

        

def model_score_save(problem_type, score_dict, best_model_dict, iteration, have_cluster):
    if 'multiclass_classification' in problem_type:
        s_name = ["macro_f1", "roc_auc", "silhouette"]
    elif 'classification' in problem_type:
        s_name = ["f1","roc_auc", "silhouette"]
    elif 'likelihood' in problem_type:
        s_name = ["test_likelihood", "syn_likelihood"]
    else:
        s_name = ["r2", "mean_squared_error", 'silhouette']
    
    mean_score = score_dict
    mean_score["iter"] = iteration
    change = False
    num = 3 if have_cluster else 2
    for each_name in s_name[:num]:
        if mean_score[each_name] > best_model_dict[each_name][each_name]:
            change = True
            best_model_dict[each_name] = mean_score
    
    

    if 'likelihood' not in problem_type:
        geo = (mean_score["distance"] * mean_score[s_name[0]]) ** 0.5
        if geo > best_model_dict["geomean_dist_score"]["geomean"]:
            change = True
            best_model_dict["geomean_dist_score"] = mean_score.copy()
            best_model_dict["geomean_dist_score"]["geomean"] = geo
        
        num = 3 if 'regression' not in problem_type else 2
        for each_func in ["mean", "harmony", "geo"][:num]:
            each_mean = func_dict[each_func](mean_score[s_name[0]], mean_score[s_name[1]])
            if each_mean > best_model_dict[each_func][each_func]:
                best_model_dict[each_func] = mean_score.copy()
                best_model_dict[each_func][each_func] = each_mean

        index_number = int(mean_score[s_name[0]] * 100)
        if index_number not in best_model_dict or mean_score["distance"] > best_model_dict[index_number]["distance"]:
            change = True
            best_model_dict[index_number] = mean_score

    return change, best_model_dict
