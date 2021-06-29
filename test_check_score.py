import torch, argparse, os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
pd.set_option('display.max_rows', None)
arg_dic = { event_accumulator.COMPRESSED_HISTOGRAMS: 500, event_accumulator.IMAGES: 4, event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0, event_accumulator.HISTOGRAMS: 1,}
func = event_accumulator.EventAccumulator

def extract_score(data_name):
    score_dic = {"adult" : "f1","news" : "mean_squared_error"}
    cluster_info = {"adult" : "KMeans2", "news" : "KMeans4"}
    score_lst = []
    for i in sorted(os.listdir("last_result/score_info/{}/".format(data_name))):
        score_dict = torch.load("last_result/score_info/{}/{}".format(data_name, i))
        if "result" in score_dict:
            score_series = score_dict["result"]
        else:
            score_series = score_dict[score_dic[data_name]]
            loc = "last_result/runs/{}/{}/".format(data_name,i[:-5])
            ea = func(loc + os.listdir(loc)[-1],arg_dic)
            ea.Reload()
            score = ea.Scalars(cluster_info[data_name] + "/silhouette")[int(score_series["iter"])].value
            score_series[cluster_info[data_name] + "/silhouette"] = score
        score_series.name = i[:-5]
        score_series[["error" in i for i in score_series.index]] = -score_series[["error" in i for i in score_series.index]]
        score_lst.append(score_series)
    result = pd.concat(score_lst, axis=1)
    result = result.drop("iter")
    result = result.drop([i for i in result.index if "silhouette" in i and cluster_info[data_name] not in i])
    result.rename(index= {cluster_info[data_name] + "/silhouette" : "silhouette"},inplace=True)
    return result.T


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Check All Score')
    parser.add_argument('--data', type=str, default = 'adult', choices=["adult", "news"])
    arg_of_parser = parser.parse_args()
    print(extract_score(arg_of_parser.data))

