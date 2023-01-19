import torch, json, copy
import pandas as pd
from .evaluate import _compute_for_distribution, compute_scores
from .evaluate_cluster import compute_cluster_scores
from .model_test import save_index
class BaseSynthesizer:
    """Base class for all default synthesizers of ``SDGym``."""

    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns=tuple(), ordinal_columns=tuple()):
        #추상함수
        pass

    def sample(self, samples):
        #추상함수
        pass

    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])

    def draw_distribution(self, number_of_sample_for_dist = None):
        if number_of_sample_for_dist == None:
            synthesized_data = self.sample(self.train.shape[0])
        else:
            synthesized_data = self.sample(number_of_sample_for_dist)
        if "meta2" in vars(self).keys():
            metrics = _compute_for_distribution(self.train, self.test, synthesized_data, self.meta2)
        else:
            metrics = _compute_for_distribution(self.train, self.test, synthesized_data, self.meta)
        return metrics
    
    def save_result_type1(self, train_data, test_data, meta, name):
        syn_data = self.sample(train_data.shape[0])
        score = compute_scores(train_data, test_data, syn_data, meta) 
        
        if "likelihood" not in meta["problem_type"]:
            cluster_score = compute_cluster_scores(train_data, test_data, syn_data, meta) 
            cluster_score["name"] = cluster_score["name"].apply(lambda x: x + "/" + "silhouette")
            cluster_score = cluster_score.set_index("name")
            result = pd.concat([score.mean(),cluster_score.iloc[:,0]],axis=0)
        else:
            result = score.mean()
        
        with open(self.save_loc + "/score_info/"+ self.data_name + "/" + self.test_name + '.json', 'w', encoding='utf-8') as f:
            json.dump({"columns" : list(result.index), "result":list(result.values)}, f, indent="\t")
        if name != "IdentitySynthesizer":
            torch.save({"name": name, "arg" : self.save_arg}, self.save_loc + "/save_model/"+ self.data_name + "/" + self.test_name + '.pth')
    
    def save_result_type2(self, i, track_score_dict, save_score_dict, every_model_dict):
        syn_data = self.sample(self.train.shape[0])
        score = compute_scores(self.train, self.test, syn_data, self.meta) 
        s = score.loc[0].index.to_list()

        for k in range(1, score.shape[1]):
            self.writer.add_scalar('average/'+s[k], score.iloc[:, k].mean(), i)
            for j in range(len(score)):
                self.writer.add_scalar(score['name'][j]+'/'+s[k], score.iloc[j, k], i)
        
        if "likelihood" not in self.meta["problem_type"]:
            cluster_score = compute_cluster_scores(self.train, self.test, syn_data, self.meta) 
            s = cluster_score.loc[0].index.to_list()
            for k in range(1, cluster_score.shape[1]):
                self.writer.add_scalar('average/'+s[k], cluster_score.iloc[:, k].mean(), i)
                for j in range(len(cluster_score)):
                    self.writer.add_scalar(cluster_score['name'][j]+'/'+s[k], cluster_score.iloc[j, k], i)
            
            cluster_score["name"] = cluster_score["name"].apply(lambda x: x + "/" + "silhouette")
            cluster_score = cluster_score.set_index("name")
            result = pd.concat([score.mean(),cluster_score.iloc[:,0]],axis=0)
        else:
            result = score.mean()

        save_index_lst = save_index(self.data_name, score.mean(), track_score_dict, self.meta["problem_type"])
        if len(save_index_lst) != 0:
            model_state = {}
            if "generator" in vars(self):
                model_state["generator"] = copy.deepcopy(self.generator.state_dict())
            if "decoder" in vars(self):
                model_state["decoder"] = copy.deepcopy(self.decoder.state_dict())

            save_score_dict["columns"] = list(result.index)
            for each_index in save_index_lst:
                every_model_dict["model"][each_index] = model_state
                save_score_dict[each_index] = list(result.values)
            torch.save(every_model_dict, self.save_loc + "/save_model/"+ self.data_name + "/" + self.test_name + '.pth')
            with open(self.save_loc + "/score_info/"+ self.data_name + "/" + self.test_name + '.json', 'w', encoding='utf-8') as f:
                json.dump(save_score_dict, f, indent="\t")

