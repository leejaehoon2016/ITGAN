import json
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation, MeanShift, Birch
from sklearn.preprocessing import OneHotEncoder

from .constants import CATEGORICAL, CONTINUOUS, ORDINAL

import warnings
warnings.filterwarnings(action='ignore')

LOGGER = logging.getLogger(__name__)
K_MEANS_PP = 'k-means++'

EVAL_MODELS = [
    {
        'class': KMeans,
        'kwargs': {
            'n_clusters': 2,
            'init': K_MEANS_PP
        }
    },
    {
        'class': Birch,
        'kwargs': {
            'n_clusters': 5,

        }

    },
    {
        'class': AgglomerativeClustering,
        'kwargs': {
            'n_clusters': 5,
        }
    }
]

_MODELS = [
    {
        'class': KMeans,
        'kwargs': {
            'n_clusters': 1,
            'init': K_MEANS_PP
        }
    },
    {
        'class': KMeans,
        'kwargs': {
            'n_clusters': 2,
            'init': K_MEANS_PP
        }
    },
    {
        'class': KMeans,
        'kwargs': {
            'n_clusters': 3,
            'init': K_MEANS_PP
        }
    },
]



class FeatureMaker:
    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features

def _prepare_cluster_problem(train, test, metadata, evaluate):
    fm = FeatureMaker(metadata)
    train = fm.make_features(train)
    test = fm.make_features(test)
    if not evaluate:
        return train, test, _MODELS
    return  train, test, EVAL_MODELS


def _evaluate_cluster(train, test, metadata, evaluate):
    train, test, classifiers = _prepare_cluster_problem(train, test, metadata, evaluate)

    performance = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict()).copy()
        if 'n_clusters' in model_kwargs:
            if "classification" in metadata["problem_type"]:
                num_of_cluster = [i["size"] for i in metadata["columns"] if i['name'] == "label"][0] * model_kwargs['n_clusters']
            else:
                num_of_cluster = max(len(metadata["columns"]) // 12, 2) * model_kwargs['n_clusters']
                
            model_kwargs['n_clusters'] =  num_of_cluster
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)
        train = train.astype(np.float)
        test =  test.astype(np.float)
        if model_repr == "KMeans":
            model_repr = "KMeans" + str(num_of_cluster)
            model.fit(train)
            predicted_label = model.predict(test)
        else:
            predicted_label = model.fit_predict(np.concatenate([train,test],axis=0))
            predicted_label = predicted_label[len(train):]
        try:
            score = silhouette_score(test, predicted_label, metric='euclidean', sample_size=100)
        except MemoryError:
            score = 0
        
        performance.append(
            {
                "name": model_repr,
                "silhouette": score
            }
        )

    return pd.DataFrame(performance)



def compute_cluster_scores(train, test, synthesized_data, metadata, evaluate = False):
    score = _evaluate_cluster(synthesized_data, train[:len(synthesized_data)], metadata, evaluate)
    return score

