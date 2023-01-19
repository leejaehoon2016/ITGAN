import json
import logging

import numpy as np
import pandas as pd
from pomegranate import BayesianNetwork
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from scipy.stats import wasserstein_distance
from xgboost import XGBClassifier

from .constants import CATEGORICAL, CONTINUOUS, ORDINAL
from .full_black_box import fbb
import warnings
warnings.filterwarnings(action='ignore')

LOGGER = logging.getLogger(__name__)


_MODELS = {
    'binary_classification': [
        {
            'class': XGBClassifier,
            'kwargs': {
                'max_depth': 3,
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr' # 'auc'

            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50
            },
        }
    ],
    'multiclass_classification': [
        {
            'class': XGBClassifier,
            'kwargs': {
                'max_depth': 3,
                'eval_metric': 'multi:softprob'

            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50
            },
        }
    ],
    'multiclass_classification2' : [
        {
            'class': XGBClassifier,
            'kwargs': {
                'max_depth': 3,
                'eval_metric': 'multi:softprob'
            }
        },
        {
            'class': RandomForestClassifier,
            'kwargs': {
                'n_estimators': 50,
            },
        }
    ],
    'regression': [
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50
            },
        }
    ],

        'regression2': [
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50,
                'learning_rate_init' : 0.1,
            },
        }
    ]
}


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
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

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

        return features, labels


def _prepare_ml_problem(train, test, metadata):
    
    fm = FeatureMaker(metadata)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)

    return x_train, y_train, x_test, y_test, _MODELS[metadata['problem_type']]


def _evaluate_multi_classification(train, test, metadata):

    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)

    performance = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using multiclass classifier %s', model_repr)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
            pred_prob = np.array([1.] * len(x_test))
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_prob = model.predict_proba(x_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        macro_precision = precision_score(y_test, pred, average='macro')
        micro_precision = precision_score(y_test, pred, average='micro')

        macro_recall = recall_score(y_test, pred, average='macro')
        micro_recall = recall_score(y_test, pred, average='micro')

        size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
        rest_label = set(range(size)) - set(unique_labels)
        tmp = []
        j = 0
        for i in range(size):
            if i in rest_label:
                tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
            else:
                try:
                    tmp.append(pred_prob[:,[j]])
                except ValueError as e:
                    print(str(e))
                    tmp.append(pred_prob[:, np.newaxis])
                j += 1
        roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))



        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "macro_precision": macro_precision,
                "micro_precision": micro_precision,
                "macro_recall": macro_recall,
                "micro_recall": micro_recall,
                "roc_auc": roc_auc
            }
        )

    return pd.DataFrame(performance)


def _evaluate_binary_classification(train, test, metadata):
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)

    performance = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using binary classifier %s', model_repr)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
            pred_prob = np.array([1.] * len(x_test))
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_prob = model.predict_proba(x_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro')

        precision = precision_score(y_test, pred, average='binary')
        recall = recall_score(y_test, pred, average='binary')

        size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
        rest_label = set(range(size)) - set(unique_labels)
        tmp = []
        j = 0
        for i in range(size):
            if i in rest_label:
                tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
            else:
                try:
                    tmp.append(pred_prob[:,[j]])
                except ValueError as e:
                    print(str(e))
                    tmp.append(pred_prob[:, np.newaxis])
                j += 1
        roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))


        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc
            }
        )

    return pd.DataFrame(performance)


def _evaluate_regression(train, test, metadata):
    x_train, y_train, x_test, y_test, regressors = _prepare_ml_problem(train, test, metadata)

    performance = []
    if metadata["problem_type"] == "regression":
        y_train = np.log(np.clip(y_train, 1, 20000))
        y_test = np.log(np.clip(y_test, 1, 20000))
    else:
        y_train = np.log(np.clip(y_train, 1, None))
        y_test = np.log(np.clip(y_test, 1, None))
    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using regressor %s', model_repr)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        r2 = r2_score(y_test, pred)
        if r2 < -1:
            r2 = -1.
        explained_variance = explained_variance_score(y_test, pred)
        mean_squared = -mean_squared_error(y_test, pred)
        mean_absolute = -mean_absolute_error(y_test, pred)



        performance.append(
            {
                "name": model_repr,
                "r2": r2,
                "explained_variance" : explained_variance,
                "mean_squared_error" : mean_squared,
                "mean_absolute_error" : mean_absolute
            }
        )

    return pd.DataFrame(performance)


def _evaluate_gmm_likelihood(train, test, metadata, components=[10, 30]):
    results = list()
    for n_components in components:
        gmm = GaussianMixture(n_components, covariance_type='diag')
        LOGGER.info('Evaluating using %s', gmm)
        gmm.fit(test)
        l1 = gmm.score(train)

        gmm.fit(train)
        l2 = gmm.score(test)

        results.append({
            "name": repr(gmm),
            "syn_likelihood": l1,
            "test_likelihood": l2,
        })

    return pd.DataFrame(results)


def _mapper(data, metadata):
    data_t = []
    for row in data:
        row_t = []
        for id_, info in enumerate(metadata['columns']):
            row_t.append(info['i2s'][int(row[id_])])

        data_t.append(row_t)

    return data_t


def _evaluate_bayesian_likelihood(train, test, metadata):
    LOGGER.info('Evaluating using Bayesian Likelihood.')
    structure_json = json.dumps(metadata['structure'])
    bn1 = BayesianNetwork.from_json(structure_json)

    train_mapped = _mapper(train, metadata)
    test_mapped = _mapper(test, metadata)
    prob = []
    for item in train_mapped:
        try:
            prob.append(bn1.probability(item))
        except Exception:
            prob.append(1e-8)

    l1 = np.mean(np.log(np.asarray(prob) + 1e-8))

    bn2 = BayesianNetwork.from_structure(train_mapped, bn1.structure)
    prob = []

    for item in test_mapped:
        try:
            prob.append(bn2.probability(item))
        except Exception:
            prob.append(1e-8)

    l2 = np.mean(np.log(np.asarray(prob) + 1e-8))
    return pd.DataFrame([{
        "name": "Bayesian Likelihood",
        "syn_likelihood": l1,
        "test_likelihood": l2,
    }])

def _compute_distance(train, syn, metadata, sample=300):
    mask_d = np.zeros(len(metadata['columns']))

    for id_, info in enumerate(metadata['columns']):
        if info['type'] in [CATEGORICAL, ORDINAL]:
            mask_d[id_] = 1
        else:
            mask_d[id_] = 0

    std = np.std(train, axis=0) + 1e-6

    dis_all = []
    for i in range(min(sample, len(train))):
        current = syn[i]
        distance_d = np.abs((train - current) * mask_d) > 1e-6
        distance_d = np.sum(distance_d, axis=1)

        distance_c = (train - current) * (1 - mask_d) / 2 / std
        distance_c = np.sum(distance_c ** 2, axis=1)
        distance = np.sqrt(np.min(distance_c + distance_d))
        dis_all.append(distance)

    return np.mean(dis_all)


_EVALUATORS = {
    'bayesian_likelihood': _evaluate_bayesian_likelihood,
    'binary_classification': _evaluate_binary_classification,
    'gaussian_likelihood': _evaluate_gmm_likelihood,
    'multiclass_classification': _evaluate_multi_classification,
    'multiclass_classification2': _evaluate_multi_classification,
    'multiclass_classification_cov': _evaluate_multi_classification,
    'regression': _evaluate_regression,
    'regression2': _evaluate_regression,
}

def _compute_EMD(train,syn):
    col = train.shape[1]
    result = []
    for i in range(col):
        result.append(wasserstein_distance(train[:,i], syn[:,i]))
    return sum(result) / len(result)

def change_data_for_dist(data, meta):
    values =[]
    for id_, info in enumerate(meta['columns']):
        if info['type'] == "categorical":
            current = np.zeros([len(data), info['size']])
            idx = data[:, id_].astype(int)
            current[np.arange(len(data)), idx] = 1
            values.append(current)
        else:
            values += [data[:, id_].reshape([-1, 1])]

    return np.concatenate(values, axis=1)

def _compute_for_distribution(train, test, syn, metadata, components=[1, 3]):
    mask_d = np.zeros(len(metadata['columns']))

    for id_, info in enumerate(metadata['columns']):
        if info['type'] in [CATEGORICAL, ORDINAL]:
            mask_d[id_] = 1
        else:
            mask_d[id_] = 0

    std = np.std(train, axis=0) + 1e-6

    dis_all = []
    for i in range(len(syn)):
        current = syn[i]
        distance_d = np.abs((train - current) * mask_d) > 1e-6
        distance_d = np.sum(distance_d, axis=1)

        distance_c = (train - current) * (1 - mask_d) / 2 / std
        distance_c = np.sum(distance_c ** 2, axis=1)
        distance = np.sqrt(np.min(distance_c + distance_d))
        dis_all.append(distance)
    dis_all = pd.Series(dis_all)
    return dis_all


def compute_scores(train, test, synthesized_data, metadata):
    evaluator = _EVALUATORS[metadata['problem_type']]
    scores = evaluator(synthesized_data, test, metadata)
    scores['distance'] = _compute_distance(train, synthesized_data, metadata)
    train_change, test_change, synthesized_change = change_data_for_dist(train, metadata), change_data_for_dist(test, metadata), change_data_for_dist(synthesized_data, metadata)
    scores['EMD'] = _compute_EMD(train_change, synthesized_change)
    return scores

