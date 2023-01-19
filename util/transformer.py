import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
import torch
from .constants import CATEGORICAL, CONTINUOUS, ORDINAL


class Transformer:
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError


class BGMTransformer(Transformer):
    """Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.

    Discrete and ordinal columns are converted to a one-hot vector.
    """

    def __init__(self, meta, n_clusters=10, eps=0.005, random_seed=777):
        """n_cluster is the upper bound of modes."""
        self.meta = meta['columns']
        self.random_seed = random_seed
        self.n_clusters = n_clusters
        self.eps = eps

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        model = []

        self.output_info = []
        self.output_dim = 0
        self.components = []
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                gm = BayesianGaussianMixture(
                    n_components = self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001,
                    n_init=1,
                    random_state = self.random_seed)
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                comp = gm.weights_ > self.eps
                self.components.append(comp)

                self.output_info += [(1, 'tanh'), (np.sum(comp), 'softmax')]
                self.output_dim += 1 + np.sum(comp)
            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']

        self.model = model

    def transform(self, data):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == CONTINUOUS:
                current = current.reshape([-1, 1])

                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (4 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))

                n_opts = sum(self.components[id_])
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(data), dtype='int')
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99)

                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                values += [features, probs_onehot]
            else:
                col_t = np.zeros([len(data), info['size']])
                idx = current.astype(int)
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas=None):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                u = data[:, st]
                v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]

                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = v_t
                st += 1 + np.sum(self.components[id_])
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t + mean_t
                data_t[:, id_] = tmp

            else:
                current = data[:, st:st + info['size']]
                st += info['size']
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = idx
        return data_t

    def inverse_transform_torch(self, data, sigmas=None):
        data_t = torch.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                u = data[:, st]
                v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]

                if sigmas is not None:
                    sig = sigmas[st]
                    u = torch.random.normal(u, sig)

                u = torch.clamp(u, -1, 1)
                v_t = torch.ones((data.shape[0], self.n_clusters)).cuda() * -100
                v_t[:, self.components[id_]] = v
                v = v_t
                st += 1 + np.sum(self.components[id_])
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t + mean_t
                data_t[:, id_] = tmp

            else:
                current = data[:, st:st + info['size']]
                st += info['size']
                idx = torch.argmax(current, axis=1)
                data_t[:, id_] = idx
        return data_t