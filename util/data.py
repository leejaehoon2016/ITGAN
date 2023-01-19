import json
import os
import urllib.request

import numpy as np
import pandas as pd

from .constants import CATEGORICAL, ORDINAL, CONTINUOUS


import warnings
warnings.filterwarnings(action='ignore')

BASE_URL = 'http://sdgym.s3.amazonaws.com/datasets/' 
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        urllib.request.urlretrieve(BASE_URL + filename, local_path)

    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_dataset(name, benchmark=False):
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']

    if benchmark:
        return train, data['test'], meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns


# for new data use this
def get_metadata(data, categorical_columns=tuple(), ordinal_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                    "name": index,
                    "type": CATEGORICAL,
                    "size": len(mapper),
                    "i2s": mapper
                })

            elif index in ordinal_columns:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
                meta.append({
                    "name": index,
                    "type": ORDINAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            else:
                meta.append({
                    "name": index,
                    "type": CONTINUOUS,
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

# if __name__ == "__main__":
    # import numpy as np
    # train, test, meta, categorical_columns, ordinal_columns = load_dataset("shoppers", benchmark=True)
    # for i in (categorical_columns + ordinal_columns)[1:-3]:
        # train[:,i] = train[:,i] - 1
        # test[:,i] = test[:,i] - 1
        # print(np.unique(train[:,i]))
    # np.savez("util/data/shoppers.npz", train = train, test = test)

    