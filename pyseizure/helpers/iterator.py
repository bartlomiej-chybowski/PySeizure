import gc
import logging
import sqlite3
import time
from ujson import loads
import numpy as np
import pandas as pd
import xgboost as xgb
from random import shuffle, sample
import pyarrow.parquet as pq
from typing import Callable, Tuple, List
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import WeightedRandomSampler
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector


class Iterator(xgb.DataIter):
    def __init__(self, filename: List[str], label_encoder: LabelEncoder,
                 feature_selector: FeatureSelector = None,
                 normaliser: Normaliser = None, subsampling: bool = True,
                 batch_size: int = None, evaluation: bool = False,
                 use_scoring: bool = True, shuffle: bool = True,
                 binary: bool = True, artefacts: bool = False):
        super().__init__()
        self._filename = filename
        self._it = 0
        self._label_encoder = label_encoder
        self.feature_selector = feature_selector
        self.features_names: List = []
        self.normaliser = normaliser
        self.use_scoring = use_scoring
        self.subsampling = subsampling
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.binary = binary
        self.artefacts = artefacts
        self.samples = []
        self.database = sqlite3.connect(filename, check_same_thread=False)
        self.weights = 0.5
        self.base_score = 0.5

        cur = self.database.cursor()
        columns = cur.execute("SELECT name FROM columns ").fetchall()
        self.all_columns = [x[0] for x in columns]
        del columns

        self.evaluation = evaluation

        self.keys = self.get_keys()

        self.samples = list(range(1, self.length + 1))
        if self.batch_size is not None:
            self.length = self.length // self.batch_size

    def get_keys(self, store: bool = True):
        if self.subsampling:
            keys = self._get_keys()
        else:
            self.length = self._calculate_length()
            keys = self.keys
        if store:
            self.keys = keys

        return keys

    def set_length(self, length: int):
        self.length = length
        self.samples = list(range(1, self.length + 1))
        if self.batch_size is not None:
            self.length = self.length // self.batch_size

    def set_keys(self, keys):
        self.keys = keys

    def _calculate_length(self):
        cur = self.database.cursor()
        keys = cur.execute('SELECT id FROM data').fetchall()
        self.keys = [x[0] for x in keys]

        return len(self.keys)

    def _get_keys(self):
        start = time.time()

        cur = self.database.cursor()
        scoring = ''
        if self.use_scoring:
            scoring = 'AND score_a > 0.8 AND score_b > 0.8'
        seiz_keys = cur.execute(f"SELECT id FROM data "
                                f"WHERE seizure = 1 "
                                f"{scoring}").fetchall()
        seiz_keys = [x[0] for x in seiz_keys]
        bckg_keys = cur.execute(f"SELECT id FROM data "
                                f"WHERE seizure = 0 "
                                f"AND artefact = 0 "
                                f"{scoring}").fetchall()
        bckg_keys = [x[0] for x in bckg_keys]
        bckg_keys = sample(bckg_keys, k=len(seiz_keys))

        keys_list = []
        keys_list.extend(seiz_keys)
        keys_list.extend(bckg_keys)
        keys_list.sort()
        self.length = len(keys_list)

        if self.shuffle:
            shuffle(keys_list)

        logging.debug(f"\nGet {self._filename} keys. "
                      f"Run time {(time.time() - start):0.3f}, "
                      f"Background keys: {len(bckg_keys)}, "
                      f"Seizure keys: {len(seiz_keys)}.")
        del bckg_keys
        del seiz_keys

        return keys_list

    def load_file(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load file with data. Split dependent and independent features.
        Assumption: dependent variable is on last position

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str]]
            independent features, dependent features, feature names
        """
        sample = pq.read_table(self._file_paths[self._it]).to_pandas()
        sample = sample.astype({c: "float64" for c in sample.columns[:-1]}
                               ).replace([np.nan, np.inf, -np.inf], 0)
        # sampling - balance data
        if self.subsampling:
            n = 40
            i = np.flatnonzero(sample['class'] != 'bckg') + 1
            sample = pd.concat([s.tail(n + 1) for s in np.split(sample, i)])
        # sampling - balance data

        score_cols = 3
        if 'brain_state' in sample.columns:
            score_cols = 4
        y = self._label_encoder.transform(sample.iloc[:, -1])
        y = np.concatenate([y.reshape(-1, 1),
                            sample.iloc[:, -score_cols:-1].values],
                           axis=1)
        x = sample.iloc[:, :-score_cols]
        del sample

        if self.normaliser:
            x = self.normaliser.transform(x)
        self.features_names = list(x.columns.values)

        if self.feature_selector:
            self.features_names = list(
                x.iloc[:, self.feature_selector.columns[:-score_cols]]
                 .loc[:, self.feature_selector.get_selected()].columns)
            x = self.feature_selector.transform(x.values)
        x = np.array(x, dtype=np.float32)

        try:
            x[:, np.argwhere(np.isnan(x).all(axis=0))] = 0
        except ValueError:
            logging.debug('no nans')

        assert x.shape[0] == y.shape[0]
        gc.collect()

        x = np.array(x, dtype=np.float32)
        if self.shuffle:
            np.random.shuffle(x)

        return x, y, self.features_names

    def next(self, input_data: Callable) -> int:
        score_cols = 4 if 'brain_state' in self.all_columns else 3
        index = self._it + 1

        if index > self.length:
            return 0

        range_start = (index - 1) * self.batch_size + 1
        range_stop = range_start + self.batch_size

        keys = str(self.keys[range_start:range_stop])[1:-1]
        cur = self.database.cursor()
        data = cur.execute(f'SELECT vals FROM data '
                           f'WHERE id IN ({keys})').fetchall()
        data = np.array([loads(x[0]) for x in data])
        sample = pd.DataFrame(data=data, columns=self.all_columns)

        sample = sample.replace([np.nan, np.inf, -np.inf], 0)

        if not self.subsampling and self.binary and self.artefacts:
            sample.loc[sample.iloc[:, -1] == 'x_artefact', 'class'] = 'bckg'

        y = self._label_encoder.transform(sample.iloc[:, -1])
        y = np.concatenate([y.reshape(-1, 1),
                            sample.iloc[:, -score_cols:-1].values], axis=1)
        sample = sample.iloc[:, :-score_cols]

        # TODO: probably f'd up order; change order of fs and norm
        self.features_names = list(sample.columns.values)
        if self.feature_selector:
            self.features_names = list(
                 sample.loc[:, self.feature_selector.get_selected()].columns)
            x = self.feature_selector.transform(sample.values)
        else:
            x = sample
        del sample

        if self.normaliser:
            x = self.normaliser.transform(
                pd.DataFrame(data=x, columns=self.features_names))
            self.feature_names = list(x.columns)
            x = np.array(x.values, dtype=np.float32)

        x = np.array(x, dtype=np.float32)
        if self.evaluation:
            input_data(data=x, feature_names=self.features_names)
        else:
            input_data(data=x, label=y[:, 0],
                       feature_names=self.features_names)
        self._it += 1
        del x

        return 1

    def next_old(self, input_data: Callable) -> int:
        """
        Advance the iterator by 1 step and pass the data to XGBoost.
        This function is called by XGBoost during the construction of DMatrix.

        Parameters
        ----------
        input_data: Callable
            a function passed in by XGBoost who has the exact same signature
            of DMatrix

        Returns
        -------
        int
            a flag: 0 - end of iteration, 1 - more files to process
        """
        index = self._it + 1
        batch_size = self.batch_size if self.batch_size is not None else 1

        # x, y, features = self.load_file()
        # input_data(data=x, label=y[:, 0], feature_names=features)
        if self.batch_size is not None:
            if index == self.length:
                return 0
            tmp_data = []
            range_start = (index - 1) * batch_size + 1
            range_stop = range_start + batch_size
            for key in self.samples[range_start:range_stop]:
                tmp_data.append(self.database['json_db'][str(key)]['values'])
            sample = pd.DataFrame(tmp_data, columns=self.all_columns)
            del tmp_data
        else:
            if index == self.length:
                return 0
            sample = pd.DataFrame(
                data=[self.database['json_db'][str(index)]['values']],
                columns=self.all_columns)

        score_cols = 4 if 'brain_state' in sample.columns.values else 3
        sample = sample.replace([np.nan, np.inf, -np.inf], 0)
        y = self._label_encoder.transform(sample.iloc[:, -1])
        y = np.concatenate([y.reshape(-1, 1),
                            sample.iloc[:, -score_cols:-1].values],
                           axis=1)
        x = sample.iloc[:, :-score_cols]
        del sample

        # self.features_names = list(x.columns.values)
        if self.feature_selector:
            self.features_names = list(
                 x.loc[:, self.feature_selector.get_selected()].columns)
            x = self.feature_selector.transform(x.values)
        if self.normaliser:
            x = self.normaliser.transform(
                pd.DataFrame(data=x, columns=self.features_names))

        x = np.array(x.values, dtype='f')
        if self.evaluation:
            input_data(data=x, feature_names=self.features_names)
        else:
            input_data(data=x, label=y[:, 0],
                       feature_names=self.features_names)
        self._it += 1
        del x

        return 1

    def __len__(self):
        return self.length

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0

    def get_weights_old(self, reversed_weights: bool = False):
        start = time.time()
        pos = 0
        neg = 0
        self.reset()
        for _ in self._file_paths:
            _, y, _ = self.load_file()
            self._it += 1
            pos += y[:, 0].sum()
            neg += len(y) - y[:, 0].sum()

        self.reset()
        end = time.time()

        logging.info(f"\nCases: positive: {pos}, negative: {neg}, "
                     f"sum: {pos + neg}, weight: {neg / pos}\n"
                     f"Elapsed time: {end - start}")

        if reversed_weights:
            return round(pos / neg, 3), round(neg / (neg + pos), 3)
        return round(neg / pos, 3), round(pos / (neg + pos), 3)

    def get_weights(self):
        global_start = time.time()
        pos, neg, y = 0, 0, []
        with self.database['db'].begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            keys = [x.decode("utf-8") for x in keys]
            keys.remove('columns')
            keys = list(map(int, keys))
            keys.sort()
        db = self.database['json_db']
        for key in keys:
            neg += 1 if db[str(key)]['values'][-1] == 'bckg' else 0
            pos += 1 if db[str(key)]['values'][-1] == 'seiz' else 0
            y.append(1 if db[str(key)]['values'][-1] == 'seiz' else 0)
        weight = np.array([round(pos / (neg + pos), 3),
                           round(neg / (neg + pos), 3)])
        logging.info(f"\nCases: negative: {neg}, positive: {pos}, "
                     f"sum: {pos + neg}, weights: {weight}")
        # calc weights for each sample (not class)
        sampler = WeightedRandomSampler(
           weight[np.array(y, dtype='int')], len(keys), True)
        del weight
        del y

        self.samples = [str(x + 1) for x in sampler]
        del sampler

        neg, pos = 0, 0
        for key in self.samples:
            neg += 1 if db[str(key)]['values'][-1] == 'bckg' else 0
            pos += 1 if db[str(key)]['values'][-1] == 'seiz' else 0

        self.weights = round(neg / pos, 3)
        self.base_score = round(pos / (neg + pos), 3)

        logging.info(f"\nSampler cases: negative: {neg}, positive: {pos}, "
                     f"sum: {pos + neg}, weights: {self.weights}\n"
                     f"Elapsed time: {time.time() - global_start}")

    def get_y_true(self):
        keys = str(self.keys)[1:-1]
        cur = self.database.cursor()
        tmp = cur.execute(f'SELECT seizure FROM data '
                          f'WHERE id IN ({keys})').fetchall()

        return np.array([x[0] for x in tmp], dtype='int')
