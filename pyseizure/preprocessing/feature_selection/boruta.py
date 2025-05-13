import time
import logging
import sqlite3
import numpy as np
import pandas as pd
from ujson import loads
from random import sample
from typing import List, Dict, Union
from pyseizure.helpers.boruta_py import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from pyseizure.data_classes.dataset import Dataset
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector

VERBOSITY = 2


class Boruta(FeatureSelector):
    def __init__(self, train_samples: List[str], chunk: int = None,
                 weight: Union[str, Dict] = 'balanced', max_iter: int = 100,
                 columns: List[int] = [], subsampling: bool = True,
                 labels: List[str] = [], normaliser: Normaliser = None,
                 dataset: Dataset = None):
        super().__init__(train_samples, chunk)
        self.name = 'boruta'
        self.columns = columns
        self.normaliser = normaliser
        self.labels = labels
        self.subsampling = subsampling
        self.max_iter = max_iter
        self.dataset = dataset
        self.database = self.open()
        # TODO: make dynamic
        self.use_scoring = True
        self.keys = []
        if weight == 'auto':
            self.weight = self._get_weights()
        else:
            self.weight = weight

        cur = self.database['db'].cursor()
        columns = cur.execute("SELECT name FROM columns ").fetchall()
        self.all_columns = [x[0] for x in columns]

        self.selector = BorutaPy(
            RandomForestClassifier(n_jobs=-1, class_weight=self.weight,
                                   max_depth=5),
            n_estimators='auto',
            perc=99,
            verbose=VERBOSITY,
            random_state=42,
            alpha=0.05,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=10)

    def __del__(self):
        self.close()

    def close(self):
        # self.database['db'].close()
        # self.database['json_db'].close()
        pass

    def open(self):
        return {
            'db': sqlite3.connect(self.train_samples, check_same_thread=False),
        }

    def reopen(self):
        self.database = self.open()

    def _get_keys(self):
        start = time.time()

        cur = self.database['db'].cursor()
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

        keys_list = []

        keys_list.extend(seiz_keys)
        keys_list.extend(sample(bckg_keys, k=len(seiz_keys)))
        keys_list.sort()

        logging.info(f"Get dataset keys run time {(time.time() - start):0.3f}")

        return keys_list

    def _get_weights(self):
        self.keys = self._get_keys()
        weight = {
            'bckg': 1,
            'seiz': 1
        }

        return weight

    def fit(self):
        keys = str(self.keys)[1:-1]
        cur = self.database['db'].cursor()
        tmp = cur.execute(f'SELECT vals FROM data '
                          f'WHERE id IN ({keys})').fetchall()
        tmp = np.array([loads(x[0]) for x in tmp])

        df = pd.DataFrame(data=tmp, columns=self.all_columns)

        del tmp
        df = df.astype({c: "float64" for c in df.columns[:-1]}
                       ).replace([np.nan, np.inf, -np.inf], 0)
        score_cols = 4 if 'brain_state' in df.columns.values else 3

        if len(self.columns) > 0:
            df_x = df.iloc[:, self.columns]
        else:
            df_x = df.iloc[:, :-score_cols]
        df_x = df_x.replace('nan', np.nan).dropna()
        if self.normaliser:
            df_x = self.normaliser.transform(df_x)
        x = np.array(df_x, dtype='f')
        df_y = df.iloc[:, -score_cols:]
        del df
        del df_x

        try:
            x[:, np.argwhere(np.isnan(x).all(axis=0)).flatten()] = 0
        except ValueError:
            logging.debug('no nans')
        try:
            x[:, np.argwhere(np.isinf(x).all(axis=0)).flatten()] = 0
        except ValueError:
            logging.debug('no infs')

        self.selector.fit(x, df_y.iloc[:, -1].values.ravel())

        del df_y
        del x

    def transform(self, data: np.array) -> np.array:
        """
        Transform data to include only selected features.

        Parameters
        ----------
        data: numpy.array

        Returns
        -------
        numpy.array
        """
        if 0 < len(self.columns) < len(data.flatten()):
            data = data[:, self.columns]

        return self.selector.transform(data)

    def get_ranking(self) -> np.array:
        """
        Get ranking of features.

        Returns
        -------
        numpy.array
        """
        return self.selector.ranking_

    def get_selected(self) -> np.array:
        """
        Get selected features.

        Returns
        -------
        numpy.array
        """
        score_cols = 4 if 'brain_state' in self.all_columns else 3
        support_ = self.selector.support_
        support = []
        for i in range(0, len(self.columns)):
            if i == len(support_):
                break
            support.append(self.columns[i] if support_[i] else np.nan)
        support = np.array(support)
        support = support[~np.isnan(support)]
        normalised = []
        for i in range(0, len(self.all_columns[:-score_cols])):
            normalised.append(True if i in support else False)
        normalised = np.array(normalised)
        del support
        del support_

        return normalised
