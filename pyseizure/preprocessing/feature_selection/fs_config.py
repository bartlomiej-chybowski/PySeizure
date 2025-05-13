import sqlite3
import numpy as np
import pandas as pd
from typing import List

from pyseizure.helpers.normaliser import Normaliser
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector

VERBOSITY = 2


class FSConfig(FeatureSelector):
    def __init__(self, train_samples: List[str], columns: List[str] = [],
                 normaliser: Normaliser = None):
        super().__init__(train_samples, 0)
        self.name = 'generic'
        self.col_names = columns
        self.columns = []
        self.normaliser = normaliser
        self.database = self.open()

    def open(self):
        return {
            'db': sqlite3.connect(self.train_samples, check_same_thread=False),
        }

    def reopen(self):
        self.database = self.open()

    def fit(self):
        """
        Fit feature selection model.

        Returns
        -------
        None
        """
        cur = self.database['db'].cursor()
        columns = cur.execute("SELECT name FROM columns ").fetchall()
        columns = [x[0] for x in columns]

        score_cols = 4 if 'brain_state' in self.col_names else 3
        columns = columns[:-score_cols]
        self.close()
        if self.normaliser is not None:
            columns = self.normaliser.get_none_inf_cols()

        self.columns = np.argwhere(pd.Index(columns).get_indexer(
            self.col_names[:-score_cols]) != -1).flatten()

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
        return data[:, self.columns]

    def get_ranking(self) -> np.array:
        """
        Get ranking of features.

        Returns
        -------
        numpy.array
        """
        score_cols = 3
        if 'brain_state' in self.col_names:
            score_cols = 4

        return np.ones(len(self.col_names))[:-score_cols]

    def get_selected(self) -> np.array:
        """
        Get selected features.

        Returns
        -------
        numpy.array
        """
        return np.array(self.col_names)[self.columns]
