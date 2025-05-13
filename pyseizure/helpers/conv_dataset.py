import time
import torch
import logging
import sqlite3
import numpy as np
import pandas as pd
from ujson import loads
from typing import List, Tuple
from random import sample
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector


class ConvDataset(Dataset):
    def __init__(self, filename: str, label_encoder: LabelEncoder,
                 feature_selector: FeatureSelector = None,
                 normaliser: Normaliser = None, batch_size: int = 32,
                 channels: int = 18, use_scoring: bool = True,
                 subsampling: bool = True,
                 binary: bool = True, artefacts: bool = False):
        self.filename = filename
        self.channels = channels
        self._label_encoder = label_encoder
        self.feature_selector = feature_selector
        self.features_names: List = []
        self.normaliser = normaliser
        self.subsampling = subsampling
        self.binary = binary
        self.artefacts = artefacts
        self.batch_size = batch_size
        self.use_scoring = use_scoring
        self.database = sqlite3.connect(filename, check_same_thread=False)

        cur = self.database.cursor()
        columns = cur.execute("SELECT name FROM columns ").fetchall()
        self.all_columns = [x[0] for x in columns]
        del columns

        self.keys = self.get_keys()

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

    def set_keys(self, keys):
        self.keys = keys

    # def __del__(self):
    #     pass

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

        logging.debug(f"\nGet {self.filename} keys. "
                      f"Run time {(time.time() - start):0.3f}, "
                      f"Background keys: {len(bckg_keys)}, "
                      f"Seizure keys: {len(seiz_keys)}.")
        del bckg_keys
        del seiz_keys

        return keys_list

    def _calculate_length(self):
        cur = self.database.cursor()
        keys = cur.execute('SELECT id FROM data').fetchall()

        self.keys = [x[0] for x in keys]

        return len(self.keys)

    def __len__(self) -> int:
        """
        Return number of files.

        Returns
        -------
        int
        """
        return self.length

    def __getitem__(self, index: int) -> Tuple:
        """
        Read and extract data and lables.

        Parameters
        ----------
        index: int
            index of the chunk

        Returns
        -------
        Tuple
            data and labels
        """
        cur = self.database.cursor()
        data = cur.execute(f'SELECT vals FROM data '
                           f'WHERE id = {self.keys[index]}').fetchone()
        sample = pd.DataFrame(data=loads(data[0]), columns=self.all_columns)
        score_cols = 4 if 'brain_state' in self.all_columns else 3
        sample = sample.replace([np.nan, np.inf, -np.inf], 0)

        if not self.subsampling and self.binary and self.artefacts:
            sample.loc[sample.iloc[:, -1] == 'x_artefact', 'class'] = 'bckg'

        y = self._label_encoder.transform(sample.iloc[:, -1])
        y = np.concatenate([y.reshape(-1, 1),
                            sample.iloc[:, -score_cols:-1].values], axis=1)
        x = sample.iloc[:, :-score_cols]
        del sample

        if self.normaliser:
            x = self.normaliser.transform(x)
        self.features_names = list(x.columns.values)

        x = np.array(x, dtype='f')
        # TODO: return scores and brain state
        return torch.tensor(x).squeeze(1), torch.tensor(np.average(y[:, 0]))
