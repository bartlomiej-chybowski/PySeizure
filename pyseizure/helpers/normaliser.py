import gc
from itertools import repeat
from multiprocessing import Pool
from ujson import loads
import logging
import sqlite3
from typing import Union, List
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax.numpy as jnp


class Normaliser:
    def __init__(self, no_threads_per_core: int, raw: bool = False):
        self.no_threads_per_core = no_threads_per_core
        self.min_max: pd.DataFrame = pd.DataFrame(dtype='float64')
        self.raw = raw

    def fit(self, data: Union[pd.DataFrame, List[str], str]) -> None:
        if isinstance(data, list):
            self._fit_list(data)
        elif isinstance(data, str) and '.db' in data:
            self._fit_from_db(data)
        else:
            self._fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self, 'raw') and self.raw:
            columns = data.columns
            data = pd.DataFrame(jnp.apply_along_axis(
                lambda x: self._np_norm(
                    x,
                    jnp.array(self.min_max_t.iloc[:, 0]),
                    jnp.array(self.min_max_t.iloc[:, 1])
                ), 0, jnp.array(data.values)), columns=columns)
        else:
            data = data.astype('float64').apply(self._norm, axis=1)

        try:
            data = data.iloc[:, np.argwhere(
                ~np.isnan(data.values).all(axis=0)).flatten()]
        except ValueError:
            logging.debug('no nans')
        try:
            data = data.iloc[:, np.argwhere(
                ~np.isinf(data.values).all(axis=0)).flatten()]
        except ValueError:
            logging.debug('no infs')

        return data

    def get_none_inf_cols(self):
        return self.min_max.columns[(self.min_max.diff().iloc[1] != 0).values]

    def _fit_from_db(self, path: str):
        conn = sqlite3.connect(path, check_same_thread=False)
        cur = conn.cursor()
        keys = cur.execute('SELECT id FROM data').fetchall()
        columns = cur.execute("SELECT name FROM columns").fetchall()
        columns = [x[0] for x in columns]
        keys = [x[0] for x in keys]
        score_cols = 4 if 'brain_state' in columns else 3
        columns = columns[:-score_cols]
        if self.raw:
            data = cur.execute(f'SELECT vals FROM data '
                               f'WHERE id = {keys[0]}').fetchall()
            data = np.array([loads(x[0]) for x in data])
            columns = list(range(data.shape[1]))
            del data
        df = pd.DataFrame(dtype='float64', columns=columns)

        chunks = [keys[x:x + 5000] for x in range(0, len(keys), 5000)]
        with Pool(processes=self.no_threads_per_core) as mp_pool:
            min_max_list = mp_pool.starmap(self._fit_chunk, zip(
                chunks, repeat(path), repeat(columns), repeat(score_cols)))
        for min_max in min_max_list:
            df = pd.concat([df, min_max])

        means = df.iloc[2:-1:4]
        std = df.iloc[3::4]
        variance = (np.mean(std**2, axis=0) + np.var(means))
        std_corrected = np.sqrt(variance)

        self.min_max = pd.concat([
            df.iloc[:-3:4].agg([min]),
            df.iloc[1:-2:4].agg([max]),
            means.agg([np.mean]),
            std_corrected.to_frame().T
        ]).astype('float64')
        self.min_max.index = [
            'min',
            'max',
            'mean',
            'std'
        ]
        self.min_max_t = self.min_max.T
        conn.close()
        del df, columns, keys, score_cols

    def _fit_chunk(self, chunk, path, columns, score_cols):
        conn = sqlite3.connect(path, check_same_thread=False)
        cur = conn.cursor()

        data = cur.execute(f'SELECT vals FROM data '
                           f'WHERE id IN ({str(chunk)[1:-1]})').fetchall()
        data = np.array([loads(x[0]) for x in data])

        if len(data.shape) == 3:
            data = data[:, :, :-score_cols]
            data = np.transpose(data, (0, 2, 1))
            data = data.reshape(-1, data.shape[-1])
            columns = list(range(data.shape[-1]))
        else:
            data = data[:, :-score_cols]
        df_tmp = pd.DataFrame(data=data, columns=columns)
        df_tmp = df_tmp.astype('float64').replace(
            [np.inf, -np.inf, np.nan], 0)

        min_max = df_tmp.agg([
            min,
            max,
            # np.mean,
            # lambda x: x.std(ddof=0)
        ])
        del df_tmp
        del data

        return min_max

    def _fit_list(self, data: List[str]) -> None:
        df = pd.DataFrame(dtype='float64')
        for index, filename in enumerate(data):
            sample = pq.read_table(filename).to_pandas()
            score_columns = 3
            if 'brain_state' in sample.columns:
                score_columns = 4
            sample = sample.iloc[:, :-score_columns].astype('float64').replace(
                [np.inf, -np.inf], 0)
            min_max = sample.agg([min, max])
            df = pd.concat([df, min_max])
            del sample
            gc.collect()
        self.min_max = pd.concat([df.iloc[:-1:2].agg([min]),
                                  df.iloc[1::2].agg([max])]).astype('float64')

    def _fit(self, data: pd.DataFrame) -> None:
        score_columns = 3
        if 'brain_state' in data.columns:
            score_columns = 4
        sample = data.iloc[:, :-score_columns].astype('float64').replace(
            [np.inf, -np.inf], 0)
        self.min_max = sample.agg([min, max]).astype('float64')

    def _norm(self, row: pd.Series) -> pd.Series:
        return ((row - self.min_max.iloc[0]) /
                (self.min_max.iloc[1] - self.min_max.iloc[0]))

    @staticmethod
    def _np_norm(row, min_arr, max_arr) -> np.array:
        return (row - min_arr) / (max_arr - min_arr)
