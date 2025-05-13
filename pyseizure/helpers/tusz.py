import os
import re
import mne
import numpy as np
import pandas as pd
import multiprocessing as mp
from ast import literal_eval
from typing import List, Tuple, Dict
from multiprocessing.pool import ThreadPool
from pyseizure.data_classes.montage import Montage
from pyseizure.helpers.dataset import Dataset


class TUSZ(Dataset):
    def __init__(self, root_dir: str, montage_def: Dict[str, str],
                 symbols: str):
        super().__init__(root_dir)
        self.symbols: List[str] = self._get_symbols(symbols)
        self.montage_def: Dict[str, List[Dict]] = self._get_montages(
            montage_def)
        self.samples: List = []

        # self.channels = ['FP1', 'F7', 'T7', 'P7', 'F3', 'C3', 'P3', 'FP2',
        #                  'F4', 'C4', 'P4', 'F8', 'T8', 'P8', 'FZ', 'CZ',
        #                  'O1', 'O2', 'PZ', 'T3', 'T4', 'T5', 'T6']
        #
        self.channel_pairs = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP1-F3',
                              'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                              'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T4', 'T4-T6',
                              'T6-O2', 'FZ-CZ', 'CZ-PZ']
        # TUSZ has no [('FZ', 'CZ'), ('CZ', 'PZ')]

    @staticmethod
    def _get_montages(montage_def: Dict[str, str]) -> Dict[str, List]:
        """
        Get montages from the file.

        Parameters
        ----------
        montage_def: Dict
            montage type and path to file pairs

        Returns
        -------
        Dict
            a dict with type list of montages pairs
        """
        def get_channels(definitions: List) -> np.array:
            channels = pd.DataFrame([x['channel'] for x in definitions])
            channels = pd.DataFrame(np.repeat(channels.values, 2, axis=0))
            channels = np.where(channels.index % 2,
                                channels[0].str.split('-', expand=True)[1],
                                channels[0].str.split('-', expand=True)[0])
            _, idx = np.unique(channels, return_index=True)

            return channels[np.sort(idx)]

        def extract_montage_channels(source: str) -> List:
            regex = re.compile(
                r'\s[A-Z]{1,3}\d?-[A-Z]{1,3}\d?|EEG\s[A-Z]{1,3}\d?-(?:REF|LE)')
            return [y.strip() for y in regex.findall(source)]

        montages = {}
        for key, value in montage_def.items():
            montage = []
            with open(value, 'r') as f:
                data = f.read().splitlines()
                for x in data:
                    if x.startswith('montage = '):
                        montage_channels = extract_montage_channels(x)
                        if len(montage_channels) == 3:
                            montage.append({
                                'channel': montage_channels[0],
                                'ref1_channel': montage_channels[1],
                                'ref2_channel': montage_channels[2]
                            })
                f.close()
                del data
            montages[key] = montage, get_channels(montage)
        return montages

    @staticmethod
    def _get_symbols(symbols: str) -> List[str]:
        """
        Get seizure types from the file.

        Parameters
        ----------
        symbols: str
            file path

        Returns
        -------
        List[str]
            a list of seizure types
        """
        with open(symbols, 'r') as f:
            data = f.read().splitlines()
            symbols = list(literal_eval(
                re.sub(r"([a-z_]+)",
                       r'"\1"',
                       re.split('=', next(
                           x for x in data if x.startswith('symbols')))[1])
            ).values())
            f.close()
            del data

        return symbols

    def traverse_data(self) -> None:
        """
        Traverse data.

        Search recursively a directory for EDF and annotations files.

        Returns
        -------
        None
        """
        samples = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith('edf'):
                    sample = re.split("_", file)
                    file_name_no_ext = os.path.join(root, file)[:-4]
                    edf_file = os.path.join(root, file)
                    samples.append({
                        'patient': sample[0],
                        'session': sample[1],
                        'recording': sample[2][:-4],
                        'montage_type': Montage(root.split('/')[-1][7:]).value,
                        'edf': edf_file,
                        'length': self.get_edf_length(edf_file),
                        'annotations': f'{file_name_no_ext}.csv'
                    })
        self.samples = samples

    @staticmethod
    def get_edf_length(file: str) -> int:
        data = mne.io.read_raw_edf(file, preload=False, verbose='WARNING')
        length = data.n_times
        data.close()

        return length

    def decode_annotations(self) -> None:
        """
        Decode annotations.

        This function extracts annotations from multichannel and binary
        annotations files. If necessary completes annotations.

        results are saved as object attribute `samples`

        Returns
        -------
        None
        """
        with ThreadPool(processes=mp.cpu_count()) as mp_pool:
            self.samples = mp_pool.map(self._decode_annotations, self.samples)

    def _decode_annotations(self, obj: Dict):
        obj['annotations'] = self._decode_annotation(obj)
        return obj

    def _decode_annotation(self, annotation: Dict
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, List, bool]:
        """
        Decode multichannel and binary annotation.

        Parameters
        ----------
        annotation: Dict
            a dictionary with file paths

        Returns
        -------
        Tuple[pandas.DataFrame, pandas.DataFrame, List, bool]
            a tuple with annotations for multichannel, binary, a list of
            montages, and a boolean indicating seizure existence
        """
        # get stop time
        with open(annotation['annotations']) as f:
            lines = f.readlines()
        stop_time = re.findall(r"\d+.\d+",
                               next(x for x in lines if 'duration' in x))[0]

        # read csvs as dataframe
        df = pd.read_csv(annotation['annotations'], comment='#')
        df_bi = pd.read_csv(f"{annotation['annotations']}_bi", comment='#')

        # complete binary
        df_bi = self._complete_binary(df_bi, float(stop_time))

        # complete multi-channel
        df = df.loc[df['channel'].isin(self.channel_pairs)]
        for pair in self.channel_pairs:
            if pair not in df['channel'].values:
                df = pd.concat([df, pd.DataFrame(data=[{
                    'channel': pair,
                    'start_time': 0.0,
                    'stop_time': float(stop_time),
                    'label': 'bckg',
                    'confidence': 1. if pair not in ['FZ-CZ', 'CZ-PZ'] else 0.
                }])], ignore_index=True)
        completed_df = pd.DataFrame()
        for name, group in df.groupby('channel'):
            completed_df = pd.concat([completed_df,
                                      self._complete(group, float(stop_time))])
        completed_df = completed_df.astype({
            'channel': 'str',
            'start_time': 'float64',
            'stop_time': 'float64',
            'label': 'str',
            'confidence': 'float64'
        })
        df_bi = df_bi.astype({
            'channel': 'str',
            'start_time': 'float64',
            'stop_time': 'float64',
            'label': 'str',
            'confidence': 'float64'
        })

        return (completed_df.reset_index(drop=True).drop_duplicates(),
                df_bi.reset_index(drop=True),
                self.montage_def[annotation['montage_type']][0],
                bool(df.loc[df['label'] != 'bckg']['label'].any()))

    @staticmethod
    def _complete_binary(df_bi: pd.DataFrame,
                         stop_time: float) -> pd.DataFrame:
        """
        Complete binary annotations with background times.

        Binary definitions include only annotations of seizure, background
        time is missing

        Parameters
        ----------
        df_bi: pandas.DataFrame
            dataframe with binary annotations
        stop_time: float
            stop time for the sample

        Returns
        -------
        pandas.DataFrame
            a dataframe with background times
        """
        if df_bi.iloc[0]['start_time'] != 0.0:
            df_bi.loc[-1] = ['TERM', 0.0, df_bi.iloc[0]['start_time'], 'bckg',
                             1.0]
            df_bi.index += 1
            df_bi = df_bi.sort_index().reset_index(drop=True)

        if df_bi.iloc[-1]['stop_time'] != stop_time:
            df_bi.loc[-1] = ['TERM', df_bi.iloc[-1]['stop_time'], stop_time,
                             'bckg', 1.0]
            df_bi.index += 1
            df_bi = df_bi.reset_index(drop=True)

        df_bi = TUSZ._add_in_between(df_bi, 'TERM')

        return df_bi

    @staticmethod
    def _complete(df: pd.DataFrame, stop_time: float) -> pd.DataFrame:
        """
        Complete annotations with background times.

        Binary definitions include only annotations of seizure, background
        time is missing

        Parameters
        ----------
        df: pandas.DataFrame
            dataframe with binary annotations
        stop_time: float
            stop time for the sample

        Returns
        -------
        pandas.DataFrame
            a dataframe with background times
        """
        df = df.astype({
            'channel': 'str',
            'start_time': 'float64',
            'stop_time': 'float64',
            'label': 'str',
            'confidence': 'float64'
        })
        df = df.reset_index(drop=True).drop_duplicates()
        df = df.sort_values(['start_time']).reset_index(drop=True)
        channel = df.iloc[0]['channel']
        # add starting background
        if df.iloc[0]['start_time'] != 0.0:
            df.loc[-1] = [channel, 0.0, df.iloc[0]['start_time'], 'bckg', 1.0]
            df.index += 1
            df = df.sort_index().reset_index(drop=True)

        # add ending background
        if df.iloc[-1]['stop_time'] != stop_time:
            df.loc[-1] = [channel, df.iloc[-1]['stop_time'], stop_time, 'bckg',
                          1.0]
            df.index += 1
            df = df.reset_index(drop=True)

        # add in-between background
        if len(df) < 2:
            return df
        df = TUSZ._add_in_between(df, channel)

        # remove overlapping background
        for idx in range(len(df) * 2):
            if idx >= len(df):
                break
            if idx <= 1:
                continue
            if df.loc[idx - 1, 'start_time'] == df.loc[idx, 'start_time']:
                if df.loc[idx - 1, 'label'] == 'bckg':
                    df = df.drop(index=[idx-1]).reset_index(drop=True)
                if df.loc[idx, 'label'] == 'bckg':
                    df = df.drop(index=[idx]).reset_index(drop=True)
            if float(df.loc[idx - 1, 'stop_time']) > \
                    float(df.loc[idx, 'start_time']):
                if df.loc[idx - 1, 'label'] == 'bckg':
                    df.loc[idx - 1, 'stop_time'] = df.loc[idx, 'start_time']

        # add in-between background
        df = TUSZ._add_in_between(df.reset_index(drop=True), channel)

        return df

    @staticmethod
    def _add_in_between(df, channel):
        for index in range(2, len(df) * 2):
            df = df.sort_index().reset_index(drop=True)
            if index > len(df):
                break
            index -= 1
            if float(df.loc[index - 1, 'stop_time']) < \
                    float(df.loc[index, 'start_time']):
                df.loc[index - 0.5] = [channel,
                                       df.loc[index - 1, 'stop_time'],
                                       df.loc[index, 'start_time'],
                                       'bckg',
                                       1.0]

        df = df.sort_values(['start_time', 'stop_time'])

        return df.reset_index(drop=True)
