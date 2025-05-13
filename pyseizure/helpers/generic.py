import os
import re
import mne
import pandas as pd
from typing import List, Dict

from pyseizure.data_classes.montage import Montage
from pyseizure.helpers.dataset import Dataset


class Generic(Dataset):
    def __init__(self, root_dir: str, montage: Montage = Montage.BIPOLAR,
                 seiz_label: str = 'seiz', rereference: bool = True):
        super().__init__(root_dir)
        self.samples: List = []
        self.montage = montage
        self.rereference = rereference
        self.seiz_label = seiz_label
        self.channels = [('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'),
                         ('T5', 'O1'), ('FP1', 'F3'), ('F3', 'C3'),
                         ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'),
                         ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
                         ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'),
                         ('T6', 'O2'), ('FZ', 'CZ'), ('CZ', 'PZ')]
        if montage != Montage.BIPOLAR and not self.rereference:
            self.channels = ['FP1', 'F7', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4',
                             'P4', 'F8', 'FZ', 'CZ', 'O1', 'O2', 'PZ', 'T3',
                             'T4', 'T5', 'T6']

    def traverse_data(self) -> None:
        """
        Traverse data.

        Search recursively a directory for EDF.

        Returns
        -------
        None
        """
        samples = []
        for root, dirs, files in os.walk(self.root_dir):
            if root == self.root_dir:
                continue
            for file in files:
                if file.lower().endswith('edf'):
                    patient = re.findall(r'\d+', file)[0]
                    edf_file = os.path.join(root, file)
                    data = mne.io.read_raw_edf(edf_file,
                                               preload=False,
                                               verbose='WARNING')
                    seizures = len(data.annotations) > 0
                    data.close()
                    del data

                    samples.append({
                        'patient': patient,
                        'session': file[:-4],
                        'montage_type': self.montage.value,
                        'edf': edf_file,
                        'length': self.get_edf_length(edf_file),
                        'annotations': ('',
                                        self.channels,
                                        self._get_montages(),
                                        seizures)
                    })
        self.samples = samples

    @staticmethod
    def get_edf_length(file: str) -> int:
        data = mne.io.read_raw_edf(file, preload=False, verbose='WARNING')
        length = data.n_times
        data.close()

        return length

    def _get_montages(self) -> List[Dict]:
        # TODO: unify naming
        if self.rereference or self.montage == Montage.BIPOLAR:
            montage = [{
                'montage': self.montage.value,
                'channel_no': i,
                'ref1_channel': x[0],
                'ref2_channel': x[1],
                'channel': f'{x[0]}-{x[1]}'
            } for i, x in enumerate(self.channels)]
        else:
            montage = [{
                'montage': self.montage.value,
                'channel_no': i,
                'ref1_channel': x,
                'ref2_channel': '',
                'channel': x
            } for i, x in enumerate(self.channels)]

        return montage

    def _get_edf_info(self, summary: pd.DataFrame, file: str) -> Dict:
        summary = summary.dropna(subset=['Title'])
        return {
            'name': file,
            'start': summary.iloc[0, 0],
            'end': summary.iloc[-1, 0],
            'seizures': summary.loc[
                summary['Title'].str.contains(rf'seizure|{self.seiz_label}',
                                              regex=True)].shape[0]
        }
