import os
import re
import mne
import wfdb
import shutil
from typing import List, Tuple, Dict
from pyseizure.data_classes.montage import Montage
from pyseizure.helpers.dataset import Dataset


class CHBMIT(Dataset):
    def __init__(self, root_dir: str, embed_annotations: bool = False):
        super().__init__(root_dir)
        self.samples: List = []
        self.embed_annotations = embed_annotations

        self._channels = [('FP1', 'F7'), ('F7', 'T7'), ('T7', 'P7'),
                          ('P7', 'O1'), ('FP1', 'F3'), ('F3', 'C3'),
                          ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'),
                          ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
                          ('FP2', 'F8'), ('F8', 'T8'), ('T8', 'P8'),
                          ('P8', 'O2'), ('FZ', 'CZ'), ('CZ', 'PZ')]
        self.channel_pairs = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP1-F3',
                              'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                              'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T4', 'T4-T6',
                              'T6-O2', 'FZ-CZ', 'CZ-PZ']

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
            if root == self.root_dir:
                continue
            patient = root.split("/")[-1]
            if ('train' in patient or 'eval' in patient or 'dev' in patient
                    or 'test' in patient):
                continue
            with open(os.path.join(root, f"{patient}-summary.txt"), "r") as f:
                summary = f.read().splitlines()
            for file in files:
                if file.lower().endswith('edf'):
                    info = self._get_info(summary, file)
                    if len(info[0]) == 0:
                        continue
                    edf_file = os.path.join(root, file)
                    annotation = None
                    if info[1]['seizures'] > 0:
                        annotation = f'{edf_file}.seizure'
                        if self.embed_annotations:
                            self._add_annotation(edf_file)
                    # TODO: unify annotations
                    samples.append({
                        'patient': patient,
                        'session': file.split("_")[1][:-4],
                        'montage_type': Montage.BIPOLAR.value,
                        'edf': edf_file,
                        'length': self.get_edf_length(edf_file),
                        'annotations': (annotation, None,
                                        info[0],
                                        bool(info[1]['seizures']))
                    })
        self.samples = samples

    def _get_info(self, summary: List, file: str) -> Tuple:
        montages = self._get_montages(summary, file)
        edf_info = self._get_edf_info(summary, file)

        return montages, edf_info

    @staticmethod
    def get_edf_length(file: str) -> int:
        data = mne.io.read_raw_edf(file, preload=False, verbose='WARNING')
        length = data.n_times
        data.close()

        return length

    def _get_montages(self, summary: List, file: str) -> List[Dict]:
        try:
            file_line = summary.index(f'File Name: {file}')
        except ValueError:
            file_line = len(summary)
        montage_line = [i for i, x in enumerate(summary[:file_line])
                        if 'Channels' in x][-1]
        montages_list = [re.split(r': |-', x.strip())
                         for x in summary[montage_line:file_line]
                         if 'Channel ' in x]
        montages_list = [x for x in montages_list
                         if len(x) == 3 and (x[1], x[2]) in self._channels]

        return list(
            {
                montage['channel']: montage
                for montage in [{
                    'channel': f'{x[1]}-{x[2]}'.replace('T7', 'T3')
                                               .replace('T8', 'T4')
                                               .replace('P7', 'T5')
                                               .replace('P8', 'T6'),
                    'ref1_channel': x[1],
                    'ref2_channel': x[2],
                    'channel_no': int(x[0].split(" ")[1])
                } for x in montages_list]
            }.values())

    @staticmethod
    def _get_edf_info(summary: List, file: str) -> Dict:
        try:
            index = summary.index(f"File Name: {file}")
        except ValueError:
            return {'name': file, 'start': '', 'end': '', 'seizures': 0}
        if "Number of Seizures in File" in summary[index + 1]:
            return {
                'name': summary[index].split(': ')[1].strip(),
                'start': '',
                'end': '',
                'seizures': int(summary[index + 1].split(': ')[1].strip()),
            }
        return {
            'name': summary[index].split(': ')[1].strip(),
            'start': summary[index + 1].split(': ')[1].strip(),
            'end': summary[index + 2].split(': ')[1].strip(),
            'seizures': int(summary[index + 3].split(': ')[1].strip()),
        }

    @staticmethod
    def _add_annotation(edf_file: str):
        shutil.copy2(edf_file, f"{edf_file}_original")
        raw = mne.io.read_raw_edf(edf_file, verbose='WARNING')
        annotations = wfdb.rdann(edf_file, extension='seizures')
        seizures = annotations.sample / annotations.fs
        raw.set_annotations(
            mne.Annotations(onset=seizures[::2],
                            duration=seizures[1::2] - seizures[::2],
                            description=['seiz']*len(seizures[::2])))
        mne.export.export_raw(f"{edf_file}", raw, 'edf', overwrite=True)
        raw.close()
