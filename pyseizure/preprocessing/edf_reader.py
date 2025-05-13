import multiprocessing as mp
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import mne
import gc
import math
import joblib
import logging
from operator import itemgetter
from multiprocessing.pool import ThreadPool as Pool
from typing import List, Dict, Tuple
from mne.io.constants import FIFF
from resampy import resample
from scipy.signal import decimate, hilbert
from pyseizure.data_classes.montage import Montage
from pyseizure.data_classes.raw_sample import RawSample
from pyseizure.data_classes.resampling import Resampling
from pyseizure.helpers.helpers import notch_filter
from pyseizure.preprocessing.eeg_reader import EEGReader
from pyseizure.preprocessing.filters.current_source_density import \
    CurrentSourceDensity
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

BACKGROUND = 'bckg'


class EDFReader(EEGReader):
    def __init__(self,
                 epoch: int = 256,
                 frequency: int = 256,
                 external_annotation: bool = True,
                 annotation_labels: List = [],
                 single_channel: bool = False,
                 save_result: bool = True,
                 channels: List[str] = [],
                 channel_pairs: List[str] = [],
                 csd: bool = False,
                 montage: Montage = Montage.BIPOLAR,
                 data_augmentation: bool = True,
                 size_factor: float = 1.0,
                 rereference: bool = True,
                 generalised: bool = True,
                 seiz_label: str = 'seiz',
                 artefacts: bool = True):
        super().__init__(epoch, frequency)
        self.external_annotation = external_annotation
        self.annotation_labels = annotation_labels
        self.artefacts = artefacts
        self.csd = csd
        self.data_augmentation = data_augmentation
        self.single_channel = single_channel
        self.save_result = save_result
        self.size_factor = size_factor
        self.rereference = rereference
        self.generalised = generalised
        self.seiz_label = seiz_label
        if len(channels) > 0:
            self.channels = channels
        else:
            self.channels = ['T7', 'P7', 'T8', 'P8', 'FP1', 'F7', 'F3', 'C3',
                             'P3', 'FP2', 'F4', 'C4', 'P4', 'F8', 'FZ', 'CZ',
                             'O1', 'O2', 'PZ', 'T3', 'T4', 'T5', 'T6']
        self.montage = montage

        if len(channel_pairs) > 0:
            self.channel_pairs = channel_pairs
        else:
            # T3 is now T7; T4 is now T8; T5 is now P7; T6 is now P8
            self.channel_pairs = [('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'),
                                  ('T5', 'O1'), ('FP1', 'F3'), ('F3', 'C3'),
                                  ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'),
                                  ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
                                  ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'),
                                  ('T6', 'O2'), ('FZ', 'CZ'), ('CZ', 'PZ')]

    def read_signals(self, files=None) -> List:
        """
        Read signals.

        Update dictionary of files with raw EEG data, data info and channel
        names.

        Parameters
        ----------
        files: List[Dict]
            a list of files for single recording

        Returns
        -------
        None
        """
        if files is None:
            raise Exception('No files provided')
        with Pool(processes=mp.cpu_count()) as mp_pool:
            start = time.time()
            raw_results = mp_pool.map(self._read_raw_edf, files)
            raw_results = [x for x in raw_results if x is not None]
            logging.info(f"read_raw_edf {time.time() - start}s")

            start = time.time()
            results = mp_pool.map(self.prepare_samples, raw_results)
            # results = []
            logging.info(f"prepare samples {time.time() - start}s")

            if self.data_augmentation:
                start = time.time()
                results.extend(mp_pool.map(self.prepare_sign_flipped,
                                           raw_results))
                logging.info(f"prepare_sign_flipped {time.time() - start}s")

                start = time.time()
                results.extend(mp_pool.map(self.prepare_time_reversed,
                                           raw_results))
                logging.info(f"prepare_time_reversed {time.time() - start}s")

                start = time.time()
                results.extend(mp_pool.map(self.prepare_augmented,
                                           raw_results))
                logging.info(f"prepare_augmented {time.time() - start}s")
            del raw_results
        gc.collect()

        return results

    @staticmethod
    def _reverse_annotations(result: Dict) -> None:
        results = [
            result['annotations'][0].apply(
                lambda x: pd.Series(
                    [x.channel, result['length'] - x.stop_time,
                     result['length'] - x.start_time, x.label, x.confidence],
                    index=x.index), axis=1).sort_values('start_time',
                                                        ignore_index=True),
            result['annotations'][1].apply(
                lambda x: pd.Series(
                    [x.channel, result['length'] - x.stop_time,
                     result['length'] - x.start_time, x.label, x.confidence],
                    index=x.index), axis=1).sort_values('start_time',
                                                        ignore_index=True),
            result['annotations'][2],
            result['annotations'][3]
        ]

        if len(result['annotations']) == 5:
            results.append(result['annotations'][4].apply(
                lambda x: pd.Series(
                    [x.channel, result['length'] - x.stop_time,
                     result['length'] - x.start_time, x.label, x.confidence],
                    index=x.index), axis=1).sort_values('start_time',
                                                        ignore_index=True))

        result['annotations'] = tuple(results)

    def prepare_time_reversed(self, results: Dict) -> List:
        local = deepcopy(results)
        local['edf'] = local['edf'].replace('.edf', '_time_reversed.edf')
        local['raw_data'] = np.flip(local['raw_data'], axis=1)
        self._reverse_annotations(local)

        return self.prepare_samples(local)

    def prepare_sign_flipped(self, results: Dict) -> List:
        local = deepcopy(results)
        local['edf'] = local['edf'].replace('.edf', '_sign_flipped.edf')
        local['raw_data'] = local['raw_data'] * -1

        return self.prepare_samples(local)

    def prepare_augmented(self, results: Dict) -> List:
        local = deepcopy(results)
        local['edf'] = local['edf'].replace('.edf', '_augmented.edf')
        local['raw_data'] = np.flip(local['raw_data'] * -1, axis=1)
        self._reverse_annotations(local)

        return self.prepare_samples(local)

    def prepare_samples(self, result: Dict) -> List:
        """
        Prepare raw samples.

        Parameters
        ----------
        result

        Returns
        -------
        List
            list of RawSample objects
        """
        frequency = int(result['info']['sfreq'])
        eeg_signal = self.resample(result['raw_data'], frequency,
                                   self.frequency)
        samples = []

        for start in range(0, eeg_signal.shape[1], self.epoch):
            annotations = self._get_annotation_for_sample(
                result['annotations'], start, result['channels'])
            if annotations is None:
                continue  # if there is no annotation - sample is corrupted
            brain_state = self._get_brain_state_for_sample(
                result['annotations'], start, result['channels'])
            montages = result['annotations'][2]
            end = start + self.epoch
            if end > eeg_signal.shape[1]:
                end = eeg_signal.shape[1]

            signal = eeg_signal[:, start:end]
            # plt.figure(figsize=(20, 3))
            # plt.ylim((-0.06, 0.07))
            # plt.plot(signal)
            # plt.show()
            # Remove DC per epoch
            signal = signal - np.average(signal, axis=1).reshape(-1, 1)
            # plt.clf()
            # plt.figure(figsize=(20, 3))
            # plt.ylim((-0.06, 0.07))
            # plt.plot(signal)
            # plt.show()

            samples.append(RawSample(patient_id=result['patient'],
                                     signal=signal,
                                     channels=result['channels'],
                                     montages=montages,
                                     frequency=self.frequency,
                                     annotations=annotations,
                                     brain_state=brain_state,
                                     annotation_labels=self.annotation_labels,
                                     name=result['edf']))
            del signal
        if self.save_result:
            joblib.dump(samples,
                        f"{result['edf'][:-4]}_raw"
                        f"{'_sc' if self.single_channel else ''}.joblib",
                        compress=3)
        del eeg_signal
        gc.collect()

        return samples

    def _get_annotation_for_sample(self, annotations: Tuple, start: int,
                                   channels: np.array) -> np.array:
        """
        Get annotations for sample.

        Parameters
        ----------
        annotations: Tuple
            Annotations for multichannel/multiclass and binary
        start: int
            Start sample
        channels: numpy.array
            List of channels
        Returns
        -------
        numpy.array
        """
        start_seconds = int(start / self.frequency)
        if self.single_channel or self.generalised:
            labels = self._extract_labels_for_sample(
                annotations[1], start_seconds, ["TERM"])
            labels = list(labels.values())
        else:
            labels = self._extract_labels_for_sample(
                annotations[0], start_seconds, channels)
            labels = [value for channel in channels
                      for key, value in labels.items() if key in channel]

        logging.debug('_get_annotation_for_sample')

        # if len(next((x for x in labels if 'x_artefact' in x), [])):
        #     return None

        return np.array(labels)

    def _get_brain_state_for_sample(self, annotations: Tuple, start: int,
                                    channels: np.array) -> np.array:
        """
        Get brain state for sample.

        Parameters
        ----------
        annotations: Tuple
            Annotations for multichannel/multiclass and binary
        start: int
            Start sample
        channels: numpy.array
            List of channels
        Returns
        -------
        numpy.array
        """
        if len(annotations) < 5:
            return np.array([[-1]])

        def _extract_brain_state_for_sample(annotation, start_seconds,
                                            channels):
            brain_state = annotation.loc[
                (start_seconds >= np.floor(annotation['start_time'])) &
                (start_seconds + 1 <= np.ceil(annotation['stop_time']))]
            if len(brain_state) < len(channels):
                try:
                    label = int(brain_state.iloc[0,  3])
                except IndexError:
                    label = int(annotation.iloc[-1,  3])
                brain_state = pd.concat(
                    [brain_state, pd.DataFrame(data=[{
                        'channel': channel,
                        'start_time': start_seconds,
                        'stop_time': start_seconds + 1,
                        'label': label,
                        'confidence': 1.0
                    } for channel in channels
                        if channel not in brain_state['channel'].values])])
                brain_state.reset_index(inplace=True, drop=True)
            if len(brain_state) == 1:
                return {
                    brain_state.iloc[0, 0]: [
                        brain_state.iloc[0, 3]
                    ]
                }
            labels = {}
            for _, annotation in brain_state.iterrows():
                labels[annotation[0]] = [
                    self._set_time_proportions(annotation,
                                               start_seconds)[0].astype('int')]
            return labels

        start_seconds = int(start / self.frequency)
        if self.single_channel:
            labels = _extract_brain_state_for_sample(
                annotations[4], start_seconds, ["TERM"])
            labels = list(labels.values())
        else:
            labels = _extract_brain_state_for_sample(
                annotations[4], start_seconds, channels)
            labels = [value for channel in channels
                      for key, value in labels.items() if key in channel]

        logging.debug('_get_brain_state_for_sample')

        return np.array(labels)

    def _extract_labels_for_sample(self,
                                   annotations: pd.DataFrame,
                                   start_seconds: int,
                                   channels: np.array) -> Dict:
        """
        Extract labels for sample.

        Parameters
        ----------
        annotations: pandas.DataFrame
        start_seconds: int

        Returns
        -------
        Dict
        """
        channel_annotations = annotations.loc[
            (start_seconds >= np.floor(annotations['start_time'])) &
            (start_seconds + 1 <= np.ceil(annotations['stop_time']))]
        if len(channel_annotations) < len(channels):
            channel_annotations = pd.concat(
                [channel_annotations, pd.DataFrame(data=[{
                    'channel': channel,
                    'start_time': start_seconds,
                    'stop_time': start_seconds + 1,
                    'label': 'bckg',
                    'confidence': 1.0
                } for channel in channels
                    if channel not in channel_annotations['channel'].values])])
        if len(channel_annotations) == 1:
            return {
                channel_annotations.iloc[0, 0]: [
                    channel_annotations.iloc[0, 3]
                ]
            }
        channel_annotations = channel_annotations.groupby(
            channel_annotations['channel'])
        labels = {}
        for annotation in channel_annotations:
            labels[annotation[0]] = list(set(
                annotation[1].apply(
                    lambda x: self._set_time_proportions(x, start_seconds),
                    axis=1).itertuples(index=False, name=None)))

        # Return only one label per channel
        logging.debug('_extract_labels_for_sample')

        return self._get_single_labels(labels)

    @staticmethod
    def _get_single_labels(labels: Dict) -> Dict:
        """
        Get single labels.

        Parameters
        ----------
        labels: Dict

        Returns
        -------
        Dict
        """
        for key, value in labels.items():
            label = [x[0] for x in value
                     if x[1] == max(value, key=itemgetter(1))[1]]
            if len(label) > 1:
                if len(set([x[1] for x in label])) > 1:
                    label = [next(x for x in label if x != BACKGROUND)]
                else:
                    label = [max(value, key=itemgetter(1))[1]]
            labels[key] = label

        logging.debug('_get_single_labels')

        return labels

    @staticmethod
    def _set_time_proportions(annotation: pd.DataFrame,
                              start_seconds: int) -> List:
        """
        Set time proportions.

        Parameters
        ----------
        annotation: pandas.DataFrame
        start_seconds: int

        Returns
        -------
        List
        """
        proportion = 1
        if start_seconds >= annotation['start_time'] \
                and start_seconds + 1 <= annotation['stop_time']:
            proportion = 1
        elif start_seconds + 1 == math.ceil(annotation['stop_time']) \
                and start_seconds >= annotation['start_time']:
            proportion = annotation['stop_time'] - start_seconds
        elif start_seconds == math.floor(annotation['start_time']) \
                and start_seconds + 1 <= annotation['stop_time']:
            proportion = start_seconds + 1 - annotation['start_time']

        # annotation[5:] = [round(item * proportion, 4)
        #                   for item in annotation[5:]]

        logging.debug('_set_time_proportions')

        return pd.Series([annotation['label'],
                          round(annotation['confidence'] * proportion, 4)])

    @staticmethod
    def resample(raw_data: np.array, source_frequency: int,
                 destination_frequency: int,
                 method: Resampling = Resampling.RESAMPY) -> np.array:
        """
        Resample signals along axis.

        Parameters
        ----------
        raw_data: numpy.array
        source_frequency: int
        destination_frequency: int
        method: Resampling
            what method of resampling to use

        Returns
        -------
        numpy.array
        """
        if source_frequency == destination_frequency:
            return raw_data
        if method == Resampling.DECIMATE:
            downsamplink_factor = int(source_frequency / destination_frequency)
            return np.apply_along_axis(decimate,
                                       axis=1,
                                       arr=raw_data,
                                       q=downsamplink_factor)

        return np.apply_along_axis(resample, axis=1, arr=raw_data,
                                   sr_orig=source_frequency,
                                   sr_new=destination_frequency,
                                   parallel='True',
                                   filter='kaiser_best')  # 'sinc_window'

    def _smooth_amplitude(self, signal: np.array) -> np.array:
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)

        median_amplitude = np.median(amplitude_envelope)
        std_amplitude = np.std(amplitude_envelope)

        lower_threshold = median_amplitude - 5 * std_amplitude
        upper_threshold = median_amplitude + 5 * std_amplitude

        modified_signal = signal.copy()
        for i in range(len(signal)):
            if (amplitude_envelope[i] < lower_threshold or
                    amplitude_envelope[i] > upper_threshold):
                modified_signal[i] = signal[i] * (median_amplitude /
                                                  amplitude_envelope[i])

        # window_size = int(self.frequency * 2)
        # smoothed_amplitude = np.convolve(
        #     amplitude_envelope, np.ones(window_size) / window_size,
        #     mode='same')
        #
        # global_target_amplitude = np.median(smoothed_amplitude)
        # modified_signal = signal * (global_target_amplitude /
        #                             smoothed_amplitude)

        return modified_signal

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(signal, label='Raw Signal', alpha=0.7)
        plt.title('Raw Signal')
        plt.ylabel('Amplitude')
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(amplitude_envelope, label='Amplitude Envelope', alpha=0.7)
        plt.axhline(median_amplitude, color='orange', linestyle=':',
                    label='Median Amplitude')
        plt.axhline(lower_threshold, color='red', linestyle='--',
                    label='Lower Threshold (Median - 1 Std)')
        plt.axhline(upper_threshold, color='red', linestyle='--',
                    label='Upper Threshold (Median + 1 Std)')
        plt.title('Amplitude Envelope and Thresholds')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

        # plt.subplot(3, 1, 2)
        # plt.plot(amplitude_envelope, label='Amplitude Envelope', alpha=0.7)
        # plt.plot(smoothed_amplitude, label='Smoothed Envelope',
        #          linestyle='--', linewidth=2)
        # plt.axhline(global_target_amplitude, color='orange',
        #             linestyle=':', label='Global Target Amplitude')
        # plt.title('Amplitude Envelope and Global Target')
        # plt.ylabel('Amplitude')
        # plt.legend()
        # plt.grid()

        # Plot modified signal
        plt.subplot(3, 1, 3)
        plt.plot(modified_signal, label='Modified Signal', color='green',
                 linewidth=2)
        plt.title('Signal with Amplitude Adjusted to Global Amplitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()

        plt.tight_layout()
        plt.show()

    def _read_raw_edf(self, sample: Dict) -> Dict:
        """
        Read raw EDF file.

        Parameters
        ----------
        sample: Dict

        Returns
        -------
        Dict
        """
        def _rename(x):
            x = x.replace('T7', 'T3').replace('T8', 'T4').replace('P7', 'T5')\
                 .replace('P8', 'T6')
            x = x.replace('EEG ', '').replace('-REF', '').replace('-LE', '')
            return x

        data = mne.io.read_raw_edf(sample['edf'],
                                   preload=True,
                                   verbose='WARNING')
        mne.rename_channels(data.info, _rename)
        # data.plot()
        data = notch_filter(data, 50)
        data = notch_filter(data, 60)
        # fig = data.plot()
        # fig.savefig('output/notch.png')
        # fig = mne.viz.plot_raw_psd(data)
        # fig.savefig('output/notch_psd.png')
        data = data.filter(l_freq=0.6,
                           h_freq=None,
                           filter_length=2048,
                           l_trans_bandwidth=0.05,
                           method='fir',
                           phase='zero-double',
                           fir_window='hann',
                           fir_design='firwin2')

        similar_indexes = self._filter_similarity(data)

        eeg_data = data.get_data()
        for idx in mne.pick_types(data.info, eeg=True):
            eeg_data[idx] = self._smooth_amplitude(eeg_data[idx])
        data._data = eeg_data
        del eeg_data

        if len(similar_indexes) > 0:
            data.set_annotations(data.annotations + mne.Annotations(
                onset=similar_indexes,
                duration=[1 for _ in range(len(similar_indexes))],
                description=['x_artefact'
                             for _ in range(len(similar_indexes))],
                orig_time=data.annotations.orig_time))

        # fig = data.plot()
        # fig.savefig('output/dc.png')
        # fig = mne.viz.plot_raw_psd(data)
        # fig.savefig('output/dc_psd.png')
        # filter_params = mne.filter.create_filter(data=data.get_data(),
        #                                  sfreq=data.info['sfreq'],
        #                                  l_freq=0.6,
        #                                  h_freq=None,
        #                                  filter_length=2048,
        #                                  l_trans_bandwidth=0.05,
        #                                  method='fir',
        #                                  phase='zero',
        #                                  fir_window='hann',
        #                                  fir_design='firwin2',
        #                                  verbose=2)
        # hann = mne.viz.plot_filter(filter_params, data.info['sfreq'],
        #                            flim=(0.01, 50))
        # hann.savefig('hann_filter.png')
        # data.plot()
        # mne.viz.plot_raw_psd(data, picks=['FP1', 'FP2', 'F3', 'F4', 'C3',
        #                                   'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
        #                                   'F8', 'T3', 'T4', 'T5', 'T6', 'FZ',
        #                                   'CZ', 'PZ'])

        if self.rereference and self.montage != Montage.BIPOLAR:
            old = [x for x in self.channels
                   if x not in ['T7', 'P7', 'T8', 'P8']]
            new = [x for x in self.channels
                   if x not in ['T3', 'T5', 'T4', 'T6']]
            if (all(x in [y.upper() for y in data.ch_names] for x in old) or
                    all(x in [y.upper() for y in data.ch_names] for x in new)):
                data = self._prepare_bipolar_montage(data, sample)
            else:
                data.close()

                return None
        else:
            sample['raw_data'] = data.get_data(verbose='WARNING')
            sample['info'] = data.info
            sample['channels'] = [x.replace("-0", "").replace("-1", "")
                                  for x in data.info['ch_names']]

        if self.external_annotation or (self.rereference and
                                        self.montage != Montage.BIPOLAR):
            channels = self.channels
            if self.rereference or self.montage == Montage.BIPOLAR:
                channels = [f'{a}-{b}' for a, b in self.channel_pairs]

            index = [sample['channels'].index(x)
                     for x in sample['channels']
                     if x in channels]
        else:
            tmp = []
            if self.rereference and self.montage == Montage.BIPOLAR:
                for x in [f'{a}-{b}' for a, b in self.channel_pairs]:
                    tmp.append(next(s for s in sample['annotations'][2]
                                    if s['channel'] == x))
            else:
                for x in self.channels:
                    tmp.append(next(s for s in sample['annotations'][2]
                                    if s['channel'] == x))
            index = [sample['channels'].index(x['channel'])
                     for x in tmp]
        sample['channels'] = np.array(sample['channels'])[index]
        sample['raw_data'] = sample['raw_data'][index]
        sample['length'] = data.n_times / data.info['sfreq']
        if not self.external_annotation:
            sample['annotations'] = self._prepare_internal_annotations(data,
                                                                       sample)
        logging.debug('_read_raw_edf')

        # Comment if not necessary
        # data.set_annotations(mne.Annotations(
        #     onset=sample['annotations'][1].iloc[:, 1],
        #     duration=sample['annotations'][1].iloc[:, 2]
        #              - sample['annotations'][1].iloc[:, 1],
        #     description=sample['annotations'][1].iloc[:, 3],
        #     orig_time=data.annotations.orig_time))
        # mne.export.export_raw(f"{sample['edf']}", data, 'edf',
        #                       overwrite=True)
        # Comment if not necessary

        data.close()

        del data
        del index
        gc.collect()

        return sample

    @staticmethod
    def _filter_similarity(data):
        # TODO: Worth checking if the first eight signals are valid EEG signals
        eeg_signals = data.get_data()[:8]
        window_size = int(data.info['sfreq'])

        def compute_self_similarity(signals):
            segments = cp.array([
                signals[:, j:j + window_size]
                for j in range(0, signals.shape[1] - window_size, window_size)
            ])
            similarity_matrix = cosine_similarity(
                cp.asnumpy(segments.reshape(len(segments), -1)))

            return similarity_matrix

        similar_indexes = []
        for i in range((eeg_signals.shape[1] // (window_size * 30)) - 1):
            similarity = compute_self_similarity(cp.array(
                eeg_signals[:, (int(window_size * 30) * i):
                               (int(window_size * 30) * (i + 1))]))
            indexes = np.argwhere(
                ((similarity[0] > 0.95) & (similarity[0] <= 0.99)) |
                ((similarity[0] < -0.95) & (similarity[0] >= -0.99))
            ).flatten().tolist()
            similar_indexes.extend([x + (i * 30) for x in indexes])

        del eeg_signals

        return similar_indexes

    def _prepare_bipolar_montage(self, data, sample):
        fig, ax = plt.subplots(figsize=(24, 12))

        if self.montage == Montage.LINKED_EARS:
            ref_channels = [x for x in data.ch_names if "A1" == x or "A2" == x]
            data.set_eeg_reference(ref_channels=ref_channels)
            # data.plot()
            # spectrum = data.compute_psd()
            # spectrum.plot(average=False, picks="data", exclude="bads")
        if (self.montage == Montage.AVERAGE_REFERENCE or
                self.montage == Montage.AVERAGE_REFERENCE_A):
            data.set_eeg_reference(ref_channels="average")
        if self.montage == Montage.OTHER:
            ref_channels = [x for x in data.ch_names if "A1" == x or "A2" == x]
            if len(ref_channels) > 0:
                data.set_eeg_reference(ref_channels=ref_channels)
            else:
                data.set_eeg_reference(ref_channels="average")

        if self.csd:
            csd = CurrentSourceDensity(channels=self.channels,
                                       size_factor=self.size_factor)
            data = csd.transform(data)

        first_chs, second_chs = list(zip(*self.channel_pairs))
        first_chs = self._full_channels_list(data, first_chs)
        first_data = data.get_data(picks=first_chs)
        second_chs = self._full_channels_list(data, second_chs)
        second_data = data.get_data(picks=second_chs)
        info = mne.create_info(
            ch_names=[f'{a}-{b}' for a, b in self.channel_pairs],
            sfreq=data.info.get('sfreq'),
            ch_types='eeg'
        )
        if self.csd:
            for val in info['chs']:
                val['coil_type'] = FIFF.get('FIFFV_COIL_EEG_CSD')
        info['subject_info'] = data.info['subject_info']
        bipolar = mne.io.RawArray(
            data=first_data - second_data,
            info=info,
            first_samp=data.first_samp
        )
        bipolar.set_meas_date(data.info['meas_date'])

        data.info = bipolar.info
        sample['info'] = bipolar.info
        sample['channels'] = bipolar.ch_names
        sample['raw_data'] = bipolar.get_data(verbose='WARNING')

        del bipolar
        del info
        del first_data
        del second_data
        if self.csd:
            del csd
        gc.collect()

        return data

    @staticmethod
    def _full_channels_list(data, channels):
        tmp = []
        for x in channels:
            for y in data.ch_names:
                if y.upper() == x:
                    tmp.append(y)
                    break

        return tmp

    @staticmethod
    def _concat_if_no_overlap(df1: pd.DataFrame,
                              df2: pd.DataFrame) -> pd.DataFrame:
        last = 0
        df1_len = len(df1) - 1
        for j, rs in df1.iterrows():
            for i, r in df2.iterrows():
                if i <= last:
                    continue
                if r['start_time'] >= rs['stop_time']:
                    if j == df1_len:
                        last += 1
                    break
                if (r['stop_time'] <= rs['start_time'] or
                        r['start_time'] >= rs['stop_time']):
                    df1 = pd.concat([df1, r.to_frame().T])
                last += 1

        df1 = pd.concat([df1, df2.iloc[last:]])

        return df1.sort_values(by=['start_time'])

    def _prepare_internal_annotations(self, data, sample):
        seizures = False
        if len(data.annotations) > 0:
            correction = 0
            if self.artefacts:
                correction = 1

            df_bi = pd.DataFrame(
                columns=['channel', 'start_time', 'stop_time', 'label',
                         'confidence'],
                data=[['TERM', annot['onset'],
                       annot['onset'] + annot['duration'],
                       'seiz', 1.0]
                      for annot in data.annotations
                      if annot['description'] == self.seiz_label])
            df_bi_rest = pd.DataFrame(
                columns=['channel', 'start_time', 'stop_time', 'label',
                         'confidence'],
                data=[['TERM', annot['onset'] - correction,
                       annot['onset'] + annot['duration'] + correction,
                       annot['description'], 1.0]
                      for annot in data.annotations
                      if annot['description'] != self.seiz_label])

            df_bi = self._concat_if_no_overlap(df_bi, df_bi_rest)

            df = pd.DataFrame(
                columns=['channel', 'start_time', 'stop_time', 'label',
                         'confidence'],
                data=[[channel, annot['onset'],
                       annot['onset'] + annot['duration'],
                       'seiz', 1.0]
                      for annot in data.annotations
                      for channel in sample['channels']
                      if annot['description'] == self.seiz_label])
            df_rest = pd.DataFrame(
                columns=['channel', 'start_time', 'stop_time', 'label',
                         'confidence'],
                data=[[channel, annot['onset'] - correction,
                       annot['onset'] + annot['duration'] + correction,
                       annot['description'], 1.0]
                      for annot in data.annotations
                      for channel in sample['channels']
                      if annot['description'] != self.seiz_label])

            df_tmp = []
            for df_ch, df_rest_ch in zip(df.groupby('channel'),
                                         df_rest.groupby('channel')):
                df_tmp.append(self._concat_if_no_overlap(df_ch[1],
                                                         df_rest_ch[1]))
            if len(df_tmp) != 0:
                df = pd.concat(df_tmp).drop_duplicates(keep='first')
            else:
                if len(df_rest) > 0:
                    df = df_rest.drop_duplicates(keep='first')
                else:
                    df = df.drop_duplicates(keep='first')
            df = df.sort_values(by=['start_time'])
            del df_tmp

            if self.artefacts:
                df_bi.loc[
                    df_bi['label'].str.lower() != 'seiz', 'label'
                ] = 'x_artefact'
                df.loc[
                    df['label'].str.lower() != 'seiz', 'label'
                ] = 'x_artefact'
                df_bi = df_bi.drop_duplicates(keep='first')
                df = df.drop_duplicates(keep='first')

        else:
            df_bi = pd.DataFrame(
                columns=['channel', 'start_time', 'stop_time', 'label',
                         'confidence'],
                data=[['TERM', 0, (data.n_times / data.info['sfreq']), 'bckg',
                       1.0]])
            df = pd.DataFrame(
                columns=['channel', 'start_time', 'stop_time', 'label',
                         'confidence'],
                data=[[channel, 0, (data.n_times / data.info['sfreq']), 'bckg',
                       1.0]
                      for channel in sample['channels']])

        return df, df_bi, sample['channels'], seizures
