import gc
import logging
import time
from itertools import combinations
import numpy as np
import pandas as pd
import pyarrow as pa
import multiprocessing as mp
import pyarrow.parquet as pq
from typing import List, Tuple
from scipy.signal import welch
from pyseizure.data_classes.feature import Feature
from multiprocessing.pool import ThreadPool as Pool
from pyseizure.data_classes.raw_sample import RawSample
from pyseizure.preprocessing.feature_engineering.correlation_feature import \
    CorrelationFeature
from pyseizure.preprocessing.feature_engineering.entropy_feature import \
    EntropyFeature
from pyseizure.preprocessing.feature_engineering.frequency_feature import \
    FrequencyFeatures
from pyseizure.preprocessing.feature_engineering.graph_feature import \
    GraphFeature
from pyseizure.preprocessing.feature_engineering.temporal_feature import \
    TemporalFeature
import warnings


warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class FeatureExtractor:
    def __init__(self,
                 raw_samples: List[List[RawSample]],
                 temporal_features: List[Feature] = [Feature.MEAN],
                 frequency_features: List[Feature] = [Feature.SIGNAL_MONOTONY],
                 entropy_features: List[Feature] = [],
                 correlation_features: List[Feature] = [],
                 graph_features: List[Feature] = [],
                 binary: bool = False,
                 single_channel: bool = False,
                 save_result: bool = False,
                 suffix: str = '_calc',
                 generalised: bool = True,
                 artefacts: bool = True):
        self.raw_samples = raw_samples
        self.temporal_features = temporal_features
        self.frequency_features = frequency_features
        self.entropy_features = entropy_features
        self.graph_features = graph_features
        self.correlation_features = correlation_features
        self.binary = binary
        self.single_channel = single_channel
        self.threshold = 1.0
        self.save_result = save_result
        self.suffix = suffix
        self.generalised = generalised
        self.artefacts = artefacts

    def calculate_features(self, raw: bool = False):
        results = []
        with Pool(processes=mp.cpu_count()) as mp_pool:
            scores = mp_pool.map(self._calculate_score, self.raw_samples)
        with mp.Manager() as manager:
            start = time.time()
            queue = manager.Queue(mp.cpu_count())
            for i, data in enumerate(zip(self.raw_samples, scores)):
                if raw:
                    process = mp.Process(target=self._calculate_raw,
                                         args=(i, data, queue))
                else:
                    process = mp.Process(target=self._calculate_features,
                                         args=(i, data, queue))
                process.start()
            for _ in range(len(self.raw_samples)):
                results.append(queue.get())
            results = sorted(results, key=lambda x: x[0])
            gc.collect()
            end = time.time()
            logging.info("calculate_features time = %s" % (end - start))

        return results

    # def _calculate_raw(self, raw_samples: List[RawSample]):
    def _calculate_raw(self, index: int, input_data: Tuple,
                       queue: mp.Queue, return_data: bool = False):
        raw_samples, scores = input_data
        name = raw_samples[0].name[:-4]
        logging.debug("Calculating features: %s" % name)

        no_channels = raw_samples[0].signal.shape[0]

        annotations = []
        data = []
        brain_state = []
        columns = list(range(raw_samples[0].signal.shape[1]))

        columns.extend(['score_m', 'score_b'])

        for i, raw_sample in enumerate(raw_samples):
            if not self.single_channel and self.generalised:
                if self.binary:
                    output = np.repeat(raw_sample.annotations,
                                       repeats=no_channels,
                                       axis=0)
                    if self.artefacts:
                        output = np.where(np.all(output != 'bckg') and
                                          np.all(output != 'seiz'),
                                          'x_artefact',
                                          output)
                    else:
                        output = np.where(np.all(output == 'bckg'),
                                          output,
                                          np.repeat([['seiz']],
                                                    repeats=no_channels,
                                                    axis=0))
            elif self.single_channel and self.generalised:
                output = raw_sample.annotations
                if self.binary:
                    if self.artefacts:
                        output = np.where(
                            (output != 'bckg') and (output != 'seiz'),
                            'x_artefact', output)
                    else:
                        output = np.where(output != 'bckg', 'seiz', 'bckg')
            annotations.append(output)
            brain_state.append(np.array([[raw_sample.brain_state[0][0]]]
                                        * no_channels))
            data.append(raw_sample.signal)
        data = np.concatenate(data, axis=0).astype('float32')
        data = np.concatenate([data, scores], axis=1).astype('float32')
        annotations = np.concatenate(annotations, axis=0).astype(str)
        brain_state = np.concatenate(brain_state, axis=0).astype('int32')
        del raw_samples

        if self.save_result:
            df = pd.DataFrame(data=data, columns=columns)
            df = df.join(pd.DataFrame(data=brain_state,
                                      columns=['brain_state']))
            df = df.join(pd.DataFrame(data=annotations, columns=['class']))
            dtype = {i: 'float32' for i in range(df.shape[1])}
            dtype['class'] = 'string'
            ch1 = False
            logging.info(f"Saving:  {name}{self.suffix}")
            if len(df) > 1000000:
                num_rows = 50000 * no_channels
                last_row = list(range(num_rows, len(df), num_rows))
                first_row = [0]
                first_row.extend(last_row)
                last_row.append(len(df) - 1)
                for i, (start, end) in enumerate(zip(first_row, last_row)):
                    pq.write_table(pa.Table.from_pandas(df.iloc[start:end, :]),
                                   f"{name}{self.suffix}"
                                   f"{'_sc' if self.single_channel else ''}"
                                   f"{'_1' if ch1 else ''}_pt{i + 1}.parquet")
            else:
                pq.write_table(pa.Table.from_pandas(df),
                               f"{name}{self.suffix}"
                               f"{'_sc' if self.single_channel else ''}"
                               f"{'_1' if ch1 else ''}.parquet")

            del df

        del annotations

        if return_data:
            queue.put((index, data, columns))
        queue.put((index, 'done'))

    def _calculate_features(self, index: int, input_data: Tuple,
                            queue: mp.Queue, return_data: bool = False):
        raw_samples, scores = input_data
        name = raw_samples[0].name[:-4]
        logging.debug("Calculating features: %s" % name)
        annotations = []
        brain_state = []
        results_temporal = []
        results_frequency = []
        results_correlation = []
        results_graph = []
        columns = []

        """Quian Quiroga et al. (2004)"""
        samples = [x.signal for x in raw_samples]
        samples = np.concatenate(samples, axis=1)
        std = np.median(np.abs(samples) / 0.6745, axis=1)

        if len(self.entropy_features) > 0:
            results_entropy, columns = self._calculate_entropy(samples)
        del samples

        for i, raw_sample in enumerate(raw_samples):
            annotations.append(self._add_annotations(raw_sample))
            brain_state.append(raw_sample.brain_state.astype('int32'))

            results, temporal_columns = \
                self._calculate_temporal_features(raw_sample, std)
            results_temporal.append(results)

            results, frequency_columns = \
                self._calculate_frequency_features(raw_sample)
            results_frequency.append(results)
            results, graph_columns = self._calculate_graph_features(raw_sample)
            results_graph.append(results)

            if self.single_channel:
                results, correlation_columns = \
                    self._calculate_correlation_features(raw_sample)
                results_correlation.append(results)

        del raw_samples

        columns.extend(temporal_columns)
        columns.extend(frequency_columns)
        columns.extend(graph_columns)
        if self.single_channel:
            columns.extend(correlation_columns)
            annotations = np.vstack(annotations)
            brain_state = np.vstack(brain_state)
            if len(results_temporal) > 0:
                results_temporal = np.vstack(results_temporal)
            if len(results_frequency) > 0:
                results_frequency = np.vstack(results_frequency)
            if len(results_correlation) > 0:
                results_correlation = np.vstack(results_correlation)
            if len(results_graph) > 0:
                results_graph = np.vstack(results_graph)
        else:
            results_temporal = np.concatenate(results_temporal, axis=0)
            annotations = np.concatenate(annotations, axis=0)
            brain_state = np.concatenate(brain_state, axis=0)
            results_frequency = np.concatenate(results_frequency, axis=0)
            results_graph = np.concatenate(results_graph, axis=0)


        result = np.concatenate(
            tuple(x for x in [results_temporal.astype('float32'),
                              results_frequency.astype('float32'),
                              results_correlation.astype('float32'),
                              results_graph.astype('float32')]
                  if (len(x) > 0 and x.shape[1] > 0 and
                      isinstance(x, type(np.array([]))))),
            axis=1)
        result = np.concatenate((result, scores.astype('float32')), axis=1)
        columns.extend(['score_m', 'score_b'])

        result = np.concatenate((result, brain_state.astype('float32')),
                                axis=1)
        columns.extend(['brain_state'])

        df = pd.DataFrame(data=result, columns=columns)
        df = df.join(pd.DataFrame(data=annotations.astype('str'),
                                  columns=['class']))
        if self.save_result:
            pq.write_table(pa.Table.from_pandas(df),
                           f"{name}{self.suffix}"
                           f"{'_sc' if self.single_channel else ''}.parquet")

        del result
        del columns
        del annotations
        del results_temporal
        del results_frequency
        del results_correlation
        del results_entropy

        if return_data:
            queue.put((index, df))
        del df
        queue.put((index, 'done'))

    def _calculate_score(self, raw_samples: List[RawSample]):
        total = len(raw_samples[0].signal)
        res_list = []
        for i, sample in enumerate(raw_samples):
            freq, psd = welch(x=sample.signal, fs=sample.frequency,
                              window='hann', nperseg=sample.frequency,
                              noverlap=sample.frequency // 2,
                              scaling='density')
            res = np.polynomial.polynomial.polyfit(x=freq[1:48],
                                                   y=psd.T[1:48, :],
                                                   deg=1)  # intercept, slope
            res_list.append(res)

        intercept = np.concatenate([x[[0]] for x in res_list])
        slope = np.concatenate([x[[1]] for x in res_list])
        del res_list
        int_thresh = (int(np.mean(intercept, axis=0).max())
                      + 3 * np.std(intercept, axis=0).max())
        slope_value = np.mean(slope, axis=0).min()
        slope_std = 3 * np.std(slope, axis=0)
        slope_min = round(slope_value - slope_std.max(),
                          int(np.abs(np.log10(np.abs(slope_std.max()))) + 2))
        slope_std = np.std(slope, axis=0) / 3
        slope_max = round(slope_std.min(),
                          int(np.abs(np.log10(np.abs(slope_std.max()))) + 2))
        if self.single_channel:
            # Fasol et al. slope detection
            result1 = ((total - np.sum((np.where(intercept > int_thresh, 1, 0)
                                        | np.where(slope < slope_min, 1, 0)),
                                       axis=1)) / total).reshape(-1, 1)
            # Flat detection
            result2 = ((total - np.sum(np.where(slope > slope_max, 1, 0),
                                       axis=1)) / total).reshape(-1, 1)
        else:
            # Fasol et al. slope detection
            result1 = (np.where(intercept > int_thresh, 0, 1) |
                       np.where(slope < slope_min, 0, 1)
                       ).flatten().reshape(-1, 1)
            # Flat detection
            result2 = (np.where(slope > slope_max, 0, 1)
                       ).flatten().reshape(-1, 1)
        del intercept
        del slope
        del total
        del psd

        return np.concatenate([result1, result2], axis=1)

    def _calculate_temporal_features(self,
                                     sample: RawSample,
                                     std: np.array) -> Tuple[np.array, List]:
        temporal_features_obj = TemporalFeature(sample.signal, std,
                                                self.temporal_features)
        temporal_features = temporal_features_obj.calculate_features()
        columns = [x.name for x in self.temporal_features]
        if self.single_channel:
            temporal_features = temporal_features.flatten()
            columns = [f'{channel}_{feature.name}'
                       for channel in sample.channels
                       for feature in self.temporal_features]

        return temporal_features, columns

    def _calculate_entropy(self, samples: np.array):
        entropy_features_obj = EntropyFeature(samples, self.entropy_features)
        entropy = entropy_features_obj.calculate_entropy()

        return entropy

    def _calculate_frequency_features(self, sample: RawSample
                                      ) -> Tuple[np.array, List]:
        frequency_features_obj = FrequencyFeatures(sample.signal,
                                                   self.frequency_features)

        frequency_features = frequency_features_obj.calculate_features()
        columns = [f'{band.name}_{feature.name}'
                   for feature in self.frequency_features
                   for band in frequency_features_obj.bands]
        if Feature.DISCRETE_WAVELET_TRANSFORM in self.frequency_features:
            for band in frequency_features_obj.bands:
                idx = columns.index(
                    f"{band.name}_{Feature.DISCRETE_WAVELET_TRANSFORM.name}")
                columns.remove(columns[idx])
                for i in range(7, -1, -1):
                    columns.insert(idx,
                                   f"{band.name}_"
                                   f"{Feature.DISCRETE_WAVELET_TRANSFORM.name}"
                                   f"_{i}")
        if self.single_channel:
            frequency_features = frequency_features.flatten()
            columns = [f'{channel}_{column}'
                       for channel in sample.channels
                       for column in columns]

        return frequency_features, columns

    def _calculate_correlation_features(self, sample: RawSample
                                        ) -> Tuple[np.array, List]:
        correlation_features_obj = CorrelationFeature(
            sample.signal, self.correlation_features)
        correlation_features = correlation_features_obj.calculate_features()
        columns = [
            f'{sample.channels[x1]}_{sample.channels[x2]}_{feature.name}'
            for x1, x2 in combinations(range(len(sample.signal)), 2)
            for feature in self.correlation_features
        ]

        return correlation_features, columns

    def _calculate_graph_features(self, sample: RawSample
                                  ) -> Tuple[np.array, List]:
        single = [Feature.LOCAL_EFFICIENCY, Feature.GLOBAL_EFFICIENCY,
                  Feature.DIAMETER, Feature.RADIUS,
                  Feature.CHARACTERISTIC_PATH]
        if not self.single_channel:
            self.graph_features = [x for x in self.graph_features
                                   if x not in single]
        graph_features_obj = GraphFeature(sample.signal, self.graph_features,
                                          single_channel=self.single_channel)
        graph_features = graph_features_obj.calculate_features()

        if self.single_channel:
            columns = [f"{channel}_{feature.name}"
                       for feature in self.graph_features
                       for channel in sample.channels
                       if feature not in single]
            columns.extend([feature.name for feature in self.graph_features
                            if feature in single])
        else:
            columns = [feature.name for feature in self.graph_features]

        return graph_features, columns

    def _add_annotations(self, sample: RawSample) -> np.array:
        output = sample.annotations
        if self.binary:
            if self.artefacts:
                output = np.where(
                    (output != 'bckg') and (output != 'seiz'),
                    'x_artefact', output)
            else:
                output = np.where(output != 'bckg', 'seiz', 'bckg')

        return output
