import numpy as np

from typing import List

import pywt
from scipy.signal import periodogram, welch
from librosa.feature import spectral_centroid
from pyseizure.data_classes.feature import Feature
from pyseizure.data_classes.filter_type import FilterType
from pyseizure.data_classes.frequency_band import FrequencyBand
from pyseizure.preprocessing.filters.fir import FIR


class FrequencyFeatures:
    bands = [
        FrequencyBand.DELTA, FrequencyBand.THETA, FrequencyBand.ALPHA,
        FrequencyBand.BETA, FrequencyBand.GAMMA
    ]

    def __init__(self, raw_signal: np.array, features: List[Feature] = [],
                 frequency: int = 256):
        self.frequency = frequency
        self.raw_signal = raw_signal
        self.features = features
        self.band_signals = self._extract_signal()

    def calculate_features(self):
        result = np.array([list() for _ in range(len(self.raw_signal))])
        for feature in self.features:
            result = np.concatenate((result, getattr(self, feature.value)),
                                    axis=1)

        return result

    def _extract_signal(self) -> List:
        result = []
        for band in self.bands:
            # result.append(bandPassFilter(data=self.raw_signal,
            #                              sampleRate=self.frequency,
            #                              highpass=band.value[0],
            #                              lowpass=band.value[1]))
            fir = FIR(order=6,
                      filter_type=FilterType.BANDPASS,
                      frequency=self.frequency,
                      frequencies=[band.value[0], band.value[1]])
            result.append(fir.transform(self.raw_signal))
        return result

    @property
    def power_spectral_density(self, use_welch: bool = True):
        result = np.array([list() for _ in range(len(self.raw_signal))])
        for band_signal in self.band_signals:
            if use_welch:
                _, psd = welch(band_signal, self.frequency, scaling='density')
            else:
                _, psd = periodogram(band_signal, self.frequency,
                                     scaling='density')
            result = np.concatenate((result, psd.mean(axis=1).reshape(-1, 1)),
                                    axis=1)

        return result

    @property
    def signal_to_noise(self, ):
        result = np.array([list() for _ in range(len(self.raw_signal))])
        for band_signal in self.band_signals:
            signal_mean = band_signal.mean(axis=1)
            signal_std = band_signal.std(axis=1)
            result = np.concatenate((result, np.where(
                signal_std == 0, 0, signal_mean / signal_std
            ).reshape(-1, 1)), axis=1)

        return result

    @property
    def power_spectral_centroid(self):
        result = np.array([list() for _ in range(len(self.raw_signal))])
        for band_signal in self.band_signals:
            centroid = spectral_centroid(y=band_signal, sr=self.frequency,
                                         n_fft=self.frequency)
            result = np.concatenate((result,
                                     centroid.mean(axis=1).reshape(-1, 1)),
                                    axis=1)

        return result

    @property
    def signal_monotony(self):
        result = np.array([list() for _ in range(len(self.raw_signal))])
        for band_signal in self.band_signals:
            monotony = []
            for row in band_signal:
                signs, counts = np.unique(np.sign(np.diff(row)),
                                          return_counts=True)
                positive = counts[np.where(signs > 0)] if np.array(
                    np.where(signs > 0)).size else 0
                negative = counts[np.where(signs < 0)] if np.array(
                    np.where(signs < 0)).size else 0
                monotony.append(
                    np.absolute(positive - negative) / (len(row)-1))
            result = np.concatenate((result,
                                     np.array(monotony).reshape(-1, 1)),
                                    axis=1)

        return result

    @property
    def coastline(self):
        """
        The sum of the absolute changes in signal value from one sample to
        the next, divided by the number of samples in the interval, divided by
        the range of the signal in the interval.
        The range of the signal is its maximum value minus its minimum value.
        If the range is zero, we set the coastline measurement to zero also.

        Returns
        -------
        numpy.array
        """
        result = np.array([list() for _ in range(len(self.raw_signal))])
        for band_signal in self.band_signals:
            length = np.sum(np.abs(np.diff(band_signal)), axis=1)
            samples_no = band_signal.shape[1]
            signal_min = np.min(band_signal, axis=1)
            signal_max = np.max(band_signal, axis=1)
            line = length / samples_no / signal_max - signal_min
            line[line == np.inf] = 0
            result = np.concatenate((result, line.reshape(-1, 1)), axis=1)

        return result

    @property
    def discrete_wavelet_transform(self):
        """https://pywavelets.readthedocs.io/en/0.2.2/ref/dwt-discrete-wavelet-transform.html"""
        result = np.array([list() for _ in range(len(self.raw_signal))])
        for band_signal in self.band_signals:
            coeffs = pywt.wavedec(band_signal, 'db4', level=7, axis=1)[1]
            result = np.concatenate((result, coeffs), axis=1)

        return result

    @property
    def energy_percentage(self):
        total_energy = np.sum(self.raw_signal**2, axis=1)
        result = np.array([list() for _ in range(len(self.raw_signal))])
        for band_signal in self.band_signals:
            energy = np.sum(band_signal**2, axis=1) / total_energy
            result = np.concatenate((result, energy.reshape(-1, 1)), axis=1)

        return result
