import numpy as np
from typing import List, Tuple
from pyseizure.data_classes.feature import Feature
from pyseizure.preprocessing.filters.fir import FIR
from pyseizure.data_classes.filter_type import FilterType
from pyseizure.data_classes.frequency_band import FrequencyBand


class EntropyFeature:
    def __init__(self,
                 raw_signal: np.array,
                 features: List[Feature] = [],
                 entropy: List[Tuple[float, float]] = None,
                 frequency: int = 256):
        self.raw_signal = raw_signal
        self.features = features
        self.entropy = entropy
        self.frequency = frequency
        self.bands = [
            FrequencyBand.DELTA, FrequencyBand.THETA, FrequencyBand.ALPHA,
            FrequencyBand.BETA, FrequencyBand.GAMMA
        ]
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
            fir = FIR(order=6,
                      filter_type=FilterType.BANDPASS,
                      frequency=self.frequency,
                      frequencies=[band.value[0], band.value[1]])
            result.append(fir.transform(self.raw_signal))
        return result

    @property
    def phase_synchrony_index(self):
        result = []
        for band_signal in self._extract_signal():
            pass
        return np.array(result)

    def calculate_entropy(self) -> Tuple[List[np.array], List]:
        result = []
        for band_signal in self._extract_signal():
            result.append(self._band_entropy(band_signal))

        return np.concatenate(result, axis=1), [
            f"entropy_{letter}_{band.name}"
            for band in self.bands for letter
            in ['x', 'y', 'y_min']
        ]
