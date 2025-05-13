import numpy as np

from scipy.signal import butter, filtfilt
from typing import Tuple

from pyseizure.data_classes.filter_type import FilterType
from pyseizure.preprocessing.filters.filter import Filter


class Butterworth(Filter):
    def __init__(self,
                 order: int = 6,
                 filter_type: FilterType = FilterType.BANDPASS,
                 frequency: float = 256.0,
                 frequencies: list = [0.5, 45.0],
                 output: str = 'ba') -> Tuple:
        super().__init__(order, filter_type, frequency, frequencies)
        self.output = output
        self.filter = self._design_filter()

    def _design_filter(self):
        return butter(N=self.order,
                      Wn=self.frequencies,
                      btype=self.filter_type.value,
                      analog=False,
                      output=self.output,
                      fs=self.frequencies)

    def transform(self, raw_signal: np.array) -> np.array:
        """
        Filter data using Butterworth filter

        Parameters
        ----------
        raw_signal: numpy.array

        Returns
        -------
        numpy.array
        """
        return filtfilt(*self.filter, raw_signal)
