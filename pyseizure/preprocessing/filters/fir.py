import numpy as np

from scipy.signal import firwin, filtfilt

from pyseizure.data_classes.filter_type import FilterType
from pyseizure.preprocessing.filters.filter import Filter


class FIR(Filter):
    def __init__(self,
                 order: int = 6,
                 filter_type: FilterType = FilterType.BANDPASS,
                 frequency: float = 256.0,
                 frequencies: list = [0.5, 45.0],
                 output: str = ''):
        super().__init__(order, filter_type, frequency, frequencies)
        self.output = output
        self.filter = self._design_filter()

    def _design_filter(self):
        return firwin(numtaps=self.order + 1,
                      cutoff=self.frequencies,
                      pass_zero=self.filter_type.value,
                      fs=self.frequency)

    def transform(self, raw_signal: np.array) -> np.array:
        """
        Filter data using FIR filter

        Parameters
        ----------
        raw_signal: numpy.array

        Returns
        -------
        numpy.array
        """
        return filtfilt(self.filter, 1, raw_signal)
