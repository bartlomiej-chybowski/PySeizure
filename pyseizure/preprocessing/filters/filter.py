from abc import ABC, abstractmethod
from typing import Union

import mne
import numpy as np

from pyseizure.data_classes.filter_type import FilterType


class Filter(ABC):
    def __init__(self,
                 order: int = 6,
                 filter_type: FilterType = FilterType.BANDPASS,
                 frequency: float = 256.0,
                 frequencies: list = [0.5, 45.0]):
        """
        Initialise Filter

        Parameters
        ----------
        order: int
            Order of the filter
        filter_type: FilterType
            Type of the filter
        frequency: float
            frequency of the signal
        frequencies: list
            a list of frequencies to filter
        """
        self.order = order
        self.filter_type = filter_type
        self.frequency = frequency
        self.frequencies = frequencies

    @abstractmethod
    def transform(self, raw_signal: Union[mne.io.Raw, np.array]
                  ) -> Union[mne.io.Raw, np.array]:
        raise NotImplementedError
