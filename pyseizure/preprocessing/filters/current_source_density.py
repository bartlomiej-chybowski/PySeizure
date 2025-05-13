from typing import List
import mne
import numpy as np
from pyseizure.data_classes.filter_type import FilterType
from pyseizure.preprocessing.filters.filter import Filter


class CurrentSourceDensity(Filter):
    def __init__(self,
                 order: int = 6,
                 filter_type: FilterType = FilterType.BANDPASS,
                 frequency: float = 256.0,
                 frequencies: list = [0.5, 45.0],
                 channels: List[str] = [],
                 size_factor: float = 1.0):
        super().__init__(order, filter_type, frequency, frequencies)
        self.channels = channels
        self.size_factor = size_factor

    def prepare(self, raw_signal: mne.io.Raw) -> mne.io.Raw:
        if raw_signal.info['dig'] is None:
            raw_signal.info["bads"].extend([x for x in raw_signal.ch_names
                                            if x.upper() not in self.channels])
        raw_signal.set_montage(
            self._prepare_dig_montage(raw_signal.ch_names,
                                      size_factor=self.size_factor),
            on_missing='ignore')

        return raw_signal.pick(None, exclude='bads')

    def _prepare_dig_montage(self, raw_channels: List,
                             size_factor: float = 1.0):
        """
        Parameters
        ----------
        raw_channels: List
            Original channels names
        size_factor: float
            Montage is prepared for adult human.
            To get children's size, ratio factor is used.

        Returns
        -------

        """
        dig_montage = mne.channels.make_standard_montage('standard_1020')
        if len(self.channels) > 0:
            ch_index, ch_names = zip(*[
                (i + 3, x.lower()) for i, x in enumerate(dig_montage.ch_names)
                if x.upper() in [y.upper() for y in raw_channels
                                 if y.upper() in self.channels]])
            ch_index = [0, 1, 2] + list(ch_index)
            dig_montage.ch_names = self._get_original_names(ch_names,
                                                            raw_channels)
            dig_montage.dig = list(np.take(dig_montage.dig, ch_index))
        digs = []
        for dig in dig_montage.dig:
            dig['r'] = dig['r'] * size_factor
            digs.append(dig)
        dig_montage.dig = digs

        return dig_montage

    @staticmethod
    def _get_original_names(ch_names, raw_channels):
        original_names = []
        for ch_name in ch_names:
            for raw_channel in raw_channels:
                if ch_name == raw_channel.lower():
                    original_names.append(raw_channel)
                    break

        return original_names

    def transform(self, raw_signal: mne.io.Raw) -> mne.io.Raw:
        """
        Transform data using CSD filter

        Parameters
        ----------
        raw_signal: mne.io.Raw

        Returns
        -------
        mne.io.Raw
        """
        raw_signal = self.prepare(raw_signal)

        return mne.preprocessing.compute_current_source_density(
            raw_signal.drop_channels(raw_signal.info['bads']))
