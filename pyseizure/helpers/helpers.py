from datetime import time, timedelta

import mne
import numpy as np


def notch_filter(raw: mne.io.Raw, freq: int) -> mne.io.Raw:
    harmonic = int(raw.info['sfreq'] // freq / 2) * freq
    if harmonic > freq:
        raw = raw.notch_filter(np.arange(freq, harmonic, freq))

    return raw


def time_to_timedelta(timestamp: time):
    return timedelta(hours=timestamp.hour,
                     minutes=timestamp.minute,
                     seconds=timestamp.second,
                     microseconds=timestamp.microsecond)
