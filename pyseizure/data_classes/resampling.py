from enum import Enum


class Resampling(Enum):
    RESAMPY = 1  # Resampy resample
    DECIMATE = 2  # Scipy decimate
