from enum import Enum


class FrequencyBand(Enum):
    DELTA = (1, 4)
    THETA = (4, 8)
    ALPHA = (8, 12)
    BETA = (12, 20)
    GAMMA = (20, 45)
    HIGH_GAMMA = (65, 80)
    RIPPLE = (80, 250)
    FAST_RIPPLE = (250, 600)
