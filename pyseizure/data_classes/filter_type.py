from enum import Enum


class FilterType(Enum):
    BANDPASS = 'bandpass'
    LOWPASS = 'lowpass'
    HIGHPASS = 'highpass'
    BANDSTOP = 'bandstop'
