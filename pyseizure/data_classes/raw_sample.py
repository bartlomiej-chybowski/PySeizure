import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class RawSample:
    """ Class for storing raw data sample with corresponding information. """
    signal: np.array
    channels: List[str]
    montages: List[Dict] = field(repr=True, default_factory=[{}])
    frequency: int = field(repr=True, default=256)
    annotations: Tuple = field(repr=True, default_factory=tuple())

    def __init__(self,
                 patient_id: int,
                 signal: np.array,
                 channels: List[str],
                 montages: List[Dict] = [{}],
                 frequency: int = 256,
                 annotations: np.array = None,
                 brain_state: np.array = None,
                 annotation_labels: np.array = None,
                 name: str = ''):
        self.patient_id = patient_id
        self.signal = signal
        self.channels = channels
        self.montages = montages
        self.frequency = frequency
        self.annotations = annotations
        self.brain_state = brain_state
        self.annotation_labels = annotation_labels
        self.name = name
        self._prune()

    def _prune(self):
        if len(self.signal) != len(self.channels):
            self.signal = self.signal[:len(self.channels)]
