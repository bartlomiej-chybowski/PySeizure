from typing import List
from abc import ABC, abstractmethod

import numpy as np


class FeatureSelector(ABC):
    def __init__(self, train_samples: List[str], chunk: int = None):
        self.train_samples = train_samples
        self.chunk = chunk
        self.selector = None
        self.name = 'Feature Selector'

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: np.array):
        raise NotImplementedError

    @abstractmethod
    def get_ranking(self):
        raise NotImplementedError

    @abstractmethod
    def get_selected(self):
        raise NotImplementedError
