from __future__ import annotations

import pandas as pd
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import multiprocessing as mp
from sklearn.preprocessing import LabelEncoder
from torch import amp, device, cuda
from torch.utils.data import DataLoader
from pyseizure.helpers.iterator import Iterator
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector


class Classifier(ABC):
    def __init__(self, train_samples: List[str], test_samples: List[str],
                 valid_samples: List[str], normaliser: Normaliser = None,
                 feature_selector: FeatureSelector = None,
                 eval_only: bool = False, config: Dict = None):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.valid_samples = valid_samples
        self.normaliser = normaliser
        self.feature_selector = feature_selector
        self.eval_only = eval_only
        self.model = None

        self.dev = device("cuda") if cuda.is_available() else device("cpu")

        self.scaler = amp.GradScaler(self.dev.type)

        self.dataset_name = config.get('dataset_name', None)
        self.no_channels = config.get('no_channels', 18)
        self.use_scheduler = config.get('use_scheduler', True)
        self.use_scoring = config.get('use_scoring', True)
        self.use_contrastive = config.get('use_contrastive', False)
        self.binary = config.get('binary', False)
        self.artefacts = config.get('artefacts', False)
        self.labels = config.get('labels', None)
        self.epoch = config.get('epoch', 100)
        self.no_threads_per_core = config.get('no_threads_per_core',
                                              int(mp.cpu_count() / 8) | 1)

        self.config = {}
        if (not config.get('tune') and
                config.get('hyperparameters', None) is not None):
            self.config = config['hyperparameters']
            self.config['hyperparameters'] = True
        else:
            self.config['hyperparameters'] = False

        self.label_encoder = LabelEncoder()
        if self.labels is not None:
            self.label_encoder.fit(self.labels)

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def tune(self, config: Dict):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: DataLoader | Iterator = None) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, post_processing: bool = False) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_path: str):
        raise NotImplementedError
