import gc
import os
import re
import ast
import mne
import sys
import time
import json
import shap
import optuna
import torch
import joblib
import random
import sqlite3
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import multiprocessing as mp
import pyarrow.parquet as pq

from copy import copy
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from functools import cmp_to_key
from itertools import combinations
from matplotlib import pyplot as plt
from pyseizure.classifier.lr import LR
from pyseizure.helpers.tusz import TUSZ
from pyseizure.classifier.xgb import XGB
from pyseizure.classifier.cnn import CNN
from pyseizure.classifier.mlp import MLP
from pyseizure._version import __version__
from pyseizure.helpers.chbmit import CHBMIT
from pyseizure.helpers.generic import Generic
from pyseizure.helpers.dataset import Dataset
from pyseizure.classifier.eegnet import EEGNet
from pyseizure.data_classes.montage import Montage
from pyseizure.data_classes.feature import Feature
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.classifier.conv_lstm import ConvLSTM
from pyseizure.data_classes.objective import Objective
from pyseizure.data_classes.classifier import Classifier
from pyseizure.preprocessing.edf_reader import EDFReader
from pyseizure.data_classes.frequency_band import FrequencyBand
from pyseizure.classifier.conv_transformer import ConvTransformer
from pyseizure.data_classes.dataset import Dataset as Dataset_enum
from pyseizure.preprocessing.feature_selection.boruta import Boruta
from pyseizure.data_classes.feature_selector import FeatureSelector
from pyseizure.preprocessing.feature_selection.fs_config import FSConfig
from pyseizure.preprocessing.feature_selection.svm_rfecv import SVMRFECV
from pyseizure.preprocessing.feature_engineering.feature_extractor import \
    FeatureExtractor
from pyseizure.preprocessing.feature_engineering.frequency_feature import \
    FrequencyFeatures
from pyseizure.helpers.metrics import get_confusion_matrix, get_mse, get_roc, \
    get_accuracy, get_precision, get_f1, get_prauc, get_sensitivity, \
    get_specificity, PostProcessor

logging.basicConfig(encoding='utf-8', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(name)s:%(message)s')

mne.set_config('MNE_LOGGING_LEVEL', 'WARNING')


class PySeizure:
    def __init__(self, config: Dict):
        version = f'version: {__version__}'
        logging.info(f"""\n\
 ____        ____       _                     
|  _ \ _   _/ ___|  ___(_)_____   _ _ __ ___  
| |_) | | | \___ \ / _ | |_  | | | | '__/ _ \ 
|  __/| |_| |___) |  __| |/ /| |_| | | |  __/ 
|_|    \__, |____/ \___|_/___|\__,_|_|  \___| 
       |___/             {version: >20}
""")
        self.config = config

    def run(self):
        self.set_seed(1)
        date_start = datetime.now()
        global_start = time.time()

        if self.config['pipeline']['feature_extraction']:
            dataset = self._dataset()
            samples = self._get_samples(dataset)
            self._extract_features(samples)
            sys.exit('check if working')

        dataset_name = self.config['pipeline']['dataset']
        config = self.config['dataset_config'][dataset_name]
        create_db = self.config['pipeline'].get('create_db', False)
        train_samples, eval_samples, test_samples = None, None, None
        if create_db or not config.get('db', False):
            train_samples, eval_samples, test_samples = self._split_samples()
            if create_db:
                self._create_dbs(train_samples, eval_samples, test_samples)
        if config.get('db', False):
            train_samples, eval_samples, test_samples = self._db_samples()
        assert train_samples is not None
        assert eval_samples is not None
        assert test_samples is not None

        feature_selector = None
        normaliser = self._normalise(train_samples)
        if not self.config['pipeline'].get('raw', False):
            feature_selector = self._feature_selection(train_samples,
                                                       normaliser)

        self._classify(train_samples, eval_samples, test_samples, normaliser,
                       feature_selector)

        logging.info(f"Runtime {(time.time() - global_start):0.3f}")
        raw = 'raw' if self.config['pipeline'].get('raw', False) else 'calc'
        model = self.config['pipeline']['classification'].get('models')
        model = ', '.join([x.value for x in model])
        logging.info(
            f'End of {raw} {dataset_name.value} {model} trial: '
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. '
            f'Trial started on: {date_start.strftime("%Y-%m-%d %H:%M:%S")}')

    @staticmethod
    def set_seed(seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _dataset(self):
        logging.info('Traverse data and decode annotations')

        dataset_name = self.config['pipeline']['dataset']
        config = self.config['dataset_config'][dataset_name]
        edf_config = self.config['edf_reader']

        if dataset_name == Dataset_enum.TUSZ:
            dataset = TUSZ(self.config['root_dir'],
                           config['montage'],
                           config['symbols'])
            dataset.traverse_data()
            dataset.decode_annotations()
        elif dataset_name == Dataset_enum.CHBMIT:
            dataset = CHBMIT(self.config['root_dir'],
                             config['embed_annotations'])
            dataset.traverse_data()
        else:
            dataset = Generic(self.config['root_dir'],
                              edf_config['montage'],
                              edf_config.get('seiz_label', 'seiz'),
                              edf_config.get('rereference', True))
            dataset.traverse_data()

        return dataset

    def _get_samples(self, dataset: Dataset):
        logging.info('Get samples for feature extraction')
        seizure_only = False
        montage = None
        pipeline_filter = self.config['pipeline'].get('filter', False)
        if pipeline_filter:
            seizure_only = pipeline_filter.get('seizures_only', False)
            montage = pipeline_filter.get('montage', None)
            if montage is not None:
                montage = montage.value

        if montage is not None:
            if seizure_only:
                samples = [x for x in dataset.samples
                           if x['annotations'][3]
                           and x['montage_type'] == montage]
            else:
                samples = [x for x in dataset.samples
                           if x['montage_type'] == montage]
        else:
            if seizure_only:
                samples = [x for x in dataset.samples
                           if x['annotations'][3]]
            else:
                samples = [x for x in dataset.samples]

        return sorted(samples, key=lambda x: x['length'])

    def _db_samples(self, key: str = 'db'):
        dataset_name = self.config['pipeline']['dataset']
        config = self.config['dataset_config'][dataset_name][key]

        return config['train'], config['valid'], config['test']

    def _split_samples(self):
        logging.info('Train/Test splitting')

        def _get_unique(filenames: List):
            files = [os.path.split(x)[-1] for x in filenames]
            files = [x.split('_augmented')[0] for x in files]
            files = [x.split('_sign')[0] for x in files]
            files = [x.split('_time')[0] for x in files]
            files = [x.split(f'_{suffix}')[0] for x in files]

            return list(np.unique(np.array(files)))

        def _get_full_path(samples):
            tmp = []
            for element in samples:
                tmp.extend([x for x in all_samples if element in x])

            return tmp

        suffix = 'raw' if self.config['pipeline']['raw'] else 'calc'
        single_channel = '_sc' if self.config['single_channel'] else ''

        train_samples = []
        test_samples = []
        eval_samples = []
        all_samples = []

        for root, dirs, files in os.walk(self.config['root_dir']):
            for file in files:
                if (file.lower().endswith(f'.parquet') and
                        f'{suffix}{single_channel}' in file.lower()):
                    if 'train' in root:
                        train_samples.append(os.path.join(root, file))
                    elif 'eval' in root:
                        eval_samples.append(os.path.join(root, file))
                    elif 'dev' in root:
                        test_samples.append(os.path.join(root, file))
                    else:
                        all_samples.append(os.path.join(root, file))

        if len(train_samples) > 0 and (len(eval_samples) == 0 or
                                       len(test_samples) == 0):
            all_samples = train_samples

        if len(all_samples) > 0:
            # split to train/test/eval sets
            # make sure single recording (augmented or original) data
            # is only in one of train/test/eval at the same time.

            unique = _get_unique(all_samples)
            train_samples = random.sample(unique, k=int(len(unique) * 0.8))
            unique = [x for x in unique if x not in train_samples]
            eval_samples = random.sample(unique, k=int(len(unique) * 0.5) | 1)
            test_samples = [x for x in unique if x not in eval_samples]
            test_samples = _get_full_path(test_samples)
            train_samples = _get_full_path(train_samples)
            eval_samples = _get_full_path(eval_samples)

            if len(test_samples) == 0:
                test_samples = copy(eval_samples)

        subset = self.config['pipeline'].get('subset', False)
        if subset:
            train_samples = train_samples[:subset[0]]
            eval_samples = eval_samples[:subset[1]]
            test_samples = test_samples[:subset[2]]

        if not self.config['edf_reader'].get('data_augmentation', False):
            train_samples = [x for x in train_samples
                             if 'augmented' not in x
                             and 'sign_flipped' not in x
                             and 'time_reversed' not in x]
            test_samples = [x for x in test_samples
                            if 'augmented' not in x
                            and 'sign_flipped' not in x
                            and 'time_reversed' not in x]
            eval_samples = [x for x in eval_samples
                            if 'augmented' not in x
                            and 'sign_flipped' not in x
                            and 'time_reversed' not in x]

        logging.info(f"Train set: {len(train_samples)} files, "
                     f"Test set: {len(test_samples)} files, "
                     f"Eval set: {len(eval_samples)} files")

        train_samples.sort(key=cmp_to_key(self._sorter))
        eval_samples.sort(key=cmp_to_key(self._sorter))
        test_samples.sort(key=cmp_to_key(self._sorter))

        return train_samples, eval_samples, test_samples

    @staticmethod
    def _sorter(a, b):
        a_path = os.path.normpath(a).split(os.sep)
        b_path = os.path.normpath(b).split(os.sep)
        if os.sep.join(a_path[:-1]) > os.sep.join(b_path[:-1]):
            return 1
        elif os.sep.join(a_path[:-1]) == os.sep.join(b_path[:-1]):
            a_part = a_path[-1].replace('.parquet', '').split('_')
            b_part = b_path[-1].replace('.parquet', '').split('_')
            if 'pt' in a_part[-1] and a_part[:-1] == b_part[:-1]:
                a_part = int(re.findall(r'\d+', a_part[-1])[0])
                b_part = int(re.findall(r'\d+', b_part[-1])[0])
                if a_part > b_part:
                    return 1
                elif a_part == b_part:
                    return 0
                else:
                    return -1
            else:
                if a_path[-1] > b_path[-1]:
                    return 1
                elif a_path[-1] == b_path[-1]:
                    return 0
                else:
                    return -1
        else:
            return -1

    def _extract_features(self, samples: List):
        extract_start = time.time()
        logging.info('Extract features')
        extractor_config = self.config['feature_extractor']
        edf_config = self.config['edf_reader']
        feature_config = self.config['features']
        raw = self.config['pipeline']['raw']
        single_channel = self.config.get('single_channel', False)

        edf_reader = EDFReader(
            epoch=edf_config.get('epoch', 256),
            frequency=edf_config.get('frequency', 256),
            external_annotation=edf_config.get('external_annotation', True),
            annotation_labels=edf_config.get('annotation_labels', []),
            single_channel=self.config['single_channel'],
            save_result=edf_config.get('save_result', True),
            channels=edf_config.get('channels', []),
            channel_pairs=edf_config.get('channel_pairs', []),
            csd=edf_config.get('csd', False),
            montage=edf_config.get('montage', Montage.BIPOLAR),
            data_augmentation=edf_config.get('data_augmentation', True),
            size_factor=edf_config.get('size_factor', 1.0),
            rereference=edf_config.get('rereference', True),
            generalised=self.config.get('generalised', False),
            seiz_label=edf_config.get('seiz_label', 'seiz'),
            artefacts=edf_config.get('artefacts', False))

        if not extractor_config.get('override', True):
            tmp = []
            for sample in samples:
                if len(list(Path(sample['edf']).parent.glob(
                        f'**/{Path(sample["edf"]).stem}*'
                        f'_{"raw" if raw else "calc"}'
                        f'{"_sc" if single_channel else ""}*.parquet'))) == 0:
                    tmp.append(sample)
            samples = tmp
            del tmp
        for i in range(0, len(samples), extractor_config['parallel_files']):
            j = i + extractor_config['parallel_files']
            if j > len(samples):
                j = len(samples)

            batch = samples[i:j]
            start = time.time()
            raw_signals = edf_reader.read_signals(batch)
            logging.info(f"Read signals runtime time "
                         f"{(time.time() - start):0.3f}")

            start = time.time()
            feature_extractor = FeatureExtractor(
                raw_samples=raw_signals,
                temporal_features=feature_config.get('temporal',
                                                     [Feature.MIN]),
                frequency_features=feature_config.get('frequency', []),
                correlation_features=feature_config.get('correlation', []),
                entropy_features=feature_config.get('entropy', []),
                graph_features=feature_config.get('graph', []),
                binary=self.config.get('binary', True),
                single_channel=self.config.get('single_channel', False),
                save_result=extractor_config.get('save_result', False),
                suffix=f'_{"raw" if raw else "calc"}',
                artefacts=edf_config.get('artefacts', False),
                generalised=self.config.get('generalised', False))
            # df = None # uncomment later
            df = feature_extractor.calculate_features(raw=raw)
            logging.info(f'Feature engineering {i + 1} out of {len(samples)}, '
                         f'run time = {(time.time() - start):0.3f}')
            del raw_signals
            del feature_extractor
            del df
            gc.collect()
        logging.info(f"Extract features run time "
                     f"{(time.time() - extract_start):0.3f}")

        return None

    def _normalise(self, samples: List):
        start = time.time()

        normaliser = None
        if self.config['pipeline']['normalisation']:
            logging.info('Fit normaliser')
            raw = self.config["pipeline"].get("raw", False)
            normaliser = Normaliser(self.config['no_threads_per_core'], raw)
            normaliser.fit(samples)
            joblib.dump(normaliser,
                        f'output/models/normaliser_'
                        f'{"raw" if raw else "calc"}_'
                        f'{self.config["pipeline"]["dataset"].value}.joblib')

            logging.info(f"Fit normaliser run time "
                         f"{(time.time() - start):0.3f}")

        return normaliser

    def _feature_selection(self, samples: List, normaliser: Normaliser):
        start = time.time()

        pipeline = self.config['pipeline']

        selection_config = pipeline['feature_selection']
        classification_conf = pipeline['classification']
        labels = classification_conf.get('labels')
        columns = self._get_columns()
        fs = FSConfig(samples, columns, normaliser)
        fs.fit()

        if selection_config.get('selector') is not None:
            logging.info('Fit feature selector')
            if selection_config['selector'] == FeatureSelector.BORUTA:
                fs = Boruta(
                    train_samples=samples,
                    weight=selection_config.get('weight'),
                    max_iter=selection_config.get('max_iter'),
                    columns=fs.columns,
                    labels=labels,
                    normaliser=normaliser,
                    subsampling=classification_conf.get('subsampling', False),
                    dataset=pipeline['dataset'])
            elif selection_config['selector'] == FeatureSelector.SVMRFECV:
                fs = SVMRFECV(
                    train_samples=samples,
                    labels=labels,
                    normaliser=normaliser,
                    columns=fs.columns,
                    subsampling=classification_conf.get('subsampling', False))
            fs.fit()
            logging.info(f"Fit feature selector run time "
                         f"{(time.time() - start):0.3f}")

        fs.close()
        fs.database = {}
        joblib.dump(fs, f'output/models/fs_'
                        f'{pipeline["classification"]["models"][0].value}_'
                        f'{pipeline["dataset"].value}.joblib')
        fs.database = fs.open()

        return fs

    def _get_columns(self):
        features = self.config['features']
        if self.config['edf_reader'].get('rereference', False):
            channels = [f"{x[0]}-{x[1]}"
                        for x in self.config['edf_reader']['channel_pairs']]
        else:
            channels = self.config['edf_reader']['channels']

        columns = [x.name for x in features['temporal']]
        if self.config['single_channel']:
            columns = [f'{channel}_{feature.name}'
                       for channel in channels
                       for feature in features['temporal']]

        frequency_columns = [f'{band.name}_{feature.name}'
                             for feature in features['frequency']
                             for band in FrequencyFeatures.bands]
        if Feature.DISCRETE_WAVELET_TRANSFORM in features['frequency']:
            for band in FrequencyFeatures.bands:
                idx = frequency_columns.index(
                    f"{band.name}_{Feature.DISCRETE_WAVELET_TRANSFORM.name}")
                frequency_columns.remove(frequency_columns[idx])
                for i in range(7, -1, -1):
                    frequency_columns.insert(
                        idx,
                        f"{band.name}"
                        f"_{Feature.DISCRETE_WAVELET_TRANSFORM.name}"
                        f"_{i}")
        if self.config['single_channel']:
            frequency_columns = [f'{channel}_{column}'
                                 for channel in channels
                                 for column in frequency_columns]
        columns.extend(frequency_columns)
        del frequency_columns

        single = [Feature.LOCAL_EFFICIENCY, Feature.GLOBAL_EFFICIENCY,
                  Feature.DIAMETER, Feature.RADIUS,
                  Feature.CHARACTERISTIC_PATH]
        if self.config['single_channel']:
            columns.extend([f"{channel}_{feature.name}"
                            for feature in features['graph']
                            for channel in channels
                            if feature not in single])
            columns.extend([feature.name for feature in features['graph']
                            if feature in single])
        else:
            columns.extend([feature.name for feature in features['graph']
                            if feature not in single])

        if self.config['single_channel']:
            columns.extend([
                f'{channels[x1]}_{channels[x2]}_{feature.name}'
                for x1, x2 in combinations(range(len(channels)), 2)
                for feature in features['correlation']
            ])

        columns.extend(['score_m', 'score_b', 'class'])

        return columns

    def _classify(self, train_samples: List, eval_samples: List,
                  test_samples: List, normaliser: Normaliser,
                  feature_selector):
        start = time.time()
        logging.info('Fit Classifiers')
        classification_conf = self.config['pipeline']['classification']
        weight = None
        if classification_conf.get('weight', False):
            if isinstance(classification_conf.get('weight'), dict):
                weight = [x for x in classification_conf['weight'].values()]
            else:
                weight = classification_conf.get('weight')

        no_channels = len(self.config['edf_reader'].get('channels', []))
        if self.config['edf_reader'].get('rereference', False):
            no_channels = len(self.config['edf_reader']['channel_pairs'])
        dataset_name = self.config['pipeline']['dataset']
        for model_class in classification_conf['models']:
            config = {
                'labels': classification_conf.get('labels'),
                'artefacts': self.config['edf_reader'].get('artefacts', False),
                'binary': self.config['binary'],
                'epoch': classification_conf.get('epoch'),
                'weight': weight,
                'subsampling': classification_conf.get('subsampling', False),
                'no_channels': no_channels,
                'dataset_name': dataset_name.value,
                'use_scheduler': classification_conf.get('use_scheduler',
                                                         True),
                'use_scoring': classification_conf.get('use_scoring', True),
                'use_contrastive': classification_conf.get('use_contrastive',
                                                           False),
                'no_threads_per_core': self.config.get('no_threads_per_core'),
                'hyperparameters': classification_conf.get('hyperparameters',
                                                           None),
                'tune': classification_conf.get('tune', False),
            }
            model = getattr(sys.modules[__name__], model_class.value)(
                train_samples=train_samples,
                test_samples=test_samples,
                valid_samples=eval_samples,
                feature_selector=feature_selector,
                normaliser=normaliser,
                eval_only=False,
                config=config)
            model_name = model_class.name.replace("_", " ").title()
            if classification_conf.get('tune', False):
                logging.info(f'Tune {model_name}')
                model.tune(classification_conf.get('tune_config', {
                    'n_jobs': 4,
                    'n_trials': 16,
                    'n_epochs': 20,
                    'tune_objective': Objective.FPFN,
                    'sampler': optuna.samplers.TPESampler(),
                    'pruner': optuna.pruners.HyperbandPruner()
                }))
            logging.info(f'Train {model_name}')
            model.train()
            logging.info(f'Evaluate on test data {model_name}')
            model.evaluate(post_processing=True)

            gc.collect()
        logging.info(f"Fit classifiers run time {(time.time() - start):0.3f}")

    def _create_dbs(self, train_samples, eval_samples, test_samples):
        dataset = self.config['pipeline']['dataset']
        db_config = self.config['dataset_config'][dataset].get('db', {})
        montage = None
        if self.config['edf_reader'].get('rereference', False):
            montage = 0
        with mp.Manager() as manager:
            start = time.time()
            queue = manager.Queue(mp.cpu_count())
            mp.Process(target=self._create_db,
                       args=(train_samples, db_config['train'], dataset,
                             montage, queue)).start()
            mp.Process(target=self._create_db,
                       args=(test_samples, db_config['test'], dataset,
                             montage, queue)).start()
            mp.Process(target=self._create_db,
                       args=(eval_samples, db_config['valid'], dataset,
                             montage, queue)).start()
            for _ in range(3):
                queue.get()
            gc.collect()
            logging.info(f"DB creation run time {(time.time() - start):0.3f}")

    def _create_db(self, samples, name, dataset, montage, queue):
        connection = sqlite3.connect(name)
        cursor = connection.cursor()
        cursor.execute(f"CREATE TABLE data ("
                       f"id INTEGER PRIMARY KEY NOT NULL, "
                       f"subject VARCHAR(255), "
                       f"dataset VARCHAR(255), "
                       f"ses_date VARCHAR(255), "
                       f"montage INTEGER NOT NULL, "
                       f"kind INTEGER NOT NULL, "
                       f"seizure INTEGER NOT NULL, "
                       f"artefact INTEGER NOT NULL, "
                       f"brain_state INTEGER NOT NULL, "
                       f"score_a REAL NOT NULL, "
                       f"score_b REAL NOT NULL, "
                       f"vals BLOB NOT NULL)")

        cursor.execute(f"CREATE TABLE columns ("
                       f"id INTEGER PRIMARY KEY NOT NULL, "
                       f"name VARCHAR(255) NOT NULL)")

        columns = pq.ParquetFile(samples[0]).metadata.schema.names
        for column in columns:
            cursor.execute(f"INSERT INTO columns (name) "
                           f"VALUES ('{column}')")
        connection.commit()

        index = 1
        edf_config = self.config['edf_reader']
        no_channels = len(edf_config.get('channels', []))
        if edf_config.get('rereference', False):
            no_channels = len(edf_config['channel_pairs'])
        size = sum([pq.ParquetFile(x).metadata.num_rows for x in samples])
        if self.config['pipeline'].get('raw', False):
            size = size / no_channels
        with tqdm(total=size,
                  bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            pbar.set_description(name)
            dataset_config = self.config['dataset_config'][dataset]
            data_dir = self.config['root_dir']
            if dataset_config.get('data_dir', False):
                data_dir = dataset_config['data_dir']
            for filename in samples:
                tmp = filename.replace(data_dir, '').split('/')
                if dataset == Dataset_enum.TUSZ:
                    subject = tmp[3].split('_')[0]
                    ses_date = tmp[1]
                else:
                    subject = ''
                    ses_date = ''
                if montage is None:
                    if 'le' in tmp:
                        montage = 1
                    elif 'ar_a' in tmp:
                        montage = 3
                    elif 'ar' in tmp:
                        montage = 2
                    else:
                        montage = 0
                if 'time_reversed' in tmp:
                    kind = 1
                elif 'sign_flipped' in tmp:
                    kind = 2
                elif 'augmented' in tmp:
                    kind = 3
                else:
                    kind = 0
                file = pq.ParquetFile(filename)
                num_rows = file.metadata.num_rows
                no_rows = 1000
                if self.config['pipeline'].get('raw', False):
                    no_rows *= no_channels
                for i in range(1, num_rows + 1, no_rows):
                    take_range = list(range(i - 1, i + no_rows - 1))
                    if num_rows < i + no_rows - 1:
                        take_range = list(
                            range(i - 1, i + num_rows % no_rows - 1))
                    vals = file.read_row_group(0).take(
                        take_range).to_pandas()
                    # Should not be needed later
                    score_cols = 0
                    if 'brain_state' in vals.columns:
                        score_cols = 1
                    if self.config['pipeline'].get('raw', False):
                        vals = np.squeeze(
                            np.lib.stride_tricks.sliding_window_view(
                                vals, (no_channels, vals.shape[1]),
                                writeable=True)[::no_channels], 1)
                    else:
                        vals = vals.values
                    for val in vals:
                        if not self.config['pipeline'].get('raw', False):
                            val = val.flatten()
                            seiz = bool((val[-1] != 'bckg') and
                                        (val[-1] != 'x_artefact'))
                            artefact = bool(val[-1] == 'x_artefact')
                            if score_cols == 0:
                                brain_state = -1
                                score_a = val[-3]
                                score_b = val[-2]
                            else:
                                score_a = val[-4]
                                score_b = val[-3]
                                brain_state = val[-2]
                        else:
                            seiz = bool((np.any(val[:, -1] != 'bckg')) and
                                        (np.any(val[:, -1] != 'x_artefact')))
                            artefact = bool(np.any(
                                val[:, -1] == 'x_artefact'))
                            if score_cols == 0:
                                brain_state = -1
                                score_a = np.mean(val[:, -3])
                                score_b = np.mean(val[:, -2])
                            else:
                                score_a = np.mean(val[:, -4])
                                score_b = np.mean(val[:, -3])
                                brain_state = np.median(val[:, -2])
                        cursor.execute(f"INSERT INTO data ("
                                       f"subject, "
                                       f"dataset, "
                                       f"ses_date, "
                                       f"montage, "
                                       f"kind, "
                                       f"seizure, "
                                       f"artefact, "
                                       f"brain_state, "
                                       f"score_a, "
                                       f"score_b, "
                                       f"vals"
                                       f") VALUES ("
                                       f"'{subject}', "
                                       f"'{dataset.value}', "
                                       f"'{ses_date}', "
                                       f"'{montage}', "
                                       f"'{kind}', "
                                       f"'{int(seiz)}', "
                                       f"'{int(artefact)}', "
                                       f"'{brain_state}', "
                                       f"'{score_a}', "
                                       f"'{score_b}', "
                                       f"'{json.dumps(val.tolist())}')")
                        index += 1
                    connection.commit()
                    pbar.update(len(vals))

        connection.close()
        queue.put(f'{name} done')

    def evaluate(self):
        normaliser = None
        if self.config['pipeline'].get('normalisation', False):
            normaliser = joblib.load(os.path.join(self.config['root_dir'],
                                                  'normaliser.joblib'))
        feature_selector = joblib.load(os.path.join(self.config['root_dir'],
                                                    'fs.joblib'))

        params_file = open(os.path.join(self.config['root_dir'], 'model.txt'))
        model_params = ast.literal_eval(params_file.readlines()[1])

        dataset_name = self.config['pipeline']['dataset']
        config = self.config['dataset_config'][dataset_name]
        classification_conf = self.config['pipeline']['classification']
        model_class = classification_conf['models'][0]
        if config.get('db', False):
            train_samples, eval_samples, test_samples = self._db_samples()

        weight = None
        if classification_conf.get('weight', False):
            if isinstance(classification_conf.get('weight'), dict):
                weight = [x for x in classification_conf['weight'].values()]
            else:
                weight = classification_conf.get('weight')
        no_channels = len(self.config['edf_reader'].get('channels', []))
        if self.config['edf_reader'].get('rereference', False):
            no_channels = len(self.config['edf_reader']['channel_pairs'])

        config = {
                'labels': classification_conf.get('labels'),
                'binary': self.config['binary'],
                'epoch': classification_conf.get('epoch'),
                'weight': weight,
                'subsampling': classification_conf.get('subsampling', False),
                'no_channels': no_channels,
                'dataset_name': dataset_name.value,
                'use_scheduler': classification_conf.get('use_scheduler',
                                                         True),
                'use_scoring': classification_conf.get('use_scoring', True),
                'use_contrastive': classification_conf.get('use_contrastive',
                                                           False)
            }
        model = getattr(sys.modules[__name__], model_class.value)(
            train_samples=train_samples,
            test_samples=test_samples,
            valid_samples=eval_samples,
            feature_selector=feature_selector,
            normaliser=normaliser,
            eval_only=True,
            config=config)

        model.config = model_params
        model._init_loaders_and_model()
        model.load_model(os.path.join(self.config['root_dir'], 'model.bin'))
        logging.info(f'{model_class.value} model evaluation on '
                     f'{dataset_name.value} test data')
        model.evaluate(model.data_loader['test'])
        logging.info(f'{model_class.value} model evaluation on '
                     f'{dataset_name.value} train data')
        model.evaluate(model.data_loader['train'])
        logging.info(f'{model_class.value} model evaluation on '
                     f'{dataset_name.value} validation data')
        model.evaluate(model.data_loader['valid'])

    def evaluate_vote(self):
        dataset = self.config['pipeline']['dataset']
        source_dataset = self.config['pipeline'].get('source_dataset', dataset)
        classification_conf = self.config['pipeline']['classification']

        models_configs = []
        for model_class in classification_conf['models']:

            _calc = 'raw'
            if model_class in [Classifier.LOGISTIC_REGRESSION,
                               Classifier.XGBOOST,
                               Classifier.MLP]:
                _calc = 'calc'

            try:
                fs_name = self.config['pipeline'].get(
                    'feature_selection', {})['selector'].value
                fs_name = f'_{fs_name}'
            except (KeyError, AttributeError):
                fs_name = '_generic'
            if _calc == 'raw':
                fs_name = ''

            train_db, eval_db, test_db = None, None, None
            if self.config['dataset_config'][dataset].get(_calc, False):
                train_db, eval_db, test_db = self._db_samples(_calc)

            normaliser = None
            if self.config['pipeline'].get('normalisation', False):
                normaliser = joblib.load(os.path.join(
                    self.config['root_dir'],
                    f'normaliser_{_calc}_{source_dataset.value}.joblib'))

            try:
                feature_selector = joblib.load(os.path.join(
                    self.config['root_dir'],
                    f'fs_{model_class.value}_{source_dataset.value}.joblib'))
                feature_selector.train_samples = eval_db
                feature_selector.reopen()
            except FileNotFoundError:
                feature_selector = None

            try:
                params_file = open(os.path.join(
                    self.config['root_dir'],
                    f'{model_class.value.lower()}_{source_dataset.value}.txt'))
                model_params = ast.literal_eval(params_file.readlines()[1])
                tune_objective = model_params.get('tune_objective', 'ssf')
                model_params['tune_objective'] = Objective(tune_objective)
            except FileNotFoundError:
                model_params = ''

            model_file = os.path.join(
                self.config['root_dir'],
                f'{model_class.value.lower()}_'
                f'{source_dataset.value}{fs_name}.bin')
            models_configs.append({
                'model': model_class,
                'model_path': model_file,
                'normaliser': normaliser,
                'model_params': model_params,
                'feature_selector_name': fs_name,
                'feature_selector': feature_selector,
                'train_samples': train_db,
                'eval_samples': eval_db,
                'test_samples': test_db,
            })

        weight = None
        if classification_conf.get('weight', False):
            if isinstance(classification_conf.get('weight'), dict):
                weight = [x for x in classification_conf['weight'].values()]
            else:
                weight = classification_conf.get('weight')
        no_channels = len(self.config['edf_reader'].get('channels', []))
        if self.config['edf_reader'].get('rereference', False):
            no_channels = len(self.config['edf_reader']['channel_pairs'])

        config = {
                'labels': classification_conf.get('labels'),
                'binary': self.config['binary'],
                'artefacts': self.config['edf_reader'].get('artefacts', False),
                'epoch': classification_conf.get('epoch'),
                'weight': weight,
                'subsampling': classification_conf.get('subsampling', False),
                'no_channels': no_channels,
                'dataset_name': dataset.value,
                'use_scheduler': classification_conf.get('use_scheduler',
                                                         True),
                'use_scoring': classification_conf.get('use_scoring', True),
                'use_contrastive': classification_conf.get('use_contrastive',
                                                           False)
            }
        results = []
        post_results = []
        for model_config in models_configs:
            model = getattr(sys.modules[__name__],
                            model_config['model'].value)(
                train_samples=model_config['train_samples'],
                test_samples=model_config['test_samples'],
                valid_samples=model_config['eval_samples'],
                feature_selector=model_config['feature_selector'],
                normaliser=model_config['normaliser'],
                eval_only=False,
                config=config)
            if model_config['model_params'] != '':
                model.config = model_config['model_params']
            model._init_loaders_and_model()
            model.load_model(model_config['model_path'])

            def predict(input_values):
                input_values = torch.tensor(input_values,
                                            dtype=torch.float32).to(model.dev)
                return np.array(model.model(input_values).detach().cpu())

            bckg, seiz = [], []
            if model_config['model'] in [Classifier.LOGISTIC_REGRESSION,
                                         Classifier.XGBOOST,
                                         Classifier.MLP]:
                if 'brain_state' in model.feature_selector.all_columns:
                    score_cols = 4
                else:
                    score_cols = 3
                if model_config['model'] != Classifier.XGBOOST:
                    for x in model.data_loader['test']:
                        label = x[1][:, :, 0].flatten()
                        if torch.any(label == 0) and len(bckg) < 200:
                            bckg_idx = (x[1].flatten(1)[:, 0] == 0).nonzero()
                            for indice in list(bckg_idx.numpy().flatten()):
                                bckg.append(x[0][indice])
                        if torch.any(label == 1) and len(seiz) < 200:
                            seiz_idx = (x[1].flatten(1)[:, 0] > 0).nonzero()
                            for indice in list(seiz_idx.numpy().flatten()):
                                seiz.append(x[0][indice])
                        if len(seiz) > 199 and len(bckg) > 199:
                            break
                    explainer = shap.KernelExplainer(
                        model=predict,
                        data=torch.concat(bckg).flatten(1).numpy())
                    shap_values = explainer.shap_values(
                        torch.concat(seiz).flatten(1).numpy())
                    df = pd.DataFrame(
                        shap_values[:, :, 1],
                        columns=np.array(
                            model.feature_selector.all_columns[:-score_cols]
                        )[model.feature_selector.get_selected()])
                else:
                    explainer = shap.TreeExplainer(model=model.model)
                    explanation = explainer(
                        xgb.QuantileDMatrix(data=model.data_loader['test']))
                    shap_values = explanation.values
                    df = pd.DataFrame(
                        shap_values,
                        columns=np.array(
                            model.feature_selector.all_columns[:-score_cols]
                        )[model.feature_selector.get_selected()])

                df.to_csv(
                    f"shap_values_{model_config['model'].value}"
                    f"_{self.config['pipeline']['source_dataset'].value}"
                    f"_{self.config['pipeline']['dataset'].value}.csv",
                    index=False)
                df = pd.read_csv(
                    f"shap_values_{model_config['model'].value}"
                    f"_{self.config['pipeline']['source_dataset'].value}"
                    f"_{self.config['pipeline']['dataset'].value}.csv")

                bands = [FrequencyBand.DELTA.name, FrequencyBand.THETA.name,
                         FrequencyBand.ALPHA.name, FrequencyBand.BETA.name,
                         FrequencyBand.GAMMA.name]
                feature_names = [
                    x.value for x in self.config['features'].get('temporal')]
                feature_names.extend([
                    f"{b_name}_{f_name.value}"
                    for b_name in bands
                    for f_name in self.config['features'].get('frequency')])
                feature_names.extend([
                    x.value
                    for x in self.config['features'].get('correlation')])
                feature_names.extend([
                    x.value for x in self.config['features'].get('graph')])

                grouped = pd.DataFrame({
                    substr.upper(): df.filter(
                        like=substr.upper()).stack().reset_index(drop=True)
                    for substr in feature_names})

                grouped.columns = [x.replace('_', ' ').title()
                                   for x in grouped.columns]
                mean_cols = grouped.mean(axis=0).sort_values(
                    ascending=False).head(10).index

                plt.figure(figsize=(10, 6))
                sns.set_palette("husl")
                sns.violinplot(data=grouped[mean_cols], orient='h',
                               scale='width', cut=0, inner=None, linewidth=1)
                plt.axvline(x=0.0, color='grey', linestyle='--', linewidth=1)
                plt.title('Global Feature Importance Using SHAP Values')
                plt.xlabel('SHAP Value')
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig(
                    f"shap_values_{model_config['model'].value}"
                    f"_{self.config['pipeline']['source_dataset'].value}"
                    f"_{self.config['pipeline']['dataset'].value}.png")

            else:
                for x in model.data_loader['test']:
                    label = x[1]
                    if torch.any(label == 0) and len(bckg) < 200:
                        for indice in list(
                                (x[1] == 0).nonzero().numpy().flatten()):
                            bckg.append(x[0][indice])
                    if torch.any(label == 1) and len(seiz) < 200:
                        for indice in list(
                                (label > 0).nonzero().numpy().flatten()):
                            seiz.append(x[0][indice])
                    if len(seiz) > 199 and len(bckg) > 199:
                        break
                explainer = shap.DeepExplainer(
                    model=model.model,
                    data=torch.reshape(torch.cat(bckg, dim=0),
                                       (-1, 18, 256)).to(model.dev))
                model.model.train()
                shap_values = explainer.shap_values(
                    torch.reshape(torch.cat(seiz, dim=0),
                                  (-1, 18, 256)).to(model.dev),
                    check_additivity=False)
                model.model.eval()
                logging.info(f'shap size: {shap_values.shape}')
                df = pd.DataFrame(shap_values[:, :, :, 1].reshape(
                    shap_values.shape[0] * shap_values.shape[1],
                    shap_values.shape[2]), columns=list(range(256)))
                df.to_csv(f"shap_values_{model_config['model'].value}"
                          f"_{self.config['pipeline']['source_dataset'].value}"
                          f"_{self.config['pipeline']['dataset'].value}.csv",
                          index=False)

                shap.image_plot(shap_values, torch.reshape(
                    torch.cat(seiz, dim=0), (-1, 18, 256)).numpy())
                plt.tight_layout()
                plt.savefig(
                    f"shap_values_{model_config['model'].value}"
                    f"_{self.config['pipeline']['source_dataset'].value}"
                    f"_{self.config['pipeline']['dataset'].value}.png")

            logging.info(f'{model_config["model"].value} model evaluation on '
                         f'{source_dataset.value} test data')
            tmp_out = model.predict(model.data_loader['test'])
            results.append({
                'y_pred': tmp_out[0],
                'y_true': tmp_out[1],
                'model': model_config['model'].value
            })

            tn, fp, fn, tp = get_confusion_matrix(
                (tmp_out[0] + 0.000000001).round(), tmp_out[1])
            logging.info(
                f"\n{model_config['model'].value} evaluation results:"
                f"\nmse\t\t{get_mse(tmp_out[0], tmp_out[1]):.5f}\n"
                f"roc\t\t{get_roc(tmp_out[0], tmp_out[1]):.5f}\n"
                f"accuracy\t{get_accuracy(tmp_out[0], tmp_out[1]):.5f}\n"
                f"precision\t{get_precision(tmp_out[0], tmp_out[1]):.5f}\n"
                f"f1\t\t{get_f1(tmp_out[0], tmp_out[1]):.5f}\n"
                f"prauc\t\t{get_prauc(tmp_out[0], tmp_out[1]):.5f}\n"
                f"sensitivity\t{get_sensitivity(tmp_out[0], tmp_out[1]):.5f}\n"
                f"specificity\t{get_specificity(tmp_out[0], tmp_out[1]):.5f}\n"
                f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")

            if classification_conf.get('post_processing', False):
                post_processor = PostProcessor(tmp_out[1], tmp_out[0])
                results_proc, y_pred_proc = post_processor.metrics_with_drift()

                post_results.append({
                    'y_pred': y_pred_proc,
                    'y_true': tmp_out[1],
                    'model': model_config['model'].value
                })
                logging.info(
                    f"\n{model_config['model'].value} "
                    f"results (Post-processing):"
                    f"\nmse\t\t{results_proc['mse']:.5f}\n"
                    f"roc\t\t{results_proc['roc']:.5f}\n"
                    f"accuracy\t{results_proc['accuracy']:.5f}\n"
                    f"precision\t{results_proc['precision']:.5f}\n"
                    f"f1\t\t{results_proc['f1']:.5f}\n"
                    f"prauc\t\t{results_proc['prauc']:.5f}\n"
                    f"sensitivity\t{results_proc['sensitivity']:.5f}\n"
                    f"specificity\t{results_proc['specificity']:.5f}\n"
                    f"TN: {results_proc['tn']} FP: {results_proc['fp']} "
                    f"FN: {results_proc['fn']} TP: {results_proc['tp']}")

        self._vote(results)
        if classification_conf.get('post_processing', False):
            self._vote(post_results, ' (Post-processing)')

    def _vote(self, results: Dict, comment: str = ''):
        min_len = np.min([x['y_true'].shape for x in results])
        y_pred = [(result['y_pred'] + 0.000000001).round()[:min_len]
                  for result in results]
        y_pred = np.median(np.array(y_pred), axis=0)
        y_mean_pred = np.array([result['y_pred'][:min_len]
                                for result in results]).mean(axis=0)
        y_true = np.array(results[0]['y_true'][:min_len])

        tn, fp, fn, tp = get_confusion_matrix(
            (y_pred + 0.000000001).round(), y_true)
        logging.info(
            f"\nBinary voting evaluation results{comment}:"
            f"\nmse\t\t{get_mse(y_pred, y_true):.5f}\n"
            f"roc\t\t{get_roc(y_pred, y_true):.5f}\n"
            f"accuracy\t{get_accuracy(y_pred, y_true):.5f}\n"
            f"precision\t{get_precision(y_pred, y_true):.5f}\n"
            f"f1\t\t{get_f1(y_pred, y_true):.5f}\n"
            f"prauc\t\t{get_prauc(y_pred, y_true):.5f}\n"
            f"sensitivity\t{get_sensitivity(y_pred, y_true):.5f}\n"
            f"specificity\t{get_specificity(y_pred, y_true):.5f}\n"
            f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")
        tn, fp, fn, tp = get_confusion_matrix(
            (y_mean_pred + 0.000000001).round(), y_true)
        logging.info(
            f"\nMean voting evaluation results{comment}:"
            f"\nmse\t\t{get_mse(y_mean_pred, y_true):.5f}\n"
            f"roc\t\t{get_roc(y_mean_pred, y_true):.5f}\n"
            f"accuracy\t{get_accuracy(y_mean_pred, y_true):.5f}\n"
            f"precision\t{get_precision(y_mean_pred, y_true):.5f}\n"
            f"f1\t\t{get_f1(y_mean_pred, y_true):.5f}\n"
            f"prauc\t\t{get_prauc(y_mean_pred, y_true):.5f}\n"
            f"sensitivity\t{get_sensitivity(y_mean_pred, y_true):.5f}\n"
            f"specificity\t{get_specificity(y_mean_pred, y_true):.5f}\n"
            f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")

