# PySeizure

---
[![DOI](https://zenodo.org/badge/DOI/DOI.NUMBER/zenodo.NUMBER.svg)](https://doi.org/DOI.NUMBER/zenodo.NUMBER)
[![Project Status: Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)

Analyse EEG data and detect epileptic seizures.

## Introduction
PySeizure is an open-source machine learning framework for automated and generalisable seizure detection across diverse clinical EEG datasets. 
It includes standardised preprocessing, automated feature extraction, and a majority voting ensemble of models for robust classification. 
Tested on CHB-MIT and TUSZ datasets, PySeizure achieves strong within- and cross-dataset performance without dataset-specific manual tuning. 

## Documentation
To read the documentation please open `docs/index.html`

## Example usage
Below code presents configuration that will:
1. Travers the TUSZ data folder (it should contain EDF files).
2. Apply filters and split recordings into 1 second epochs.
2. Augment data and extract features from all files recorded with average reference montage.
3. Create Train, Test, and Eval databases specified by `db`.
4. Normalise data.
5. Select best performing features using Boruta feature selector.
5. Tune hyperparameters and train selected models.
6. Run evaluation within TUSZ dataset (specified by `calc` and `raw`) using selected models with post-processing.

```python
import optuna

from pyseizure.py_seizure import PySeizure
from pyseizure.data_classes.feature import Feature
from pyseizure.data_classes.montage import Montage
from pyseizure.data_classes.dataset import Dataset
from pyseizure.data_classes.objective import Objective
from pyseizure.data_classes.classifier import Classifier
from pyseizure.data_classes.feature_selector import FeatureSelector



def main():
    config = {
        'no_threads_per_core': 4,
        'single_channel': False,  # True for calculated features
        'generalised': True, 
        'binary': True,
        'root_dir': 'path/to/the/data/TUSZ/data',
        'dataset_config': {
            Dataset.TUSZ: {
                'montage': {
                    'ar': 'path/to/the/data/TUSZ/DOCS/01_tcp_ar_montage.txt',
                    'le': 'path/to/the/data/TUSZ/DOCS/02_tcp_le_montage.txt',
                    'ar_a': 'path/to/the/data/TUSZ/DOCS/03_tcp_ar_a_montage.txt',
                    'le_a': 'path/to/the/data/TUSZ/DOCS/04_tcp_le_a_montage.txt',
                },
                'symbols': 'path/to/the/data/TUSZ/DOCS/nedc_ann_eeg_tools_map_v01.txt',
                'db': {
                    'train': 'path/to/the/data/input/tusz/train.db',
                    'valid': 'path/to/the/data/input/tusz/eval.db',
                    'test': 'path/to/the/data/input/tusz/test.db'
                },
                'calc': {
                    'train': 'path/to/the/data/input/tusz/train.db',
                    'valid': 'path/to/the/data/input/tusz/eval.db',
                    'test': 'path/to/the/data/input/tusz/test.db'
                },
                'raw': {
                    'train': 'path/to/the/data/input/tusz/train_raw.db',
                    'valid': 'path/to/the/data/input/tusz/eval_raw.db',
                    'test': 'path/to/the/data/input/tusz/test_raw.db'
                },
            },
            Dataset.CHBMIT: {
                'embed_annotations': False,
                'db': {
                    'train': 'path/to/the/data/input/chbmit/train.db',
                    'valid': 'path/to/the/data/input/chbmit/eval.db',
                    'test': 'path/to/the/data/input/chbmit/test.db'
                },
                'calc': {
                    'train': 'path/to/the/data/input/chbmit/train.db',
                    'valid': 'path/to/the/data/input/chbmit/eval.db',
                    'test': 'path/to/the/data/input/chbmit/test.db'
                },
                'raw': {
                    'train': 'path/to/the/data/input/chbmit/train_raw.db',
                    'valid': 'path/to/the/data/input/chbmit/eval_raw.db',
                    'test': 'path/to/the/data/input/chbmit/test_raw.db'
                },
            },
            Dataset.GENERIC: {
                'db': {
                    'train': 'path/to/the/data/input/generic/train.db',
                    'valid': 'path/to/the/data/input/generic/eval.db',
                    'test': 'path/to/the/data/input/generic/test.db'
                },
                'calc': {
                    'train': 'path/to/the/data/input/generic/train.db',
                    'valid': 'path/to/the/data/input/generic/eval.db',
                    'test': 'path/to/the/data/input/generic/test.db'
                },
                'raw': {
                    'train': 'path/to/the/data/input/generic/train_raw.db',
                    'valid': 'path/to/the/data/input/generic/eval_raw.db',
                    'test': 'path/to/the/data/input/generic/test_raw.db'
                }
            }
        },
        'features': {
            'temporal': [
                Feature.MEAN, Feature.VARIANCE, Feature.SKEWNESS,
                Feature.KURTOSIS, Feature.INTERQUARTILE_RANGE, Feature.MIN,
                Feature.MAX, Feature.HJORTH_MOBILITY,
                Feature.HJORTH_COMPLEXITY, Feature.PETROSIAN_FRACTAL_DIMENSION,
                Feature.SPIKE_COUNT, Feature.COASTLINE, Feature.INTERMITTENCY,
                Feature.VOLTAGE_AUC, Feature.SPIKINESS,
                Feature.STANDARD_DEVIATION, Feature.ZERO_CROSSING,
                Feature.PEAK_TO_PEAK, Feature.ABSOLUTE_AREA_UNDER_SIGNAL,
                Feature.TOTAL_SIGNAL_ENERGY
            ],
            'frequency': [
                Feature.POWER_SPECTRAL_DENSITY,
                Feature.POWER_SPECTRAL_CENTROID, Feature.SIGNAL_MONOTONY,
                Feature.SIGNAL_TO_NOISE, Feature.COASTLINE,
                Feature.DISCRETE_WAVELET_TRANSFORM,
                Feature.ENERGY_PERCENTAGE
            ],
            'correlation': [
                Feature.CROSS_CORRELATION_MAX_COEF, Feature.COHERENCE,
                Feature.PHASE_SLOPE_INDEX, Feature.IMAGINARY_COHERENCE
            ],
            'entropy': [],
            'graph': [
                Feature.ECCENTRICITY, Feature.CLUSTERING_COEFFICIENT,
                Feature.BETWEENNESS_CENTRALITY, Feature.LOCAL_EFFICIENCY,
                Feature.GLOBAL_EFFICIENCY, Feature.DIAMETER, Feature.RADIUS,
                Feature.CHARACTERISTIC_PATH
            ]
        },
        'edf_reader': {
            'external_annotation': True, # TUSZ
            # 'external_annotation': False, # CHBMIT, Generic
            'csd': False,
            'size_factor': 1.0,
            'rereference': True,
            'data_augmentation': True,
            'save_result': False,
            'epoch': 256,
            'frequency': 256,
            'annotation_labels': ['bckg', 'seiz'],
            'artefacts': False, # Set to TRUE if artefacts are annotated
            'seiz_label': 'seiz',
            'channels': ['T7', 'P7', 'T8', 'P8', 'FP1', 'F7', 'F3', 'C3', 'P3',
                         'FP2', 'F4', 'C4', 'P4', 'F8', 'FZ', 'CZ', 'O1', 'O2',
                         'PZ', 'T3', 'T4', 'T5', 'T6'],
            'channel_pairs': [('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'),
                              ('T5', 'O1'), ('FP1', 'F3'), ('F3', 'C3'),
                              ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'),
                              ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
                              ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'),
                              ('T6', 'O2'), ('FZ', 'CZ'), ('CZ', 'PZ')],
            'montage': Montage.AVERAGE_REFERENCE,
            # 'montage': Montage.AVERAGE_REFERENCE_A,
            # 'montage': Montage.LINKED_EARS,
            # 'montage': Montage.BIPOLAR,
            # 'montage': Montage.OTHER,
        },
        'feature_extractor': {
            'parallel_files': 1,
            'save_result': True,
            'override': True,
        },
        'pipeline': {
            'dataset': Dataset.TUSZ,
            # 'dataset': Dataset.CHBMIT,
            # 'dataset': Dataset.GENERIC,
            'source_dataset': Dataset.TUSZ,
            'raw': False,
            'feature_extraction': True,
            'create_db': True,
            'filter': {
                'montage': Montage.AVERAGE_REFERENCE,
                # 'montage': Montage.AVERAGE_REFERENCE_A,
                # 'montage': Montage.LINKED_EARS,
                # 'montage': Montage.BIPOLAR,
                # 'montage': Montage.OTHER,
                'seizures_only': False,
            },
            'normalisation': True,
            'feature_selection': {
                # 'selector': None,
                'selector': FeatureSelector.BORUTA,
                # 'selector': FeatureSelector.SVMRFECV,
                'max_iter': 20,
                'weight': 'auto'
            },
            'classification': {
                'models': [
                    Classifier.LOGISTIC_REGRESSION,
                    Classifier.XGBOOST,
                    Classifier.MLP,
                    # Classifier.CNN,
                    # Classifier.CONV_LSTM,
                    # Classifier.EEGNET,
                    # Classifier.CONV_TRANSFORMER
                ],
                'tune': True,
                'tune_config': {
                    'n_jobs': 1,
                    'n_trials': 12,
                    'n_epochs': 10,
                    'tune_objective': Objective.FPFN,
                    'sampler': optuna.samplers.TPESampler(), # Tree-structured Parzen Estimator
                    'pruner': optuna.pruners.HyperbandPruner()
                },
                'epoch': 100,
                'labels': ['bckg', 'seiz'],
                'subsampling': True,
                'post_processing': True,
                'weight': 'auto',
                'use_scheduler': True,
                'use_contrastive': False,
                'use_scoring': True
            }
        }
    }

    pyseizure = PySeizure(config)
    pyseizure.run()
    pyseizure.evaluate_vote()


if __name__ == '__main__':
    main()
```

## Credits

If you use this code in your project use the citation below:

    @article{Chybowski_PySeizure_2025,
        title={PySeizure: A single machine learning classifier framework to detect seizures in diverse datasets},
        author={Bartlomiej Chybowski},
        year={2025},
        doi={},
        url={},
        howpublished={\url{https://github.com/bartlomiej-chybowski/PySeizure}},
    }
