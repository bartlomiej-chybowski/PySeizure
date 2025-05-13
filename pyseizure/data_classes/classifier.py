from enum import Enum


class Classifier(Enum):
    LOGISTIC_REGRESSION = 'LR'
    XGBOOST = 'XGB'
    CNN = 'CNN'
    EEGNET = 'EEGNet'
    CONV_LSTM = 'ConvLSTM'
    MLP = 'MLP'  # MultilayerPerceptron
    CONV_TRANSFORMER = 'ConvTransformer'
