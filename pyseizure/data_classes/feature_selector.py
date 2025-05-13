from enum import Enum


class FeatureSelector(Enum):
    BORUTA = 'boruta'
    SVMRFECV = 'svmrfecv'
