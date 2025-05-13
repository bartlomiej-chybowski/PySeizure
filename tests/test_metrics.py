from unittest import TestCase
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

from pyseizure.helpers.metrics import PostProcessor, get_f1
from pyseizure.helpers.metrics import PostProcessor, get_mse, get_accuracy, \
    get_roc, get_precision, get_f1, get_prauc, get_sensitivity, get_specificity

class TestPostProcessor(TestCase):
    def setUp(self) -> None:
        self.postprocessor1 = PostProcessor(
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0]), # e: 6-8; n: 0-5, 9-9
            np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0])) # e: 1-6; n: 0-0, 7-9
        self.postprocessor2 = PostProcessor(
            np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]), # e: 1-1, 4-6, 9-10; n: 0-0, 2-3, 7-8, 11-11
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])) # e: 3-11; n: 0-2
        self.postprocessor3 = PostProcessor(
            np.array([0, 0, 0, 0]), # e: ; n: 0-3
            np.array([0, 1, 1, 0])) # e: 1-2; n: 0-0, 3-3
        self.postprocessor4 = PostProcessor(
            np.array([0, 0, 0, 0]), # e: ; n: 0-3
            np.array([0, 0, 0, 0])) # e: ; n: 0-3

        self.postprocessor_gap1 = PostProcessor(
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]),
            np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0])
        )

        self.ppmd1 = PostProcessor(
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      1, 1, 1, 0, 1, 1, 1, 0, 0, 0]), # e: 6-13, 18-20, 22-24
            np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), # e: 1-1, 5-7, 9-12, 18-25
        )

        self.ppmd2 = PostProcessor(
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      1, 1, 1, 0, 1, 1, 1, 0, 0, 0]), # e: 6-13, 18-20, 22-24
            np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                      1, 1, 1, 0, 0, 1, 1, 1, 0, 0]), # e: 1-2, 5-7, 9-12, 18-20, 23-25
        )

        self.ppmd3 = PostProcessor(
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                      1, 1, 1, 0, 1, 1, 1, 0, 0, 0]), # e: 6-8, 11-13, 18-20, 22-24
            np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), # e: 1-1, 5-7, 9-12, 18-25
        )

    @staticmethod
    def _is_close(a, b, rtol=1e-05, atol=1e-08):
        for key, value in a.items():
            yield np.isclose(value, b[key], rtol=rtol, atol=atol)

    def test_ovlp(self):
        assert self.postprocessor1.ovlp() == (0, 0, 0, 1)
        assert self.postprocessor2.ovlp() == (0, 0, 1, 2)
        assert self.postprocessor3.ovlp() == (0, 1, 0, 0)
        assert self.postprocessor4.ovlp() == (1, 0, 0, 0)


    def test__get_event_segments(self):
        assert np.all(self.postprocessor1._get_event_segments(
            self.postprocessor1.y_true) == np.array([[6, 8]]))
        assert np.all(self.postprocessor1._get_event_segments(
            self.postprocessor1.y_pred) == np.array([1, 6]))

        assert np.all(self.postprocessor2._get_event_segments(
            self.postprocessor2.y_true) == np.array([[1, 1], [4, 6], [9, 10]]))
        assert np.all(self.postprocessor2._get_event_segments(
            self.postprocessor2.y_pred) == np.array([3, 11]))

        assert np.all(self.postprocessor3._get_event_segments(
            self.postprocessor3.y_true) == np.array([[]]).reshape(0, 2))
        assert np.all(self.postprocessor3._get_event_segments(
            self.postprocessor3.y_pred) == np.array([1, 2]))

    def test_get_score(self):
        assert self.postprocessor1.get_score() == -188.0

    def test_process(self):

        assert np.all(self.postprocessor_gap1.process() == np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]))
        no_postprocessing = get_f1(self.postprocessor_gap1.y_pred,
                                   self.postprocessor_gap1.y_true)
        postprocessing = get_f1(self.postprocessor_gap1.process(),
                                self.postprocessor_gap1.y_true)

        assert np.isclose(no_postprocessing, 0.88, atol=0.01)
        assert np.isclose(postprocessing, 1.0, atol=0.01)

    def test_compute_metrics_with_drift(self):
        results1 = self.ppmd1.metrics_with_drift()
        assert np.all(results1[1] == np.array([0, 0, 0, 0, 0, 0.49, 1, 1, 1, 1,
                                               1, 1, 1, 0.51, 0, 0, 0, 0, 1, 1,
                                               1, 1, 1, 1, 1, 0.49, 0, 0]))
        results2 = self.ppmd2.metrics_with_drift()
        assert np.all(results2[1] == np.array([0, 1, 1, 0, 0, 0.49, 1, 1, 1, 1,
                                               1, 1, 1, 0.51, 0, 0, 0, 0, 1, 1,
                                               1, 0, 0, 1, 1, 0.49, 0, 0]))
        results3 = self.ppmd3.metrics_with_drift()
        assert np.all(results3[1] == np.array([0, 0, 0, 0, 0, 0.49, 1, 1, 1, 1,
                                               1, 1, 1, 0.51, 0, 0, 0, 0, 1, 1,
                                               1, 1, 1, 1, 1, 0.49, 0, 0]))

        expected = {
            'mse': 0.02,
            'roc': 1.0,
            'accuracy': 1.0,
            'precision': 1.0,
            'f1': 1.0,
            'prauc': 1.0,
            'sensitivity': 1.0,
            'specificity': 1.0
        }
        for key, value in results1[0].items():
            assert np.isclose(value, expected[key], atol=0.01)

        expected ={
            'mse': 0.16,
            'roc': 0.84,
            'accuracy': 0.85,
            'precision': 0.85,
            'f1': 0.85,
            'prauc': 0.88,
            'sensitivity': 0.85,
            'specificity': 0.85
        }
        for key, value in results2[0].items():
            assert np.isclose(value, expected[key], atol=0.01)

        expected = {
            'mse': 0.09,
            'roc': 0.92,
            'accuracy': 0.92,
            'precision': 0.93,
            'f1': 0.92,
            'prauc': 0.92,
            'sensitivity': 0.92,
            'specificity': 0.93
        }
        for key, value in results3[0].items():
            assert np.isclose(value, expected[key], atol=0.01)
