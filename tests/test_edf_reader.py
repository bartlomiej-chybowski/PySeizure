from unittest import TestCase

import numpy as np
import pandas as pd

from pyseizure.preprocessing.edf_reader import EDFReader

annotations = [
    [0, 0, 0.0, 325.9697, 'EEG FP1-REF', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
     0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0],
    [0, 0, 0.0, 325.9697, 'EEG F7-REF', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0],
    [0, 0, 325.9697, 507.1212, 'EEG FP1-REF', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 1.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0],
    [0, 0, 325.9697, 507.1212, 'EEG F7-REF', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 1.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0],
    [0, 0, 0.0, 349.1212, 'EEG FP1-REF', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
     0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0],
    [0, 0, 0.0, 349.1212, 'EEG F3-REF', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0]]
channel = {
    'channel': 0,
    'ref1_channel': 'EEG FP1-REF',
    'ref2_channel': 'EEG F7-REF'
}
edf_reader = EDFReader()


class TestEDFReader(TestCase):
    def test__set_time_proportions(self):
        assert edf_reader._set_time_proportions(annotations[0], 325) == [
            0, 0, 0.0, 325.9697, 'EEG FP1-REF', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.9697, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert edf_reader._set_time_proportions(annotations[2], 325) == [
            0, 0, 325.9697, 507.1212, 'EEG FP1-REF', 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0303, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert edf_reader._set_time_proportions(annotations[4], 325) == [
            0, 0, 0.0, 349.1212, 'EEG FP1-REF', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test__extract_labels_for_sample(self):
        assert edf_reader._extract_labels_for_sample(
            annotations, 'EEG FP1-REF', 325) == ['EEG FP1-REF', 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.9849, 0.0,
                                                 0.0, 0.0303, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0]

    def test__reverse_annotations(self):
        data = {
            'length': 256,
            'annotations': (
                pd.DataFrame(data=[['TERM', 0.0, 15.9455, 'bckg', 1.0],
                                   ['TERM', 15.9455, 71.9028, 'seiz', 1.0],
                                   ['TERM', 71.9028, 135.9218, 'bckg', 1.0],
                                   ['TERM', 135.9218, 163.0113, 'seiz', 1.0],
                                   ['TERM', 163.0113, 192.0569, 'bckg', 1.0],
                                   ['TERM', 192.0569, 243.0943, 'seiz', 1.0],
                                   ['TERM', 243.0943, 256.0, 'bckg', 1.0]],
                             columns=['channel', 'start_time', 'stop_time',
                                      'label', 'confidence']),
                pd.DataFrame(data=[['TERM', 0.0, 15.9455, 'bckg', 1.0],
                                   ['TERM', 15.9455, 71.9028, 'seiz', 1.0],
                                   ['TERM', 71.9028, 135.9218, 'bckg', 1.0],
                                   ['TERM', 135.9218, 163.0113, 'seiz', 1.0],
                                   ['TERM', 163.0113, 192.0569, 'bckg', 1.0],
                                   ['TERM', 192.0569, 243.0943, 'seiz', 1.0],
                                   ['TERM', 243.0943, 256.0, 'bckg', 1.0]],
                             columns=['channel', 'start_time', 'stop_time',
                                      'label', 'confidence']),
                [1, 2, 3],
                True
            )
        }

        edf_reader._reverse_annotations(data)

        result = pd.DataFrame(data=[['TERM', 0.0, 12.9057, 'bckg', 1.0],
                                    ['TERM', 12.9057, 63.9431, 'seiz', 1.0],
                                    ['TERM', 63.9431, 92.9887, 'bckg', 1.0],
                                    ['TERM', 92.9887, 120.0782, 'seiz', 1.0],
                                    ['TERM', 120.0782, 184.0972, 'bckg', 1.0],
                                    ['TERM', 184.0972, 240.0545, 'seiz', 1.0],
                                    ['TERM', 240.0545, 256.0, 'bckg', 1.0]],
                              columns=['channel', 'start_time', 'stop_time',
                                       'label', 'confidence'])

        assert np.allclose(data['annotations'][0]['start_time'].values,
                           result['start_time'].values)
        assert np.allclose(data['annotations'][0]['stop_time'].values,
                           result['stop_time'].values)
        assert np.equal(data['annotations'][0]['label'].values,
                        result['label'].values).all()

    def test__concat_if_no_overlap(self):
        df1 = pd.DataFrame(columns=['channel', 'start_time', 'stop_time',
                                    'label', 'confidence'],
                           data=[['TERM', 1176.0, 1180.0, 'seiz', 1.0],
                                 ['TERM', 1273.0, 1277.0, 'seiz', 1.0]])
        df2 = pd.DataFrame(columns=['channel', 'start_time', 'stop_time',
                                    'label', 'confidence'],
                           data=[['TERM', 0.0, 2584.7, 'test long', 1.0],
                                 ['TERM', 33.0, 35.0, 'test1', 1.0],
                                 ['TERM', 1160.0, 1162.0, 'test2', 1.0],
                                 ['TERM', 1175.0, 1176.0, 'test3', 1.0],
                                 ['TERM', 1176.0, 1178.0, 'test4', 1.0],
                                 ['TERM', 1177.0, 1178.0, 'test5', 1.0],
                                 ['TERM', 1177.0, 1179.0, 'test6', 1.0],
                                 ['TERM', 1180.0, 1182.0, 'test7', 1.0],
                                 ['TERM', 1182.0, 1184.0, 'test8', 1.0],
                                 ['TERM', 1260.0, 1262.0, 'test9', 1.0],
                                 ['TERM', 1270.0, 1274.0, 'test10', 1.0],
                                 ['TERM', 1272.0, 1287.0, 'test13', 1.0],
                                 ['TERM', 1273.0, 1277.0, 'test11', 1.0],
                                 ['TERM', 1274.0, 1276.0, 'test12', 1.0],
                                 ['TERM', 1276.0, 1277.0, 'test13', 1.0],
                                 ['TERM', 1276.0, 1278.0, 'test13', 1.0],
                                 ['TERM', 1276.0, 1279.0, 'test13', 1.0],
                                 ['TERM', 1280.0, 1282.0, 'test14', 1.0],
                                 ['TERM', 2583.0, 2585.0, 'test15', 1.0]])

        expected = pd.DataFrame(columns=['channel', 'start_time', 'stop_time',
                                         'label', 'confidence'],
                                data=[['TERM', 33.0, 35.0, 'test1', 1.0],
                                      ['TERM', 1160.0, 1162.0, 'test2', 1.0],
                                      ['TERM', 1175.0, 1176.0, 'test3', 1.0],
                                      ['TERM', 1176.0, 1180.0, 'seiz', 1.0],
                                      ['TERM', 1180.0, 1182.0, 'test7', 1.0],
                                      ['TERM', 1182.0, 1184.0, 'test8', 1.0],
                                      ['TERM', 1260.0, 1262.0, 'test9', 1.0],
                                      ['TERM', 1273.0, 1277.0, 'seiz', 1.0],
                                      ['TERM', 1280.0, 1282.0, 'test14', 1.0],
                                      ['TERM', 2583.0, 2585.0, 'test15', 1.0]])

        result = edf_reader._concat_if_no_overlap(df1, df2)
        assert pd.testing.assert_frame_equal(result, expected)
