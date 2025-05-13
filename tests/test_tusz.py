from unittest import TestCase
from unittest.mock import patch
from pandas.testing import assert_frame_equal
import pandas as pd

from pyseizure.helpers.tusz import TUSZ


class TestTUSZ(TestCase):
    def setUp(self) -> None:
        with patch('pyseizure.helpers.tusz.TUSZ._get_symbols',
                   return_value=['null', 'spsw', 'gped', 'pled', 'eybl',
                                 'artf', 'bckg', 'seiz', 'fnsz', 'gnsz',
                                 'spsz', 'cpsz', 'absz', 'tnsz', 'cnsz',
                                 'tcsz', 'atsz', 'mysz', 'nesz', 'intr',
                                 'slow', 'eyem', 'chew', 'shiv', 'musc',
                                 'elpp', 'elst', 'calb', 'hphs', 'trip',
                                 'elec', 'eyem_chew', 'eyem_shiv', 'eyem_musc',
                                 'eyem_elec', 'chew_shiv', 'chew_musc',
                                 'chew_elec', 'shiv_musc', 'shiv_elec',
                                 'musc_elec']), \
             patch('pyseizure.helpers.tusz.TUSZ._get_montages',
                   return_value={
                       'ar': [{
                           'channel': 'FP1-F7',
                           'ref1_channel': 'EEG FP1-REF',
                           'ref2_channel': 'EEG F7-REF'
                       }, {
                           'channel': 'F7-T3',
                           'ref1_channel': 'EEG F7-REF',
                           'ref2_channel': 'EEG T3-REF'
                       }, {
                           'channel': 'T3-T5',
                           'ref1_channel': 'EEG T3-REF',
                           'ref2_channel': 'EEG T5-REF'
                       }, {
                           'channel': 'T5-O1',
                           'ref1_channel': 'EEG T5-REF',
                           'ref2_channel': 'EEG O1-REF'
                       }, {
                           'channel': 'FP2-F8',
                           'ref1_channel': 'EEG FP2-REF',
                           'ref2_channel': 'EEG F8-REF'
                       }, {
                           'channel': 'F8-T4',
                           'ref1_channel': 'EEG F8-REF',
                           'ref2_channel': 'EEG T4-REF'
                       }, {
                           'channel': 'T4-T6',
                           'ref1_channel': 'EEG T4-REF',
                           'ref2_channel': 'EEG T6-REF'
                       }, {
                           'channel': 'T6-O2',
                           'ref1_channel': 'EEG T6-REF',
                           'ref2_channel': 'EEG O2-REF'
                       }, {
                           'channel': 'A1-T3',
                           'ref1_channel': 'EEG A1-REF',
                           'ref2_channel': 'EEG T3-REF'
                       }, {
                           'channel': 'T3-C3',
                           'ref1_channel': 'EEG T3-REF',
                           'ref2_channel': 'EEG C3-REF'
                       }, {
                           'channel': 'C3-CZ',
                           'ref1_channel': 'EEG C3-REF',
                           'ref2_channel': 'EEG CZ-REF'
                       }, {
                           'channel': 'CZ-C4',
                           'ref1_channel': 'EEG CZ-REF',
                           'ref2_channel': 'EEG C4-REF'
                       }, {
                           'channel': 'C4-T4',
                           'ref1_channel': 'EEG C4-REF',
                           'ref2_channel': 'EEG T4-REF'
                       }, {
                           'channel': 'T4-A2',
                           'ref1_channel': 'EEG T4-REF',
                           'ref2_channel': 'EEG A2-REF'
                       }, {
                           'channel': 'FP1-F3',
                           'ref1_channel': 'EEG FP1-REF',
                           'ref2_channel': 'EEG F3-REF'
                       }, {
                           'channel': 'F3-C3',
                           'ref1_channel': 'EEG F3-REF',
                           'ref2_channel': 'EEG C3-REF'
                       }, {
                           'channel': 'C3-P3',
                           'ref1_channel': 'EEG C3-REF',
                           'ref2_channel': 'EEG P3-REF'
                       }, {
                           'channel': 'P3-O1',
                           'ref1_channel': 'EEG P3-REF',
                           'ref2_channel': 'EEG O1-REF'
                       }, {
                           'channel': 'FP2-F4',
                           'ref1_channel': 'EEG FP2-REF',
                           'ref2_channel': 'EEG F4-REF'
                       }, {
                           'channel': 'F4-C4',
                           'ref1_channel': 'EEG F4-REF',
                           'ref2_channel': 'EEG C4-REF'
                       }, {
                           'channel': 'C4-P4',
                           'ref1_channel': 'EEG C4-REF',
                           'ref2_channel': 'EEG P4-REF'
                       }, {
                           'channel': 'P4-O2',
                           'ref1_channel': 'EEG P4-REF',
                           'ref2_channel': 'EEG O2-REF'
                       }],
                       'le': [{
                           'channel': 'FP1-F7',
                           'ref1_channel': 'EEG FP1-LE',
                           'ref2_channel': 'EEG F7-LE'
                       }, {
                           'channel': 'F7-T3',
                           'ref1_channel': 'EEG F7-LE',
                           'ref2_channel': 'EEG T3-LE'
                       }, {
                           'channel': 'T3-T5',
                           'ref1_channel': 'EEG T3-LE',
                           'ref2_channel': 'EEG T5-LE'
                       }, {
                           'channel': 'T5-O1',
                           'ref1_channel': 'EEG T5-LE',
                           'ref2_channel': 'EEG O1-LE'
                       }, {
                           'channel': 'FP2-F8',
                           'ref1_channel': 'EEG FP2-LE',
                           'ref2_channel': 'EEG F8-LE'
                       }, {
                           'channel': 'F8-T4',
                           'ref1_channel': 'EEG F8-LE',
                           'ref2_channel': 'EEG T4-LE'
                       }, {
                           'channel': 'T4-T6',
                           'ref1_channel': 'EEG T4-LE',
                           'ref2_channel': 'EEG T6-LE'
                       }, {
                           'channel': 'T6-O2',
                           'ref1_channel': 'EEG T6-LE',
                           'ref2_channel': 'EEG O2-LE'
                       }, {
                           'channel': 'A1-T3',
                           'ref1_channel': 'EEG A1-LE',
                           'ref2_channel': 'EEG T3-LE'
                       }, {
                           'channel': 'T3-C3',
                           'ref1_channel': 'EEG T3-LE',
                           'ref2_channel': 'EEG C3-LE'
                       }, {
                           'channel': 'C3-CZ',
                           'ref1_channel': 'EEG C3-LE',
                           'ref2_channel': 'EEG CZ-LE'
                       }, {
                           'channel': 'CZ-C4',
                           'ref1_channel': 'EEG CZ-LE',
                           'ref2_channel': 'EEG C4-LE'
                       }, {
                           'channel': 'C4-T4',
                           'ref1_channel': 'EEG C4-LE',
                           'ref2_channel': 'EEG T4-LE'
                       }, {
                           'channel': 'T4-A2',
                           'ref1_channel': 'EEG T4-LE',
                           'ref2_channel': 'EEG A2-LE'
                       }, {
                           'channel': 'FP1-F3',
                           'ref1_channel': 'EEG FP1-LE',
                           'ref2_channel': 'EEG F3-LE'
                       }, {
                           'channel': 'F3-C3',
                           'ref1_channel': 'EEG F3-LE',
                           'ref2_channel': 'EEG C3-LE'
                       }, {
                           'channel': 'C3-P3',
                           'ref1_channel': 'EEG C3-LE',
                           'ref2_channel': 'EEG P3-LE'
                       }, {
                           'channel': 'P3-O1',
                           'ref1_channel': 'EEG P3-LE',
                           'ref2_channel': 'EEG O1-LE'
                       }, {
                           'channel': 'FP2-F4',
                           'ref1_channel': 'EEG FP2-LE',
                           'ref2_channel': 'EEG F4-LE'
                       }, {
                           'channel': 'F4-C4',
                           'ref1_channel': 'EEG F4-LE',
                           'ref2_channel': 'EEG C4-LE'
                       }, {
                           'channel': 'C4-P4',
                           'ref1_channel': 'EEG C4-LE',
                           'ref2_channel': 'EEG P4-LE'
                       }, {
                           'channel': 'P4-O2',
                           'ref1_channel': 'EEG P4-LE',
                           'ref2_channel': 'EEG O2-LE'
                       }],
                       'ar_a': [{
                           'channel': 'FP1-F7',
                           'ref1_channel': 'EEG FP1-REF',
                           'ref2_channel': 'EEG F7-REF'
                       }, {
                           'channel': 'F7-T3',
                           'ref1_channel': 'EEG F7-REF',
                           'ref2_channel': 'EEG T3-REF'
                       }, {
                           'channel': 'T3-T5',
                           'ref1_channel': 'EEG T3-REF',
                           'ref2_channel': 'EEG T5-REF'
                       }, {
                           'channel': 'T5-O1',
                           'ref1_channel': 'EEG T5-REF',
                           'ref2_channel': 'EEG O1-REF'
                       }, {
                           'channel': 'FP2-F8',
                           'ref1_channel': 'EEG FP2-REF',
                           'ref2_channel': 'EEG F8-REF'
                       }, {
                           'channel': 'F8-T4',
                           'ref1_channel': 'EEG F8-REF',
                           'ref2_channel': 'EEG T4-REF'
                       }, {
                           'channel': 'T4-T6',
                           'ref1_channel': 'EEG T4-REF',
                           'ref2_channel': 'EEG T6-REF'
                       }, {
                           'channel': 'T6-O2',
                           'ref1_channel': 'EEG T6-REF',
                           'ref2_channel': 'EEG O2-REF'
                       }, {
                           'channel': 'T3-C3',
                           'ref1_channel': 'EEG T3-REF',
                           'ref2_channel': 'EEG C3-REF'
                       }, {
                           'channel': 'C3-CZ',
                           'ref1_channel': 'EEG C3-REF',
                           'ref2_channel': 'EEG CZ-REF'
                       }, {
                           'channel': 'CZ-C4',
                           'ref1_channel': 'EEG CZ-REF',
                           'ref2_channel': 'EEG C4-REF'
                       }, {
                           'channel': 'C4-T4',
                           'ref1_channel': 'EEG C4-REF',
                           'ref2_channel': 'EEG T4-REF'
                       }, {
                           'channel': 'FP1-F3',
                           'ref1_channel': 'EEG FP1-REF',
                           'ref2_channel': 'EEG F3-REF'
                       }, {
                           'channel': 'F3-C3',
                           'ref1_channel': 'EEG F3-REF',
                           'ref2_channel': 'EEG C3-REF'
                       }, {
                           'channel': 'C3-P3',
                           'ref1_channel': 'EEG C3-REF',
                           'ref2_channel': 'EEG P3-REF'
                       }, {
                           'channel': 'P3-O1',
                           'ref1_channel': 'EEG P3-REF',
                           'ref2_channel': 'EEG O1-REF'
                       }, {
                           'channel': 'FP2-F4',
                           'ref1_channel': 'EEG FP2-REF',
                           'ref2_channel': 'EEG F4-REF'
                       }, {
                           'channel': 'F4-C4',
                           'ref1_channel': 'EEG F4-REF',
                           'ref2_channel': 'EEG C4-REF'
                       }, {
                           'channel': 'C4-P4',
                           'ref1_channel': 'EEG C4-REF',
                           'ref2_channel': 'EEG P4-REF'
                       }, {
                           'channel': 'P4-O2',
                           'ref1_channel': 'EEG P4-REF',
                           'ref2_channel': 'EEG O2-REF'
                       }],
                       'le_a': [{
                           'channel': 'FP1-F7',
                           'ref1_channel': 'EEG FP1-LE',
                           'ref2_channel': 'EEG F7-LE'
                       }, {
                           'channel': 'F7-T3',
                           'ref1_channel': 'EEG F7-LE',
                           'ref2_channel': 'EEG T3-LE'
                       }, {
                           'channel': 'T3-T5',
                           'ref1_channel': 'EEG T3-LE',
                           'ref2_channel': 'EEG T5-LE'
                       }, {
                           'channel': 'T5-O1',
                           'ref1_channel': 'EEG T5-LE',
                           'ref2_channel': 'EEG O1-LE'
                       }, {
                           'channel': 'FP2-F8',
                           'ref1_channel': 'EEG FP2-LE',
                           'ref2_channel': 'EEG F8-LE'
                       }, {
                           'channel': 'F8-T4',
                           'ref1_channel': 'EEG F8-LE',
                           'ref2_channel': 'EEG T4-LE'
                       }, {
                           'channel': 'T4-T6',
                           'ref1_channel': 'EEG T4-LE',
                           'ref2_channel': 'EEG T6-LE'
                       }, {
                           'channel': 'T6-O2',
                           'ref1_channel': 'EEG T6-LE',
                           'ref2_channel': 'EEG O2-LE'
                       }, {
                           'channel': 'T3-C3',
                           'ref1_channel': 'EEG T3-LE',
                           'ref2_channel': 'EEG C3-LE'
                       }, {
                           'channel': 'C3-CZ',
                           'ref1_channel': 'EEG C3-LE',
                           'ref2_channel': 'EEG CZ-LE'
                       }, {
                           'channel': 'CZ-C4',
                           'ref1_channel': 'EEG CZ-LE',
                           'ref2_channel': 'EEG C4-LE'
                       }, {
                           'channel': 'C4-T4',
                           'ref1_channel': 'EEG C4-LE',
                           'ref2_channel': 'EEG T4-LE'
                       }, {
                           'channel': 'FP1-F3',
                           'ref1_channel': 'EEG FP1-LE',
                           'ref2_channel': 'EEG F3-LE'
                       }, {
                           'channel': 'F3-C3',
                           'ref1_channel': 'EEG F3-LE',
                           'ref2_channel': 'EEG C3-LE'
                       }, {
                           'channel': 'C3-P3',
                           'ref1_channel': 'EEG C3-LE',
                           'ref2_channel': 'EEG P3-LE'
                       }, {
                           'channel': 'P3-O1',
                           'ref1_channel': 'EEG P3-LE',
                           'ref2_channel': 'EEG O1-LE'
                       }, {
                           'channel': 'FP2-F4',
                           'ref1_channel': 'EEG FP2-LE',
                           'ref2_channel': 'EEG F4-LE'
                       }, {
                           'channel': 'F4-C4',
                           'ref1_channel': 'EEG F4-LE',
                           'ref2_channel': 'EEG C4-LE'
                       }, {
                           'channel': 'C4-P4',
                           'ref1_channel': 'EEG C4-LE',
                           'ref2_channel': 'EEG P4-LE'
                       }, {
                           'channel': 'P4-O2',
                           'ref1_channel': 'EEG P4-LE',
                           'ref2_channel': 'EEG O2-LE'
                       }]
                   }):
            self.tusz = TUSZ('', '', '')

    def test__complete_binary(self):
        df_bi = pd.DataFrame(
            columns=['channel', 'start_time', 'stop_time', 'label',
                     'confidence'],
            data=[['TERM', 92.03130, 160.88280, 'seiz', 1.0]])

        result = self.tusz._complete_binary(df_bi, 301.0)

        assert_frame_equal(result, pd.DataFrame(
            columns=['channel', 'start_time', 'stop_time', 'label',
                     'confidence'],
            data=[['TERM', 0.0, 92.03130, 'bckg', 1.0],
                  ['TERM', 92.03130, 160.88280, 'seiz', 1.0],
                  ['TERM', 160.88280, 301.0, 'bckg', 1.0]]))

    def test__complete_binary_2_rows(self):
        df_bi = pd.DataFrame(
            columns=['channel', 'start_time', 'stop_time', 'label',
                     'confidence'],
            data=[['TERM', 92.03130, 160.88280, 'seiz', 1.0],
                  ['TERM', 201.04120, 234.18570, 'seiz', 1.0]]
        )
        result = self.tusz._complete_binary(df_bi, 301.0)

        assert_frame_equal(result, pd.DataFrame(
            columns=['channel', 'start_time', 'stop_time', 'label',
                     'confidence'],
            data=[['TERM', 0.0, 92.03130, 'bckg', 1.0],
                  ['TERM', 92.03130, 160.88280, 'seiz', 1.0],
                  ['TERM', 160.88280, 201.04120, 'bckg', 1.0],
                  ['TERM', 201.04120, 234.18570, 'seiz', 1.0],
                  ['TERM', 234.18570, 301.0, 'bckg', 1.0]]))

    def test__complete_binary_multiple_rows(self):
        df_bi = pd.DataFrame(
            columns=['channel', 'start_time', 'stop_time', 'label',
                     'confidence'],
            data=[['TERM', 1.0000, 32.1156, 'seiz', 1.0000],
                  ['TERM', 150.0160, 229.2979, 'seiz', 1.0000],
                  ['TERM', 339.7800, 419.9534, 'seiz', 1.0000],
                  ['TERM', 499.5920, 565.1040, 'seiz', 1.0000],
                  ['TERM', 668.0240, 754.2560, 'seiz', 1.0000],
                  ['TERM', 845.7000, 887.1320, 'seiz', 1.0000],
                  ['TERM', 995.9520, 1058.0760, 'seiz', 1.0000],
                  ['TERM', 1180.4520, 1199.9400, 'seiz', 1.0000]]
        )
        result = self.tusz._complete_binary(df_bi, 1201.0)

        assert_frame_equal(result, pd.DataFrame(
            columns=['channel', 'start_time', 'stop_time', 'label',
                     'confidence'],
            data=[['TERM', 0.0000, 1.0000, 'bckg', 1.0],
                  ['TERM', 1.0000, 32.1156, 'seiz', 1.0],
                  ['TERM', 32.1156, 150.0160, 'bckg', 1.0],
                  ['TERM', 150.0160, 229.2979, 'seiz', 1.0],
                  ['TERM', 229.2979, 339.7800, 'bckg', 1.0],
                  ['TERM', 339.7800, 419.9534, 'seiz', 1.0],
                  ['TERM', 419.9534, 499.5920, 'bckg', 1.0],
                  ['TERM', 499.5920, 565.1040, 'seiz', 1.0],
                  ['TERM', 565.1040, 668.0240, 'bckg', 1.0],
                  ['TERM', 668.0240, 754.2560, 'seiz', 1.0],
                  ['TERM', 754.2560, 845.7000, 'bckg', 1.0],
                  ['TERM', 845.7000, 887.1320, 'seiz', 1.0],
                  ['TERM', 887.1320, 995.9520, 'bckg', 1.0],
                  ['TERM', 995.9520, 1058.0760, 'seiz', 1.0],
                  ['TERM', 1058.0760, 1180.4520, 'bckg', 1.0],
                  ['TERM', 1180.4520, 1199.9400, 'seiz', 1.0],
                  ['TERM', 1199.9400, 1201.0000, 'bckg', 1.0]]))
