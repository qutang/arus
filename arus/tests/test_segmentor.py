import pandas as pd
import numpy as np
from .. import segmentor
from .. import moment
import pytest


@pytest.fixture
def sensor_data(spades_lab_data):
    sensor_file = spades_lab_data['subjects']['SPADES_47']['sensors']['DW'][0]
    df = pd.read_csv(sensor_file, header=0, parse_dates=[0])
    return df.iloc[:1000, :]


class TestSegmentor:
    def test_segment(self, sensor_data):
        df = pd.DataFrame(data=sensor_data)
        seg = segmentor.Segmentor()
        i = 0
        for sample, context in seg.segment(df):
            assert len(sample.shape) == 1
            assert len(sample) == 4
            np.testing.assert_array_equal(
                sample.values, sensor_data.iloc[i, :].values)
            i += 1


class TestSlidingWindowSegmentor:
    @pytest.mark.parametrize("window_size", [0, 1, 10])
    @pytest.mark.parametrize('ref_st', [None, pd.Timestamp.now()])
    def test_segment(self, sensor_data, window_size, ref_st):
        if window_size == 0:
            with pytest.raises(ValueError):
                seg = segmentor.SlidingWindowSegmentor(
                    window_size=window_size, ref_st=ref_st, st_col=0, et_col=None)
        else:
            seg = segmentor.SlidingWindowSegmentor(
                window_size=window_size, ref_st=ref_st, st_col=0, et_col=None)
            i = 0
            for sample, context in seg.segment(sensor_data):
                assert moment.Moment.get_duration(
                    context['start_time'], context['stop_time'], unit='s') == window_size
                assert sample.shape[0] <= window_size * 80
                assert sample.shape[1] == 4
                i += 1
            if ref_st is not None:
                assert i == 0
