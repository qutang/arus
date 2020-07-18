from ..plugins import actigraph
from .. import testing
from .. import mhealth_format as mh
from .. import dataset
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=['actigraph_imu', 'actigraph_accel'])
def test_file(request):
    return dataset.get_sample_datapath(request.param)


class TestActigraphReader:
    @pytest.mark.parametrize('chunksize', [None, 5])
    def test_read(self, test_file, chunksize):
        reader = actigraph.ActigraphReader(
            test_file, has_ts=True, has_header=True)
        reader.read(chunksize=chunksize)
        results = []
        for data in reader.get_data():
            results.append(data)
        if chunksize is None:
            assert len(results) == 1
        else:
            assert len(results) > 1
        if 'imu' in test_file:
            assert results[0].shape[1] == 11
        else:
            assert results[0].shape[1] == 4
        assert results[0].columns[0] == mh.constants.TIMESTAMP_COL

        meta = reader.get_meta()
        assert type(meta) == dict
        assert len(meta['VERSION_CODE']) == 6
        assert meta['SAMPLING_RATE'] > 30
        assert meta['SENSOR_ID'].startswith('TAS')
        if 'imu' in test_file:
            assert meta['IMU']
            assert meta['DYNAMIC_RANGE'] == 16
        else:
            assert not meta['IMU']
            assert meta['DYNAMIC_RANGE'] == 8


class TestActigraphSensorFileGenerator:
    @pytest.mark.parametrize('buffer_size', [None, 5])
    def test_generate(self, test_file, buffer_size):
        gr = actigraph.ActigraphSensorFileGenerator(
            test_file, buffer_size=buffer_size)
        sizes = []
        gr.run()
        for data, _ in gr.get_result():
            if data is None:
                break
            assert type(data) == pd.DataFrame
            sizes.append(data.shape[0])
        if buffer_size is None:
            assert len(sizes) == 1
            assert sizes[0] > 5
        else:
            sizes = sizes[:-1]
            assert np.all(np.array(sizes) == 5)
