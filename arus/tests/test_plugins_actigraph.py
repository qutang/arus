from ..plugins import actigraph
from .. import testing
from .. import mhealth_format as mh
import pytest


@pytest.fixture
def test_file():
    sensor_file, _ = testing.load_test_data(file_type='actigraph',
                                            file_num='single',
                                            exception_type='consistent_sr')
    return sensor_file


class TestActigraphReader:
    @pytest.mark.parametrize('chunksize', [None, 1000])
    def test_read_csv(self, test_file, chunksize):
        reader = actigraph.ActigraphReader(test_file)
        reader.read_csv(chunksize=chunksize)
        results = []
        for data in reader.get_data():
            results.append(data)
        if chunksize is None:
            assert len(results) == 1
            assert results[0].shape[0] > 1000
        else:
            assert len(results) > 1
            assert results[0].shape[0] == 1000
        assert results[0].shape[1] == 4
        assert results[0].columns[0] == mh.constants.TIMESTAMP_COL

    def test_read_meta(self, test_file):
        reader = actigraph.ActigraphReader(test_file)
        reader.read_meta()
        meta = reader.get_meta()
        assert type(meta) == dict
        assert len(meta['VERSION_CODE']) == 6
        assert meta['SAMPLING_RATE'] > 30
        assert meta['SENSOR_ID'].startswith('TAS')


class TestActigraphSensorFileGenerator:
    @pytest.mark.parametrize('buffer_size', [None, 1000])
    def test_generate(self, test_file, buffer_size):
        gr = actigraph.ActigraphSensorFileGenerator(
            test_file, buffer_size=buffer_size)
        results = []
        for data in gr.generate():
            results.append(data)
        if buffer_size is None:
            assert len(results) == 1
            assert results[0].shape[0] > 1000
        else:
            assert len(results) > 1
            assert results[0].shape[0] == 1000
        assert results[0].shape[1] == 4
        assert results[0].columns[0] == mh.constants.TIMESTAMP_COL
