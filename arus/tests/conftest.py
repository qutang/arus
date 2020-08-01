import pandas as pd
import pytest

from .. import dataset as ds


@pytest.fixture(scope="module")
def spades_lab_data():
    return ds.load_dataset('spades_lab')


@pytest.fixture(scope="module")
def spades_lab_ds():
    return ds.MHDataset(path=ds.get_dataset_path('spades_lab'),
                        name='spades_lab', input_type=ds.InputType.MHEALTH_FORMAT)


@pytest.fixture(scope="module")
def single_mhealth_sensor(spades_lab_ds):
    sensor = spades_lab_ds.get_sensors(pid='SPADES_1', placement='DW')[0]
    sensor_file = sensor.paths[0]
    data = pd.read_csv(sensor_file, parse_dates=[0])
    return data
