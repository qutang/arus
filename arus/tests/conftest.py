import pytest
from .. import dataset


@pytest.fixture(scope="module")
def spades_lab_data():
    return dataset.load_dataset('spades_lab')
