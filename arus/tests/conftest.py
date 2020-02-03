import pytest
from .. import dataset


@pytest.fixture(scope="module")
def spades_lab():
    return dataset.load_dataset('spades_lab')
