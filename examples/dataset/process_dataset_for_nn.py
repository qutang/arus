

from arus import dataset
from arus import developer

if __name__ == "__main__":
    developer.set_default_logging()
    dataset.process_dataset('spades_lab', approach='nn')
