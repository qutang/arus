"""
Process raw dataset
============================================================================

"""

# %%
# Imports
# -----------
from arus import dataset
from arus import developer
import multiprocessing as mp

# %%
# process raw dataset
# ---------------------------------
if __name__ == "__main__":
    developer.set_default_logging()
    mp.freeze_support()
    dataset.process_raw_dataset('spades_lab')
