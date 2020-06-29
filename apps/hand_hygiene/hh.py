"""hh

Usage:
  hh clean ROOT PID [SR] [--date_range=<date_range>] [--auto_range=<auto_range>] [--skip-to-mhealth] [--debug]
  hh --help
  hh --version

Arguments:
  FOLDER                                        Dataset folder.
  PID                                           Participant ID.
  SR                                            Sampling rate in Hz.

Options:
  --date_range=<date_range>                     Date range. E.g., "--date_range 2020-06-01,2020-06-10", or "--date_range 2020-06-01," or "--date_range ,2020-06-10".
  --auto_range=<auto_range>                     Auto date freq. Default is "W-SUN", or weekly starting from Sunday.
  -h, --help                                    Show help message.
  -v, --version                                 Program/app version.
"""

import _version
import clean_up
import sync
import arus
import sys

from docopt import docopt
from loguru import logger


def main():
    arguments = docopt(__doc__, version=f'hand hygiene {_version.__version__}')
    logger.remove()
    if arguments['--debug']:
        logger.add(sys.stderr, level='DEBUG')
    else:
        logger.add(sys.stderr, level='INFO')
    logger.debug(arguments)
    if arguments['clean']:
        root = arguments['ROOT']
        pid = arguments['PID']
        sr = int(arguments['SR']) if arguments['SR'] is not None else None
        date_range = arguments['--date_range'].split(
            ',') if arguments['--date_range'] is not None else None
        auto_range = arguments['--auto_range'] or "W-SUN"
        if not arguments['--skip-to-mhealth']:
            clean_up.convert_to_mhealth(root, pid)
        sync.sync(root, pid)
        if sr is None:
            logger.warning(
                'Signaligner conversion is skipped because SR is None')
        else:
            arus.cli.convert_to_signaligner_both(
                root, pid, sr, date_range, auto_range)


if __name__ == "__main__":
    main()
