"""arus command line application

Usage:
  arus signaligner [<file_type> -f | --folder <folder> -p | --pid <pid> --sr <sampling_rate> --method <method> --debug]
  arus app [<app_command> -f | --folder <folder> --name <name> --version <version>]
  arus -h | --help
  arus -v | --version

Options:
  -h, --help                        Show help message
  -v, --version                     Program/app version
  --pattern <file_pattern>          Filepath pattern to get the files
  --sr <sampling_rate>              The sampling rate of the converted data
  -f <folder>, --folder <folder>    Dataset folder.
  --name <name>                     Provided name string.
  --method <method>                 The method used for conversion: "interp" or "closest"
  --flat                            Flat mhealth folder structure (date folder only)
  --hourly                          Split into hourly files
  -p <pid>, --pid <pid>             Participant ID
  --data_id <data_id>               ID for the data file
"""

from docopt import docopt
from loguru import logger
from . import developer
from . import mhealth_format as mh
from .plugins import signaligner
import glob
import os
import alive_progress as progress
import sys
import subprocess


def cli():
    arguments = docopt(__doc__, version='arus {}'.format(
        developer._find_current_version(".", "arus")))
    if arguments['--debug']:
        logger.remove()
        logger.add(sys.stderr, level='DEBUG')
    else:
        logger.remove()
        logger.add(sys.stderr, level='INFO')
    logger.debug(arguments)

    if arguments['signaligner']:
        signaligner_command(arguments)
    elif arguments['app']:
        app_command(arguments)
    elif arguments['mhealth']:
        raise NotImplementedError("This command is not implemented yet")


def signaligner_command(arguments):
    pid = arguments['--pid'][0]
    root = arguments['--folder'][0]
    file_type = arguments['<file_type>']
    if file_type == 'sensor':
        sr = int(arguments['--sr'])
        method = arguments['--method']
        convert_to_signaligner(root, pid, file_type, sr=sr, method=method)
    elif file_type == 'annotation':
        sr = None
        method = None
        convert_to_signaligner(root, pid, file_type, sr=sr, method=method)
    elif file_type is None:
        sr = int(arguments['--sr'])
        method = arguments['--method']
        convert_to_signaligner_both(root, pid, sr=sr, method=method)


def convert_to_signaligner_both(root, pid, sr, method):
    convert_to_signaligner(root, pid, 'annotation', None, None)
    convert_to_signaligner(root, pid, 'sensor', sr, method)


def convert_to_signaligner(root, pid, file_type, sr, method):
    if file_type == 'annotation':
        filepaths = mh.get_annotation_files(pid, root)
        groups = set(map(mh.parse_annotation_type_from_filepath, filepaths))
    elif file_type == 'sensor':
        filepaths = mh.get_sensor_files(pid, root)
        session_span = mh.get_session_span(pid, root)
        groups = set(map(mh.parse_sensor_id_from_filepath, filepaths))
    logger.debug(groups)
    for group in groups:
        logger.info('Convert files of {}'.format(group))
        group_files = list(filter(lambda f: group in f, filepaths))
        sensor_type = mh.parse_sensor_type_from_filepath(group_files[0])

        # call signify functions
        if file_type == 'annotation':
            output_path = os.path.join(
                root, pid, mh.DERIVED_FOLDER, 'signaligner', '{}-labelsets'.format(pid), '{0}.{0}-Signaligner.{1}.csv'.format(group, file_type))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            signaligner.signify_annotation_files(
                group_files, group, output_path)
        elif file_type == 'sensor':
            # set output file paths
            output_path = os.path.join(
                root, pid, mh.DERIVED_FOLDER, 'signaligner', '{}-sensors'.format(pid), '{0}-{1}.{1}-Signaligner.{2}.csv'.format(sensor_type, group, file_type))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_annotation_path = os.path.join(os.path.dirname(
                output_path.replace('sensors', 'labelsets')), 'Missing.{}.annotation.csv'.format(group))
            os.makedirs(os.path.dirname(output_annotation_path), exist_ok=True)
            signaligner.signify_sensor_files(
                group_files, group, output_path, output_annotation_path, session_span, sr)


def app_command(arguments):
    if arguments['<app_command>'] == 'build':
        root = arguments['--folder'][0]
        name = arguments['--name']
        version = arguments['--version'] or developer._find_current_version(
            '.', 'arus')
        developer.build_arus_app(root, name, version)
    elif arguments['<app_command>'] == 'run':
        root = arguments['--folder'][0]
        name = arguments['--name']
        subprocess.run(['python', os.path.join(
            root, 'apps', name, 'main.py')], shell=True)


if __name__ == "__main__":
    cli()
