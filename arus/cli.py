"""arus command line application

Usage:
  arus signaligner FOLDER PID [SR] [-t <file_type>] [--date_range=<date_range>] [--auto_range=<auto_range>] [--debug]
  arus app APP_COMMAND FOLDER NAME [--app_version=<app_version>]
  arus dataset DATASET_COMMAND DATASET_NAME [FOLDER] [OUTPUT_FOLDER] [--debug]
  arus package PACK_COMMAND [NEW_VERSION] [--dev] [--release]
  arus --help
  arus --version

Arguments:
  FOLDER                                        Dataset folder.
  PID                                           Participant ID.
  SR                                            Sampling rate in Hz.
  APP_COMMAND                                   Sub commands for app command. Either "build" or "run".
  NAME                                          Name of the app.
  PACK_COMMAND                                  "release", "docs"
  NEW_VERSION                                   "major", "minor", "patch" or number.

Options:
  -t <file_type>, --file_type=<file_type>       File type: either "sensor" or "annotation". If omit, both are included.
  --date_range=<date_range>                     Date range. E.g., "--date_range 2020-06-01,2020-06-10", or "--date_range 2020-06-01," or "--date_range ,2020-06-10".
  --auto_range=<auto_range>                     Auto date freq. Default is "W-SUN", or weekly starting from Sunday.
  --app_version=<app_version>                   App version. If omit, default is the same as the version of arus package.
  -h, --help                                    Show help message.
  -v, --version                                 Program/app version.
"""

from docopt import docopt
from loguru import logger
from . import developer
from . import dataset
from . import mhealth_format as mh
from .plugins import signaligner
import glob
from datetime import datetime, timedelta
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
    elif arguments['dataset']:
        dataset_command(arguments)
    elif arguments['package']:
        package_command(arguments)


def signaligner_command(arguments):
    pid = arguments['PID']
    root = arguments['FOLDER']
    file_type = arguments['--file_type']
    date_range = arguments['--date_range'].split(
        ',') if arguments['--date_range'] is not None else None
    # auto range would be weekly files starting from Sunday
    auto_range = arguments['--auto_range'] or 'W-SUN'
    sr = int(arguments['SR']) if arguments['SR'] is not None else None

    logger.debug('Signaligner arguments: {}', [
                 root, pid, file_type, sr, date_range, auto_range])

    if file_type == 'sensor':
        convert_to_signaligner(root, pid, file_type, sr=sr,
                               date_range=date_range, auto_range=auto_range)
    elif file_type == 'annotation':
        convert_to_signaligner(root, pid, file_type, sr=sr,
                               date_range=date_range, auto_range=auto_range)
    elif file_type is None:
        convert_to_signaligner_both(
            root, pid, sr=sr, date_range=date_range, auto_range=auto_range)


def convert_to_signaligner_both(root, pid, sr, date_range=None, auto_range='W-SUN'):
    convert_to_signaligner(root, pid, 'annotation', sr, date_range, auto_range)
    convert_to_signaligner(root, pid, 'sensor', sr, date_range, auto_range)


def convert_to_signaligner(root, pid, file_type, sr=None, date_range=None, auto_range='W-SUN'):
    session_span = mh.get_session_span(pid, root)
    session_span = signaligner.shrink_session_span(session_span, date_range)
    logger.info('Session span: {}'.format(session_span))
    if file_type == 'annotation':
        filepaths = mh.get_annotation_files(pid, root)
        groups = set(map(mh.parse_annotation_type_from_filepath, filepaths))
    elif file_type == 'sensor':
        assert sr is not None
        filepaths = mh.get_sensor_files(pid, root)
        logger.debug(session_span)
        groups = set(map(mh.parse_data_id_from_filepath, filepaths))
    logger.debug(groups)
    for group in groups:
        logger.info('Convert files of {}'.format(group))
        group_files = list(filter(lambda f: group in f, filepaths))
        sensor_type = mh.parse_sensor_type_from_filepath(group_files[0])

        sub_session_markers = signaligner.auto_split_session_span(
            session_span, auto_range)
        logger.debug(sub_session_markers)
        for i in range(len(sub_session_markers) - 1):
            sub_session_span = sub_session_markers[i:i + 2]
            st_display = sub_session_span[0].strftime('%Y%m%d%H')
            et_display = sub_session_span[1].strftime('%Y%m%d%H')
            logger.info(
                f'Process sub session: {st_display} - {et_display} based on {auto_range}')

            # call signify functions
            if file_type == 'annotation':
                output_path = os.path.join(
                    root, pid, mh.DERIVED_FOLDER, 'signaligner', f'{pid}_{st_display}_{et_display}_labelsets', f'{pid}_{group}_{st_display}_{et_display}.{file_type}.csv')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                signaligner.signify_annotation_files(
                    group_files, group, output_path, sub_session_span)
            elif file_type == 'sensor':
                # set output file paths
                output_path = os.path.join(
                    root, pid, mh.DERIVED_FOLDER, 'signaligner', f'{pid}_{st_display}_{et_display}_sensors', f'{pid}_{sensor_type}_{group}_{st_display}_{et_display}.{file_type}.csv')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                output_annotation_path = os.path.join(os.path.dirname(
                    output_path.replace('sensors', 'labelsets')), f'{pid}_missing_{group}_{st_display}_{et_display}.annotation.csv')
                os.makedirs(os.path.dirname(
                    output_annotation_path), exist_ok=True)
                signaligner.signify_sensor_files(
                    group_files, group, output_path, output_annotation_path, sub_session_span, sr)


def app_command(arguments):
    if arguments['APP_COMMAND'] == 'build':
        root = arguments['FOLDER']
        name = arguments['NAME']
        version = arguments['--app_version'] or developer._find_current_version(
            '.', 'arus')
        developer.build_arus_app(root, name, version)
    elif arguments['APP_COMMAND'] == 'run':
        root = arguments['FOLDER']
        name = arguments['NAME']
        subprocess.run(['python', os.path.join(
            root, 'apps', name, 'main.py')], shell=True)


def dataset_command(arguments):
    if arguments['DATASET_COMMAND'] == 'compress':
        root = arguments['FOLDER']
        name = arguments['DATASET_NAME']
        output_folder = arguments['OUTPUT_FOLDER']
        logger.info(
            f'Compressing dataset as {name}.tar.gz in folder {output_folder}')
        developer.compress_dataset(root, output_folder, f'{name}.tar.gz')
    elif arguments['DATASET_COMMAND'] == 'query':
        name = arguments['DATASET_NAME']
        if name == 'all':
            logger.info("Available sample data names")
            logger.info(dataset.get_available_sample_data())
        else:
            logger.info(f'Query dataset: {name} in ARUS package')
            logger.info(dataset.get_sample_datapath(name))


def package_command(arguments):
    if arguments['PACK_COMMAND'] == 'release':
        is_dev = arguments['--dev'] or False
        nver = arguments['NEW_VERSION']
        assert nver is not None
        release = arguments['--release'] or False
        _release_package(nver, dev=is_dev, release=release)
    elif arguments['PACK_COMMAND'] == 'docs':
        is_dev = arguments['--dev'] or True
        release = arguments['--release'] or False
        if is_dev and not release:
            developer.dev_website()
        elif release:
            developer.build_website()


def _release_package(nver, dev=False, release=False):
    new_version = developer.bump_package_version('.', 'arus', nver, dev)
    if new_version is not None and developer.command_is_available('git') and release:
        developer.commit_repo(f"Bump version to {new_version}")
        developer.tag_repo(new_version)
        developer.push_repo()
        developer.push_tag(new_version)
