import os
import arus
import pandas as pd
import joblib
from joblib import Memory
from loguru import logger
import numpy as np

memory = Memory(os.path.join(arus.env.get_data_home(), 'joblib'), verbose=0)


def adjust_class_on_side(class_label, sensor_df):
    pass


def _annot_to_activity(annot_df, pid, start_time, stop_time):
    if annot_df is None:
        return "Unknown"
    if annot_df.shape[0] == 0:
        return "Unknown"
    label_list = annot_df[arus.mh.ANNOTATION_LABEL_COL].unique().tolist()
    durations = arus.ClassSet.get_annotation_durations(
        annot_df, arus.mh.ANNOTATION_LABEL_COL)
    interval = int(arus.Moment.get_duration(
        start_time, stop_time, unit='ms'))

    if not np.all(durations.values >= np.timedelta64(interval, 'ms')) or len(label_list) > 1:
        return "Transition"
    else:
        return label_list[0]


def _annot_to_side(annot_df, pid, start_time, stop_time):
    if annot_df is None:
        return None
    if annot_df.shape[0] == 0:
        return None
    label_list = annot_df[arus.mh.ANNOTATION_LABEL_COL].unique(
    ).tolist()
    durations = arus.ClassSet.get_annotation_durations(
        annot_df, arus.mh.ANNOTATION_LABEL_COL)
    interval = int(arus.Moment.get_duration(
        start_time, stop_time, unit='ms'))

    if not np.all(durations.values >= np.timedelta64(interval, 'ms')) or len(label_list) > 1:
        return None
    else:
        return label_list[0]


@memory.cache
def _load_placement_map(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return None


@memory.cache
def _load_class_map(filepath):
    logger.info('Loading class map...')
    return pd.read_csv(filepath)


def get_placement(root, pid, sid):
    placement_map_file = os.path.join(root, pid, arus.mh.SUBJECT_META_FOLDER,
                                      arus.mh.META_LOCATION_MAPPING_FILENAME)
    placement_map = _load_placement_map(placement_map_file)
    if placement_map is None:
        return None
    else:
        p = placement_map.loc[placement_map['SENSOR_ID']
                              == sid, 'PLACEMENT'].values[0]
        return p


def get_class_set(*annot_dfs, task_names, st, et, pid, aids, **kwargs):
    class_labels = {arus.mh.TIMESTAMP_COL: [st],
                    arus.mh.START_TIME_COL: [st],
                    arus.mh.STOP_TIME_COL: [et]}

    class_map_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'task_class_map.csv')
    task_class_map = _load_class_map(class_map_file)
    for aid, annot_df in zip(aids, annot_dfs):
        if aid == 'HandHygieneSide':  # experted annotated hand side for face touching activities; for hand washing this should always be "Both"
            side = _annot_to_side(annot_df, pid, st, et)
        elif aid == 'HandHygiene':
            activity = _annot_to_activity(annot_df, pid, st, et)

    for task_name in task_names:
        if task_name not in task_class_map.columns:
            logger.warning(
                f"{task_name} is not a valid task name for the current dataset")
        else:
            class_label = task_class_map.loc[task_class_map.ANNOTATION ==
                                             activity, task_name]
            if class_label.empty:
                class_label = 'Unknown'
            else:
                class_label = class_label.values[0]
            class_label = f'{class_label}' if side is None else f'{class_label}-{side}'
            class_labels[task_name] = class_label
    class_vector = pd.DataFrame.from_dict(class_labels)
    return class_vector
