from .. import moment
from .. import mhealth_format as mh
import numpy as np
import pkg_resources
import pandas as pd
from loguru import logger


def _get_annotation_durations(annot_df):
    durations = annot_df.groupby(annot_df.columns[3]).apply(
        lambda rows: np.sum(rows.iloc[:, 2] - rows.iloc[:, 1]))
    return durations


def _annot_to_activity(annot_df, pid, start_time, stop_time):
    if annot_df is None:
        return "Unknown"
    if annot_df.shape[0] == 0:
        return "Unknown"
    label_list = annot_df[mh.ANNOTATION_LABEL_COL].str.lower()
    annot_df[mh.ANNOTATION_LABEL_COL] = label_list
    annot_df = annot_df.loc[(label_list != 'wear on')
                            & (label_list != 'wearon'), :]
    if annot_df.shape[0] == 0:
        return "Unknown"
    labels = annot_df.iloc[:, 3].unique()
    labels.sort()
    label = ' '.join(labels).lower().strip()

    # filter if it does not cover the entire 12.8s
    durations = _get_annotation_durations(annot_df)
    interval = int(moment.Moment.get_duration(
        start_time, stop_time, unit='ms'))

    if not np.all(durations.values >= np.timedelta64(interval, 'ms')):
        return "Transition"
    # special cases
    if pid == 'SPADES_26':
        if 'biking' in label and start_time.hour == 11 and stop_time.minute > 26:
            return "Stationary cycle ergometry"
    elif pid == 'SPADES_19':
        if '3 mph' in label and 'arms on desk' in label and 'treadmill' in label:
            return "Level treadmill walking at 3 mph with arms on desk"

    if 'stairs' in label and 'up' in label:
        return 'Walking upstairs'
    elif 'stairs' in label and 'down' in label:
        return 'Walking downstairs'
    if 'mbta' in label or 'city' in label or 'outdoor' in label:
        return 'Unknown'
    if "sitting" in label and 'writing' in label:
        return 'Sitting and writing'
    elif 'stand' in label and 'writ' in label:
        return 'Standing and writing at a table'
    elif 'sit' in label and ('web' in label or 'typ' in label):
        return 'Sitting and typing on a keyboard'
    elif 'reclin' in label and ('text' in label or 'web' in label):
        return 'Reclining and using phone'
    elif 'sit' in label and 'story' in label and ('city' not in label and 'outdoor' not in label):
        return "Sitting and talking"
    elif "reclin" in label and 'story' in label:
        return 'Reclining and talking'
    elif "stand" in label and ('web' in label or 'typ' in label):
        return "Standing and typing on a keyboard"
    elif 'bik' in label and ('stationary' in label or '300' in label):
        return "Stationary cycle ergometry"
    elif ('treadmill' in label or 'walk' in label) and '1' in label:
        return "Level treadmill walking at 1 mph with arms on desk"
    elif ('treadmill' in label or 'walk' in label) and '2' in label:
        return "Level treadmill walking at 2 mph with arms on desk"
    elif 'treadmill' in label and 'phone' in label:
        return "Level treadmill walking at 3-3.5 mph while holding a phone to the ear and talking"
    elif 'treadmill' in label and 'bag' in label:
        return "Level treadmill walking at 3-3.5 mph and carrying a bag"
    elif 'treadmill' in label and 'story' in label:
        return "Level treadmill walking at 3-3.5 mph while talking"
    elif ('treadmill' in label or 'walk' in label) and 'drink' in label:
        return 'Level treadmill walking at 3-3.5 mph and carrying a drink'
    elif ('treadmill' in label or 'walk' in label) and ('3.5' in label or '3' in label):
        return 'Level treadmill walking at 3-3.5 mph'
    elif '5.5' in label or 'jog' in label or 'run' in label:
        return 'Treadmill running at 5.5 mph & 5% grade'
    elif 'laundry' in label:
        return 'Standing and folding towels'
    elif 'sweep' in label:
        return 'Standing and sweeping'
    elif 'shelf' in label and 'load' in label:
        return 'Standing loading/unloading shelf'
    elif 'lying' in label:
        return "Lying on the back"
    elif label == 'sitting' or ('sit' in label and 'still' in label):
        return "Sitting still"
    elif label == "still" or 'standing' == label or label == 'standing still':
        return "Self-selected free standing"
    else:
        return 'Unknown'


def _load_task_class_map():
    map_filepath = pkg_resources.resource_filename(
        'arus', 'spades_lab/task_class_map.csv')
    return pd.read_csv(map_filepath)


def class_set(*annot_dfs, task_names=None, st=None, et=None, pid=None, **kwargs):
    class_labels = {mh.TIMESTAMP_COL: [st],
                    mh.START_TIME_COL: [st],
                    mh.STOP_TIME_COL: [et]}

    task_class_map = _load_task_class_map()
    annot_df = annot_dfs[0]
    activity = _annot_to_activity(annot_df, pid, st, et)

    for task_name in task_names:
        if task_name not in task_class_map.columns:
            logger.warning(
                f"{task_name} is not a valid task name for the current dataset")
        else:
            class_label = task_class_map.loc[task_class_map.ACTIVITY ==
                                             activity, task_name]
            class_labels[task_name] = class_label.values.tolist()
    class_vector = pd.DataFrame.from_dict(class_labels)
    return class_vector
