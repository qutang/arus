from .. import mhealth_format as mh
from .. import class_label
from dataclasses import dataclass, field
from loguru import logger
import pandas as pd
import typing
import os
import enum


@dataclass
class SensorObj:
    paths: str
    sid: str
    data_type: str = None
    device_type: str = None
    placement: str = None
    sr: int = None
    grange: float = None
    data: pd.DataFrame = None


@dataclass
class AnnotationObj:
    paths: str
    aid: str
    annotation_type: str = None
    annotator: str = None
    data: pd.DataFrame = None


@dataclass
class SubjectObj:
    path: str
    pid: str
    demography: pd.DataFrame = None
    sensors: typing.List[SensorObj] = field(default_factory=list)
    annotations: typing.List[AnnotationObj] = field(default_factory=list)
    processed: typing.Dict[str, typing.Any] = field(default_factory=dict)


class InputType(enum.Enum):
    MHEALTH_FORMAT = enum.auto()


@dataclass
class MHDataset:
    name: str
    path: str = None
    load: bool = True
    input_type: InputType = None
    subjects: typing.List[SubjectObj] = field(default_factory=list)
    processed: typing.Dict[str, typing.Any] = field(default_factory=dict)
    class_func: object = None
    placement_func: object = None

    def __post_init__(self):
        if not self.load:
            return
        if self.input_type == InputType.MHEALTH_FORMAT:
            self._load_mhealth_format()
        else:
            logger.error(
                f"Unrecognized dataset input type: {self.input_type.name}")

    def _load_mhealth_format(self):
        self.subjects = []
        pids = mh.get_pids(self.path)
        for pid in pids:
            subj = SubjectObj(path=os.path.join(self.path, pid),
                              pid=pid)
            subj.sensors = self._load_sensors(pid)
            subj.annotations = self._load_annotations(pid)
            self.subjects.append(subj)

    def _load_sensors(self, pid):
        sensor_files = mh.get_sensor_files(pid, self.path)
        data_ids = set(
            map(lambda f: mh.parse_data_id_from_filepath(f), sensor_files))
        sensors = []
        for data_id in data_ids:
            paths = list(filter(lambda f: data_id in f, sensor_files))
            sid = mh.parse_sensor_id_from_filepath(paths[0])
            if self.placement_func is not None:
                placement = self.placement_func(self.path, pid, sid)
            else:
                placement = None
            data_type = mh.parse_data_type_from_filepath(paths[0])
            device_type = mh.parse_sensor_type_from_filepath(paths[0])
            sensor = SensorObj(paths=paths, sid=sid, data_type=data_type,
                               device_type=device_type, placement=placement)
            sensors.append(sensor)
        return sensors

    def _load_annotations(self, pid):
        annotation_files = mh.get_annotation_files(pid, self.path)
        annotations = []
        data_ids = set(
            map(lambda f: mh.parse_data_id_from_filepath(f), annotation_files))
        for data_id in data_ids:
            paths = list(filter(lambda f: data_id in f, annotation_files))
            annotation_type = mh.parse_annotation_type_from_filepath(
                paths[0])
            annotator = mh.parse_annotator_from_filepath(paths[0])
            annot = AnnotationObj(
                paths=paths, aid=data_id, annotation_type=annotation_type, annotator=annotator)
            annotations.append(annot)
        return annotations

    def clear_processed(self):
        self.processed = {}
        for subj in self.subjects:
            subj.processed = {}

    def set_class_func(self, class_func):
        self.class_func = class_func

    def set_placement_func(self, placement_func=None):
        self.placement_func = placement_func
        if placement_func is not None:
            logger.info('Set placement func, loading placements...')
            for pid in self.get_pids():
                for sensor in self.get_sensors(pid):
                    sensor.placement = placement_func(
                        self.path, pid, sensor.sid)

    def has_pid(self, pid):
        for subj in self.subjects:
            if pid == subj.pid:
                return True
        return False

    def get_pids(self):
        return list(map(lambda subj: subj.pid, self.subjects))

    def get_subject_obj(self, pid):
        for subj in self.subjects:
            if pid == subj.pid:
                return subj
        return None

    def get_sensors(self, pid, sid=None, placement=None, data_type=None, device_type=None):
        subj = self.get_subject_obj(pid)

        def filter_func(s):
            if sid is not None and s.sid is not None and sid != s.sid:
                return False
            if placement is not None and s.placement is not None and placement != s.placement:
                return False
            if data_type is not None and s.data_type is not None and data_type != s.data_type:
                return False
            if device_type is not None and s.device_type is not None and device_type != s.device_type:
                return False
            return True
        if subj is not None:
            sensors = subj.sensors
            sensor_objs = list(filter(filter_func, sensors))
            return sensor_objs
        else:
            logger.error(f'{pid} does not exist.')
            return []

    def get_annotations(self, pid, annotation_type="", annotator="", aid="", return_field=None):
        subj = self.get_subject_obj(pid)
        if subj is not None:
            annots = subj.annotations
            annot_objs = list(filter(
                lambda a: annotation_type in a.annotation_type
                and annotator in a.annotator and aid in a.aid, annots))
            if return_field is None:
                return annot_objs
            else:
                return list(map(lambda a: a.__getattribute__(return_field), annot_objs))
        else:
            logger.error(f'{pid} does not exist.')
            return []

    def get_annotation_field(self, pid, field_name):
        subj = self.get_subject_obj(pid)
        if subj is not None:
            annots = subj.annotations
            field_values = set(map(
                lambda a: a.__getattribute__(field_name), annots))
            return field_values
        else:
            logger.error(f'{pid} does not exist.')
            return []

    def get_sensor_field(self, pid, field_name):
        subj = self.get_subject_obj(pid)
        if subj is not None:
            sensors = subj.sensors
            field_values = set(map(
                lambda s: s.__getattribute__(field_name), sensors))
            return field_values
        else:
            logger.error(f'{pid} does not exist.')
            return []

    def get_class_set(self, pid, task_names, window_size, step_size=None, start_time=None, stop_time=None):
        step_size = step_size or window_size
        aids = self.get_annotation_field(pid, 'aid')
        raw_resources = [mh.MhealthFileReader.read_csvs(
            *self.get_annotations(
                pid, aid=aid, return_field='paths')[0], datetime_cols=[0, 1, 2]) for aid in aids]
        class_set = class_label.ClassSet(raw_resources, aids)
        class_set.compute_offline(window_size, self.class_func,
                                  task_names, start_time, stop_time, step_size=step_size, pid=pid)
        return class_set.get_class_set()

    def subset(self, name, pids=[]):
        new_ds = MHDataset(path=self.path, name=name,
                           input_type=self.input_type, load=False)
        for pid in pids:
            new_ds.subjects.append(self.get_subject_obj(pid))
        new_ds.set_class_func(self.class_func)
        new_ds.set_placement_func(self.placement_func)
        return new_ds

    def get_session_span(self, pid):
        if self.input_type == InputType.MHEALTH_FORMAT:
            return mh.get_session_span(pid, self.path)
        else:
            logger.error(
                f'Unrecognized dataset input type {self.input_type.name}')
            return None
