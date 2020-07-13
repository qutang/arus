from .. import mhealth_format as mh
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
    processed: typing.Any = field(init=False)


class InputType(enum.Enum):
    MHEALTH_FORMAT = enum.auto()


@dataclass
class MHDataset:
    name: str
    path: str = None
    load: bool = True
    input_type: InputType = None
    subjects: typing.List[SubjectObj] = field(default_factory=list)
    processed: typing.Any = field(default_factory=dict)

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
            data_type = mh.parse_data_type_from_filepath(paths[0])
            device_type = mh.parse_sensor_type_from_filepath(paths[0])
            sensor = SensorObj(paths=paths, sid=sid, data_type=data_type,
                               device_type=device_type)
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

    def get_sensors(self, pid, sid="", placement="", data_type="", device_type=""):
        subj = self.get_subject_obj(pid)
        if subj is not None:
            sensors = subj.sensors
            sensor_objs = list(filter(
                lambda s: placement == s.placement
                and sid == s.sid and data_type == s.data_type and device_type == s.device_type, sensors))
            return sensor_objs
        else:
            logger.error(f'{pid} does not exist.')
            return []

    def get_annotations(self, pid, annotation_type="", annotator=""):
        subj = self.get_subject_obj(pid)
        if subj is not None:
            annots = subj.annotations
            annot_objs = list(filter(
                lambda a: annotation_type == a.annotation_type
                and annotator in a.annotator, annots))
            return annot_objs
        else:
            logger.error(f'{pid} does not exist.')
            return []

    def subset(self, name, pids=[]):
        new_ds = MHDataset(path=self.path, name=name,
                           input_type=self.input_type, load=False)
        for pid in pids:
            new_ds.subjects.append(self.get_subject_obj(pid))
        return new_ds

    def get_session_span(self, pid):
        if self.input_type == InputType.MHEALTH_FORMAT:
            return mh.get_session_span(pid, self.path)
        else:
            logger.error(
                f'Unrecognized dataset input type {self.input_type.name}')
            return None
