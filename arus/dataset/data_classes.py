from .. import mhealth_format as mh
from .. import generator, plugins, segmentor, stream2
from .. import class_label
from dataclasses import dataclass, field
from loguru import logger
import pandas as pd
import typing
import os
import enum
import re


class InputType(enum.Enum):
    MHEALTH_FORMAT = "MHEALTH_FORMAT"
    SIGNALIGNER = "SIGNALIGNER"


@dataclass
class SensorObj:
    paths: str
    sid: str = None
    data_type: str = None
    device_type: str = None
    placement: str = None
    sr: int = None
    grange: float = None
    data: pd.DataFrame = None
    start_time: typing.Any = None
    input_type: InputType = InputType.MHEALTH_FORMAT

    def __post_init__(self):
        if self.input_type == InputType.SIGNALIGNER:
            reader = plugins.actigraph.ActigraphReader(
                self.paths[0], has_ts=False, has_header=True)
            reader.read_meta()
            meta = reader.get_meta()
            self.start_time = meta['START_TIME']
            self.sr = meta['SAMPLING_RATE']
            self.grange = meta['DYNAMIC_RANGE']
            self.sid = meta['SENSOR_ID']
            self.data_type = os.path.basename(self.paths[0]).split('.csv')[0]
        elif self.input_type == InputType.MHEALTH_FORMAT:
            self.sid = mh.parse_sensor_id_from_filepath(self.paths[0])
            self.data_type = mh.parse_data_type_from_filepath(self.paths[0])
            self.device_type = mh.parse_sensor_type_from_filepath(
                self.paths[0])
            if self.device_type == 'ActigraphGT9X':
                if self.data_type == 'AccelerationCalibrated' or self.data_type == 'AccelerometerCalibrated':
                    self.sr = 80
                elif self.data_type == 'IMUMagnetometer' or self.data_type == 'IMUAccelerometerCalibrated' or self.data_type == 'IMUGyroscope':
                    self.sr = 100
                else:
                    self.sr = None
            self.start_time = mh.parse_timestamp_from_filepath_content(
                self.paths[0])

    def get_data(self):
        if self.input_type == InputType.MHEALTH_FORMAT:
            return mh.MhealthFileReader.read_csvs(*self.paths)
        elif self.input_type == InputType.SIGNALIGNER:
            reader = plugins.actigraph.ActigraphReader(
                self.paths[0], False, False)
            reader.read()
            return next(reader.get_data())

    def get_stream(self, seg: segmentor.Segmentor = None, stream_id: str = None, **seg_kwargs):
        if self.input_type == InputType.MHEALTH_FORMAT:
            gen = generator.MhealthSensorFileGenerator(
                *self.paths, buffer_size=3600 * 80)
        elif self.input_type == InputType.SIGNALIGNER:
            gen = plugins.actigraph.ActigraphSensorFileGenerator(
                *self.paths, has_ts=False, has_header=True, buffer_size=3600 * 80)
        else:
            logger.error(
                f'Unrecognized input type: {self.input_type.name}, failed to get stream')
            return None
        seg = seg or segmentor.SlidingWindowSegmentor(**seg_kwargs)
        stream_name = self.sid + '-' + self.data_type + '-stream'
        sensor_stream = stream2.Stream(gen, seg, name=stream_name)
        sensor_stream.set_essential_context(
            stream_id=stream_id or self.placement or stream_name)
        return sensor_stream


@dataclass
class AnnotationObj:
    paths: str
    aid: str = None
    annotation_type: str = None
    annotator: str = None
    data: pd.DataFrame = None
    input_type: InputType = InputType.MHEALTH_FORMAT

    def __post_init__(self):
        if self.input_type == InputType.MHEALTH_FORMAT:
            self.annotation_type = mh.parse_annotation_type_from_filepath(
                self.paths[0])
            self.annotator = mh.parse_annotator_from_filepath(self.paths[0])
            self.aid = mh.parse_data_id_from_filepath(self.paths[0])

    def get_stream(self, window_size: float, start_time: "str, pandas.Timestamp", input_type: InputType):
        if input_type == InputType.MHEALTH_FORMAT:
            gen = generator.MhealthSensorFileGenerator(
                *self.paths, buffer_size=3600 * 80)
        elif input_type == InputType.SIGNALIGNER:
            gen = plugins.actigraph.ActigraphSensorFileGenerator(
                *self.paths, buffer_size=3600 * 80)
        else:
            logger.error(
                f'Unrecognized input type: {input_type.name}, failed to get stream')
            return None
        seg = segmentor.SlidingWindowSegmentor(
            window_size=window_size, ref_st=start_time, st_col=0, et_col=None)
        sensor_stream = stream2.Stream(gen, seg, name=self.sid + '-' +
                                       self.data_type + '-stream')
        return sensor_stream


@dataclass
class SubjectObj:
    path: str
    pid: str
    demography: pd.DataFrame = None
    sensors: typing.List[SensorObj] = field(default_factory=list)
    annotations: typing.List[AnnotationObj] = field(default_factory=list)
    processed: typing.Dict[str, typing.Any] = field(default_factory=dict)

    def get_sensor(self, field_name, field_value):
        for sensor in sensors:
            if sensor.__getattribute__(field_name) == field_value:
                return sensor
        return None

    def subset_sensors(self, field_name, field_value):
        sensors = []
        for sensor in self.sensors:
            if re.fullmatch(field_value, sensor.__getattribute__(field_name)) is not None:
                sensors.append(sensor)
        return sensors


@dataclass
class MHDataset:
    name: str
    path: str = None
    load: bool = True
    input_type: InputType = InputType.MHEALTH_FORMAT
    subjects: typing.List[SubjectObj] = field(default_factory=list)
    processed: typing.Dict[str, typing.Any] = field(default_factory=dict)
    class_set_parser: object = None
    placement_parser: object = None

    def __post_init__(self):
        if not self.load:
            return
        self._load()

    def _load(self):
        self.subjects = []
        if self.path is None:
            return
        pids = mh.get_pids(self.path)
        for pid in pids:
            subj = SubjectObj(path=os.path.join(self.path, pid),
                              pid=pid)
            subj.sensors = self._load_sensors(pid)
            subj.annotations = self._load_annotations(pid)
            self.subjects.append(subj)

    def _load_sensors(self, pid):
        sensor_files = mh.get_sensor_files(pid, self.path)
        data_ids = sorted(list(set(
            map(lambda f: mh.parse_data_id_from_filepath(f), sensor_files))))
        sensors = []
        for data_id in data_ids:
            paths = list(filter(lambda f: data_id in f, sensor_files))
            sid = mh.parse_sensor_id_from_filepath(paths[0])
            if self.placement_parser is not None:
                placement = self.placement_parser(self.path, pid, sid)
            else:
                placement = None
            sensor = SensorObj(paths=paths, sid=sid, placement=placement)
            sensors.append(sensor)
        return sensors

    def _load_annotations(self, pid):
        annotation_files = mh.get_annotation_files(pid, self.path)
        annotations = []
        data_ids = sorted(list(set(
            map(lambda f: mh.parse_data_id_from_filepath(f), annotation_files))))
        for data_id in data_ids:
            paths = list(filter(lambda f: data_id ==
                                mh.parse_data_id_from_filepath(f), annotation_files))
            annot = AnnotationObj(
                paths=paths, aid=data_id)
            annotations.append(annot)
        return annotations

    def clear_processed(self):
        self.processed = {}
        for subj in self.subjects:
            subj.processed = {}

    def set_class_set_parser(self, class_set_parser):
        self.class_set_parser = class_set_parser

    def set_placement_parser(self, placement_parser=None):
        self.placement_parser = placement_parser
        if placement_parser is not None:
            logger.info('Set placement func, loading placements...')
            for pid in self.get_pids():
                for sensor in self.get_sensors(pid):
                    # logger.debug(f'Set placement for {pid} - {sensor.sid}')
                    sensor.placement = placement_parser(
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

    def get_annotations(self, pid, annotation_type=None, annotator=None, aid=None):
        subj = self.get_subject_obj(pid)

        def filter_func(s):
            if annotation_type is not None and s.annotation_type is not None and annotation_type != s.annotation_type:
                return False
            if annotator is not None and s.annotator is not None and annotator != s.annotator:
                return False
            if aid is not None and s.aid is not None and aid != s.aid:
                return False
            return True
        if subj is not None:
            annots = subj.annotations
            annot_objs = list(filter(filter_func, annots))
            return annot_objs
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

    def get_class_set(self, pid, task_names, window_size, step_size=None, start_time=None, stop_time=None, show_progress=True):
        step_size = step_size or window_size
        aids = self.get_annotation_field(pid, 'aid')
        raw_resources = [mh.MhealthFileReader.read_csvs(
            *self.get_annotations(pid, aid=aid)[0].paths, datetime_cols=[0, 1, 2]) for aid in aids]
        class_set = class_label.ClassSet(raw_resources, aids)
        class_set.compute_offline(window_size, self.class_set_parser,
                                  task_names, start_time, stop_time, step_size=step_size, pid=pid, show_progress=show_progress)
        return class_set.get_class_set()

    def subset(self, name, pids=None):
        new_ds = MHDataset(path=self.path, name=name,
                           input_type=self.input_type, load=False)
        if pids is None:
            pids = self.get_pids()
        for pid in pids:
            new_ds.subjects.append(self.get_subject_obj(pid))
        new_ds.set_class_set_parser(self.class_set_parser)
        new_ds.set_placement_parser(self.placement_parser)
        return new_ds

    def get_session_span(self, pid):
        if self.input_type == InputType.MHEALTH_FORMAT:
            return mh.get_session_span(pid, self.path)
        else:
            logger.error(
                f'Unrecognized dataset input type {self.input_type.name}')
            return None


@dataclass
class SGDataset(MHDataset):
    name: str
    path: str = None
    sensor_folder: str = None
    label_folder: str = None
    load: bool = True
    input_type: InputType = InputType.SIGNALIGNER
    subjects: typing.List[SubjectObj] = field(default_factory=list)
    processed: typing.Dict[str, typing.Any] = field(default_factory=dict)
    class_set_parser: object = None
    placement_parser: object = None

    def _load(self):
        self.subjects = []
        pid = self.name
        subj = SubjectObj(path=self.path, pid=pid)
        subj.sensors = self._load_sensors(pid)
        subj.annotations = self._load_annotations(pid)
        self.subjects.append(subj)

    def _load_sensors(self, pid):
        sensors = []
        if self.sensor_folder is None:
            logger.error('sensor_folder must be set to load sensor files.')
            return sensors
        path = os.path.join(self.path, self.sensor_folder)
        sensor_filenames = sorted(os.listdir(path))

        for sensor_filename in sensor_filenames:
            data_id = sensor_filename.split('.csv')[0]
            paths = [os.path.join(path, sensor_filename)]
            sid = data_id
            if self.placement_parser is not None:
                placement = self.placement_parser(self.path, pid, sid)
            else:
                placement = None
            data_type = data_id
            device_type = data_id
            sensor = SensorObj(paths=paths, sid=sid, data_type=data_type,
                               device_type=device_type, placement=placement, input_type=self.input_type)
            sensors.append(sensor)
        return sensors

    def _load_annotations(self, pid):
        annotations = []
        if self.label_folder is None:
            logger.error(
                'label_folder must be set to load annotation/label files.')
            return annotations
        path = os.path.join(self.path, self.label_folder)
        label_filenames = sorted(os.listdir(path))

        for label_filename in label_filenames:
            data_id = label_filename.split('.csv')[0]
            paths = [data_id]
            annotation_type = data_id
            annotator = data_id
            annot = AnnotationObj(
                paths=paths, aid=data_id, annotation_type=annotation_type, annotator=annotator, input_type=self.input_type)
            annotations.append(annot)
        return annotations
