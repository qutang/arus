
from ._model import HARModel
from .. import feature as feat
from .. import mhealth_format as mh
from .. import dataset as ds
from dataclasses import dataclass, field
import typing
import sklearn.svm as svm
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing as preprocess
from sklearn.utils import shuffle
import pandas as pd
from loguru import logger
import os
import joblib


@dataclass
class MUSSHARModel(HARModel):
    name: str = "MUSS_HAR"
    window_size: float = 12.8
    sr: int = 80
    used_placements: typing.List[str] = field(default_factory=lambda: ['DW'])
    model: typing.Any = None
    train_perf: typing.Dict[str, typing.Any] = field(default_factory=dict)

    def reset_model(self):
        self.model = None
        self.scaler = None
        self.used_placements = ['DW']

    def reset_processed_data(self):
        self.data_set.clear_processed()

    def load_dataset(self, data_set: ds.MHDataset):
        for subj in data_set.subjects:
            # Exclude all other data types
            sensors = []
            for p in self.used_placements:
                sensors += data_set.get_sensors(
                    subj.pid, data_type='AccelerationCalibrated', placement=p)
            subj.sensors = sensors
        self.data_set = data_set

    def compute_features(self):
        fss = []
        for subj in self.data_set.subjects:

            start_time, stop_time = self.data_set.get_session_span(subj.pid)

            sensor_dfs, placements, srs = self._import_data_per_subj(
                subj, self.data_set.input_type)

            subj_fs, subj_fs_names = self._compute_features_per_subj(
                sensor_dfs, placements, srs, start_time=start_time, stop_time=stop_time, window_size=self.window_size, sr=self.sr)
            if subj_fs is not None and subj_fs_names is not None:
                subj.processed = {**subj.processed,
                                  'fs': subj_fs, 'fs_names': subj_fs_names}
                fss.append(subj_fs)
                fs_names = subj_fs_names
            else:
                logger.warning(
                    f'Subject {subj.pid} failed to compute features, this may due to the sensor data is out of range, incorrect sensor data, or invalid sensor data values. Ignore it from the feature set.')
        if len(fss) > 0:
            fs = pd.concat(fss, axis=0, ignore_index=True, sort=False)
            self.data_set.processed = {
                **self.data_set.processed, 'fs': fs, 'fs_names': fs_names}
        else:
            logger.warning('No feature set is computed.')

    def compute_class_set(self, task_names):
        css = []
        for subj in self.data_set.subjects:

            start_time, stop_time = self.data_set.get_session_span(subj.pid)

            subj_class_set = self.data_set.get_class_set(
                subj.pid, task_names=task_names, window_size=self.window_size, step_size=None, start_time=start_time, stop_time=stop_time)

            if subj_class_set is not None:
                subj.processed = {**subj.processed,
                                  'cs': subj_class_set, 'cs_names': task_names}
                css.append(subj_class_set)
            else:
                logger.warning(
                    f'Subject {subj.pid} failed to compute class set, this may due to the annotation data is out of range, incorrect annotation data, or invalid annotation data values. Ignore it from the class set.')
        if len(css) > 0:
            cs = pd.concat(css, axis=0, ignore_index=True, sort=False)
            self.data_set.processed = {
                **self.data_set.processed, 'cs': cs, 'cs_names': task_names}
        else:
            logger.warning('No class set is computed.')

    def get_feature_names(self):
        if 'fs_names' in self.data_set.processed:
            fs_names = self.data_set.processed['fs_names']
        else:
            logger.warning(
                'No feature set available, please run compute_features at first.')
            fs_names = None
        return fs_names

    def get_processed_path(self):
        output_path = os.path.join(
            self.data_set.path, mh.PROCESSED_FOLDER, self.mid)
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def train(self, task_name, ignore_classes=["Unknown", "Transition"], pids=[], verbose=False, **kwargs):
        fcs, fs_names, cs_names = self._preprocess_feature_class_set(pids)
        if task_name not in cs_names:
            logger.error(
                f'{task_name} is not in the class set, aborting training.')
            return

        fcs = MUSSHARModel.ignore_classes(
            fcs, task_name, remove_classes=ignore_classes)

        X = fcs.loc[:, fs_names].values
        y = fcs.loc[:, task_name].values

        self.model, train_acc = self._train_classifier(
            X, y, verbose=verbose, **kwargs)
        self.train_perf = {**self.train_perf, 'acc': train_acc}

    def save_fcs(self, pids=[]):
        fcs, fs_names, cs_names = self._preprocess_feature_class_set(pids)
        if fcs is not None:
            output_path = os.path.join(
                sefl.get_processed_path(), 'fcs.csv')
            fcs.to_csv(output_path, float_format='%.6f')
            logger.info(f'Saved feature and class set to {output_path}')
        else:
            logger.warning('No available feature and class set to save!')

    def save_model(self):
        joblib.dump(self, filename=os.path.join(
            self.get_processed_path(), self.name + '.har'))

    @staticmethod
    def load_model(path):
        muss_har = joblib.load(path)
        return muss_har

    def predict_ds(self, data_set: ds.MHDataset):
        pass

    def predict_fs(self, feature_set: pd.DataFrame):
        pass

    def _preprocess_feature_class_set(self, pids):
        if 'fs' in self.data_set.processed:
            if len(pids) > 0:
                fss = [self.data_set.get_subject_obj(
                    pid).processed['fs'] for pid in pids]
                fs = pd.concat(fss, axis=0, sort=False, ignore_index=True)
                fs_names = self.data_set.get_subject_obj(
                    pids[0]).processed['fs_names']
            else:
                fs = self.data_set.processed['fs']
                fs_names = self.data_set.processed['fs_names']
        else:
            logger.error(
                'No feature set available, please run compute_features at first.')
            return
        if 'cs' in self.data_set.processed:
            if len(pids) > 0:
                css = [self.data_set.get_subject_obj(
                    pid).processed['cs'] for pid in pids]
                cs = pd.concat(css, axis=0, sort=False, ignore_index=True)
                cs_names = self.data_set.get_subject_obj(
                    pids[0]).processed['cs_names']
            else:
                cs = self.data_set.processed['cs']
                cs_names = self.data_set.processed['cs_names']
        else:
            logger.error(
                'No class set available, please run compute_class_set at first.')
            return

        fcs = pd.merge(fs, cs, how='inner',
                       on=mh.FEATURE_SET_TIMESTAMP_COLS, sort=False)
        return fcs, fs_names, cs_names

    def _train_classifier(self, X, y, C=16, kernel='rbf', gamma=0.25, tol=0.0001, output_probability=True, class_weight='balanced', verbose=False):
        input_matrix, input_classes = shuffle(X, y)
        classifier = svm.SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            tol=tol,
            probability=output_probability,
            class_weight=class_weight,
            verbose=verbose
        )

        scaler = preprocess.MinMaxScaler((-1, 1))
        pipe = make_pipeline(scaler, classifier,
                             memory=self.get_processed_path())
        logger.debug(pipe)
        pipe.fit(input_matrix, input_classes)
        train_accuracy = pipe.score(input_matrix, input_classes)
        return pipe, train_accuracy

    def _import_data_per_subj(self, subj, input_type: ds.InputType):
        sensor_dfs = []
        placements = []
        srs = []
        for sensor in subj.sensors:
            if input_type == ds.InputType.MHEALTH_FORMAT:
                sensor.data = mh.MhealthFileReader.read_csvs(*sensor.paths)
            else:
                logger.error(
                    f'Unrecognized dataset input type: {input_type.name}')
            sensor_dfs.append(sensor.data)
            placements.append(sensor.placement)
            srs.append(sensor.sr)
        return sensor_dfs, placements, srs

    def _compute_features_per_subj(self, sensor_dfs, placements, srs, **kwargs):
        feature_set = feat.FeatureSet(sensor_dfs, placements, srs)
        feature_set.compute_offline(window_size=kwargs['window_size'],
                                    feature_func=feat.preset,
                                    feature_names=feat.preset_names(),
                                    sr=kwargs['sr'],
                                    start_time=kwargs['start_time'],
                                    stop_time=kwargs['stop_time'],
                                    featureset_name=feat.PresetFeatureSet.MUSS)
        return feature_set.get_feature_set(), feature_set.get_feature_names()
