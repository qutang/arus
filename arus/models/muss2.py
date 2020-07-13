
from ._model import HARModel
from .. import feature as feat
from .. import mhealth_format as mh
from .. import dataset as ds
from dataclasses import dataclass, field
import typing
import sklearn.svm as svm
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_model_selection
from sklearn import preprocessing as sk_preprocessing
from sklearn import utils as sk_utils
import pandas as pd
from loguru import logger


@dataclass
class MUSSHARModel(HARModel):
    name: str = "MUSS_HAR"
    used_placements: typing.List[str] = field(default_factory=lambda: ['DW'])
    model: svm.SVC = field(default=None)
    scaler: sk_preprocessing.MinMaxScaler = field(default=None)

    def reset_model(self):
        self.model = None
        self.scaler = None
        self.used_placements = ['DW']

    def reset_features(self):
        for subj in self.data_set.subjects:
            subj.processed = dict()
        self.data_set.processed = dict()

    def load_dataset(self, data_set: ds.MHDataset):
        for subj in data_set.subjects:
            # Exclude all other data types
            sensors = []
            for p in self.used_placements:
                sensors += data_set.get_sensors(
                    subj.pid, data_type='AccelerationCalibrated', placement=p)
            subj.sensors = sensors
        self.data_set = data_set

    def compute_features(self, *, window_size, sr):
        fss = []
        for subj in self.data_set.subjects:

            start_time, stop_time = self.data_set.get_session_span(subj.pid)

            sensor_dfs, placements = self._import_data_per_subj(
                subj, self.data_set.input_type)

            subj_fs, subj_fs_names = self._compute_features_per_subj(
                sensor_dfs, placements, start_time=start_time, stop_time=stop_time, window_size=window_size, sr=sr)
            if subj_fs is not None and subj_fs_names is not None:
                subj.processed = {'fs': subj_fs, 'fs_names': subj_fs_names}
                fss.append(subj_fs)
                fs_names = subj_fs_names
            else:
                logger.warning(
                    f'Subject {subj.pid} failed to compute features, this may due to the sensor data is out of range, incorrect sensor data, or invalid sensor data values. Ignore it from the feature set.')
        if len(fss) > 0:
            fs = pd.concat(fss, axis=0, ignore_index=True, sort=False)
            self.data_set.processed = {'fs': fs, 'fs_names': fs_names}
        else:
            logger.warning('No feature set is computed.')

    def compute_classes(self, **kwargs):
        pass

    def train(self, verbose=False, **kwargs):
        if 'fs' in self.data_set.processed:
            X = self.data_set.processed['fs'].loc[:, fs_names].values
            self._train_classifier(X, y, verbose=verbose, **kwargs)
        else:
            logger.warning(
                'No feature set available, please run compute_features at first.')

    def predict_ds(self, data_set: ds.MHDataset):
        pass

    def predict_fs(self, feature_set: pd.DataFrame):
        pass

    def _train_classifier(self, X, y, C=16, kernel='rbf', gamma=0.25, tol=0.0001, output_probability=True, class_weight='balanced', verbose=False):
        input_matrix, input_classes = sk_utils.shuffle(X, y)
        classifier = svm.SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            tol=tol,
            probability=output_probability,
            class_weight=class_weight,
            verbose=verbose
        )
        scaler = sk_preprocessing.MinMaxScaler((-1, 1))
        scaled_X = scaler.fit_transform(input_matrix)
        model = classifier.fit(scaled_X, input_classes)
        train_accuracy = model.score(scaled_X, input_classes)
        return model, scaler, train_accuracy

    def _import_data_per_subj(self, subj, input_type: ds.InputType):
        sensor_dfs = []
        placements = []
        for sensor in subj.sensors:
            if input_type == ds.InputType.MHEALTH_FORMAT:
                sensor.data = mh.MhealthFileReader.read_csvs(*sensor.paths)
            else:
                logger.error(
                    f'Unrecognized dataset input type: {input_type.name}')
            sensor_dfs.append(sensor.data)
            placements.append(sensor.placement)
        return sensor_dfs, placements

    def _compute_features_per_subj(self, sensor_dfs, placements, **kwargs):
        feature_set = feat.FeatureSet(sensor_dfs, placements)
        feature_set.compute_offline(window_size=kwargs['window_size'],
                                    feature_func=feat.preset,
                                    feature_names=feat.preset_names(),
                                    sr=kwargs['sr'],
                                    start_time=kwargs['start_time'],
                                    stop_time=kwargs['stop_time'],
                                    featureset_name=feat.PresetFeatureSet.MUSS)
        return feature_set.get_feature_set(), feature_set.get_feature_names()
