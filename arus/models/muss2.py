
from ._model import HARModel
from .. import feature as feat
from .. import mhealth_format as mh
from .. import dataset as ds
from .. import scheduler
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
import typing
import gc


@dataclass
class MUSSHARModel(HARModel):
    name: str = "MUSS_HAR"
    window_size: float = 12.8
    sr: int = 80
    used_placements: typing.List[str] = field(default_factory=lambda: ['DW'])

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
        sch = scheduler.Scheduler(
            scheme=scheduler.Scheduler.Scheme.SUBMIT_ORDER, max_workers=4)
        sch.reset()
        for subj in self.data_set.subjects:
            logger.info(f'Computing features for {subj.pid}')
            start_time, stop_time = self.data_set.get_session_span(subj.pid)

            sensor_dfs, placements, srs = self._import_data_per_subj(
                subj, self.data_set.input_type)

            sch.submit(
                self._compute_features_per_subj, sensor_dfs, placements, srs,
                start_time=start_time, stop_time=stop_time, window_size=self.window_size, sr=self.sr)
            gc.collect()

            # subj_fs, subj_fs_names = self._compute_features_per_subj(
            #     sensor_dfs, placements, srs, start_time=start_time, stop_time=stop_time, window_size=self.window_size, sr=self.sr)

        result = sch.get_all_remaining_results()
        for subj_fv_result, subj in zip(result, self.data_set.subjects):
            subj_fs, subj_fs_names = subj_fv_result
            if subj_fs is not None and subj_fs_names is not None:
                subj.processed = {**subj.processed,
                                  'fs': subj_fs, 'fs_names': subj_fs_names}
                fs = subj_fs.copy()
                fs['PID'] = subj.pid
                fss.append(fs)
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
            logger.info(f'Compute class set for {subj.pid}')
            start_time, stop_time = self.data_set.get_session_span(subj.pid)

            subj_class_set = self.data_set.get_class_set(
                subj.pid, task_names=task_names, window_size=self.window_size, step_size=None, start_time=start_time, stop_time=stop_time)

            if subj_class_set is not None:
                subj.processed = {**subj.processed,
                                  'cs': subj_class_set, 'cs_names': task_names}
                cs = subj_class_set.copy()
                cs['PID'] = subj.pid
                css.append(cs)
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

    def get_training_acc(self):
        return self.train_perf['acc']

    def train(self, task_name, ignore_classes=["Unknown", "Transition"], pids=[], verbose=False, **kwargs):
        if len(pids) == 0:
            pid_info = 'all participants'
        else:
            pid_info = ','.join(pids)
        logger.info(f'Train MUSS model of {task_name} for {pid_info}')
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
        logger.info(f'Training accuracy: {train_acc}')

    def save_fcs(self, pids=[]):
        fcs, fs_names, cs_names = self._preprocess_feature_class_set(pids)
        if fcs is not None:
            output_path = os.path.join(
                sefl.get_processed_path(), 'fcs.csv')
            fcs.to_csv(output_path, float_format='%.6f')
            logger.info(f'Saved feature and class set to {output_path}')
        else:
            logger.warning('No available feature and class set to save!')

    def save_model(self, save_fcs=True):
        output_filepath = os.path.join(
            self.get_processed_path(), self.name + '.har')
        logger.info(f'Save trained MUSS model to {output_filepath}')
        bundle = {
            'mid': self.mid,
            'placements': self.used_placements,
            'model': self.model,
            'dataset_name': self.data_set.name,
            'fs': None if not save_fcs else self.data_set.processed['fs'],
            'cs': None if not save_fcs else self.data_set.processed['cs'],
            'fs_names': None if not save_fcs else self.data_set.processed['fs_names'],
            'cs_names': None if not save_fcs else self.data_set.processed['cs_names']
        }
        joblib.dump(bundle, filename=output_filepath)

    @staticmethod
    def load_model(path):
        bundle = joblib.load(path)
        model = MUSSHARModel(
            mid=bundle['mid'], used_placements=bundle['placements'])
        model.model = bundle['model']
        model.data_set = ds.MHDataset(name=bundle['dataset_name'])
        model.data_set.processed['fs'] = bundle['fs']
        model.data_set.processed['cs'] = bundle['cs']
        model.data_set.processed['fs_names'] = bundle['fs_names']
        model.data_set.processed['cs_names'] = bundle['cs_names']
        return model

    def predict(self, *input_objs, **kwargs):
        result = None
        if type(input_objs[0]) is ds.SensorObj:
            if 'sr' in kwargs:
                result = self._predict_sensor(*input_objs, sr=kwargs['sr'])
            else:
                result = self._predict_sensor(*input_objs)
        elif type(input_objs[0]) is ds.SubjectObj:
            result = self._predict_subj(*input_objs)
        elif type(input_objs[0]) is pd.DataFrame and len(input_objs) == 1 and 'fs_names' in kwargs:
            result = self._predict_fs(input_objs[0], kwargs['fs_names'])
        elif type(input_objs[0]) is pd.DataFrame and 'placements' in kwargs and ('sr' in kwargs or 'srs' in kwargs):
            result = self._predict_raw_df(input_objs, **kwargs)
        elif type(input_objs[0]) is str and 'placements' in kwargs:
            result = self._predict_raw_files(input_objs, **kwargs)
        else:
            raise ValueError(
                f'Unrecognized input type {type(input_objs[0])} for predict function.')
        return result

    def _predict_subj(self, *subj_objs: typing.List[ds.SubjectObj]):
        results = []
        for subj in subj_objs:
            if 'fs' in subj.processed:
                result = self._predict_fs(
                    subj.processed['fs'], subj.processed['fs_names'])
            else:
                sensors = []
                for p in self.used_placements:
                    sensor = subj.get_sensor('placement', p)
                    sensors.append(sensor)
                result = self._predict_sensor(*sensors)
            results.append(result)
        return results

    def _predict_sensor(self, *sensor_objs, sr=None):
        raw_dfs = [sensor.get_data() for sensor in sensor_objs]
        placements = [sensor.placement for sensor in sensor_objs]
        srs = [sensor.sr for sensor in sensor_objs]
        return self._predict_raw_df(raw_dfs, placements, srs=srs, sr=sr)

    def _predict_raw_files(self, raw_files, placements, file_format, srs=None, sr=None, **kwargs):
        if srs is None:
            srs = [None] * len(raw_files)
        sensors = []
        for p in self.used_placements:
            i = placements.index(p)
            sensors.append(ds.SensorObj(paths=[raw_files[i]], sr=srs[i]
                                        or sr, placement=p, input_type=file_format))
        return self._predict_sensor(*sensors, sr=sr)

    def _predict_raw_df(self, raw_dfs, placements, srs=None, **kwargs):
        start_time = min([raw_df.iloc[0, 0] for raw_df in raw_dfs])
        logger.info(f'The start time of test data is: {start_time}')
        if 'sr' in kwargs and srs is None:
            srs = [kwargs['sr']] * len(raw_dfs)
        elif 'sr' in kwargs and srs is not None:
            srs = [sr or kwargs['sr'] for sr in srs]
        logger.info('Compute features for test data')
        fs, fs_names = self._compute_features_per_subj(
            raw_dfs, placements=placements, srs=srs, start_time=start_time, stop_time=None, window_size=self.window_size, **kwargs)
        logger.info('Make predictions for test data')
        predicts = self._predict_fs(fs, fs_names)
        return predicts

    def _predict_fs(self, fs, fs_names):
        fs = fs.dropna()
        predicts = self.model.predict(fs.loc[:, fs_names].values)
        predict_df = pd.DataFrame.from_dict({
            'HEADER_TIME_STAMP': fs['HEADER_TIME_STAMP'],
            'START_TIME': fs['START_TIME'],
            'STOP_TIME': fs['STOP_TIME'],
            'PREDICTION': predicts
        })
        return predict_df

    def _preprocess_feature_class_set(self, pids):
        if 'fs' in self.data_set.processed:
            if len(pids) > 0:
                fss = []
                for pid in pids:
                    fs = self.data_set.get_subject_obj(pid).processed['fs']
                    fs['PID'] = pid
                    fss.append(fs)
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
                css = []
                for pid in pids:
                    cs = self.data_set.get_subject_obj(pid).processed['cs']
                    cs['PID'] = pid
                    css.append(cs)
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

        if 'PID' in fs:
            on = mh.FEATURE_SET_TIMESTAMP_COLS + ['PID']
        else:
            on = mh.FEATURE_SET_TIMESTAMP_COLS
        fcs = pd.merge(fs, cs, how='inner',
                       on=on, sort=False)
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
        del sensor_dfs
        gc.collect()
        return feature_set.get_feature_set(), feature_set.get_feature_names()
