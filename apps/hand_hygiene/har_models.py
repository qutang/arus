import arus
import class_set
from dataclasses import dataclass, field
import typing
import arus
import os


@dataclass
class HandHygieneClassicModel(arus.models.MUSSHARModel):
    name: str = "HH_CLASSIC"
    window_size: float = 2
    step_size: float = 1
    sr: int = 80
    used_placements: typing.List[str] = field(
        default_factory=lambda: ['LW', 'RW'])

    def load_dataset(self, data_set: arus.dataset.MHDataset):
        for subj in data_set.subjects:
            # Exclude all other data types
            sensors = []
            for p in self.used_placements:
                sensors += data_set.get_sensors(subj.pid, placement=p)
            subj.sensors = sensors
        self.data_set = data_set

    def _import_data_per_subj(self, subj, input_type: arus.dataset.InputType):
        sensor_dfs = []
        placements = []
        srs = []
        sensors = subj.subset_sensors('data_type', 'AccelerometerCalibrated')
        for sensor in sensors:
            if input_type == arus.dataset.InputType.MHEALTH_FORMAT:
                sensor.data = arus.mh.MhealthFileReader.read_csvs(
                    *sensor.paths)
            else:
                logger.error(
                    f'Unrecognized dataset input type: {input_type.name}')
            sensor_dfs.append(sensor.data)
            placements.append(sensor.placement)
            srs.append(sensor.sr)
        return sensor_dfs, placements, srs


def _prepare_model(root, pids=None, placements=['LW', 'RW'], target_name='ACTIVITY'):
    hh = arus.dataset.MHDataset(name='hand_hygiene',
                                path=root)
    hh.set_placement_parser(class_set.get_placement)
    hh.set_class_set_parser(class_set.get_class_set)
    mid = hh.name + '-' + '-'.join(placements) + \
        '-' + target_name + '-hh-classic'
    model = HandHygieneClassicModel(mid=mid, used_placements=placements)
    model.load_dataset(hh)
    model.compute_features(pids=pids)
    model.compute_class_set(task_names=[target_name], pids=pids)
    return model


def train_hh_classic(root, pids=None, placements=['LW', 'RW']):
    target_name = 'ACTIVITY'
    model = _prepare_model(root, pids, placements, target_name)
    model.train(task_name='ACTIVITY', pids=pids, verbose=True)
    model.save_model(save_fcs=True)


def cv_hh_classic(root, pids=None, placements=['LW', 'RW']):
    target_name = 'ACTIVITY'
    model_path = os.path.join(
        root, arus.mh.PROCESSED_FOLDER, f'hand_hygiene-{"-".join(placements)}-{target_name}-hh-classic', f'HH_CLASSIC-{"-".join(pids)}.har')
    if os.path.exists(model_path):
        model = HandHygieneClassicModel.load_model(model_path)
    else:
        model = _prepare_model(root, pids, placements, target_name)
    reporter = arus.models.Reporter(
        log_dir=os.path.join(model.get_processed_path(), 'report'))
    model.cross_validation(target_name, pids=pids, n_fold=5)
    reporter.reset()
    reporter.report_cv(
        model.data_set.processed['cv'], cm_df=model.data_set.processed['cv_cm'])


if __name__ == "__main__":
    # train_hh_classic('D:/Datasets/hand_hygiene/', ['P1-1'])
    cv_hh_classic('D:/Datasets/hand_hygiene/', ['P1-1'])
