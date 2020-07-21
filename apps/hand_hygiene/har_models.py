import arus
import class_set
from dataclasses import dataclass, field
import typing
import arus


@dataclass
class HandHygieneClassicModel(arus.models.MUSSHARModel):
    name: str = "HH_CLASSIC"
    window_size: float = 2
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


def train_hh_classic(root, pids=None, placements=['LW', 'RW']):
    target_name = 'ACTIVITY'
    hh = arus.dataset.MHDataset(name='hand_hygiene',
                                path=root)
    hh.set_placement_parser(class_set.get_placement)
    hh.set_class_set_parser(class_set.get_class_set)
    sub_hh = hh.subset(name=f'hh-{"-".join(pids)}', pids=pids)
    mid = sub_hh.name + '-' + '-'.join(placements) + \
        '-' + target_name + '-hh-classic'
    model = HandHygieneClassicModel(mid=mid, used_placements=placements)
    model.load_dataset(sub_hh)
    model.compute_features()
    model.compute_class_set(task_names=[target_name])
    fs = model.data_set.processed['fs']
    cs = model.data_set.processed['cs']
    model.train(task_name='ACTIVITY', verbose=True)
    model.save_model(save_fcs=True)


if __name__ == "__main__":
    train_hh_classic('D:/Datasets/hand_hygiene/', ['P1-1'])
