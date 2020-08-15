from .. import mhealth_format as mh
from .. import dataset as ds
from dataclasses import dataclass, field
import typing
from loguru import logger
from .. import error_code

from sklearn.metrics import confusion_matrix
import pandas as pd


@dataclass
class HARModel:
    mid: str
    name: str = "HAR"
    model: typing.Any = None
    train_perf: typing.Dict[str, typing.Any] = field(default_factory=dict)
    train_pids: typing.List[str] = field(default_factory=list)
    data_set: ds.MHDataset = field(init=False)

    def __post_init__(self):
        logger.info(self)

    def reset_model(self):
        self.model = None

    def load_dataset(self, data: ds.MHDataset):
        self.data_set = data

    def compute_features(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def compute_class_set(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def train(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def predict(self, *input_objs, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def cross_validation(self, pids=None, n_fold=10, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def logo_validation(self, pids=None, group_col=None, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def learning_curve_logo(self, pids=None, group_col=None, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes')

    def learning_curve_cv(self, pids=None, n_fold=10, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes')

    @staticmethod
    def confusion_matrix(true_labels, pred_labels, label_names):
        cm = confusion_matrix(true_labels, pred_labels, labels=label_names)
        cm_df = pd.DataFrame(data=cm, index=label_names, columns=label_names)
        return cm_df

    @staticmethod
    def ignore_classes(fcs, task_name, remove_classes=['Unknown', 'Transition']):
        is_valid_label = ~fcs[task_name].isin(remove_classes).values
        fcs = fcs.loc[is_valid_label, :]
        return fcs
