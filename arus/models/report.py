from torch.utils.tensorboard import SummaryWriter
from .. import extensions as ext
from loguru import logger
import shutil


class Reporter:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir

    def reset(self):
        logger.info('Reset reporter log information...')
        shutil.rmtree(self.log_dir, ignore_errors=True)

    def report_cv(self, cv_result, cm_df=None, tag_prefix='cv'):
        writer = SummaryWriter(log_dir=self.log_dir)
        n_iter = cv_result.shape[0]
        logger.info('Report cv metrics...')
        for i in range(n_iter):
            test_metric_dict = {}
            train_metric_dict = {}
            time_dict = {}
            for j in range(len(cv_result.columns)):
                if 'test' in cv_result.columns[j]:
                    test_metric_dict[cv_result.columns[j]
                                     ] = cv_result.iloc[i, j]
                elif 'train' in cv_result.columns[j]:
                    train_metric_dict[cv_result.columns[j]
                                      ] = cv_result.iloc[i, j]
                elif 'time' in cv_result.columns[j]:
                    time_dict[cv_result.columns[j]
                              ] = cv_result.iloc[i, j]
            writer.add_scalars(f'{tag_prefix}/train', train_metric_dict, i)
            writer.add_scalars(f'{tag_prefix}/test', test_metric_dict, i)
            writer.add_scalars(f'{tag_prefix}/time', time_dict, i)
        if cm_df is not None:
            logger.info('Report cv confusion matrix')
            fig = ext.plotting.confusion_matrix(cm_df)
            writer.add_figure(
                f'{tag_prefix}/confusion_matrix', fig, close=True)
        writer.close()
