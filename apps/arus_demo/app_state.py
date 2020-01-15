from pathos import pools
import os
import dill
import datetime as dt


class _AppState:
    def __init__(self):
        self.origin_dataset = []
        self.origin_labels = []
        self.origin_label_candidates = []
        self.origin_model = None

        self.new_activity_name = ""
        self.data_collection_labels = []
        self.data_collection_label_candidates = []
        self.new_dataset = None

        self.new_labels = []
        self.new_model = None

        self.nearby_devices = None

        self.test_stack = []

        self.output_folder = AppState._path
        self.pid = 'ARUS_' + dt.datetime.now().strftime('%Y%m%d%H%M%S')

        self.task_pool = pools.ProcessPool(nodes=6)
        self.task_pool.close()
        self.io_pool = pools.ThreadPool(nodes=4)
        self.io_pool.close()


class AppState:
    def __init__(self):
        pass

    _instance = None

    _path = os.path.join(
        os.path.expanduser("~"), 'arus')

    _snapshot_path = os.path.join(
        _path, 'snapshot.pkl')

    @staticmethod
    def getInstance():
        if AppState._instance is None:
            AppState._instance = _AppState()
        return AppState._instance

    @staticmethod
    def snapshot():
        output_path = AppState._snapshot_path
        with open(output_path, 'wb') as f:
            dill.dump(AppState._instance, f)

    @staticmethod
    def restore():
        input_path = AppState._snapshot_path
        if os.path.exists(input_path):
            with open(input_path, 'rb') as f:
                AppState._instance = dill.load(f)
            return True
        else:
            return False

    @staticmethod
    def reset():
        AppState._instance = _AppState()
