from pathos import pools
import os
import dill
import datetime as dt


class _AppState:
    def __init__(self):
        self.initial_model = None
        self.updated_model = None
        self.initial_model_is_training = False
        self.update_model_is_training = False
        self.initial_model_is_validating = False
        self.update_model_is_validating = False
        self.initial_model_is_testing = False
        self.update_model_is_testing = False
        self.initial_model_training_labels = None
        self.initial_model_validation_results = None
        self.update_model_validation_results = None
        self.initial_dataset = None
        self.initial_model_pipeline = None
        self.selected_activities_for_collection = None
        self.selected_activities_for_update = None
        self.collected_feature_set = None
        self.placement_names_collected_data = None
        self.nearby_devices = None
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
