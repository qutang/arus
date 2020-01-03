from pathos import pools
import os


class AppState:
    class _AppState:
        def __init__(self):
            self.initial_model = None
            self.initial_model_is_training = False
            self.initial_model_is_validating = False
            self.initial_model_training_labels = None
            self.initial_model_validation_results = None
            self.initial_dataset = None
            self.initial_model_pipeline = None
            self.selected_activities_for_collection = None
            self.nearby_devices = None
            self.output_folder = os.path.join(
                os.path.expanduser("~"), 'arus')
            self.pid = None
            self.task_pool = pools.ProcessPool(nodes=6)
            self.task_pool.close()
            self.io_pool = pools.ThreadPool(nodes=4)
            self.io_pool.close()

    def __init__(self):
        pass

    _instance = None

    @staticmethod
    def getInstance():
        if AppState._instance is None:
            AppState._instance = AppState._AppState()
        return AppState._instance
