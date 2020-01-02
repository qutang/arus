from pathos import pools


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
            self.nearby_devices = None
            self.task_pool = pools.ProcessPool(nodes=2)
            self.task_pool.close()
            self.io_pool = pools.ThreadPool(nodes=1)
            self.io_pool.close()

    def __init__(self):
        pass

    _instance = None

    @staticmethod
    def getInstance():
        if AppState._instance is None:
            AppState._instance = AppState._AppState()
        return AppState._instance
