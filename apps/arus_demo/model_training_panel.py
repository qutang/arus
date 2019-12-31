import PySimpleGUI as sg
import app_state as app
import backend

def panel_heading(text):
    return sg.Text(text=text, relief=sg.RELIEF_FLAT,
                           font=('Helvetica', 12, 'bold'), size=(20, 1))

def panel_progressbar_text(text=""): 
    progress_text = sg.Text(text, key='_PROGRESS_MESSAGE_', size=(40, 1))
    return progress_text

def panel_progressbar_bar(max_value=100):
    progress_bar = sg.ProgressBar(max_value, orientation='h', size=(
                        40, 20), key='_PROGRESS_BAR_')
    return progress_bar

def panel_control_button(text, disabled=False, key=None):
    return sg.Button(button_text=text,
                              font=('Helvetica', 11), auto_size_button=True, size=(20, None), key=key, disabled=disabled)

class ModelTrainingPanel:
    def __init__(self, title):
        self._title = title
        self._progress_bar_text = None
        self._progress_bar = None
        self._window = None
        self._app_state = app.AppState.getInstance()

    def init_panel(self):
        heading = "Model training"

        self._progress_bar_text = panel_progressbar_text()
        self._progress_bar = panel_progressbar_bar()

        layout = [
                    [panel_heading(heading)],
                    [self._progress_bar_text],
                    [self._progress_bar]
                ]
        self._window = sg.Window(self._title, layout=layout, finalize=True)

    def start_training_task(self):
        self._app_state.initial_model_is_training = True
        feature_df, class_df = self._app_state.initial_dataset
        training_labels = self._app_state.initial_model_training_labels
        pool = self._app_state.task_pool
        pool.restart(force=True)
        yield from backend.train_initial_model(training_labels, feature_df, class_df, pool)
            
    def close_task_pool(self):
        self._app_state.task_pool.close()
        self._app_state.task_pool.join()

    def start(self):
        self.init_panel()
        training_process = self.start_training_task()
        task = None
        while True:
            event, _ = self._window.read(timeout=100)
            if event is None:
                break
            else:
                try:
                    message = next(training_process)
                except StopIteration:
                    message = None
                if type(message) == str:
                    self._progress_bar.UpdateBar(self._progress_bar.TKProgressBar.TKProgressBarForReal['value'] + 1)
                    self._progress_bar_text.Update(value=message)
                elif message is not None:
                    task = message
                if task is not None:
                    if not task.ready():
                        self._progress_bar.UpdateBar(self._progress_bar.TKProgressBar.TKProgressBarForReal['value'] + 1)
                    else:
                        self._app_state.initial_model = task.get()
                        self._progress_bar.UpdateBar(100)
                        self._progress_bar_text.Update(value='Completed. Training accuracy: ' + str(self._app_state.initial_model[2]))
        self.close_task_pool()
        self._window.close()