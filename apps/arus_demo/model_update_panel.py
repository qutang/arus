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


def panel_checkbox(text, disabled=False, default=False, key=None):
    return sg.Checkbox(text, default=default, disabled=disabled, auto_size_text=True, font=('Helvetica', 11), key=key)


def panel_radioboxes(texts, group_id, keys, disableds, direction='row', default_index=0):
    radio_boxes = []
    for text, key, disabled in zip(texts, keys, disableds):
        default = texts.index(text) == default_index
        box = sg.Radio(text, group_id, default=default, disabled=disabled,
                       auto_size_text=True, font=('Helvetica', 11), enable_events=True, key=key)
        if direction == 'row':
            radio_boxes.append(box)
        else:
            radio_boxes = radio_boxes + [box]
    return radio_boxes


class ModelUpdatePanel:
    def __init__(self, title):
        self._title = title
        self._progress_bar_text = None
        self._progress_bar = None
        self._window = None
        self._app_state = app.AppState.getInstance()

    def init_panel(self):
        heading = "Model updating"

        self._progress_bar_text = panel_progressbar_text()
        self._progress_bar = panel_progressbar_bar()
        self._start_button = panel_control_button(
            'Start training', key='_START_', disabled=False)
        if self._app_state.initial_model_training_labels is not None:
            self._strategy = '_REPLACE_'
            default_index = 0
            disableds = [False, False, False]
        else:
            self._strategy = '_NONE_'
            default_index = 2
            disableds = [True, True, True]
        self._train_strategy_selection = panel_radioboxes(
            ['Replace original', 'Combine original', 'No origin'], disableds=disableds, default_index=default_index, group_id='_UPDATE_STRATEGY_', keys=['_REPLACE_', '_COMBINE_', '_NONE_'])
        layout = [
            [panel_heading(heading)],
            [self._start_button] + self._train_strategy_selection,
            [self._progress_bar_text],
            [self._progress_bar],
        ]
        self._window = sg.Window(self._title, layout=layout, finalize=True)

    def start_updating_task(self, strategy='_NONE_'):
        self._app_state.update_model_is_training = True
        if strategy != '_NONE_':
            init_feature_df, init_class_df = self._app_state.initial_dataset
        else:
            init_feature_df = None
            init_class_df = None
        init_model = self._app_state.initial_model
        if init_model is None:
            init_model = self._app_state.initial_model_training_labels
        new_feature_set = self._app_state.collected_feature_set
        pool = self._app_state.task_pool
        pool.restart(force=True)
        yield from backend.update_initial_model(init_model, init_feature_df, init_class_df, new_feature_set, self._app_state.selected_activities_for_update, self._app_state.placement_names_collected_data, strategy, pool)

    def close_task_pool(self):
        self._app_state.task_pool.close()
        self._app_state.task_pool.join()

    def start(self):
        self.init_panel()
        task = None
        updating_process = None
        while True:
            event, values = self._window.read(timeout=100)
            if event != '__TIMEOUT__':
                print(event)
                print(values)
            if event is None:
                break
            elif event == self._start_button.Key:
                updating_process = self.start_updating_task(
                    strategy=self._strategy)
            elif event == "_COMBINE_" or event == "_REPLACE_" or event == "_NONE_":
                self._strategy = event
            else:
                if updating_process is not None:
                    try:
                        message = next(updating_process)
                    except StopIteration:
                        message = None
                    if type(message) == str:
                        self._progress_bar.UpdateBar(
                            self._progress_bar.TKProgressBar.TKProgressBarForReal['value'] + 1)
                        self._progress_bar_text.Update(value=message)
                    elif message is not None:
                        task = message
                    if task is not None:
                        if not task.ready():
                            self._progress_bar.UpdateBar(
                                self._progress_bar.TKProgressBar.TKProgressBarForReal['value'] + 1)
                        else:
                            self._app_state.updated_model = task.get()
                            self._progress_bar.UpdateBar(100)
                            self._progress_bar_text.Update(
                                value='Completed. Training accuracy: ' + str(self._app_state.updated_model[2]))
                            task = None
                            updating_process = None
        self.close_task_pool()
        self._window.close()
