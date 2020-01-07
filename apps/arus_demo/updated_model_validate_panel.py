import PySimpleGUI as sg
import app_state as app
import backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


def panel_heading(text):
    return sg.Text(text=text, relief=sg.RELIEF_FLAT,
                   font=('Helvetica', 12, 'bold'), size=(20, 1))


def panel_progressbar_text(text=""):
    progress_text = sg.Text(text, key='_PROGRESS_MESSAGE_', size=(30, 1))
    return progress_text


def panel_progressbar_bar(max_value=100):
    progress_bar = sg.ProgressBar(max_value, orientation='h', size=(
        50, 20), key='_PROGRESS_BAR_')
    return progress_bar


def panel_control_button(text, disabled=False, key=None):
    return sg.Button(button_text=text,
                     font=('Helvetica', 11), auto_size_button=True, size=(20, None), key=key, disabled=disabled)


def panel_canvas():
    return sg.Canvas(size=(3, 3), key='_CM_CANVAS_')


def panel_table(values=[["", "", "", ""]], headings=["class", "precision", "recall", "f1-score"]):
    return sg.Table(values=values, headings=headings, size=(50, None), def_col_width=12, auto_size_columns=False, key='_REPORT_', visible=False)


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


class UpdateModelValidationPanel:
    def __init__(self, title):
        self._title = title
        self._progress_bar_text = None
        self._progress_bar = None
        self._cm_canvas = None
        self._report = None
        self._window = None
        self._app_state = app.AppState.getInstance()

    def init_panel(self):
        heading = "Model validation"

        self._progress_bar_text = panel_progressbar_text()
        self._progress_bar = panel_progressbar_bar()
        self._start_button = panel_control_button(
            'Start validating', key='_START_', disabled=False)
        self._cm_canvas = panel_canvas()
        self._report = panel_table()

        if self._app_state.initial_model_training_labels is not None:
            self._strategy = '_REPLACE_'
            default_index = 0
            disableds = [False, False, False]
        else:
            self._strategy = '_NONE_'
            default_index = 2
            disableds = [True, True, True]
        self._validate_strategy_selection = panel_radioboxes(
            ['Replace original', 'Combine original', 'No origin'], disableds=disableds, default_index=default_index, group_id='_VALIDATE_STRATEGY_', keys=['_REPLACE_', '_COMBINE_', '_NONE_'])

        layout = [
            [panel_heading(heading)],
            [self._start_button] + self._validate_strategy_selection,
            [self._progress_bar_text],
            [self._progress_bar],
            [self._cm_canvas],
            [self._report],
        ]
        self._window = sg.Window(self._title, layout=layout, finalize=True)

    def start_validation_task(self, strategy='_NONE_'):
        self._app_state.update_model_is_validating = True
        feature_df, class_df = self._app_state.initial_dataset
        if self._app_state.initial_model is None:
            origin_labels = self._app_state.initial_model_training_labels
            placement_names = self._app_state.placement_names_collected_data
        else:
            origin_labels = self._app_state.initial_model[0].classes_
            placement_names = self._app_state.initial_model[-2]
        new_feature_set = self._app_state.collected_feature_set
        new_labels = self._app_state.selected_activities_for_update

        model = self._app_state.updated_model
        pool = self._app_state.task_pool
        pool.restart(force=True)
        yield from backend.validate_updated_model(feature_df, class_df, origin_labels, new_feature_set, new_labels, placement_names, strategy, pool)

    def close_task_pool(self):
        self._app_state.task_pool.close()
        self._app_state.task_pool.join()

    def _show_confusion_matrix(self):
        result = self._app_state.update_model_validation_results
        fig = backend.get_confusion_matrix_figure(result)
        _, _, figure_w, figure_h = fig.bbox.bounds
        self._cm_canvas.set_size((figure_w, figure_h))
        figure_canvas_agg = FigureCanvasTkAgg(fig, self._cm_canvas.TKCanvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

    def _show_classification_report(self):
        result = self._app_state.update_model_validation_results
        values = backend.get_classification_report_table(result)
        self._report.Update(values=values, visible=True,
                            num_rows=min(len(values), 10))

    def _show_validation_results(self):
        self._progress_bar.UpdateBar(100)
        self._progress_bar_text.Update(value='Completed.')
        self._show_confusion_matrix()
        self._show_classification_report()

    def start(self):
        self.init_panel()
        num_of_classes = len(self._app_state.selected_activities_for_update) + \
            len(self._app_state.initial_model_training_labels)
        validation_process = None
        task = None
        while True:
            event, _ = self._window.read(timeout=100 * num_of_classes)
            if event is None:
                break
            elif event == self._start_button.Key:
                validation_process = self.start_validation_task(
                    strategy=self._strategy)
                self._start_button.Update('Validating...', disabled=True)
            elif event == "_COMBINE_" or event == "_REPLACE_" or event == "_NONE_":
                self._strategy = event.strip('_')
                if self._strategy is not None:
                    self._strategy += '_TRAIN'
            else:
                if validation_process is not None:
                    try:
                        message = next(validation_process)
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
                            self._app_state.update_model_validation_results = task.get()
                            self._show_validation_results()
                            self._start_button.Update(
                                'Start validating', disabled=False)
                            task = None
        self.close_task_pool()
        self._window.close()
