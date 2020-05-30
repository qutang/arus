import enum
from loguru import logger
import queue
import time

import PySimpleGUI as sg

import app_state as app
import backend
import base
import components as comp

from arus.models.muss import Strategy


class Event(enum.Enum):
    START_TASK = enum.auto()
    PROGRESS_UPDATE = enum.auto()
    TASK_COMPLETED = enum.auto()
    RESULT_READY = enum.auto()
    PLACEMENT_SELECTED = enum.auto()


class Mode(enum.Enum):
    TRAIN = enum.auto()
    LOSO = enum.auto()


class TaskWindow(base.BaseWindow):
    def __init__(self, title, strategies=[Strategy.USE_ORIGIN_ONLY], mode=Mode.TRAIN):
        super().__init__(title, 100)
        self._strategies = strategies
        self._progress_queue = queue.Queue()
        self._task = None
        self._result = None
        self._selected_strategy = None
        self._selected_placements = None
        self._mode = mode

    def init_states(self):
        state = app.AppState.getInstance()
        self._selected_strategy = self._selected_strategy or self._strategies[0]
        self._progress_text = "Task is not started..."
        self._timer = comp.Timer()
        if self._mode == Mode.TRAIN:
            if len(self._strategies) == 1:
                self._result = state.origin_model
            else:
                self._result = state.new_model
        elif self._mode == Mode.LOSO:
            if len(self._strategies) == 1:
                self._result = state.origin_validate_results
            else:
                self._result = state.new_validate_results
        if self._result is not None:
            self._events.put(Event.RESULT_READY)
        return state

    def init_views(self):
        header_row = [
            comp.heading(
                self._title,
                fixed_column_width=self._column_width
            )
        ]

        self._start_button = comp.control_button(
            "Start task",
            disabled=False,
            key=Event.START_TASK
        )
        start_button_row = [
            self._start_button
        ]

        placements = ['DW', 'DA', 'DT']
        placement_list_row = [
            comp.selection_list(placements, default_selections=placements,
                                mode='multiple', fixed_column_width=self._column_width, rows=len(placements), key=Event.PLACEMENT_SELECTED)
        ]

        strategy_row = self.init_strategy_controls()

        self._progress_bar = comp.ProgressBar(
            self._progress_text,
            fixed_column_width=self._column_width
        )
        progress_text_row = self._progress_bar.get_component()[0]
        progress_bar_row = self._progress_bar.get_component()[1]

        if self._mode == Mode.TRAIN:
            result_view = self.init_train_view()
        elif self._mode == Mode.LOSO:
            result_view = self.init_loso_view()
        else:
            result_view = []
        layout = [
            header_row,
            start_button_row,
            placement_list_row,
            strategy_row,
            progress_text_row,
            progress_bar_row,
            result_view
        ]

        return sg.Window(
            self._title,
            layout=layout,
            finalize=True
        )

    def _update_states_and_events(self, event, values):
        self._selected_placements = values[Event.PLACEMENT_SELECTED]
        for strategy in self._strategies:
            if values[strategy]:
                self._selected_strategy = strategy
                break
        try:
            progress = self._progress_queue.get(timeout=0.02)
            if type(progress) is str:
                self._events.put(Event.PROGRESS_UPDATE)
                self._progress_text = progress
            else:
                self._task = progress
                self._events.put(Event.PROGRESS_UPDATE)
            self._timer.tick()
        except queue.Empty:
            pass
        if self._task is not None:
            since_last_tick = self._timer.since_last_tick()
            if not self._task.ready() and since_last_tick >= 1:
                self._events.put(Event.PROGRESS_UPDATE)
                self._timer.tick()
            elif self._task.ready():
                self._timer.tick()
                self._result = self._task.get()
                self._events.put(Event.TASK_COMPLETED)
                self._progress_text = 'Completed in ' + \
                    self._timer.get_total_lapsed_time(
                        formatted=True) + ' seconds.'
                self._task = None
                logger.info("Training lapsed time: " +
                            self._timer.get_total_lapsed_time(formatted=True))

    def _dispatch_events(self, event):
        if event == Event.START_TASK:
            self.disable_start_button()
            self._timer.start()
            self.start_task()
        elif event == Event.PROGRESS_UPDATE:
            self.increment_progress()
        elif event == Event.TASK_COMPLETED:
            self._timer.stop()
            self._timer.reset()
            self.increment_progress(1.0)
            self.enable_start_button()
            self.close_task_pool()
            if self._mode == Mode.TRAIN:
                self.display_model_summary()
            elif self._mode == Mode.LOSO:
                self.display_confusion_matrix()
                self.display_classification_report()
        elif event == Event.RESULT_READY:
            if self._mode == Mode.TRAIN:
                self.display_model_summary()
            elif self._mode == Mode.LOSO:
                self.display_confusion_matrix()
                self.display_classification_report()

    def init_strategy_controls(self):
        default_strategy_index = self._strategies.index(
            self._selected_strategy)
        self._control_strategies = comp.RadioGroup(
            'STRATEGY',
            len(self._strategies),
            default_index=default_strategy_index,
            fixed_column_width=self._column_width
        )
        strategy_names = [strategy.name for strategy in self._strategies]
        self._control_strategies.add_radioboxes(
            strategy_names,
            keys=self._strategies,
            disable_states=[False] * len(self._strategies)
        )
        return self._control_strategies.get_component()

    def init_train_view(self):
        summary = backend.get_model_summary(self._result)
        self._text_model_summary = comp.text(
            summary, fixed_column_width=self._column_width)
        summary_row = [
            self._text_model_summary
        ]
        return summary_row

    def init_loso_view(self):
        self._view_cm = comp.Plot(
            fixed_column_width=int(self._column_width / 2), rows=10)
        self._view_report = comp.table(cells=[["", "", "", ""]], headings=[
                                       "class", "precision", "recall", "f1-score"], fixed_column_width=int(self._column_width / 2))
        return [self._view_cm.get_component(), self._view_report]

    def disable_start_button(self):
        self._start_button.update('Training...', disabled=True)

    def enable_start_button(self):
        self._start_button.update('Start training', disabled=False)

    def start_task(self):
        if self._mode == Mode.TRAIN:
            backend.train_model(
                origin_labels=self._state.origin_labels,
                origin_dataset=self._state.origin_dataset,
                new_dataset=self._state.new_dataset,
                new_labels=self._state.new_labels,
                progress_queue=self._progress_queue,
                strategy=self._selected_strategy,
                pool=self._state.task_pool,
                placement_names=self._selected_placements
            )
        elif self._mode == Mode.LOSO:
            backend.validate_model(
                origin_labels=self._state.origin_labels,
                origin_dataset=self._state.origin_dataset,
                new_dataset=self._state.new_dataset,
                new_labels=self._state.new_labels,
                progress_queue=self._progress_queue,
                strategy=self._selected_strategy,
                pool=self._state.task_pool,
                placement_names=self._selected_placements
            )

    def increment_progress(self, percentage=None):
        if percentage is None:
            self._progress_bar.increment(text=self._progress_text)
        else:
            self._progress_bar.update(percentage, self._progress_text)

    def display_model_summary(self):
        summary = backend.get_model_summary(self._result)
        self._text_model_summary.update(summary)

    def display_confusion_matrix(self):
        result = self._result
        fig = self._view_cm.get_figure()
        if fig is None:
            fig = backend.get_confusion_matrix_figure(result)
            self._view_cm.update_plot(fig)
        else:
            backend.get_confusion_matrix_figure(result, fig)
            self._view_cm.update_plot()

    def display_classification_report(self):
        result = self._result
        values = backend.get_classification_report_table(result)
        self._view_report.Update(values=values, visible=True,
                                 num_rows=min(len(values), 10))

    def close_task_pool(self):
        self._state.task_pool.close()
        self._state.task_pool.join()

    def get_result(self):
        return self._result
