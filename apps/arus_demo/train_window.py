import enum
import logging
import queue
import time

import PySimpleGUI as sg

import app_state as app
import backend
import base
import components as comp


class Event(enum.Enum):
    TRAIN_MODEL = enum.auto()
    PROGRESS_UPDATE = enum.auto()
    TRAIN_COMPLETED = enum.auto()


class TrainWindow(base.BaseWindow):
    def __init__(self, title, train_strategies=[backend.TRAIN_STRATEGY.USE_ORIGIN_ONLY]):
        super().__init__(title, 100)
        self._train_strategies = train_strategies
        self._progress_queue = queue.Queue()
        self._task = None
        self._trained_model = None

    def init_states(self):
        state = app.AppState.getInstance()
        state.train_strategy = state.train_strategy or self._train_strategies[0]
        self._progress_text = "Training is not started..."
        self._timer = comp.Timer()
        return state

    def init_views(self):
        header_row = [
            comp.heading(
                "Train model",
                fixed_column_width=self._column_width
            )
        ]

        self._start_button = comp.control_button(
            "Start training",
            disabled=False,
            key=Event.TRAIN_MODEL
        )
        start_button_row = [
            self._start_button
        ]

        strategy_row = self.init_strategy_controls()

        self._progress_bar = comp.ProgressBar(
            self._progress_text,
            fixed_column_width=self._column_width
        )
        progress_text_row = self._progress_bar.get_component()[0]
        progress_bar_row = self._progress_bar.get_component()[1]

        if len(self._train_strategies) == 1:
            summary = backend.get_model_summary(self._state.origin_model)
        else:
            summary = backend.get_model_summary(self._state.new_model)
        self._text_model_summary = comp.text(
            summary, fixed_column_width=self._column_width)
        summary_row = [
            self._text_model_summary
        ]

        layout = [
            header_row,
            start_button_row,
            strategy_row,
            progress_text_row,
            progress_bar_row,
            summary_row
        ]

        return sg.Window(
            self._title,
            layout=layout,
            finalize=True
        )

    def _update_states_and_events(self, event, values):
        for strategy in self._train_strategies:
            if values[strategy]:
                self._state.train_strategy = strategy
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
                self._trained_model = self._task.get()
                self._events.put(Event.TRAIN_COMPLETED)
                self._progress_text = 'Completed in ' + \
                    self._timer.get_total_lapsed_time(
                        formatted=True) + ' seconds.'
                self._task = None
                logging.info("Training lapsed time: " +
                             self._timer.get_total_lapsed_time(formatted=True))

    def _dispatch_events(self, event):
        if event == Event.TRAIN_MODEL:
            self.disable_start_button()
            self._timer.start()
            self.start_train_task()
        elif event == Event.PROGRESS_UPDATE:
            self.increment_progress()
        elif event == Event.TRAIN_COMPLETED:
            self._timer.stop()
            self._timer.reset()
            self.increment_progress(1.0)
            self.enable_start_button()
            self.close_task_pool()
            self.display_model_summary()

    def init_strategy_controls(self):
        default_strategy_index = self._train_strategies.index(
            self._state.train_strategy)
        self._control_strategies = comp.RadioGroup(
            'TRAIN_STRATEGY',
            len(self._train_strategies),
            default_index=default_strategy_index,
            fixed_column_width=int(
                self._column_width / len(self._train_strategies)
            )
        )
        strategy_names = [strategy.name for strategy in self._train_strategies]
        self._control_strategies.add_radioboxes(
            strategy_names,
            keys=self._train_strategies,
            disable_states=[False] * len(self._train_strategies)
        )
        return self._control_strategies.get_component()

    def disable_start_button(self):
        self._start_button.update('Training...', disabled=True)

    def enable_start_button(self):
        self._start_button.update('Start training', disabled=False)

    def start_train_task(self):
        backend.train_model(
            origin_labels=self._state.origin_labels,
            origin_dataset=self._state.origin_dataset,
            progress_queue=self._progress_queue,
            strategy=self._state.train_strategy,
            pool=self._state.task_pool
        )

    def increment_progress(self, percentage=None):
        if percentage is None:
            self._progress_bar.increment(text=self._progress_text)
        else:
            self._progress_bar.update(percentage, self._progress_text)

    def display_model_summary(self):
        summary = backend.get_model_summary(self._trained_model)
        self._text_model_summary.update(summary)

    def close_task_pool(self):
        self._state.task_pool.close()
        self._state.task_pool.join()

    def get_trained_model(self):
        return self._trained_model
