import enum
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pathos.pools as pools
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import app_state as app
import backend
import components as comp
from data_collection_panel import DataCollectionPanel
from model_testing_panel import ModelTestingPanel
from model_training_panel import ModelTrainingPanel
from model_update_panel import ModelUpdatePanel
from model_validation_panel import ModelValidationPanel
from updated_model_test_panel import UpdatedModelTestingPanel
from updated_model_validate_panel import UpdateModelValidationPanel


class Event(enum.Enum):
    CLOSE_WINDOW = enum.auto(),
    ORIGIN_LABELS_CHANGED = enum.auto(),
    NOT_ENOUGH_ORIGIN_LABELS = enum.auto(),
    ENOUGH_ORIGIN_LABELS = enum.auto(),
    TRAIN_ORIGIN_MODEL = enum.auto(),
    TEST_ORIGIN_MODEL = enum.auto(),
    VALIDATE_ORIGIN_MODEL = enum.auto(),
    NEW_ACTIVITY_NAME_CHANGED = enum.auto(),
    ADD_NEW_ACTIVITY = enum.auto(),
    DATA_COLLECTION_LABELS_CHANGED = enum.auto(),
    NOT_ENOUGH_DATA_COLLECTION_LABELS = enum.auto(),
    ENOUGH_DATA_COLLECTION_LABELS = enum.auto(),
    COLLECT_NEW_DATA = enum.auto(),
    CHECK_NEW_DATA = enum.auto(),
    NEW_LABELS_CHANGED = enum.auto(),
    NOT_ENOUGH_NEW_LABELS = enum.auto(),
    ENOUGH_NEW_LABELS = enum.auto(),
    TRAIN_NEW_MODEL = enum.auto(),
    TEST_NEW_MODEL = enum.auto(),
    VALIDATE_NEW_MODEL = enum.auto()


class Dashboard:
    def __init__(self, title):
        self._title = title
        self._state = app.AppState.getInstance()
        self._column_width = 25
        self._events = queue.Queue()

    def start(self):
        self.init_states()
        self.init_views()
        while True:
            self.update_states_and_events()
            if self.dispatch_events():
                break
        self._window.close()

    def init_states(self):
        origin_label_candidates = backend.get_class_label_candidates(
            self._state.origin_dataset)
        if len(self._state.data_collection_label_candidates) == 0:
            self._state.data_collection_label_candidates = origin_label_candidates
        if len(self._state.origin_label_candidates) == 0:
            self._state.origin_label_candidates = origin_label_candidates

    def init_views(self):
        origin_model_column = self.init_origin_model_view()
        data_collection_column = self.init_data_collection_view()
        new_model_column = self.init_new_model_view()

        layout = [
            [
                origin_model_column,
                data_collection_column,
                new_model_column
            ]
        ]
        dashboard = sg.Window(
            self._title,
            layout=layout,
            finalize=True,
            resizable=True
        )
        self._window = dashboard

    def update_states_and_events(self):
        inherited_window_event, new_values = self._window.read(timeout=20)
        self._events.queue.clear()
        if inherited_window_event == sg.TIMEOUT_KEY:
            pass
        elif inherited_window_event is not None:
            self._events.put(inherited_window_event)

        if new_values is not None:
            if self._state.origin_labels != new_values[Event.ORIGIN_LABELS_CHANGED]:
                self._state.origin_labels = new_values[Event.ORIGIN_LABELS_CHANGED][:]
                if len(self._state.origin_labels) >= 2:
                    self._events.put(Event.ENOUGH_ORIGIN_LABELS)
                else:
                    self._events.put(Event.NOT_ENOUGH_ORIGIN_LABELS)

            if self._state.new_labels != new_values[Event.NEW_LABELS_CHANGED]:
                self._state.new_labels = new_values[Event.NEW_LABELS_CHANGED][:]
                if len(self._state.new_labels) >= 2:
                    self._events.put(Event.ENOUGH_NEW_LABELS)
                else:
                    self._events.put(Event.NOT_ENOUGH_NEW_LABELS)

            if self._state.data_collection_labels != new_values[Event.DATA_COLLECTION_LABELS_CHANGED]:
                self._state.data_collection_labels = new_values[
                    Event.DATA_COLLECTION_LABELS_CHANGED][:]
                if len(self._state.data_collection_labels) >= 2:
                    self._events.put(Event.ENOUGH_DATA_COLLECTION_LABELS)
                else:
                    self._events.put(Event.NOT_ENOUGH_DATA_COLLECTION_LABELS)

            if self._state.new_activity_name != new_values[Event.NEW_ACTIVITY_NAME_CHANGED]:
                self._state.new_activity_name = new_values[Event.NEW_ACTIVITY_NAME_CHANGED]

            if inherited_window_event == Event.ADD_NEW_ACTIVITY:
                if self._state.new_activity_name not in self._state.data_collection_label_candidates:
                    self._state.data_collection_label_candidates = [
                        self._state.new_activity_name] + self._state.data_collection_label_candidates

        if inherited_window_event is None:
            self._events.put(Event.CLOSE_WINDOW)
        return self._events

    def dispatch_events(self):
        for event in self._events.queue:
            logging.info('Dispatch event: ' + str(event))
            if event == Event.ENOUGH_ORIGIN_LABELS:
                self.enable_train_origin_model()
            elif event == Event.NOT_ENOUGH_ORIGIN_LABELS:
                self.disable_train_origin_model()
            elif event == Event.ENOUGH_DATA_COLLECTION_LABELS:
                self.enable_collect_data()
            elif event == Event.NOT_ENOUGH_DATA_COLLECTION_LABELS:
                self.disable_collect_data()
            elif event == Event.ENOUGH_NEW_LABELS:
                self.enable_train_new_model()
            elif event == Event.NOT_ENOUGH_NEW_LABELS:
                self.disable_train_new_model()
            elif event == Event.TRAIN_ORIGIN_MODEL:
                self.open_train_window(event)
            elif event == Event.VALIDATE_ORIGIN_MODEL:
                self.open_validate_window(event)
            elif event == Event.TEST_ORIGIN_MODEL:
                self.open_test_window(event)
            elif event == Event.COLLECT_NEW_DATA:
                self.open_data_collection_window()
            elif event == Event.ADD_NEW_ACTIVITY:
                self.update_data_collection_label_list()
                self.reset_new_activity_input()
            elif event == Event.CLOSE_WINDOW:
                return True
        return False

    def init_origin_model_view(self):
        heading = 'Step 1 - Use origin data'
        description = "Hold 'Ctrl' to select multiple classes you would like to train for the origin model."
        button_texts = ['Train model', 'Validate model', 'Test model']
        model_info = backend.get_model_summary(
            model=self._state.origin_model)

        header_row = [
            comp.heading(
                heading,
                fixed_column_width=self._column_width
            )
        ]

        description_row = [
            comp.text(
                description,
                fixed_column_width=self._column_width
            )
        ]

        label_selection_row = [
            comp.selection_list(
                items=self._state.origin_label_candidates,
                default_selections=self._state.origin_labels,
                fixed_column_width=self._column_width,
                mode='multiple',
                key=Event.ORIGIN_LABELS_CHANGED,
                rows=20
            )
        ]

        self._train_button_origin_model = comp.control_button(
            text=button_texts[0],
            fixed_column_width=self._column_width,
            disabled=len(self._state.origin_labels) == 0, key=Event.TRAIN_ORIGIN_MODEL
        )
        train_button_row = [
            self._train_button_origin_model
        ]

        self._validate_button_origin_model = comp.control_button(
            text=button_texts[1],
            fixed_column_width=self._column_width,
            disabled=self._state.origin_model is None, key=Event.VALIDATE_ORIGIN_MODEL
        )
        validate_button_row = [
            self._validate_button_origin_model
        ]

        self._test_button_origin_model = comp.control_button(
            text=button_texts[2],
            fixed_column_width=self._column_width,
            disabled=self._state.origin_model is None, key=Event.TEST_ORIGIN_MODEL
        )

        test_button_row = [
            self._test_button_origin_model
        ]

        self._text_origin_model = comp.text(
            text=model_info,
            fixed_column_width=self._column_width
        )

        model_info_row = [
            self._text_origin_model
        ]

        layout = [
            header_row,
            description_row,
            label_selection_row,
            train_button_row,
            validate_button_row,
            test_button_row,
            model_info_row
        ]

        return sg.Column(layout=layout, scrollable=False, element_justification='center')

    def init_data_collection_view(self):
        heading = 'Step 2 - Collect new data'
        description = "Type a new activity class or select an activity class from the dropdown list that you want to collect data for. Hold 'Ctrl' to select multiple classes you would like to collect data for."

        confirm_button_text = 'Add'

        data_info = backend.get_dataset_summary(
            dataset=self._state.new_dataset)

        button_texts = ['Collect data', 'Check data']

        header_row = [
            comp.heading(
                text=heading,
                fixed_column_width=self._column_width
            )
        ]

        description_row = [
            comp.text(
                text=description,
                fixed_column_width=self._column_width
            )
        ]

        self._input_new_activity = comp.TextInput(
            confirm_button_text=confirm_button_text,
            candidate_items=[],
            fixed_column_width=self._column_width,
            confirm_button_key=Event.ADD_NEW_ACTIVITY,
            text_input_key=Event.NEW_ACTIVITY_NAME_CHANGED,
        )
        input_row = self._input_new_activity.get_component_as_row()

        self._data_collection_label_list = comp.selection_list(
            items=self._state.data_collection_label_candidates,
            mode='multiple',
            fixed_column_width=self._column_width,
            default_selections=self._state.data_collection_labels,
            key=Event.DATA_COLLECTION_LABELS_CHANGED,
            rows=17
        )
        label_selection_row = [
            self._data_collection_label_list
        ]

        self._collect_button_new_data = comp.control_button(
            text=button_texts[0],
            key=Event.COLLECT_NEW_DATA,
            fixed_column_width=self._column_width,
            disabled=self._state.selected_activities_for_collection is None
        )
        collect_button_row = [
            self._collect_button_new_data
        ]

        self._check_button_new_data = comp.control_button(
            text=button_texts[1],
            key=Event.CHECK_NEW_DATA,
            fixed_column_width=self._column_width,
            disabled=self._state.collected_feature_set is None
        )
        check_button_row = [
            self._check_button_new_data
        ]

        self._text_new_data = comp.text(
            text=data_info,
            fixed_column_width=self._column_width
        )
        data_info_row = [
            self._text_new_data
        ]

        layout = [
            header_row,
            description_row,
            input_row,
            label_selection_row,
            collect_button_row,
            check_button_row,
            data_info_row
        ]

        return sg.Column(layout=layout, scrollable=False, element_justification='center')

    def init_new_model_view(self):
        heading = 'Step 3 - Use new data'
        description = "Hold 'Ctrl' to select multiple classes you would like to use the new data to train a new model."
        button_texts = ['Train model', 'Validate model', 'Test model']
        model_info = backend.get_model_summary(self._state.new_model)
        new_data = self._state.new_dataset
        if new_data is None:
            new_label_candidates = []
        else:
            new_label_candidates = new_data['GT_LABEL'].unique().tolist()

        header_row = [
            comp.heading(
                heading,
                fixed_column_width=self._column_width
            )
        ]

        description_row = [
            comp.text(
                description,
                fixed_column_width=self._column_width
            )
        ]

        label_selection_row = [
            comp.selection_list(
                items=new_label_candidates,
                default_selections=self._state.new_labels,
                fixed_column_width=self._column_width,
                mode='multiple',
                key=Event.NEW_LABELS_CHANGED,
                rows=20
            )
        ]

        self._train_button_new_model = comp.control_button(
            text=button_texts[0],
            fixed_column_width=self._column_width,
            disabled=self._state.new_labels is None, key=Event.TRAIN_NEW_MODEL
        )
        train_button_row = [
            self._train_button_new_model
        ]

        self._validate_button_new_model = comp.control_button(
            text=button_texts[1],
            fixed_column_width=self._column_width,
            disabled=self._state.new_model is None,
            key=Event.VALIDATE_NEW_MODEL
        )
        validate_button_row = [
            self._validate_button_new_model
        ]

        self._test_button_new_model = comp.control_button(
            text=button_texts[2],
            fixed_column_width=self._column_width,
            disabled=self._state.new_model is None,
            key=Event.TEST_NEW_MODEL
        )

        test_button_row = [
            self._test_button_origin_model
        ]

        self._text_new_model = comp.text(
            text=model_info,
            fixed_column_width=self._column_width
        )

        model_info_row = [
            self._text_new_model
        ]

        layout = [
            header_row,
            description_row,
            label_selection_row,
            train_button_row,
            validate_button_row,
            test_button_row,
            model_info_row
        ]

        return sg.Column(layout=layout, scrollable=False, element_justification='center')

    def open_train_window(self, event):
        if event == Event.TRAIN_ORIGIN_MODEL:
            panel = ModelTrainingPanel("Train origin model")
        elif event == Event.TRAIN_NEW_MODEL:
            panel = ModelUpdatePanel('Train new model')
        panel.start()

    def open_validate_window(self, event):
        if event == Event.VALIDATE_ORIGIN_MODEL:
            panel = ModelValidationPanel("Validate origin model")
        elif event == Event.VALIDATE_NEW_MODEL:
            panel = UpdateModelValidationPanel("Validate new model")
        panel.start()

    def open_test_window(self, event):
        if event == Event.TEST_ORIGIN_MODEL:
            panel = ModelTestingPanel("Test origin model")
        elif event == Event.TEST_NEW_MODEL:
            panel = UpdatedModelTestingPanel("Test new model")
        panel.start()

    def open_data_collection_window(self):
        panel = DataCollectionPanel("Collect new data")
        panel.start()

    def enable_train_origin_model(self):
        self._train_button_origin_model.update(disabled=False)
        self._validate_button_origin_model.update(disabled=False)

    def enable_train_new_model(self):
        self._train_button_new_model.update(disabled=False)
        self._validate_button_new_model.update(disabled=False)

    def enable_collect_data(self):
        self._collect_button_new_data.update(disabled=False)

    def disable_train_origin_model(self):
        self._train_button_origin_model.update(disabled=True)
        self._validate_button_origin_model.update(disabled=True)

    def disable_train_new_model(self):
        self._train_button_new_model.update(disabled=True)
        self._validate_button_new_model.update(disabled=True)

    def disable_collect_data(self):
        self._collect_button_new_data.update(disabled=True)

    def update_data_collection_label_list(self):
        self._data_collection_label_list.update(
            self._state.data_collection_label_candidates)

    def reset_new_activity_input(self):
        self._input_new_activity.update_input(value="")
