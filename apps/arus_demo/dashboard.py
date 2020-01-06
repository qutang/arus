import logging
import os
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
from model_training_panel import ModelTrainingPanel
from model_validation_panel import ModelValidationPanel
from model_testing_panel import ModelTestingPanel
from data_collection_panel import DataCollectionPanel
from model_update_panel import ModelUpdatePanel


def dashboard_heading(text):
    return sg.Text(text=text, relief=sg.RELIEF_FLAT,
                   font=('Helvetica', 12, 'bold'), size=(20, 1))


def dashboard_description(text, key=None):
    return sg.Text(
        text=text, font=('Helvetica', 10), size=(25, None), key=key)


def dashboard_listbox(items, mode, defaults=None, key=None, right_click_menu=None, num_of_rows=20):
    return sg.Listbox(
        values=items, default_values=defaults, select_mode=mode, font=('Helvetica', 10), size=(27, num_of_rows), key=key, enable_events=True, right_click_menu=right_click_menu)


def dashboard_control_button(text, disabled, key=None):
    return sg.Button(button_text=text,
                     font=('Helvetica', 11), auto_size_button=True, size=(20, None), key=key, disabled=disabled)


def dashboard_text_combo_with_button(items, text, button_key=None, input_key=None):
    add_button = sg.Button(button_text=text, font=(
        'Helvetica', 10), size=(5, None), key=button_key)
    new_activity_input = sg.Combo(values=items, font=(
        'Helvetica', 11), size=(15, None), key=input_key, enable_events=True)
    return new_activity_input, add_button


class Dashboard:
    def __init__(self, title):
        self._title = title
        self._initial_model_train_button = None
        self._initial_model_validate_button = None
        self._initial_model_test_button = None
        self._dashboard_window = None
        self._initial_model_info_text = None
        self._initial_model_class_labels = None
        self._app_state = app.AppState.getInstance()

    def _init_dashboard_model_training(self):
        heading = 'Step 1 - Train initial model'
        description = "Hold 'Ctrl' to select multiple classes you would like to train for the model."
        button_texts = ['Train model', 'Validate model', 'Test model']
        model_info = self._get_model_info()

        key_listbox = '_CLASS_LABELS_INITIAL_'
        key_buttons = ["_" + text.upper().replace(' ', '_') +
                       '_INITIAL_' for text in button_texts]
        key_model_info = '_MODEL_INFO_INITIAL_'

        header = [[dashboard_heading(heading)]]

        description = [[dashboard_description(description)]]

        class_label_list = [[dashboard_listbox(
            items=self._class_labels,
            defaults=self._app_state.initial_model_training_labels,
            mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
            key=key_listbox)
        ]]
        button_disabled = [
            self._app_state.initial_model_training_labels is None,
            self._app_state.initial_model is None,
            self._app_state.initial_model is None,
        ]
        buttons = [
            [dashboard_control_button(text=text, disabled=disabled, key=key)] for text, key, disabled in zip(button_texts, key_buttons, button_disabled)
        ]

        info = [[
                dashboard_description(text=model_info, key=key_model_info)
                ]]

        return sg.Column(layout=header + description + class_label_list + buttons + info, scrollable=False)

    def _init_dashboard_new_activities(self):
        heading = 'Step 2 - Add new activities'
        description = "Type a new activity class or select an activity class from the dropdown list that you want to collect data for. Hold 'Ctrl' to select multiple classes you would like to collect data for."

        text_combo_text = 'Add'

        data_info = self._get_data_info()

        add_button_key = '_NEW_ACTIVITY_ADD_'
        new_activity_input_key = '_NEW_ACTIVITY_INPUT_'

        activity_list_key = '_NEW_ACTIVITY_LIST_'

        button_texts = ['Collect data', 'Check data']
        button_keys = ['_NEW_ACTIVITY_COLLECT_', '_NEW_ACTIVITY_CHECK_']
        data_info_key = '_NEW_ACTIVITY_INFO_'

        header = [[dashboard_heading(text=heading)]]
        description = [[dashboard_description(text=description)]]
        text_combo = [dashboard_text_combo_with_button(
            items=[], text=text_combo_text, button_key=add_button_key, input_key=new_activity_input_key)]

        class_labels_list = [[
            dashboard_listbox(
                items=self._class_labels, mode=sg.LISTBOX_SELECT_MODE_EXTENDED, defaults=self._app_state.selected_activities_for_collection, key=activity_list_key, num_of_rows=17)
        ]]

        button_disabled = [
            self._app_state.selected_activities_for_collection is None,
            self._app_state.collected_feature_set is None,
        ]

        buttons = [[dashboard_control_button(
            text=text, disabled=disabled, key=key)] for text, key, disabled in zip(button_texts, button_keys, button_disabled)]

        info = [[
                dashboard_description(text=data_info, key=data_info_key)
                ]]

        return sg.Column(layout=header + description + text_combo + class_labels_list + buttons + info, scrollable=False)

    def _init_dashboard_model_update(self):
        heading = 'Step 3 - Update model'
        description = "Hold 'Ctrl' to select multiple classes you would like to use the new data to update the model."
        button_texts = ['Update model', 'Validate model', 'Test model']
        button_keys = ['_UPDATE_MODEL_UPDATE_',
                       '_UPDATE_MODEL_VALIDATE_', '_UPDATE_MODEL_TEST_']
        update_label_list_key = '_UPDATE_LABELS_'
        update_info_key = '_UPDATE_INFO_'

        new_data = self._app_state.collected_feature_set

        update_info = self._get_update_info()

        if new_data is None:
            update_labels = []
        else:
            update_labels = new_data['GT_LABEL'].unique().tolist()

        header = [[dashboard_heading(heading)]]
        description = [[dashboard_description(description)]]
        update_label_list = [[dashboard_listbox(
            items=update_labels, mode=sg.LISTBOX_SELECT_MODE_EXTENDED, defaults=self._app_state.selected_activities_for_update, key=update_label_list_key)]]

        buttons_disabled = [
            self._app_state.selected_activities_for_update is None,
            self._app_state.updated_model is None,
            self._app_state.updated_model is None
        ]
        buttons = [[dashboard_control_button(
            text, disabled=disabled, key=key)] for text, key, disabled in zip(button_texts, button_keys, buttons_disabled)]

        info = [[
                dashboard_description(text=update_info, key=update_info_key)
                ]]

        return sg.Column(layout=header + description + update_label_list + buttons + info, scrollable=False)

    def _get_initial_class_labels(self):
        class_df = self._app_state.initial_dataset[1]
        return class_df['MUSS_22_ACTIVITY_ABBRS'].unique().tolist()

    def init_dashboard(self):
        self._class_labels = self._get_initial_class_labels()
        col_model_training = self._init_dashboard_model_training()
        col_new_activities = self._init_dashboard_new_activities()
        col_model_update = self._init_dashboard_model_update()
        layout = [
            [col_model_training, col_new_activities,
                col_model_update]
        ]
        dashboard = sg.Window(self._title, layout=layout,
                              finalize=True)

        self._dashboard_window = dashboard
        self._initial_model_train_button = dashboard['_TRAIN_MODEL_INITIAL_']
        self._initial_model_class_labels = dashboard['_CLASS_LABELS_INITIAL_']
        self._initial_model_validate_button = dashboard['_VALIDATE_MODEL_INITIAL_']
        self._initial_model_test_button = dashboard['_TEST_MODEL_INITIAL_']
        self._initial_model_info_text = dashboard['_MODEL_INFO_INITIAL_']

        self._new_activity_add_button = dashboard['_NEW_ACTIVITY_ADD_']
        self._new_activity_check_button = dashboard['_NEW_ACTIVITY_CHECK_']
        self._new_activity_input = dashboard['_NEW_ACTIVITY_INPUT_']
        self._new_activity_list = dashboard['_NEW_ACTIVITY_LIST_']
        self._new_activity_collect_button = dashboard['_NEW_ACTIVITY_COLLECT_']
        self._new_activity_info = dashboard['_NEW_ACTIVITY_INFO_']

        self._update_model_update_button = dashboard['_UPDATE_MODEL_UPDATE_']
        self._update_model_validate_button = dashboard['_UPDATE_MODEL_VALIDATE_']
        self._update_model_test_button = dashboard['_UPDATE_MODEL_TEST_']
        self._update_label_list = dashboard['_UPDATE_LABELS_']
        self._update_info_text = dashboard['_UPDATE_INFO_']
        return dashboard

    def _get_model_info(self):
        model = self._app_state.initial_model
        if model is not None:
            name = ','.join(model[0].classes_)
            acc = model[2]
            info = 'Model:\n' + name + '\nTraining accuracy: ' + str(acc)[:5]
        else:
            info = 'No model is available'
        return info

    def _handle_initial_model_training(self):
        if self._app_state.initial_model is not None:
            no_need_retrain = np.array_equal(
                sorted(self._app_state.initial_model_training_labels),
                sorted(self._app_state.initial_model[0].classes_)
            )
        else:
            no_need_retrain = False
        if self._app_state.initial_model is not None and no_need_retrain:
            sg.Popup('Already trained.')
        else:
            panel = ModelTrainingPanel("Initial model training")
            panel.start()
        if self._app_state.initial_model is not None:
            info = self._get_model_info()
            self._initial_model_info_text.Update(value=info)
            self._initial_model_test_button.Update(disabled=False)
            self._initial_model_validate_button.Update(disabled=False)

    def _get_data_info(self):
        new_data = self._app_state.collected_feature_set
        if new_data is not None:
            labels = new_data['GT_LABEL'].unique().tolist()
            num_of_windows = new_data.shape[0]
            info = 'New data for labels:\n' + \
                str(labels) + '\nTotal windows: ' + str(num_of_windows)
        else:
            info = 'No data is available'
        return info

    def _get_update_info(self):
        model = self._app_state.updated_model
        if model is not None:
            name = ','.join(model[0].classes_)
            acc = model[2]
            info = 'Model:\n' + name + '\nTraining accuracy: ' + str(acc)[:5]
        else:
            info = 'No updated model is available'
        return info

    def _handle_initial_model_validation(self):
        panel = ModelValidationPanel("Initial model validation")
        panel.start()

    def _handle_initial_model_testing(self):
        panel = ModelTestingPanel("Initial model testing")
        panel.start()

    def _handle_data_collection(self):
        panel = DataCollectionPanel("Data collection for new activites")
        panel.start()
        new_data = self._app_state.collected_feature_set
        if new_data is not None:
            info = self._get_data_info()
            self._new_activity_info.Update(value=info)
            print(new_data)
            self._new_activity_check_button.Update(disabled=False)
            update_labels = new_data['GT_LABEL'].unique().tolist()
            self._update_label_list.Update(update_labels, disabled=False)

    def _handle_update_initial_model(self):
        panel = ModelUpdatePanel("Update model with new collected data")
        panel.start()
        if self._app_state.updated_model is not None:
            info = self._get_update_info()
            self._update_info_text.Update(value=info)
            self._update_model_validate_button.Update(disabled=False)
            self._update_model_test_button.Update(disabled=False)

    def _handle_initial_model_training_label_changed(self, labels):
        num_classes = len(labels)
        if num_classes > 1:
            self._initial_model_train_button.Update(disabled=False)
        else:
            self._initial_model_train_button.Update(disabled=True)
        self._app_state.initial_model_training_labels = labels

    def _handle_activities_for_collection_changed(self, labels):
        num_classes = len(labels)
        if num_classes >= 1:
            self._new_activity_collect_button.Update(disabled=False)
        else:
            self._new_activity_collect_button.Update(disabled=True)
        self._app_state.selected_activities_for_collection = labels

    def _handle_add_new_activity(self, name):
        if name not in self._class_labels:
            self._class_labels = [name] + self._class_labels
            self._new_activity_list.Update(self._class_labels)

    def _hanlde_activities_for_update_changed(self, labels):
        num_classes = len(labels)
        if num_classes >= 1:
            self._update_model_update_button.Update(disabled=False)
            self._update_model_validate_button.Update(disabled=False)
        else:
            self._update_model_update_button.Update(disabled=True)
            self._update_model_validate_button.Update(disabled=True)
        self._app_state.selected_activities_for_update = labels

    def handle_dashboard(self, event, values):
        if event == self._initial_model_train_button.Key:  # Train model
            self._handle_initial_model_training()
        elif event == self._initial_model_validate_button.Key:
            self._handle_initial_model_validation()
        elif event == self._initial_model_test_button.Key:
            self._handle_initial_model_testing()
        elif event == self._new_activity_collect_button.Key:
            self._handle_data_collection()
        # Only enable buttons when there are two or more classes selected
        elif event == self._initial_model_class_labels.Key:
            self._handle_initial_model_training_label_changed(
                values[self._initial_model_class_labels.Key])
        elif event == self._new_activity_add_button.Key:
            self._handle_add_new_activity(values[self._new_activity_input.Key])
        elif event == self._new_activity_list.Key:
            self._handle_activities_for_collection_changed(
                values[self._new_activity_list.Key])
        elif event == self._update_label_list.Key:
            self._hanlde_activities_for_update_changed(
                values[self._update_label_list.Key])
        elif event == self._update_model_update_button.Key:
            self._handle_update_initial_model()

    def start(self):
        sg.ChangeLookAndFeel('Reddit')
        self.init_dashboard()
        while True:
            event, values = self._dashboard_window.read()
            print(event)
            print(values)
            if event is None:
                break
            else:
                self.handle_dashboard(event, values)
        self._dashboard_window.close()
