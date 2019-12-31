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


def dashboard_heading(text):
    return sg.Text(text=text, relief=sg.RELIEF_FLAT,
                           font=('Helvetica', 12, 'bold'), size=(20, 1))

def dashboard_description(text, key=None):
    return sg.Text(
            text=text, font=('Helvetica', 10), size=(25, None), key=key)

def dashboard_listbox(items, mode, key=None):
    return sg.Listbox(
            values=items, select_mode=mode, font=('Helvetica', 10), size=(27, 20), key=key, enable_events=True)

def dashboard_control_button(text, disabled, key=None):
    return sg.Button(button_text=text,
                              font=('Helvetica', 11), auto_size_button=True, size=(20, None), key=key, disabled=disabled)

def dashboard_text_combo_with_button(items, text, key=None):
    return sg.Combo(values=items, font=('Helvetica', 11), size=(15, None), key=key), sg.Button(button_text=text, font=('Helvetica', 10), size=(5, None))


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

    def _init_dashboard_model_training(self, class_labels=[]):
        heading = 'Step 1 - Train initial model'
        description = "Hold 'Ctrl' to select multiple classes you would like to train for the model."
        button_texts = ['Train model', 'Validate model', 'Test model']
        model_info = "No model available"

        key_listbox = '_CLASS_LABELS_INITIAL_'
        key_buttons = ["_" + text.upper().replace(' ', '_') + '_INITIAL_' for text in button_texts]
        key_model_info = '_MODEL_INFO_INITIAL_'

        header = [[dashboard_heading(heading)]]

        description = [[dashboard_description(description)]]

        class_label_list = [[dashboard_listbox(
                                items=class_labels, 
                                mode=sg.LISTBOX_SELECT_MODE_EXTENDED, 
                                key=key_listbox)
                            ]]
        
        buttons = [
                [dashboard_control_button(text=text, disabled=True, key=key)] for text, key in zip(button_texts, key_buttons)
            ]
        
        info = [[
                dashboard_description(text=model_info, key=key_model_info)
            ]]

        return sg.Column(layout=header + description + class_label_list + buttons + info, scrollable=False)

    def _init_dashboard_new_activities(self, class_labels=[]):
        heading = 'Step 2 - Add new activities'
        description = "Type a new activity class or select an activity class from the dropdown list that you want to collect data for. Hold 'Ctrl' to select multiple classes you would like to collect data for."
        text_combo_text = 'Add'
        button_texts = ['Collect data', 'Check data']

        header = [[dashboard_heading(text=heading)]]
        description = [[dashboard_description(text=description)]]
        text_combo = [dashboard_text_combo_with_button(items=class_labels, text=text_combo_text)]

        class_labels_list = [[
                dashboard_listbox(
                    items=[], mode=sg.LISTBOX_SELECT_MODE_EXTENDED)
            ]]

        buttons = [[dashboard_control_button(text=text, disabled=True)] for text in button_texts]

        return sg.Column(layout=header + description + text_combo + class_labels_list + buttons, scrollable=False)

    def _init_dashboard_model_update(self, class_labels=[]):
        heading = 'Step 3 - Update model'
        description = "Hold 'Ctrl' to select multiple classes you would like to use the new data to update the model."
        button_texts = ['Update model', 'Validate model']

        header = [[dashboard_heading(heading)]]
        description = [[dashboard_description(description)]]
        class_label_list = [[dashboard_listbox(
            items=class_labels, mode=sg.LISTBOX_SELECT_MODE_EXTENDED)]]
        buttons = [[dashboard_control_button(text, disabled=True)] for text in button_texts]

        return sg.Column(layout=header + description + class_label_list + buttons, scrollable=False)

    def _get_initial_class_labels(self):
        class_df = self._app_state.initial_dataset[1]
        return class_df['MUSS_22_ACTIVITY_ABBRS'].unique().tolist()

    def init_dashboard(self):
        class_labels = self._get_initial_class_labels()
        col_model_training = self._init_dashboard_model_training(
            class_labels=class_labels)
        col_new_activities = self._init_dashboard_new_activities(
            class_labels=class_labels)
        col_model_update = self._init_dashboard_model_update(
            class_labels=class_labels)
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
        return dashboard

    def _display_model_info(self):
        model = self._app_state.initial_model
        name = ','.join(model[0].classes_)
        acc = model[2]
        info = 'Model:\n' + name + '\nTraining accuracy: ' + str(acc)[:5]
        self._initial_model_info_text.Update(
            value=info)


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
            self._display_model_info()
            self._initial_model_test_button.Update(disabled=False)
            self._initial_model_validate_button.Update(disabled=False)

    def _handle_initial_model_validation(self):
        panel = ModelValidationPanel("Initial model validation")
        panel.start()

    def _handle_initial_model_testing(self):
        pass

    def _handle_initial_model_training_label_changed(self, labels):
        num_classes = len(labels)
        if num_classes > 1:
            self._initial_model_train_button.Update(disabled=False)
        else:
            self._initial_model_train_button.Update(disabled=True)
        self._app_state.initial_model_training_labels = labels

    def handle_dashboard(self, event, values):
        if event == self._initial_model_train_button.Key:  # Train model
            self._handle_initial_model_training()
        elif event == self._initial_model_validate_button.Key:
            self._handle_initial_model_validation()
        elif event == self._initial_model_test_button.Key:
            self._handle_initial_model_testing()
        elif event == self._initial_model_class_labels.Key:  # Only enable buttons when there are two or more classes selected
            self._handle_initial_model_training_label_changed(values[self._initial_model_class_labels.Key])

    def start(self):
        sg.ChangeLookAndFeel('Reddit')
        self.init_dashboard()
        while True:
            event, values = self._dashboard_window.read()
            print(event)
            if event is None:
                break
            else:
                self.handle_dashboard(event, values)
        self._dashboard_window.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    app_state = app.AppState.getInstance()
    app_state.initial_dataset = backend.load_initial_data()
    demo = Dashboard(title='Dashboard')
    demo.start()
