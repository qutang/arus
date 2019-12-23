import PySimpleGUI as sg
import sys
import os
from arus.testing import load_test_data
from arus.models.muss import MUSSModel
from arus.plugins.metawear.scanner import MetaWearScanner
import pandas as pd
import pathos.pools as pools
import threading
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ArusDemo:
    def __init__(self, title):
        self._title = title
        self._global_data = {'TRAIN_MODEL_RUNNING': False}

    def load_initial_data(self):
        class_filepath, _ = load_test_data(file_type='mhealth', sensor_type='class_labels',
                                           file_num='single', exception_type='multi_tasks')
        feature_filepath, _ = load_test_data(file_type='mhealth', sensor_type='feature',
                                             file_num='single', exception_type='multi_placements')
        class_df = pd.read_csv(class_filepath, parse_dates=[
            0, 1, 2], infer_datetime_format=True)
        feature_df = pd.read_csv(feature_filepath, parse_dates=[
            0, 1, 2], infer_datetime_format=True)
        self._global_data['INITIAL_CLASS_DF'] = class_df
        self._global_data['INITIAL_FEATURE_DF'] = feature_df

    def train_initial_model(self, training_labels):
        muss = MUSSModel()
        feature_set = self._global_data['INITIAL_FEATURE_DF']
        class_set = self._global_data['INITIAL_CLASS_DF']
        yield 'Extracting training data for DW...', 1
        dw_features = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DW', [
            'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + muss.get_feature_names()]
        yield 'Extracting training data for DA...', 2
        da_features = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DA', [
            'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + muss.get_feature_names()]
        yield 'Combining training data together...', 3
        combined_feature_set, combined_feature_names = muss.combine_features(
            dw_features, da_features, placement_names=['DW', 'DA'])
        cleared_class_set = class_set[['HEADER_TIME_STAMP',
                                       'START_TIME', 'STOP_TIME', 'MUSS_22_ACTIVITY_ABBRS']]
        yield 'Synchronizing training data and class labels...', 4
        synced_feature, synced_class = muss.sync_feature_and_class(
            combined_feature_set, cleared_class_set)
        # only use training labels
        yield 'Filtering out unused class labels...', 5
        filter_condition = synced_class['MUSS_22_ACTIVITY_ABBRS'].isin(
            training_labels)
        input_feature = synced_feature.loc[filter_condition, :]
        input_class = synced_class.loc[filter_condition, :]

        yield 'Training SVM classifier...', 6
        pool = pools.ProcessPool(nodes=1)
        task = pool.apipe(muss.train_classifier, input_feature,
                          input_class, class_col='MUSS_22_ACTIVITY_ABBRS', feature_names=combined_feature_names, placement_names=['DW', 'DA'])
        i = 7
        while not task.ready() and self._global_data['TRAIN_MODEL_RUNNING']:
            yield 'Training SVM classifier...', min((i, 9))
            i = i + 1
        if task.ready():
            model = task.get()
            yield 'Completed. Training accuracy: ' + str(model[2]), 10
            self._global_data['INITIAL_MODEL'] = model
        else:
            return

    def validate_initial_model(self):
        muss = MUSSModel()
        feature_set = self._global_data['INITIAL_FEATURE_DF']
        class_set = self._global_data['INITIAL_CLASS_DF']
        yield 'Extracting validation data for DW...', 1
        dw_features = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DW', [
            'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME', 'PID'] + muss.get_feature_names()]
        yield 'Extracting validation data for DA...', 2
        da_features = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DA', [
            'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME', 'PID'] + muss.get_feature_names()]
        yield 'Combining validation data together...', 3
        combined_feature_set, combined_feature_names = muss.combine_features(
            dw_features, da_features, placement_names=['DW', 'DA'], group_col='PID')
        cleared_class_set = class_set[['HEADER_TIME_STAMP',
                                       'START_TIME', 'STOP_TIME', 'PID', 'MUSS_22_ACTIVITY_ABBRS']]
        yield 'Synchronizing training data and class labels...', 4
        synced_feature, synced_class = muss.sync_feature_and_class(
            combined_feature_set, cleared_class_set, group_col='PID')
        # only use training labels
        yield 'Filtering out unused class labels...', 5
        training_labels = self._global_data['INITIAL_MODEL'][0].classes_
        filter_condition = synced_class['MUSS_22_ACTIVITY_ABBRS'].isin(
            training_labels)
        input_feature = synced_feature.loc[filter_condition, :]
        input_class = synced_class.loc[filter_condition, :]

        yield 'Validating SVM classifier...', 6
        pool = pools.ProcessPool(nodes=1)
        task = pool.apipe(muss.validate_classifier, input_feature,
                          input_class, class_col='MUSS_22_ACTIVITY_ABBRS', feature_names=combined_feature_names, placement_names=['DW', 'DA'], group_col='PID')
        i = 7
        while not task.ready() and self._global_data['VALIDATE_MODEL_RUNNING']:
            yield 'Validating SVM classifier...', min((i, 99))
            i = i + 1
        if task.ready():
            result = task.get()
            yield 'Completed. Validation accuracy: ' + str(result[2]), 100
            self._global_data['INITIAL_MODEL_VALIDATION'] = result + \
                (training_labels, )
        else:
            return

    def test_initial_model(self):
        pass

    def get_nearby_devices(self):
        scanner = MetaWearScanner()
        pool = pools.ThreadPool(nodes=1)
        task = pool.apipe(scanner.get_nearby_devices, max_devices=2)
        while not task.ready():
            yield False
        self._global_data['NEARBY_DEVICES'] = task.get()[:2]
        yield True
        return

    def _get_initial_class_labels(self):
        class_df = self._global_data['INITIAL_CLASS_DF']
        return class_df['MUSS_22_ACTIVITY_ABBRS'].unique().tolist()

    def _init_dashboard_model_training(self, class_labels=[]):
        heading = 'Step 1 - Train initial model'
        button_texts = ['Train model', 'Validate model', 'Test model']

        header = [[sg.Text(text=heading, relief=sg.RELIEF_FLAT,
                           font=('Helvetica', 12, 'bold'), size=(20, 1))]]
        description = [[sg.Text(
            text="Hold 'Ctrl' to select multiple classes\nyou would like to train for the model.", font=('Helvetica', 10))]]
        class_label_list = [[sg.Listbox(
            values=class_labels, select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, font=('Helvetica', 10), size=(27, 20), key='_TRAINING_CLASS_LABELS_', enable_events=True)]]
        buttons = [[sg.Button(button_text=text,
                              font=('Helvetica', 11), auto_size_button=True, size=(20, None), key="_" + text.upper().replace(' ', '_') + '_INITIAL_', disabled=True)] for text in button_texts]
        info = [
            [sg.Text(text="No model available", font=('Helvetica', 10), size=(20, None), key='_INITIAL_MODEL_INFO_')]]

        return sg.Column(layout=header + description + class_label_list + buttons + info, scrollable=False)

    def _init_dashboard_new_activities(self, class_labels=[]):
        heading = 'Step 2 - Add new activities'
        button_texts = ['Collect data', 'Check data']

        header = [[sg.Text(text=heading, relief=sg.RELIEF_FLAT,
                           font=('Helvetica', 12, 'bold'), size=(20, 1))]]
        description = [[sg.Text(
            text="Type a new activity class or select\nan activity class from the dropdown list\nthat you want to collect data for.\nHold 'Ctrl' to select multiple classes\nyou would like to collect data for.", font=('Helvetica', 10))]]
        input_new_activity = [
            [
                sg.Combo(
                    values=class_labels, font=('Helvetica', 11), size=(15, None)),
                sg.Button(button_text='Add', font=('Helvetica', 10), size=(5, None))],
            [
                sg.Listbox(
                    values=[], select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, size=(27, 15))
            ]]

        buttons = [[sg.Button(button_text=text,
                              font=('Helvetica', 11), auto_size_button=True, size=(20, None))] for text in button_texts]

        return sg.Column(layout=header + description + input_new_activity + buttons, scrollable=False)

    def _init_dashboard_model_update(self, class_labels=[]):
        heading = 'Step 3 - Update model'
        button_texts = ['Update model', 'Validate model']

        header = [[sg.Text(text=heading, relief=sg.RELIEF_FLAT,
                           font=('Helvetica', 12, 'bold'), size=(20, 1))]]
        description = [[sg.Text(
            text="Hold 'Ctrl' to select multiple classes\nyou would like to use the new data\nto update the model.", font=('Helvetica', 10))]]
        class_label_list = [[sg.Listbox(
            values=class_labels, select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, font=('Helvetica', 10), size=(27, 19))]]
        buttons = [[sg.Button(button_text=text,
                              font=('Helvetica', 11), auto_size_button=True, size=(20, None))] for text in button_texts]

        return sg.Column(layout=header + description + class_label_list + buttons, scrollable=False)

    def _init_dashboard_model_test(self):
        heading = 'Step 4 - Test model'
        button_texts = ['Test model']

        header = [[sg.Text(text=heading, relief=sg.RELIEF_FLAT,
                           font=('Helvetica', 12, 'bold'), size=(20, 1))]]
        buttons = [[sg.Button(button_text=text,
                              font=('Helvetica', 11), auto_size_button=True, size=(20, None))] for text in button_texts]

        return sg.Column(layout=header + buttons, scrollable=False)

    def init_dashboard(self):
        class_labels = self._get_initial_class_labels()
        col_model_training = self._init_dashboard_model_training(
            class_labels=class_labels)
        col_new_activities = self._init_dashboard_new_activities(
            class_labels=class_labels)
        col_model_update = self._init_dashboard_model_update(
            class_labels=class_labels)
        col_model_test = self._init_dashboard_model_test()
        layout = [
            [col_model_training, col_new_activities,
                col_model_update, col_model_test]
        ]
        dashboard = sg.Window('Dashboard', layout=layout,
                              finalize=True)
        return dashboard

    def _display_model_info(self, model, textElement):
        name = ','.join(model[0].classes_)
        acc = model[2]
        info = 'Model:\n' + name + '\nTraining accuracy: ' + str(acc)[:5]
        textElement.Update(
            value=info)

    def popup_init_model_training(self, training_labels):
        layout = [[sg.Text('Initializing training data...', key='_TRAIN_MODEL_MESSAGE_', size=(40, 1))],
                  [sg.ProgressBar(10, orientation='h', size=(
                      40, 20), key='_TRAIN_MODEL_PROGRESS_BAR_')],
                  [sg.Button(button_text='Cancel', key='_TRAIN_MODEL_CLOSE_')]]
        popup = sg.Window('Training initial model...', layout, finalize=True)
        progress_bar = popup['_TRAIN_MODEL_PROGRESS_BAR_']
        message_bar = popup['_TRAIN_MODEL_MESSAGE_']
        self._global_data['TRAIN_MODEL_RUNNING'] = True
        for message, progress in self.train_initial_model(training_labels):
            event, _ = popup.read(timeout=500)
            if event == '_TRAIN_MODEL_CLOSE_' or event is None:
                break
            progress_bar.UpdateBar(progress)
            message_bar.Update(value=message)
        self._global_data['TRAIN_MODEL_RUNNING'] = False
        if progress == 10:
            popup['_TRAIN_MODEL_CLOSE_'].Update(text='Close')
            event, _ = popup.read()
        popup.close()

    def _display_cm_figure(self, validation_result, canvasElement):
        muss = MUSSModel()
        labels = validation_result[-1]
        fig = muss.get_confusion_matrix(
            validation_result[0], validation_result[1], labels=labels, graph=True)
        _, _, figure_w, figure_h = fig.bbox.bounds
        canvasElement.set_size((figure_w, figure_h))
        figure_canvas_agg = FigureCanvasTkAgg(fig, canvasElement.TKCanvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    def _display_classification_report(self, validation_result, textElement):
        muss = MUSSModel()
        report = muss.get_classification_report(
            validation_result[0], validation_result[1], labels=validation_result[-1])
        textElement.Update(value=report, visible=True)

    def popup_init_model_validation(self, run=True):
        layout = [[sg.Text('Initializing validation data...' if run else 'Already validated.', key='_VALIDATE_MODEL_MESSAGE_', size=(40, 1))],
                  [sg.ProgressBar(100, orientation='h', size=(
                      40, 20), key='_VALIDATE_MODEL_PROGRESS_BAR_')],
                  [sg.Canvas(size=(3, 3),
                             key='_VALIDATE_MODEL_CM_FIG_')],
                  [sg.Text(size=(50, None),
                           key='_VALIDATE_MODEL_REPORT_')],
                  [sg.Button(button_text='Cancel', key='_VALIDATE_MODEL_CLOSE_')]]
        popup = sg.Window('Validating initial model...', layout, finalize=True)
        progress_bar = popup['_VALIDATE_MODEL_PROGRESS_BAR_']
        message_bar = popup['_VALIDATE_MODEL_MESSAGE_']
        cm_canvas = popup['_VALIDATE_MODEL_CM_FIG_']
        report_text = popup['_VALIDATE_MODEL_REPORT_']
        if run:
            self._global_data['VALIDATE_MODEL_RUNNING'] = True
            for message, progress in self.validate_initial_model():
                event, _ = popup.read(timeout=500)
                if event == '_VALIDATE_MODEL_CLOSE_' or event is None:
                    break
                progress_bar.UpdateBar(progress)
                message_bar.Update(value=message)
            self._global_data['VALIDATE_MODEL_RUNNING'] = False
            if progress == 100:
                popup['_VALIDATE_MODEL_CLOSE_'].Update(text='Close')
                self._display_cm_figure(
                    self._global_data['INITIAL_MODEL_VALIDATION'], cm_canvas)
                self._display_classification_report(
                    self._global_data['INITIAL_MODEL_VALIDATION'], report_text)
                report_text.Update(visible=True)
                event, _ = popup.read()
        else:
            popup['_VALIDATE_MODEL_CLOSE_'].Update(text='Close')
            self._display_cm_figure(
                self._global_data['INITIAL_MODEL_VALIDATION'], cm_canvas)
            self._display_classification_report(
                self._global_data['INITIAL_MODEL_VALIDATION'], report_text)
            while True:
                event, _ = popup.read(timeout=500)
                if event == '_VALIDATE_MODEL_CLOSE_' or event is None:
                    break
        popup.close()

    def popup_init_model_testing(self):
        labels = list(self._global_data['INITIAL_MODEL'][0].classes_)
        layout = [
            [sg.Button(button_text='Scan nearby devices',
                       key='_TEST_MODEL_SCAN_'),
             sg.VerticalSeparator(pad=None),
             sg.Column([
                 [sg.Text('Dom Wrist')],
                 [sg.Image('./dom_wrist.png', size=(None, None))],
                 [sg.Text(text='',
                          key='_TEST_MODEL_WRIST_', size=(20, None))],
             ]),
             sg.VerticalSeparator(pad=None),
             sg.Column([
                 [sg.Text('Right Ankle')],
                 [sg.Image('./right_ankle.png', size=(None, None))],
                 [sg.Text(text='',
                          key='_TEST_MODEL_ANKLE_', size=(20, None))],
             ])
             ],
            [sg.Text('Please put on the corresponding devices on body at places shown by the pictures.\nMake sure the sensor ID matches the placement you put on.')],
            [
                sg.Listbox(
                    values=labels + ['Any'], default_values=['Any'], select_mode=sg.LISTBOX_SELECT_MODE_BROWSE, enable_events=True, auto_size_text=True, size=(None, min(10, len(labels) + 1)), key='_TEST_MODEL_LABELS_'),
                sg.VerticalSeparator(pad=None),
                sg.Column(layout=[
                          [sg.Text(label, font=('Helvetica', 13), size=(15, None), background_color='green', text_color='white')] for label in labels])
            ],
            [
                sg.Text('Current window: not started')
            ],
            [
                sg.Button(button_text='Start testing',
                          disabled=True, key='_TEST_MODEL_START_')
            ],
            [sg.Button(button_text='Close', key='_TEST_MODEL_CLOSE_')]
        ]
        popup = sg.Window('Testing initial model',
                          layout=layout, finalize=True)

        device_elements = [popup['_TEST_MODEL_WRIST_'],
                           popup['_TEST_MODEL_ANKLE_']]
        scan_button = popup['_TEST_MODEL_SCAN_']
        start_button = popup['_TEST_MODEL_START_']
        while True:
            event, values = popup.read(timeout=200)
            if event == '_TEST_MODEL_SCAN_':
                scan_button.Update(
                    text='Scanning...', disabled=True)
                for ready in self.get_nearby_devices():
                    event, _ = popup.read(timeout=200)
                    if event == '_TEST_MODEL_CLOSE_' or event is None or ready:
                        break
                device_addrs = self._global_data['NEARBY_DEVICES']
                for el, addr in zip(device_elements, device_addrs):
                    el.Update(value=addr)
                scan_button.Update(
                    text='Scan nearby devices', disabled=False)
                if 'NEARBY_DEVICES' in self._global_data:
                    start_button.Update(disabled=False)
            elif event == '_TEST_MODEL_CLOSE_' or event is None:
                break
            elif event == '_TEST_MODEL_LABELS':
                self._global_data['INITIAL_MODEL_TEST_LABEL'] = values['_TEST_MODEL_LABELS_']
        popup.close()

    def handle_dashboard(self, event, values, dashboard):
        if event == "_TRAIN_MODEL_INITIAL_":  # Train model
            training_labels = values['_TRAINING_CLASS_LABELS_']
            if 'INITIAL_MODEL' in self._global_data and np.array_equal(sorted(training_labels), sorted(self._global_data['INITIAL_MODEL'][0].classes_)):
                sg.Popup('Already trained.')
            else:
                self.popup_init_model_training(training_labels)
            if 'INITIAL_MODEL' in self._global_data:
                model = self._global_data['INITIAL_MODEL']
                textElement = dashboard['_INITIAL_MODEL_INFO_']
                self._display_model_info(model, textElement)
                dashboard['_TEST_MODEL_INITIAL_'].Update(disabled=False)
                dashboard['_VALIDATE_MODEL_INITIAL_'].Update(disabled=False)
        elif event == '_VALIDATE_MODEL_INITIAL_':
            training_labels = values['_TRAINING_CLASS_LABELS_']
            if 'INITIAL_MODEL_VALIDATION' in self._global_data and np.array_equal(sorted(self._global_data['INITIAL_MODEL_VALIDATION'][-1]), sorted(self._global_data['INITIAL_MODEL'][0].classes_)):
                self.popup_init_model_validation(run=False)
            else:
                self.popup_init_model_validation(run=True)
            # if self._global_data['INITIAL_MODEL_VALIDATION'] is not None:
        elif event == '_TEST_MODEL_INITIAL_':
            self.popup_init_model_testing()
        elif event == "_TRAINING_CLASS_LABELS_":  # Only enable buttons when there are two or more classes selected
            num_classes = len(values['_TRAINING_CLASS_LABELS_'])
            if num_classes > 1:
                dashboard['_TRAIN_MODEL_INITIAL_'].Update(disabled=False)
            else:
                dashboard['_TRAIN_MODEL_INITIAL_'].Update(disabled=True)

    def start(self):
        self.load_initial_data()
        dashboard = self.init_dashboard()

        while True:
            event, values = dashboard.read()
            print(event)
            if event is None:
                break
            else:
                self.handle_dashboard(event, values, dashboard)
        dashboard.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    demo = ArusDemo(title='Arus active training demo')
    demo.start()
