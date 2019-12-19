import PySimpleGUI as sg
import sys
from arus.testing import load_test_data
from arus.models.muss import MUSSModel
import pandas as pd
import pathos.pools as pools
import time


# Backend handler functions
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
        combined_feature_set = muss.combine_features(dw_features, da_features,
                                                     placement_names=['DW', 'DA'])
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
        task = pool.apipe(muss.train_classifier, input_feature, input_class)
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

    def _get_initial_class_labels(self):
        class_df = self._global_data['INITIAL_CLASS_DF']
        return class_df['MUSS_22_ACTIVITY_ABBRS'].unique().tolist()

    def _init_dashboard_model_training(self, class_labels=[]):
        heading = 'Step 1 - Train initial model'
        button_texts = ['Train model', 'Validate model']

        header = [[sg.Text(text=heading, relief=sg.RELIEF_FLAT,
                           font=('Helvetica', 12, 'bold'), size=(20, 1))]]
        description = [[sg.Text(
            text="Hold 'Ctrl' to select multiple classes\nyou would like to train for the model.", font=('Helvetica', 10))]]
        class_label_list = [[sg.Listbox(
            values=class_labels, select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, font=('Helvetica', 10), size=(27, 20), key='_TRAINING_CLASS_LABELS_', enable_events=True)]]
        buttons = [[sg.Button(button_text=text,
                              font=('Helvetica', 11), auto_size_button=True, size=(20, None), key="_" + text.upper().replace(' ', '_') + '_', disabled=True)] for text in button_texts]

        return sg.Column(layout=header + description + class_label_list + buttons, scrollable=False)

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

    def popup_model_training(self, training_labels):
        layout = [[sg.Text('Initializing training data...', key='_TRAIN_MODEL_MESSAGE_', size=(40, 1))],
                  [sg.ProgressBar(10, orientation='h', size=(
                      40, 20), key='_TRAIN_MODEL_PROGRESS_BAR_')],
                  [sg.Button(button_text='Cancel', key='_TRAIN_MODEL_CLOSE_')]]
        popup = sg.Window('Training initial model...', layout)
        progress_bar = popup['_TRAIN_MODEL_PROGRESS_BAR_']
        message_bar = popup['_TRAIN_MODEL_MESSAGE_']
        self._global_data['TRAIN_MODEL_RUNNING'] = True
        for message, progress in self.train_initial_model(training_labels):
            event, values = popup.read(timeout=500)
            if event == '_TRAIN_MODEL_CLOSE_' or event == '_TRAIN_MODEL_CLOSE_' or event is None:
                break
            progress_bar.UpdateBar(progress)
            message_bar.Update(value=message)
        self._global_data['TRAIN_MODEL_RUNNING'] = False
        if progress == 10:
            popup['_TRAIN_MODEL_CLOSE_'].Update(text='Close')
            event, values = popup.read()
        popup.close()

    def handle_dashboard(self, event, values, dashboard):
        if event == "_TRAIN_MODEL_":  # Train model
            training_labels = values['_TRAINING_CLASS_LABELS_']
            self.popup_model_training(training_labels)

        elif event == "_TRAINING_CLASS_LABELS_":  # Only enable buttons when there are two or more classes selected
            num_classes = len(values['_TRAINING_CLASS_LABELS_'])
            if num_classes > 1:
                dashboard['_TRAIN_MODEL_'].Update(disabled=False)
                dashboard['_VALIDATE_MODEL_'].Update(disabled=False)
            else:
                dashboard['_TRAIN_MODEL_'].Update(disabled=True)
                dashboard['_VALIDATE_MODEL_'].Update(disabled=True)

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
    demo = ArusDemo(title='Arus active training demo')
    demo.start()
