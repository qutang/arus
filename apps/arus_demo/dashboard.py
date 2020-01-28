import enum

import PySimpleGUI as sg

import app_state as app
import backend
import base
import components as comp
import task_window
import stream_window
from arus.models.muss import Strategy


class Event(enum.Enum):
    TEST_ORIGIN_MODEL = enum.auto(),
    ORIGIN_LABELS_CHANGED = enum.auto(),
    NOT_ENOUGH_ORIGIN_LABELS = enum.auto(),
    ENOUGH_ORIGIN_LABELS = enum.auto(),
    TRAIN_ORIGIN_MODEL = enum.auto(),
    ORIGIN_MODEL_READY = enum.auto(),
    VALIDATE_ORIGIN_MODEL = enum.auto(),
    NEW_ACTIVITY_NAME_CHANGED = enum.auto(),
    ADD_NEW_ACTIVITY = enum.auto(),
    DATA_COLLECTION_LABELS_CHANGED = enum.auto(),
    NOT_ENOUGH_DATA_COLLECTION_LABELS = enum.auto(),
    ENOUGH_DATA_COLLECTION_LABELS = enum.auto(),
    COLLECT_NEW_DATA = enum.auto(),
    NEW_DATASET_READY = enum.auto(),
    CHECK_NEW_DATA = enum.auto(),
    NEW_LABELS_CHANGED = enum.auto(),
    NOT_ENOUGH_NEW_LABELS = enum.auto(),
    ENOUGH_NEW_LABELS = enum.auto(),
    TRAIN_NEW_MODEL = enum.auto(),
    NEW_MODEL_READY = enum.auto(),
    TEST_NEW_MODEL = enum.auto(),
    VALIDATE_NEW_MODEL = enum.auto(),
    ORIGIN_VALIDATE_RESULT_READY = enum.auto(),
    NEW_VALIDATE_RESULT_READY = enum.auto()


class Dashboard(base.BaseWindow):
    def __init__(self, title):
        super().__init__(title, 25)

    def init_states(self):
        state = app.AppState.getInstance()
        origin_label_candidates = backend.get_class_label_candidates(
            state.origin_dataset)
        if len(state.data_collection_label_candidates) == 0:
            state.data_collection_label_candidates = origin_label_candidates
        if len(state.origin_label_candidates) == 0:
            state.origin_label_candidates = origin_label_candidates
        return state

    def init_views(self):
        self._window_train_origin = None
        self._window_data_collection = None
        self._window_train_new = None
        self._window_validate_origin = None
        self._window_validate_new = None
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
        return dashboard

    def _update_states_and_events(self, event, values):
        if self._state.origin_labels != values[Event.ORIGIN_LABELS_CHANGED]:
            self._state.origin_labels = values[Event.ORIGIN_LABELS_CHANGED][:]
            if len(self._state.origin_labels) >= 2:
                self._events.put(Event.ENOUGH_ORIGIN_LABELS)
            else:
                self._events.put(Event.NOT_ENOUGH_ORIGIN_LABELS)

        if self._state.new_labels != values[Event.NEW_LABELS_CHANGED]:
            self._state.new_labels = values[Event.NEW_LABELS_CHANGED][:]
            if len(self._state.new_labels) >= 1:
                self._events.put(Event.ENOUGH_NEW_LABELS)
            else:
                self._events.put(Event.NOT_ENOUGH_NEW_LABELS)

        if self._state.data_collection_labels != values[Event.DATA_COLLECTION_LABELS_CHANGED]:
            self._state.data_collection_labels = values[
                Event.DATA_COLLECTION_LABELS_CHANGED][:]
            if len(self._state.data_collection_labels) >= 1:
                self._events.put(Event.ENOUGH_DATA_COLLECTION_LABELS)
            else:
                self._events.put(Event.NOT_ENOUGH_DATA_COLLECTION_LABELS)

        if self._state.new_activity_name != values[Event.NEW_ACTIVITY_NAME_CHANGED]:
            self._state.new_activity_name = values[Event.NEW_ACTIVITY_NAME_CHANGED]

        if event == Event.ADD_NEW_ACTIVITY:
            if self._state.new_activity_name not in self._state.data_collection_label_candidates:
                self._state.data_collection_label_candidates = [
                    self._state.new_activity_name] + self._state.data_collection_label_candidates

        if self._window_train_origin is not None:
            origin_model = self._window_train_origin.get_result()
            if origin_model is not None:
                self._state.origin_model = origin_model
                self._window_train_origin = None
                self._events.put(Event.ORIGIN_MODEL_READY)

        if self._window_data_collection is not None:
            new_dataset = self._window_data_collection.get_result()
            if new_dataset is not None:
                self._state.new_dataset = new_dataset
                self._window_data_collection = None
                self._events.put(Event.NEW_DATASET_READY)

        if self._window_train_new is not None:
            new_model = self._window_train_new.get_result()
            if new_model is not None:
                self._state.new_model = new_model
                self._window_train_new = None
                self._events.put(Event.NEW_MODEL_READY)

        if self._window_validate_origin is not None:
            origin_validate_results = self._window_validate_origin.get_result()
            if origin_validate_results is not None:
                self._state.origin_validate_results = origin_validate_results
                self._window_validate_origin = None
                self._events.put(Event.ORIGIN_VALIDATE_RESULT_READY)

        if self._window_validate_new is not None:
            new_validate_results = self._window_validate_new.get_result()
            if new_validate_results is not None:
                self._state.new_validate_results = new_validate_results
                self._window_validate_new = None
                self._events.put(Event.NEW_VALIDATE_RESULT_READY)

    def _dispatch_events(self, event):
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
            self.open_task_window(event)
        elif event == Event.ORIGIN_MODEL_READY:
            self.enable_test_origin_model()
            self.display_model_summary(event)
        elif event == Event.VALIDATE_ORIGIN_MODEL:
            self.open_task_window(event)
        elif event == Event.TEST_ORIGIN_MODEL:
            self.open_stream_window(event)
        elif event == Event.COLLECT_NEW_DATA:
            self.open_stream_window(event)
        elif event == Event.NEW_DATASET_READY:
            self.enable_check_data_button()
            self.display_dataset_summary(event)
            self.update_new_label_list()
        elif event == Event.ADD_NEW_ACTIVITY:
            self.update_data_collection_label_list()
            self.reset_new_activity_input()
        elif event == Event.TRAIN_NEW_MODEL:
            self.open_task_window(event)
        elif event == Event.VALIDATE_NEW_MODEL:
            self.open_task_window(event)
        elif event == Event.NEW_MODEL_READY:
            self.enable_test_new_model()
            self.display_model_summary(event)
        elif event == Event.TEST_NEW_MODEL:
            self.open_stream_window(event)

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
            disabled=len(self._state.origin_labels) == 0,
            key=Event.TRAIN_ORIGIN_MODEL
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
            disabled=self._state.origin_model is None,
            key=Event.TEST_ORIGIN_MODEL
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
            disabled=self._state.data_collection_labels is None
        )
        collect_button_row = [
            self._collect_button_new_data
        ]

        self._check_button_new_data = comp.control_button(
            text=button_texts[1],
            key=Event.CHECK_NEW_DATA,
            fixed_column_width=self._column_width,
            disabled=self._state.new_dataset is None
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

        self._new_label_list = comp.selection_list(
            items=new_label_candidates,
            default_selections=self._state.new_labels,
            fixed_column_width=self._column_width,
            mode='multiple',
            key=Event.NEW_LABELS_CHANGED,
            rows=20
        )
        label_selection_row = [
            self._new_label_list
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
            self._test_button_new_model
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

    def open_task_window(self, event):
        if event == Event.TRAIN_ORIGIN_MODEL:
            self._window_train_origin = task_window.TaskWindow(
                "Train origin model")
            self._window_train_origin.start()
        elif event == Event.TRAIN_NEW_MODEL:
            train_strategies = [
                Strategy.REPLACE_ORIGIN,
                Strategy.COMBINE_ORIGIN,
                Strategy.USE_NEW_ONLY
            ]
            self._window_train_new = task_window.TaskWindow(
                'Train new model',
                strategies=train_strategies
            )
            self._window_train_new.start()
        elif event == Event.VALIDATE_ORIGIN_MODEL:
            self._window_validate_origin = task_window.TaskWindow(
                "Validate origin model", mode=task_window.Mode.LOSO)
            self._window_validate_origin.start()
        elif event == Event.VALIDATE_NEW_MODEL:
            strategies = [
                Strategy.REPLACE_ORIGIN,
                Strategy.COMBINE_ORIGIN
            ]
            self._window_validate_new = task_window.TaskWindow(
                "Validate new model", strategies=strategies, mode=task_window.Mode.LOSO)
            self._window_validate_new.start()

    def display_model_summary(self, event):
        if event == Event.ORIGIN_MODEL_READY:
            summary = backend.get_model_summary(self._state.origin_model)
            self._text_origin_model.update(summary)
        elif event == Event.NEW_MODEL_READY:
            summary = backend.get_model_summary(self._state.new_model)
            self._text_new_model.update(summary)

    def display_dataset_summary(self, event):
        if event == Event.NEW_DATASET_READY:
            summary = backend.get_dataset_summary(self._state.new_dataset)
            self._text_new_data.update(summary)

    def open_stream_window(self, event):
        if event == Event.TEST_ORIGIN_MODEL:
            panel = stream_window.StreamWindow(
                "Test origin model",
                model=self._state.origin_model,
                support_modes=[
                    backend.PROCESSOR_MODE.TEST_ONLY,
                    backend.PROCESSOR_MODE.TEST_AND_SAVE
                ]
            )
            panel.start()
        elif event == Event.TEST_NEW_MODEL:
            panel = stream_window.StreamWindow(
                "Test new model",
                model=self._state.new_model,
                support_modes=[
                    backend.PROCESSOR_MODE.TEST_ONLY,
                    backend.PROCESSOR_MODE.TEST_AND_SAVE
                ]
            )
            panel.start()
        elif event == Event.COLLECT_NEW_DATA:
            self._window_data_collection = stream_window.StreamWindow(
                'Collect new data',
                model=self._state.origin_model,
                labels=self._state.data_collection_labels,
                support_modes=[
                    backend.PROCESSOR_MODE.TEST_AND_COLLECT,
                    backend.PROCESSOR_MODE.COLLECT_ONLY,
                    backend.PROCESSOR_MODE.ACTIVE_COLLECT
                ]
            )
            self._window_data_collection.start()

    def enable_train_origin_model(self):
        self._train_button_origin_model.update(disabled=False)
        self._validate_button_origin_model.update(disabled=False)

    def enable_train_new_model(self):
        self._train_button_new_model.update(disabled=False)
        self._validate_button_new_model.update(disabled=False)

    def enable_test_origin_model(self):
        self._test_button_origin_model.update(disabled=False)

    def enable_test_new_model(self):
        self._test_button_new_model.update(disabled=False)

    def enable_collect_data(self):
        self._collect_button_new_data.update(disabled=False)

    def disable_train_origin_model(self):
        self._train_button_origin_model.update(disabled=True)
        self._validate_button_origin_model.update(disabled=True)

    def disable_train_new_model(self):
        self._train_button_new_model.update(disabled=True)
        self._validate_button_new_model.update(disabled=True)

    def disable_test_origin_model(self):
        self._test_button_origin_model.update(disabled=False)

    def disable_test_new_model(self):
        self._test_button_new_model.update(disabled=False)

    def disable_collect_data(self):
        self._collect_button_new_data.update(disabled=True)

    def disable_check_data_button(self):
        self._check_button_new_data.update(disabled=True)

    def enable_check_data_button(self):
        self._check_button_new_data.update(disabled=False)

    def update_data_collection_label_list(self):
        self._data_collection_label_list.update(
            self._state.data_collection_label_candidates)

    def reset_new_activity_input(self):
        self._input_new_activity.update_input(value="")

    def update_new_label_list(self):
        new_label_candidates = self._state.new_dataset['GT_LABEL'].unique(
        ).tolist()
        self._new_label_list.update(new_label_candidates)
