import enum
from loguru import logger
import queue
import time
import datetime as dt
import copy

import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

import app_state as app
import backend
import base
import components as comp
import arus


class Event(enum.Enum):
    CONNECT_DEVICES = enum.auto()
    DEVICE_CONNECTED = enum.auto()
    DISCONNECT_DEVICES = enum.auto()
    DEVICE_DISCONNECTED = enum.auto()
    MODE_CHANGED = enum.auto()
    SCAN_DEVICES = enum.auto()
    SCAN_COMPLETED = enum.auto()
    START_TEST = enum.auto()
    TEST_STARTED = enum.auto()
    STOP_TEST = enum.auto()
    TEST_STOPPED = enum.auto()
    INFERENCE_UPDATE = enum.auto()
    TEST_LABEL_CHANGED = enum.auto()
    TIMER_UPDATE = enum.auto()
    GUIDANCE_PLAYED = enum.auto()
    SAVE_ANNOTATION = enum.auto()
    ANNOTATION_SAVED = enum.auto()


class StreamWindow(base.BaseWindow):
    def __init__(
        self,
        title,
        model=None,
        labels=None,
        support_modes=[
            backend.PROCESSOR_MODE.TEST_ONLY,
            backend.PROCESSOR_MODE.TEST_AND_SAVE
        ]
    ):
        super().__init__(title, 100)
        self._test_model = model
        self._labels = labels
        self._support_modes = support_modes
        self._selected_mode = None
        self._selected_placements = None
        self._selected_devices = ['', '', '']
        self._selected_placement_addrs = None

        self._scan_task = None
        self._connect_task = None
        self._disconnect_task = None
        self._start_task = None
        self._stop_task = None
        self._guide_task = None
        self._save_task = None

        self._pipeline = None
        self._latest_inference = None
        self._inference_queue = []

        self._result = None

        self._selected_test_label = None
        self._timer = comp.Timer()

        self._current_annotation = {
            "HEADER_TIME_STAMP": None, "START_TIME": None, "STOP_TIME": None, "LABEL_NAME": None
        }
        self._annotation_save_queue = queue.Queue()

        self._current_guidance = {
            "HEADER_TIME_STAMP": None, "START_TIME": None, "STOP_TIME": None, "LABEL_NAME": None
        }

    def get_result(self):
        return self._result

    def init_states(self):
        state = app.AppState.getInstance()
        if self._test_model is not None:
            self._labels = self._labels or self._test_model[0].classes_.tolist(
            )
            self._selected_mode = self._support_modes[0]
            self._selected_placements = self._test_model[-2][:]
        else:
            self._selected_mode = backend.PROCESSOR_MODE.COLLECT_ONLY
            self._selected_placements = []
        if self._selected_mode in [
                backend.PROCESSOR_MODE.COLLECT_ONLY,
                backend.PROCESSOR_MODE.TEST_AND_COLLECT,
                backend.PROCESSOR_MODE.ACTIVE_COLLECT]:
            self._result = state.new_dataset
        else:
            self._result = None
        return state

    def init_views(self):
        header_row = [
            comp.heading(
                self._title,
                fixed_column_width=self._column_width
            )
        ]

        self._scan_button = comp.control_button(
            "Scan devices",
            disabled=False,
            key=Event.SCAN_DEVICES
        )
        scan_button_row = [
            self._scan_button
        ]

        device_info_row = self.init_devices_view()

        self._connect_button = comp.control_button(
            "Connect devices",
            disabled=True,
            key=Event.CONNECT_DEVICES
        )

        self._disconnect_button = comp.control_button(
            "Disconnect devices",
            disabled=True,
            key=Event.DISCONNECT_DEVICES
        )

        self._start_button = comp.control_button(
            "Start " + self._title,
            disabled=True,
            key=Event.START_TEST
        )

        self._stop_button = comp.control_button(
            "Stop " + self._title,
            disabled=True,
            key=Event.STOP_TEST
        )

        control_mode_row = self.init_mode_controls()

        self._timer_view = comp.text('00:00:00')
        self._guidance_view = comp.text('', fixed_column_width=25)
        info_row = [
            self._timer_view, self._guidance_view
        ]

        button_row = [
            self._connect_button, self._disconnect_button, self._start_button, self._stop_button
        ]

        test_info_row = self.init_test_view()

        data_summary_row = self.init_data_summary_view()

        layout = [
            header_row,
            scan_button_row,
            device_info_row,
            button_row,
            control_mode_row,
            info_row,
            test_info_row,
            data_summary_row
        ]

        return sg.Window(
            self._title,
            layout=layout,
            finalize=True
        )

    def _update_states_and_events(self, event, values):
        if self._scan_task is not None:
            if self._scan_task.ready():
                self._state.nearby_devices = sorted(self._scan_task.get())
                self._selected_devices = self._state.nearby_devices[:3]
                self._events.put(Event.SCAN_COMPLETED)
                self._scan_task = None

        if self._connect_task is not None:
            if self._connect_task.ready():
                self._pipeline = self._connect_task.get()
                self._events.put(Event.DEVICE_CONNECTED)
                self._connect_task = None

        if self._disconnect_task is not None:
            if self._disconnect_task.ready():
                self._events.put(Event.DEVICE_DISCONNECTED)
                self._disconnect_task = None

        if self._start_task is not None:
            if self._start_task.ready():
                self._events.put(Event.TEST_STARTED)
                self._start_task = None

        if self._stop_task is not None:
            if self._stop_task.ready():
                self._events.put(Event.TEST_STOPPED)
                self._stop_task = None

        if self._guide_task is not None:
            if self._guide_task.ready():
                self._events.put(Event.GUIDANCE_PLAYED)
                self._guide_task = None

        if self._save_task is not None:
            if self._save_task.done():
                self._events.put(Event.ANNOTATION_SAVED)
                self._save_task = None

        for mode in self._support_modes:
            if values[mode]:
                if mode != self._selected_mode:
                    if mode != backend.PROCESSOR_MODE.COLLECT_ONLY:
                        self._selected_placements = self._test_model[-2][:]

                    self._events.put(Event.MODE_CHANGED)
                    self._selected_mode = mode
                    break

        if event == 'DW_ADDR':
            self._selected_devices[0] = values[event]
        if event == 'DA_ADDR':
            self._selected_devices[1] = values[event]
        if event == 'DT_ADDR':
            self._selected_devices[2] = values[event]

        if event in ['DW', 'DA', 'DT']:
            if values[event]:
                self._selected_placements.append(event)
            else:
                self._selected_placements.remove(event)
            logger.info('Selected placements: ' +
                        str(self._selected_placements))

        if self._state.nearby_devices is not None and len(self._selected_devices) >= 3:
            self._selected_placement_addrs = [
                self._selected_devices[
                    ['DW', 'DA', 'DT'].index(placement)
                ] for placement in self._selected_placements
            ]

        if self._pipeline is not None:
            self._latest_inference, st, et, _, _, _ = next(
                self._pipeline.get_iterator(timeout=0.02))
            if self._latest_inference is not None:
                logger.info(self._latest_inference)
                self.format_inference_result()
                self.add_to_result()
                self._events.put(Event.INFERENCE_UPDATE)

        if event == Event.START_TEST:
            self._timer.start()
            if self._selected_mode != backend.PROCESSOR_MODE.TEST_ONLY:
                self._annotation_save_queue.queue.clear()
                self.start_new_annotation()

        if event == Event.STOP_TEST:
            self._timer.stop()
            if self._selected_mode != backend.PROCESSOR_MODE.TEST_ONLY:
                self.complete_new_annotation()
                self._events.put(Event.SAVE_ANNOTATION)

        if event == Event.DISCONNECT_DEVICES:
            self._timer.stop()
            self._timer.reset()

        if len(values[Event.TEST_LABEL_CHANGED]) > 0:
            if self._selected_test_label != values[Event.TEST_LABEL_CHANGED][0] and self._selected_mode != backend.PROCESSOR_MODE.TEST_ONLY and self._timer.is_running():
                self.complete_new_annotation()
                self._events.put(Event.SAVE_ANNOTATION)
                self._selected_test_label = values[Event.TEST_LABEL_CHANGED][0]
                self.start_new_annotation()
            elif self._selected_test_label != values[Event.TEST_LABEL_CHANGED][0]:
                self._selected_test_label = values[Event.TEST_LABEL_CHANGED][0]

        if self._timer.is_running() and self._timer.since_last_tick() >= 1:
            self._events.put(Event.TIMER_UPDATE)
            self._timer.tick()

    def _dispatch_events(self, event):
        if event == Event.SCAN_DEVICES:
            self.scan_devices()
            self.disable_scan_button()
        elif event == Event.SCAN_COMPLETED:
            self.update_device_addrs()
            self.enable_scan_button()
            self.enable_connect_button()
        elif event == Event.CONNECT_DEVICES:
            if len(set(self._selected_devices)) != len(self._selected_devices):
                sg.Popup(
                    'Please make sure the selected device addresses are all different', title='Warning')
            else:
                self.connect_devices()
                self.disable_connect_button()
                self.disable_device_selections()
        elif event == Event.DEVICE_CONNECTED:
            self.enable_disconnect_button()
            self.disable_connect_button(finished=True)
            self.enable_start_button()
            self._control_modes.disable_all()
        elif event == Event.DISCONNECT_DEVICES:
            self.disconnect_devices()
            self.disable_disconnect_button()
            self.disable_start_button()
        elif event == Event.DEVICE_DISCONNECTED:
            self.enable_connect_button()
            self.enable_device_selections()
            self.disable_disconnect_button(finished=True)
            self.disable_start_button(finished=True)
            self.disable_stop_button(finished=True)
            self._control_modes.enable_all()
        elif event == Event.START_TEST:
            self.start_test()
            self.disable_start_button()
            self.disable_disconnect_button()
        elif event == Event.TEST_STARTED:
            self.disable_start_button(finished=True)
            self.enable_stop_button()
        elif event == Event.STOP_TEST:
            self.stop_test()
            self.disable_stop_button()
        elif event == Event.TEST_STOPPED:
            self.disable_stop_button(finished=True)
            self.enable_start_button()
            self.enable_disconnect_button()
        elif event == Event.INFERENCE_UPDATE:
            if self._selected_mode != backend.PROCESSOR_MODE.COLLECT_ONLY:
                self.update_test_view()
            if self._selected_mode == backend.PROCESSOR_MODE.ACTIVE_COLLECT:
                self.play_active_guidance()
            self.update_data_summary_view()
        elif event == Event.TIMER_UPDATE:
            self.update_timer_view()
        elif event == Event.GUIDANCE_PLAYED:
            self._guidance_view.update('')
        elif event == Event.SAVE_ANNOTATION:
            self.save_annotation()
        elif event == Event.ANNOTATION_SAVED:
            logger.info('Annotation is saved.')
        elif event == Event.MODE_CHANGED:
            if self._selected_mode == backend.PROCESSOR_MODE.COLLECT_ONLY:
                self.enable_placement_selections()
            else:
                self.update_placement_selections()
                self.disable_placement_selections()

    def init_devices_view(self):
        if 'DW' in self._selected_placements and self._selected_mode != backend.PROCESSOR_MODE.COLLECT_ONLY:
            dw_selected = True
            dw_disabled = True
        elif self._selected_mode == backend.PROCESSOR_MODE.COLLECT_ONLY:
            dw_selected = False
            dw_disabled = False
        else:
            dw_selected = False
            dw_disabled = True
        self._info_wrist = comp.DeviceInfo(device_name='Dominant Wrist',
                                           placement_img_url='./assets/dom_wrist.png', fixed_column_width=30, device_name_key="DW",
                                           device_addr_key="DW_ADDR",
                                           device_selected=dw_selected,
                                           device_selection_disabled=dw_disabled)
        if 'DA' in self._selected_placements and self._selected_mode != backend.PROCESSOR_MODE.COLLECT_ONLY:
            da_selected = True
            da_disabled = True
        elif self._selected_mode == backend.PROCESSOR_MODE.COLLECT_ONLY:
            da_selected = False
            da_disabled = False
        else:
            da_selected = False
            da_disabled = True
        self._info_ankle = comp.DeviceInfo(device_name='Right Ankle',
                                           placement_img_url='./assets/right_ankle.png',
                                           fixed_column_width=30, device_name_key="DA",
                                           device_addr_key="DA_ADDR",
                                           device_selected=da_selected, device_selection_disabled=da_disabled)
        if 'DT' in self._selected_placements and self._selected_mode != backend.PROCESSOR_MODE.COLLECT_ONLY:
            dt_selected = True
            dt_disabled = True
        elif self._selected_mode == backend.PROCESSOR_MODE.COLLECT_ONLY:
            dt_selected = False
            dt_disabled = False
        else:
            dt_selected = False
            dt_disabled = True
        self._info_thigh = comp.DeviceInfo(device_name='Dominant Thigh',
                                           placement_img_url='./assets/dom_thigh.png',
                                           fixed_column_width=30, device_name_key="DT",
                                           device_addr_key='DT_ADDR',
                                           device_selected=dt_selected, device_selection_disabled=dt_disabled)
        return [sg.Column(layout=self._info_wrist.get_component(), scrollable=False), sg.Column(layout=self._info_ankle.get_component(), scrollable=False), sg.Column(layout=self._info_thigh.get_component(), scrollable=False)]

    def init_test_view(self):
        if self._selected_mode in [
            backend.PROCESSOR_MODE.TEST_AND_SAVE,
            backend.PROCESSOR_MODE.TEST_ONLY
        ]:
            extra_label = 'Any'
        elif self._selected_mode in [
            backend.PROCESSOR_MODE.ACTIVE_COLLECT,
            backend.PROCESSOR_MODE.COLLECT_ONLY,
            backend.PROCESSOR_MODE.TEST_AND_COLLECT
        ]:
            extra_label = 'Sync'
        if self._test_model is not None:
            model_labels = self._test_model[0].classes_.tolist()
        else:
            model_labels = self._labels
        n = len(model_labels)

        self._test_label_list = comp.selection_list(
            [extra_label] + self._labels,
            default_selections=extra_label,
            mode='single',
            fixed_column_width=int(self._column_width * 1 / 6),
            rows=8,
            key=Event.TEST_LABEL_CHANGED
        )

        self._labelgrid = comp.LabelGrid(
            n,
            fixed_column_width=int(self._column_width * 5 / 6),
            default_background_color='green',
            default_text_color='white',
            n_cols=4
        )
        self._labelgrid.add_labels(
            model_labels, keys=[None] * n, background_colors=[None] * n, text_colors=[None] * n)
        return [
            self._test_label_list,
            sg.Column(
                layout=self._labelgrid.get_component(),
                scrollable=False
            )
        ]

    def init_data_summary_view(self):
        cells = backend.get_data_summary_table(
            self._result)
        cells = cells or [["", "", ""]]
        self._data_summary = comp.table(
            cells, headings=["Activity", "# of windows", "Accuracy"], fixed_column_width=self._column_width)
        return [self._data_summary]

    def init_mode_controls(self):
        mode_names = [mode.name for mode in self._support_modes]
        disable_states = [False] * len(self._support_modes)
        default_index = 0
        if self._test_model is None:
            i = self._support_modes.index(
                backend.PROCESSOR_MODE.ACTIVE_COLLECT
            )
            if default_index == i:
                default_index += 1
            disable_states[i] = True
            i = self._support_modes.index(
                backend.PROCESSOR_MODE.TEST_AND_COLLECT
            )
            if default_index == i:
                default_index += 1
            disable_states[i] = True

        self._control_modes = comp.RadioGroup(
            'PROCESS_MODE',
            len(self._support_modes),
            default_index=default_index,
            fixed_column_width=self._column_width
        )
        self._control_modes.add_radioboxes(
            mode_names,
            keys=self._support_modes,
            disable_states=disable_states
        )
        return self._control_modes.get_component()

    def scan_devices(self):
        self._scan_task = backend.get_nearby_devices()

    def connect_devices(self):
        self._connect_task = backend.connect_devices(
            self._selected_placement_addrs, model=self._test_model, mode=self._selected_mode, output_folder=self._state.output_folder, pid=self._state.pid, placement_names=self._selected_placements)

    def start_test(self):
        self._start_task = backend.start_test_model(self._pipeline)

    def stop_test(self):
        self._stop_task = backend.stop_test_model(self._pipeline)

    def play_active_guidance(self):
        if (np.all(self._inference_queue[-2:]) and len(self._inference_queue) > 1) or len(self._inference_queue) >= 4:
            text = 'Switch now!'
            self._guide_task = backend.play_sound('./assets/switch_now')
            self._inference_queue.clear()
        else:
            text = 'Keep going!'
            self._guide_task = backend.play_sound('./assets/keep_going')
        self._guidance_view.update(text)

    def disconnect_devices(self):
        self._disconnect_task = backend.disconnect_devices(self._pipeline)

    def save_annotation(self):
        annotation = self._annotation_save_queue.get()
        session_name = None
        if self._selected_mode == backend.PROCESSOR_MODE.COLLECT_ONLY:
            session_name = 'PassiveSensing'
        elif self._selected_mode == backend.PROCESSOR_MODE.TEST_AND_COLLECT:
            session_name = 'PassiveSensing'
        elif self._selected_mode == backend.PROCESSOR_MODE.ACTIVE_COLLECT:
            session_name = 'ActiveTraining'
        elif self._selected_mode == backend.PROCESSOR_MODE.TEST_AND_SAVE:
            if self._test_model == self._state.origin_model:
                session_name = 'TestOriginModel'
            elif self._test_model == self._state.new_model:
                session_name = 'TestNewModel'
        self._save_task = backend.save_annotation(
            annotation,
            output_folder=self._state.output_folder,
            pid=self._state.pid,
            session_name=session_name
        )
        logger.info('Saving annotation...')

    def disable_scan_button(self):
        self._scan_button.update('Scanning...', disabled=True)

    def enable_scan_button(self):
        self._scan_button.update('Scan devices', disabled=False)

    def update_device_addrs(self):
        self._info_wrist.update_addr_list(self._state.nearby_devices, index=0)
        self._info_ankle.update_addr_list(self._state.nearby_devices, index=1)
        self._info_thigh.update_addr_list(self._state.nearby_devices, index=2)

    def add_to_result(self):
        if self._result is None:
            self._result = self._latest_inference[1]
        else:
            self._result = pd.concat([self._result,
                                      self._latest_inference[1]], sort=False
                                     )
        self._result.reset_index(drop=True)

    def parse_inference_result(self):
        correct = self._latest_inference[0][0].split(
            ':')[0] == self._selected_test_label or self._selected_test_label == 'Any'
        return correct

    def update_test_view(self):
        correct = self.parse_inference_result()
        self._inference_queue.append(correct)
        base_color = 'g' if correct else 'r'
        new_b_colors = []
        new_t_colors = []
        for value in self._latest_inference[0]:
            name = value.split(':')[0]
            num = float(value.split(':')[1])
            if not correct and name == self._selected_test_label:
                new_color = 'blue'
                text_color = 'white'
            else:
                new_color = arus.ext.plotting.adjust_lightness(
                    base_color, amount=num / 100.0, return_format='hex')
                text_color = "white" if num > 20 else "gray"
            new_b_colors.append(new_color)
            new_t_colors.append(text_color)
        self._labelgrid.update_labels(
            self._latest_inference[0], new_b_colors, new_t_colors)

    def update_data_summary_view(self):
        cells = backend.get_data_summary_table(
            self._result)
        cells = cells or [["", "", ""]]
        self._data_summary.update(cells)

    def update_timer_view(self):
        self._timer_view.update(
            self._timer.get_total_lapsed_time(formatted=True) + ' seconds')

    def format_inference_result(self):
        self._latest_inference = list(self._latest_inference)
        # sort probs
        if self._selected_mode != backend.PROCESSOR_MODE.COLLECT_ONLY:
            model_labels = self._test_model[0].classes_.tolist()
            pred_probs = self._latest_inference[0][0].tolist()
            formatted = []
            for label, value in zip(model_labels, pred_probs):
                formatted.append(
                    label + ': ' + str(int(round(value, 2) * 100)))
            sort_formatted = list(sorted(
                formatted, key=lambda v: int(v.split(':')[1]), reverse=True))
            self._latest_inference[0] = sort_formatted
            # add labels to feature dataframe
            self._latest_inference[1]['GT_LABEL'] = self._selected_test_label
            self._latest_inference[1]['PREDICTION'] = self._latest_inference[0][0].split(":")[
                0]
            self._latest_inference[1]['PID'] = self._state.pid
        else:
            self._latest_inference[1]['GT_LABEL'] = self._selected_test_label
            self._latest_inference[1]['PID'] = self._state.pid

    def disable_connect_button(self, finished=False):
        if finished:
            text = 'Connected'
        else:
            text = 'Connecting...'
        self._connect_button.update(text, disabled=True)

    def disable_disconnect_button(self, finished=False):
        if finished:
            text = 'Disconnected'
        else:
            text = 'Disconnecting...'
        self._disconnect_button.update(text, disabled=True)

    def enable_connect_button(self):
        self._connect_button.update('Connect devices', disabled=False)

    def enable_disconnect_button(self):
        self._disconnect_button.update('Disconnect', disabled=False)

    def enable_start_button(self):
        self._start_button.update('Start test', disabled=False)

    def disable_start_button(self, finished=False):
        if finished:
            text = 'Started'
        else:
            text = 'Starting...'
        self._start_button.update(text, disabled=True)

    def enable_stop_button(self):
        self._stop_button.update('Stop test', disabled=False)

    def disable_stop_button(self, finished=False):
        if finished:
            text = 'Stopped'
        else:
            text = 'Stopping...'
        self._stop_button.update(text, disabled=True)

    def start_new_annotation(self):
        start_time = dt.datetime.now()
        self._current_annotation['LABEL_NAME'] = [self._selected_test_label]
        self._current_annotation['START_TIME'] = [start_time]
        self._current_annotation['HEADER_TIME_STAMP'] = [start_time]
        self._current_annotation['STOP_TIME'] = None

    def complete_new_annotation(self):
        stop_time = dt.datetime.now()
        self._current_annotation['STOP_TIME'] = [stop_time]
        self._annotation_save_queue.put(
            copy.deepcopy(self._current_annotation))

    def is_annotation_incomplete(self):
        return self._current_annotation['STOP_TIME'] is None

    def enable_placement_selections(self):
        self._info_wrist.enable_selection()
        self._info_ankle.enable_selection()
        self._info_thigh.enable_selection()

    def disable_placement_selections(self):
        self._info_wrist.disable_selection()
        self._info_ankle.disable_selection()
        self._info_thigh.disable_selection()

    def update_placement_selections(self):
        selected = 'DW' in self._selected_placements
        self._info_wrist.update_selection(selected)
        selected = 'DA' in self._selected_placements
        self._info_ankle.update_selection(selected)
        selected = 'DT' in self._selected_placements
        self._info_thigh.update_selection(selected)

    def disable_device_selections(self):
        self._info_wrist.disable_addr_list()
        self._info_ankle.disable_addr_list()
        self._info_thigh.disable_addr_list()

    def enable_device_selections(self):
        self._info_wrist.enable_addr_list()
        self._info_ankle.enable_addr_list()
        self._info_thigh.enable_addr_list()
