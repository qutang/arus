import enum
import logging
import queue
import time

import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import app_state as app
import backend
import base
import components as comp
import arus.core.libs.plotting as arus_plot


class Event(enum.Enum):
    CONNECT_DEVICES = enum.auto()
    DEVICE_CONNECTED = enum.auto()
    DISCONNECT_DEVICES = enum.auto()
    DEVICE_DISCONNECTED = enum.auto()
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


class ProcessWindow(base.BaseWindow):
    def __init__(self, title, model=None, labels=None, mode=backend.PROCESSOR_MODE.INFERENCE):
        super().__init__(title, 100)
        self._test_model = model
        self._labels = labels
        self._mode = mode
        self._scan_task = None
        self._connect_task = None
        self._disconnect_task = None
        self._start_task = None
        self._stop_task = None
        self._guide_task = None
        self._pipeline = None
        self._latest_inference = None
        self._selected_test_label = None
        self._timer = comp.Timer()

    def init_states(self):
        state = app.AppState.getInstance()
        if self._test_model is not None:
            self._labels = self._labels or self._test_model[0].classes_.tolist(
            )
        return state

    def init_views(self):
        header_row = [
            comp.heading(
                "Test model",
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
            "Start test",
            disabled=True,
            key=Event.START_TEST
        )

        self._stop_button = comp.control_button(
            "Stop test",
            disabled=True,
            key=Event.STOP_TEST
        )

        self._timer_view = comp.text('00:00:00')
        self._guidance_view = comp.text('', fixed_column_width=25)
        info_row = [
            self._timer_view, self._guidance_view
        ]

        button_row = [
            self._connect_button, self._disconnect_button, self._start_button, self._stop_button
        ]

        test_info_row = self.init_test_view()

        layout = [
            header_row,
            scan_button_row,
            device_info_row,
            button_row,
            info_row,
            test_info_row
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

        if self._pipeline is not None:
            self._latest_inference, st, et, _, _, _ = next(
                self._pipeline.get_iterator(timeout=0.02))
            if self._latest_inference is not None:
                if self._mode == backend.PROCESSOR_MODE.ACTIVE_TRAINING:
                    self._latest_inference[1]['GT_LABEL'] = self._selected_test_label
                    self._latest_inference[1]['PID'] = self._state.pid
                    if self._state.new_dataset is None:
                        self._state.new_dataset = self._latest_inference[1]
                    else:
                        self._state.new_dataset = self._state.new_dataset.append(
                            self._latest_inference[1]
                        )
                self._events.put(Event.INFERENCE_UPDATE)

        if len(values[Event.TEST_LABEL_CHANGED]) > 0:
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
            self.connect_devices()
            self.disable_connect_button()
        elif event == Event.DEVICE_CONNECTED:
            self.enable_disconnect_button()
            self.disable_connect_button(finished=True)
            self.enable_start_button()
        elif event == Event.DISCONNECT_DEVICES:
            self.disconnect_devices()
            self.disable_disconnect_button()
            self.disable_start_button()
            self._timer.stop()
            self._timer.reset()
        elif event == Event.DEVICE_DISCONNECTED:
            self.enable_connect_button()
            self.disable_disconnect_button(finished=True)
            self.disable_start_button(finished=True)
            self.disable_stop_button(finished=True)
        elif event == Event.START_TEST:
            self.start_test()
            self.disable_start_button()
            self._timer.start()
        elif event == Event.TEST_STARTED:
            self.disable_start_button(finished=True)
            self.enable_stop_button()
        elif event == Event.STOP_TEST:
            self.stop_test()
            self.disable_stop_button()
            self._timer.stop()
        elif event == Event.TEST_STOPPED:
            self.disable_stop_button(finished=True)
            self.enable_start_button()
        elif event == Event.INFERENCE_UPDATE:
            self.update_test_view()
            if self._mode == backend.PROCESSOR_MODE.ACTIVE_TRAINING:
                self.play_active_guidance()
        elif event == Event.TIMER_UPDATE:
            self.update_timer_view()
        elif event == Event.GUIDANCE_PLAYED:
            self._guidance_view.update('')

    def init_devices_view(self):
        self._info_wrist = comp.DeviceInfo(device_name='Dominant Wrist',
                                           placement_img_url='./assets/dom_wrist.png', fixed_column_width=25)
        self._info_ankle = comp.DeviceInfo(device_name='Right Ankle',
                                           placement_img_url='./assets/right_ankle.png',
                                           fixed_column_width=25)
        return [sg.Column(layout=self._info_wrist.get_component(), scrollable=False), sg.Column(layout=self._info_ankle.get_component(), scrollable=False)]

    def init_test_view(self):
        if self._mode == backend.PROCESSOR_MODE.INFERENCE:
            model_labels = self._test_model[0].classes_.tolist()
            extra_label = 'Any'
        elif self._mode == backend.PROCESSOR_MODE.ACTIVE_TRAINING:
            model_labels = self._test_model[0].classes_.tolist()
            extra_label = 'Sync'

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

    def scan_devices(self):
        self._scan_task = backend.get_nearby_devices()

    def connect_devices(self):
        self._connect_task = backend.connect_devices(
            self._state.nearby_devices, model=self._test_model, mode=self._mode, output_folder=self._state.output_folder, pid=self._state.pid)

    def start_test(self):
        self._start_task = backend.start_test_model(self._pipeline)

    def stop_test(self):
        self._stop_task = backend.stop_test_model(self._pipeline)

    def play_active_guidance(self):
        if (np.all(self._state.test_stack[-2:]) and len(self._state.test_stack) > 1) or len(self._state.test_stack) == 4:
            text = 'Switch now!'
            self._guide_task = backend.play_sound('./assets/switch_now')
            self._state.test_stack.clear()
        else:
            text = 'Keep going!'
            self._guide_task = backend.play_sound('./assets/keep_going')
        self._guidance_view.update(text)

    def disconnect_devices(self):
        self._disconnect_task = backend.disconnect_devices(self._pipeline)

    def disable_scan_button(self):
        self._scan_button.update('Scanning...', disabled=True)

    def enable_scan_button(self):
        self._scan_button.update('Scan devices', disabled=False)

    def update_device_addrs(self):
        self._info_wrist.update_addr(self._state.nearby_devices[0])
        self._info_ankle.update_addr(self._state.nearby_devices[1])

    def parse_inference_result(self, values):
        correct = values[0].split(
            ':')[0] == self._selected_test_label or self._selected_test_label == 'Any'
        return correct

    def update_test_view(self):
        values = self.format_inference_result(self._latest_inference)
        correct = self.parse_inference_result(values)
        self._state.test_stack.append(correct)
        base_color = 'g' if correct else 'r'
        new_b_colors = []
        new_t_colors = []
        for value in values:
            name = value.split(':')[0]
            num = float(value.split(':')[1])
            if not correct and name == self._selected_test_label:
                new_color = 'blue'
                text_color = 'white'
            else:
                new_color = arus_plot.adjust_lightness(
                    base_color, amount=num / 100.0, return_format='hex')
                text_color = "white" if num > 20 else "gray"
            new_b_colors.append(new_color)
            new_t_colors.append(text_color)
        self._labelgrid.update_labels(values, new_b_colors, new_t_colors)

    def update_timer_view(self):
        self._timer_view.update(
            self._timer.get_total_lapsed_time(formatted=True) + ' seconds')

    def format_inference_result(self, result):
        model_labels = self._test_model[0].classes_.tolist()
        pred_probs = result[0][0].tolist()
        formatted = []
        for label, value in zip(model_labels, pred_probs):
            formatted.append(label + ': ' + str(int(round(value, 2) * 100)))
        sort_formatted = sorted(
            formatted, key=lambda v: int(v.split(':')[1]), reverse=True)
        return sort_formatted

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
