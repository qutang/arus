import PySimpleGUI as sg
import app_state as app
import backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import datetime as dt
import arus.core.libs.plotting as arus_plot


def panel_heading(text):
    return sg.Text(text=text, relief=sg.RELIEF_FLAT,
                   font=('Helvetica', 12, 'bold'), size=(20, 1))


def panel_control_button(text, disabled=False, key=None):
    return sg.Button(button_text=text,
                     font=('Helvetica', 11), auto_size_button=True, size=(20, None), key=key, disabled=disabled)


def panel_table(values=[["", "", "", ""]], headings=["class", "precision", "recall", "f1-score"]):
    return sg.Table(values=values, headings=headings, size=(50, None), def_col_width=12, auto_size_columns=False, key='_REPORT_', visible=False)


def panel_text(text, key=None):
    return sg.Text(
        text=text, font=('Helvetica', 10), size=(25, None), key=key)


def panel_description(text, key=None):
    return sg.Text(
        text=text, font=('Helvetica', 10), size=(50, None), key=key)


def panel_image(img_url):
    return sg.Image(img_url, size=(None, None))


def panel_device(name, img_url=None, addr_key=None):
    device_addr_text = panel_text("", key=addr_key)
    device_panel = sg.Column([
        [panel_text(name)],
        [panel_image(img_url)],
        [device_addr_text]
    ])
    return device_panel, device_addr_text


def panel_listbox(labels, key=None):
    return sg.Listbox(values=list(labels) + ['Any'], default_values=['Any'], select_mode=sg.LISTBOX_SELECT_MODE_BROWSE,
                      enable_events=True, auto_size_text=True, size=(None, min(10, len(labels) + 1)), key=key)


def panel_label_grid(labels):
    label_grid = []
    n_col = 4
    n = 1
    label_row = []
    label_grid_elements = []
    for label in labels:
        el = sg.Text(label, font=('Helvetica', 13), size=(
            15, None), background_color='green', text_color='gray')
        label_row.append(el)
        label_grid_elements.append(el)
        if n % (n_col - 1) == 0:
            label_grid.append(label_row[:])
            label_row = []
        n = n + 1
    if len(label_row) > 0:
        label_grid.append(label_row[:])
    return sg.Column(layout=label_grid), label_grid_elements


class UpdatedModelTestingPanel:
    def __init__(self, title):
        self._title = title
        self._scan_button = None
        self._connect_button = None
        self._wrist_addr_text = None
        self._window = None
        self._pipeline = None
        self._app_state = app.AppState.getInstance()
        self._test_label = 'Any'

    def init_panel(self):
        training_labels = self._app_state.updated_model[0].classes_
        heading = "Model testing"
        scan_button_text = 'Scan nearby metawears'
        scan_button_key = '_SCAN_'
        wrist_panel_text = 'Dom. Wrist'
        wrist_panel_img = './dom_wrist.png'
        wrist_panel_addr_key = '_WRIST_ADDR_'

        ankle_panel_text = 'Right Ankle'
        ankle_panel_img = './right_ankle.png'
        ankle_panel_addr_key = '_ANKLE_ADDR_'

        connect_button_text = 'Connect to devices'
        connect_button_key = '_CONNECT_'

        process_button_text = 'Start testing'
        process_button_key = '_START_'

        test_description_text = "Please put on the corresponding devices on body at places shown by the pictures. Make sure the sensor ID matches the placement you put on."

        label_listbox_key = '_LABELS_'

        self._scan_button = panel_control_button(
            scan_button_text, key=scan_button_key)

        wrist_panel, self._wrist_addr_text = panel_device(
            wrist_panel_text, img_url=wrist_panel_img, addr_key=wrist_panel_addr_key)

        ankle_panel, self._ankle_addr_text = panel_device(
            ankle_panel_text, img_url=ankle_panel_img, addr_key=ankle_panel_addr_key)

        self._connect_button = panel_control_button(
            connect_button_text, key=connect_button_key, disabled=True)

        self._process_button = panel_control_button(
            process_button_text, key=process_button_key, disabled=True)

        self._description_text = panel_description(test_description_text)

        self._labels_listbox = panel_listbox(
            training_labels, key=label_listbox_key)

        label_grid_panel, self._labels_grid = panel_label_grid(training_labels)

        layout = [
            [panel_heading(heading)],
            [self._scan_button],
            [wrist_panel, sg.VerticalSeparator(), ankle_panel],
            [self._connect_button, self._process_button],
            [self._description_text],
            [self._labels_listbox, sg.VerticalSeparator(), label_grid_panel]
        ]
        self._window = sg.Window(self._title, layout=layout, finalize=True)

    def start_scan_task(self):
        self._app_state.io_pool.restart(force=True)
        scan_task = backend.get_nearby_devices(self._app_state.io_pool)
        return scan_task

    def stop_scan_task(self):
        self._app_state.io_pool.close()
        self._app_state.io_pool.join()

    def start_start_connect_task(self):
        self._app_state.io_pool.restart(force=True)
        self._app_state.task_pool.restart(force=True)
        return self._app_state.io_pool.apipe(backend.test_model,
                                             devices=self._app_state.nearby_devices,
                                             model=self._app_state.updated_model)

    def stop_start_connect_task(self):
        self._app_state.io_pool.close()
        self._app_state.io_pool.join()

    def start_process_task(self):
        self._pipeline.process(start_time=dt.datetime.now())

    def stop_process_task(self):
        self._pipeline.pause()

    def start_stop_connect_task(self):
        self._app_state.io_pool.restart(force=True)
        task = self._app_state.io_pool.apipe(self._pipeline.stop)
        return task

    def stop_stop_connect_task(self):
        self._app_state.io_pool.close()
        self._app_state.io_pool.join()

    def close_task_pool(self):
        self._app_state.task_pool.close()
        self._app_state.task_pool.join()

    def _format_result(self, result):
        training_labels = self._app_state.updated_model[0].classes_
        formatted = []
        for label, value in zip(training_labels, result[0]):
            formatted.append(label + ': ' + str(int(round(value, 2) * 100)))
        sort_formatted = sorted(
            formatted, key=lambda v: int(v.split(':')[1]), reverse=True)
        return sort_formatted

    def _update_label_grid(self, values):
        i = 0
        correct = values[0].split(
            ':')[0] == self._test_label or self._test_label == 'Any'
        base_color = 'g' if correct else 'r'
        for value in values:
            name = value.split(':')[0]
            num = float(value.split(':')[1])
            if not correct and name == self._test_label:
                new_color = 'blue'
                text_color = 'white'
            else:
                new_color = arus_plot.adjust_lightness(
                    base_color, amount=num / 100.0, return_format='hex')
                text_color = "white" if num > 20 else "gray"
            self._labels_grid[i].Update(
                value, background_color=new_color, text_color=text_color)
            i = i + 1

    def start(self):
        self.init_panel()
        scan_task = None
        stop_connect_task = None
        start_connect_task = None
        process_task = None
        while True:
            event, values = self._window.read(timeout=200)
            if event is None:
                if self._pipeline is not None:
                    self._pipeline.stop()
                break
            elif event == self._scan_button.Key:
                scan_task = self.start_scan_task()
                self._scan_button.Update("Scanning...", disabled=True)
            elif event == self._connect_button.Key:
                if start_connect_task is None and self._pipeline is None:
                    start_connect_task = self.start_start_connect_task()
                    self._connect_button.Update(
                        'Connecting...', disabled=True)
                elif stop_connect_task is None and self._pipeline is not None:
                    stop_connect_task = self.start_stop_connect_task()
                    self._connect_button.Update(
                        'Disconnecting...', disabled=True)
                    self._process_button.Update('Stopping...', disabled=True)
            elif event == self._process_button.Key:
                if not self._pipeline._started:
                    self.start_process_task()
                    process_task = self._pipeline.get_iterator(timeout=0.1)
                    self._process_button.Update('Stop testing')
                else:
                    self.stop_process_task()
                    self._process_button.Update('Start testing')
            elif event == self._labels_listbox.Key:
                self._test_label = values[event][0]
            else:
                if scan_task is not None:
                    if scan_task.ready():
                        self._app_state.nearby_devices = sorted(
                            scan_task.get())
                        self._scan_button.Update(
                            "Scan nearby metawears", disabled=False)
                        self._wrist_addr_text.Update(
                            self._app_state.nearby_devices[0])
                        self._ankle_addr_text.Update(
                            self._app_state.nearby_devices[1])
                        self._connect_button.Update(disabled=False)
                        self.stop_scan_task()
                        scan_task = None
                    else:
                        pass
                if stop_connect_task is not None:
                    if stop_connect_task.ready():
                        if stop_connect_task.get():
                            self._connect_button.Update(
                                'Connect to devices', disabled=False)
                            self._process_button.Update(
                                'Start testing', disabled=True)
                            self._pipeline = None
                        else:
                            self._connect_button.Update(
                                'Disconnect', disabled=False)
                        self.stop_stop_connect_task()
                        stop_connect_task = None
                    else:
                        pass
                if start_connect_task is not None:
                    if start_connect_task.ready():
                        self._pipeline = start_connect_task.get()
                        self._connect_button.Update(
                            'Disconnect', disabled=False)
                        self._process_button.Update(disabled=False)
                        self.stop_start_connect_task()
                        start_connect_task = None
                    else:
                        pass
                if process_task is not None:
                    data, _, _, _, _, name = next(process_task)
                    if data is not None:
                        display_result = self._format_result(data)
                        print('Processed results: ' + str(display_result))
                        self._update_label_grid(display_result)

        self.close_task_pool()
        self._window.close()
