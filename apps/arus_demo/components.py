import PySimpleGUI as sg
import time
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

HEADING_FONT = ('Helvetica', 12, 'bold')
PRIMARY_FONT = ('Helvetica', 10)
SMALL_FONT = ('Helvetica', 8)
BIG_FONT = ('Helvetica', 14, 'bold')


def heading(text, fixed_column_width=None, key=None):
    return sg.Text(text=text,
                   relief=sg.RELIEF_FLAT,
                   font=HEADING_FONT,
                   auto_size_text=True,
                   justification='center',
                   size=(fixed_column_width, None),
                   key=key)


def text(text, fixed_column_width=None, text_color=None, background_color=None, key=None):
    return sg.Text(text=text,
                   font=PRIMARY_FONT,
                   size=(fixed_column_width, None),
                   text_color=text_color,
                   background_color=background_color,
                   key=key)


def big_text(text, fixed_column_width=None, text_color=None, background_color=None, key=None):
    return sg.Text(text=text,
                   font=HEADING_FONT,
                   size=(fixed_column_width, None),
                   text_color=text_color,
                   background_color=background_color,
                   key=key)


def selection_list(items, default_selections=None, mode='single', fixed_column_width=None, rows=20, key=None):
    if mode == 'single':
        mode = sg.LISTBOX_SELECT_MODE_SINGLE
    else:
        mode = sg.LISTBOX_SELECT_MODE_EXTENDED
    return sg.Listbox(values=items,
                      default_values=default_selections,
                      select_mode=mode,
                      font=PRIMARY_FONT,
                      size=(fixed_column_width, rows),
                      key=key,
                      enable_events=False)


def dropdown_list(items, default_item=None, fixed_column_width=None, key=None):
    return sg.Combo(items, default_value=default_item, size=(fixed_column_width, None), key=key, enable_events=True)


def control_button(text, size='normal', fixed_column_width=None, disabled=False, key=None):
    if size == 'normal':
        font = PRIMARY_FONT
    elif size == 'big':
        font = HEADING_FONT
    elif size == 'small':
        font = SMALL_FONT
    return sg.Button(button_text=text,
                     font=font,
                     auto_size_button=True,
                     size=(fixed_column_width, None),
                     key=key,
                     disabled=disabled)


class TextInput:
    def __init__(self, confirm_button_text, candidate_items=[], fixed_column_width=None, confirm_button_disabled=False, text_input_disabled=False, confirm_button_key=None, text_input_key=None):
        if fixed_column_width is not None:
            button_width = int(fixed_column_width * 1 / 4)
            input_width = int(fixed_column_width * 3 / 4)
        else:
            button_width = None
            input_width = None
        self._confirm_button = control_button(
            confirm_button_text,
            size='normal',
            fixed_column_width=button_width,
            disabled=confirm_button_disabled,
            key=confirm_button_key)
        self._text_input = sg.Combo(
            values=candidate_items,
            font=PRIMARY_FONT,
            size=(input_width, None),
            key=text_input_key,
            disabled=text_input_disabled
        )

    def get_component_as_row(self):
        return [self._text_input, self._confirm_button]

    def update_button(self, **kwargs):
        self._confirm_button.update(**kwargs)

    def update_input(self, **kwargs):
        self._text_input.update(**kwargs)


class ProgressBar:
    def __init__(self, text, fixed_column_width=None, text_key=None, bar_key=None):
        self._progress_text = sg.Text(text,
                                      key=text_key,
                                      font=PRIMARY_FONT,
                                      size=(fixed_column_width, None))

        self._progress_bar = sg.ProgressBar(100,
                                            orientation='h',
                                            auto_size_text=True,
                                            size=(fixed_column_width, 20),
                                            key=bar_key)

    def get_component(self):
        return [self._progress_text], [self._progress_bar]

    def update(self, percentage, text=None):
        if text is not None:
            self._progress_text.update(value=text)
        self._progress_bar.update_bar(int(100 * percentage))

    def increment(self, text=None):
        if text is not None:
            self._progress_text.update(value=text)
        self._progress_bar.update_bar(
            self._progress_bar.TKProgressBar.TKProgressBarForReal['value'] + 1)


def checkbox(text, fixed_column_width=None, default_checked=False, disabled=False, key=None):
    return sg.Checkbox(text,
                       default=default_checked,
                       disabled=disabled,
                       auto_size_text=True,
                       size=(fixed_column_width, None),
                       font=PRIMARY_FONT,
                       key=key, enable_events=True)


class RadioGroup:
    def __init__(self, group_id, n, default_index=0, direction='horizontal',
                 fixed_column_width=None):
        self._group_id = group_id
        self._direction = direction
        self._radio_boxes = []
        self._els = []
        self._n = n
        self._default_index = default_index
        if fixed_column_width is not None:
            if self._direction == 'horizontal':
                self._width = int(
                    fixed_column_width / n)
            else:
                self._width = fixed_column_width
        else:
            self._width = None

    def add_radiobox(self, text, key, disabled=False):
        if len(self._radio_boxes) == self._default_index:
            checked = True
        else:
            checked = False
        box = sg.Radio(text, self._group_id, default=checked, disabled=disabled,
                       auto_size_text=True, font=PRIMARY_FONT, size=(self._width, None), enable_events=False, key=key)
        self._els.append(box)
        if self._direction == 'horizontal':
            self._radio_boxes.append(box)
        else:
            self._radio_boxes = self._radio_boxes + [box]

    def add_radioboxes(self, texts, keys, disable_states):
        for text, key, disabled in zip(texts, keys, disable_states):
            self.add_radiobox(text, key, disabled=disabled)

    def disable_all(self):
        for el in self._els:
            el.update(disabled=True)

    def enable_all(self):
        for el in self._els:
            el.update(disabled=False)

    def get_component(self):
        return self._radio_boxes


def table(cells, headings, fixed_column_width=None, key=None):
    return sg.Table(values=cells,
                    headings=headings,
                    size=(fixed_column_width, None),
                    def_col_width=int(fixed_column_width/len(headings)),
                    auto_size_columns=False,
                    key=key)


def image(img_url, key=None):
    return sg.Image(img_url,
                    enable_events=False,
                    key=key)


class Plot:
    def __init__(self, fixed_column_width=None, rows=None, key=None):
        self._el = sg.Canvas(size=(fixed_column_width, rows), key=None)
        self._graph = None
        self._fig = None

    def update_plot(self, fig=None):
        if fig is not None:
            self._fig = fig
            _, _, figure_w, figure_h = fig.bbox.bounds
            self._el.set_size((figure_w, figure_h))
            self._graph = FigureCanvasTkAgg(fig, self._el.TKCanvas)
            self._graph.draw()
            self._graph.get_tk_widget().pack(side='top', fill='both', expand=1)
        else:
            self._graph.draw()
            self._graph.get_tk_widget().pack(side='top', fill='both', expand=1)

    def get_figure(self):
        return self._fig

    def get_component(self):
        return self._el


class DeviceInfo:
    def __init__(self, device_name, device_addr_list=[], placement_img_url=None, fixed_column_width=None, device_name_key=None, placement_img_key=None, device_addr_key=None, device_selected=True, device_selection_disabled=False):
        self._device_name = checkbox(device_name,
                                     fixed_column_width=fixed_column_width, default_checked=device_selected, key=device_name_key, disabled=device_selection_disabled)
        self._device_placement = image(placement_img_url,
                                       key=placement_img_key)
        self._device_addr_list = dropdown_list(
            items=device_addr_list, default_item=None, fixed_column_width=fixed_column_width, key=device_addr_key)

    def get_component(self):
        return (
            [self._device_name],
            [self._device_placement],
            [self._device_addr_list]
        )

    def update_addr_list(self, addrs, index):
        self._device_addr_list.update(values=addrs, set_to_index=index)

    def update_addr_selection(self, index):
        self._device_addr_list.update(set_to_index=index)

    def enable_selection(self):
        self._device_name.update(disabled=False)

    def disable_selection(self):
        self._device_name.update(disabled=True)

    def disable_addr_list(self):
        self._device_addr_list.update(disabled=True)

    def enable_addr_list(self):
        self._device_addr_list.update(disabled=False)

    def update_selection(self, selected=False):
        self._device_name.update(selected)


class LabelGrid:
    def __init__(self, n, fixed_column_width=None, default_background_color=None, default_text_color=None, n_cols=4):
        self._n = n
        if fixed_column_width is not None:
            self._width = int(fixed_column_width / n_cols)
        else:
            self._width = None
        self._default_background_color = default_background_color
        self._default_text_color = default_text_color
        self._n_cols = n_cols
        self._label_grid = [[]]

    def add_label(self, label, key=None, background_color=None, text_color=None):
        el = big_text(label,
                      background_color=background_color or self._default_background_color,
                      text_color=text_color or self._default_text_color, fixed_column_width=self._width,
                      key=key)
        last_row = self._label_grid[-1]
        if len(last_row) < self._n_cols:
            self._label_grid[-1].append(el)
        else:
            self._label_grid.append([el])

    def add_labels(self, labels, keys, background_colors, text_colors):
        for label, key, bcolor, tcolor in zip(labels, keys, background_colors, text_colors):
            self.add_label(label, key, background_color=bcolor,
                           text_color=tcolor)

    def get_component(self):
        return self._label_grid

    def update_label_by_key(self, key, **kwargs):
        for row in self._label_grid:
            for el in row:
                if el.Key == key:
                    el.update(**kwargs)
                    break
            break

    def update_label_by_index(self, i, j, **kwargs):
        self._label_grid[i][j].update(**kwargs)

    def update_labels(self, new_labels, new_b_colors, new_t_colors):
        i = 0
        for row in self._label_grid:
            for el in row:
                el.update(
                    new_labels[i], background_color=new_b_colors[i], text_color=new_t_colors[i])
                i = i + 1


class Timer:
    def __init__(self):
        self.reset()

    def is_running(self):
        return len(self._start_time) > 0 and len(self._start_time) > len(self._stop_time)

    def start(self):
        self._start_time.append(time.time())
        self._last_tick = self._start_time[0]

    def get_total_lapsed_time(self, formatted=False):
        if len(self._start_time) - len(self._stop_time) == 1:
            stop_time = self._stop_time + [time.time()]
        else:
            stop_time = self._stop_time
        total_time = 0
        for st, et in zip(self._start_time, stop_time):
            total_time += (et - st)
        if formatted:
            total_time = self.format_lapse_time(total_time)
        return total_time

    def get_lapsed_times(self, formatted=False):
        lapse_times = []
        if len(self._start_time) - len(self._stop_time) == 1:
            stop_time = self._stop_time + [time.time()]
        else:
            stop_time = self._stop_time
        for st, et in zip(self._start_time, stop_time):
            if formatted:
                ts = self.format_lapse_time(et - st)
            else:
                ts = et - st
            lapse_times.append(ts)
        return lapse_times

    def get_last_lapse_time(self, formatted=False):
        if len(self._start_time) - len(self._stop_time) == 1:
            stop_time = self._stop_time + [time.time()]
        else:
            stop_time = self._stop_time
        ts = stop_time[-1] - self._start_time[-1]
        if formatted:
            return self.format_lapse_time(ts)
        else:
            return ts

    def format_lapse_time(self, ts):
        seconds = np.floor(ts)
        sec_digits = str(int(seconds % 60))
        min_digits = str(int(np.floor(seconds / 60) % 60))
        hour_digits = str(int(np.floor(seconds / 3600) % 60))

        if len(sec_digits) == 1:
            sec_digits = '0' + sec_digits
        if len(min_digits) == 1:
            min_digits = '0' + min_digits
        if len(hour_digits) == 1:
            hour_digits = '0' + hour_digits
        return hour_digits + ':' + min_digits + ':' + sec_digits

    def update(self):
        self._marker_time.append(time.time())

    def tick(self):
        self._last_tick = time.time()

    def stop(self):
        if len(self._stop_time) == len(self._start_time):
            return
        self._stop_time.append(time.time())

    def reset(self):
        self._start_time = []
        self._stop_time = []
        self._marker_time = []
        self._last_tick = None

    def since_last_tick(self):
        return time.time() - self._last_tick
