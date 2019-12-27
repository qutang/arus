import logging
import time
import queue

import pandas as pd
import numpy as np
from arus.core.stream import Stream, SlidingWindowStream
from mbientlab.metawear import cbindings, libmetawear
from pymetawear.client import MetaWearClient

from .corrector import MetawearTimestampCorrector


class MetaWearSlidingWindowStream(SlidingWindowStream):
    """Data stream to syncly or asyncly read metawear device sensor stream in real-time.

    This class inherits `SlidingWindowStream` class to load metawear device data into sliding windows.

    The stream will load a metawear device (mac address) in the `data_source` and separate them into chunks specified by `window_size` to be loaded in the data queue.

    Examples:
        1. Loading a metawear device of window size 12.8s asynchronously.

        ```python
        .. include:: ../../../examples/metawear_stream.py
        ```
    """

    def __init__(self, data_source, window_size, sr, grange, max_retries=3, name='metawear-stream'):
        """
        Args:
            data_source (str or list): filepath or list of filepaths of mhealth sensor data
            sr (int): the sampling rate (Hz) for the given data
            buffer_size (float, optional): the buffer size for file reader in seconds
            storage_format (str, optional): the storage format of the files in `data_source`. It now supports `mhealth` and `actigraph`.
            simulate_reality (bool, optional): simulate real world time delay if `True`.
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source,
                         window_size=window_size, simulate_reality=False, start_time_col=0, stop_time_col=0, name=name)
        self._sr = sr
        self._grange = grange
        self._device = None
        self._corrector = MetawearTimestampCorrector(sr)
        self._max_retries = max_retries
        self._input_count = 0
        self._callback_started = False

    def get_device_name(self):
        model_code = libmetawear.mbl_mw_metawearboard_get_model(
            self._device.mw.board)
        metawear_models = cbindings.Model()
        model_names = list(
            filter(lambda attr: '__' not in attr, dir(metawear_models)))
        for name in model_names:
            if getattr(metawear_models, name) == model_code:
                return name
        return 'NA'

    def _setup_metawear(self, addr):
        count_retry = 0
        while self._device is None:
            try:
                self._device = MetaWearClient(addr, connect=True, debug=False)
                self._device_name = self.get_device_name()
            except Exception as e:
                if count_retry == self._max_retries:
                    raise Exception(
                        "Max retry reaches, still cannot connect to device: " + addr)
                logging.error(str(e))
                logging.info(str(count_retry) + ' retry connect to ' + addr)
                count_retry = count_retry + 1
                time.sleep(2)
        logging.info("New metawear connected: {0}".format(
            self._device))
        libmetawear.mbl_mw_metawearboard_set_time_for_response(
            self._device.mw.board, 4000)
        # high frequency throughput connection setup
        self._device.settings.set_connection_parameters(
            7.5, 7.5, 0, 6000)
        # Up to 4dB for Class 2 BLE devices
        # https://github.com/hbldh/pymetawear/blob/master/pymetawear/modules/settings.py
        # https://mbientlab.com/documents/metawear/cpp/0/settings_8h.html#a335f712d5fc0587eff9671b8b105d3ed
        # Hossain AKMM, Soh WS. A comprehensive study of Bluetooth signal parameters for localization. 2007 Ieee 18th International Symposium on Personal, Indoor and Mobile Radio Communications, Vols 1-9. 2007:428-32.
        self._device.settings.set_tx_power(power=4)

        self._device.accelerometer.set_settings(
            data_rate=self._sr, data_range=self._grange)
        self._device.accelerometer.high_frequency_stream = True

    def _format_data_as_mhealth(self, data):
        real_world_ts = time.time()
        ts_set = self._corrector.correct(data, real_world_ts)
        calibrated_values = self._calibrate_coord_system(data)
        formatted = {
            'HEADER_TIME_STAMP': [pd.Timestamp.fromtimestamp(ts_set[2])],
            'X': [calibrated_values[0]],
            'Y': [calibrated_values[1]],
            'Z': [calibrated_values[2]],
            'MAC_ADDRESS': [self._data_source],
            'DEVICE_NAME': [self._device_name]
        }
        return pd.DataFrame.from_dict(formatted)

    def stop(self):
        self._device.led.stop_and_clear()
        time.sleep(0.5)
        self._device.accelerometer.notifications(callback=None)
        time.sleep(0.5)
        self._device.disconnect()
        logging.debug('Disconnected.')
        self._callback_started = False
        self._buffer_data_source(None)
        super().stop()

    def _start_metawear(self):
        self._input_count = 0

        def _callback(data):
            if self._input_count == 0:
                logging.debug('Accelerometer callback starts running...')
                self._callback_started = True
            self._input_count = self._input_count + 1
            formatted = self._format_data_as_mhealth(data)
            self._buffer_data_source(formatted)
            if self._input_count == self._sr:
                logging.debug('Received data for one second')
                self._input_count = 1

        def _start():
            logging.info('starting accelerometer module...')
            self._device.accelerometer.notifications(
                callback=_callback)
            time.sleep(2)
            if not self._callback_started:
                return False
            else:
                pattern = self._device.led.load_preset_pattern(
                    'solid', repeat_count=0xff)
                self._device.led.write_pattern(pattern, 'g')
                self._device.led.play()
                return True
        return _start()

    def _calibrate_coord_system(self, data):
        # axis values are calibrated according to the coordinate system of Actigraph GT9X
        # http://www.correctline.pl/wp-content/uploads/2015/01/ActiGraph_Device_Axes_Link.png
        x = data['value'].x
        y = data['value'].y
        z = data['value'].z
        if self._device_name == 'METAMOTION_R':
            # as normal wear in the case on wrist
            calibrated_x = y
            calibrated_y = -x
            calibrated_z = z
        else:
            calibrated_x = x
            calibrated_y = y
            calibrated_z = z
        return (calibrated_x, calibrated_y, calibrated_z)

    def load_data_source_(self, data_source):
        if isinstance(data_source, str):
            addr = data_source
            self._setup_metawear(addr)
            self._start_metawear()
        else:
            raise RuntimeError(
                "Data source should be the mac address of the metawear device")
