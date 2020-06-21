

from loguru import logger
import queue
import time

import pandas as pd

from .. import generator
import threading
import importlib


def _print_extra_dep_warning(e):
    msg = (
        "Arus plugin metawear requirements are not installed. Some functionality will not be usable.\n\n"
        "Please install the metawear extra packages as follows:\n\n"
        "  pip install arus[metawear]\n\n"
    )
    print(str(e) + "\n\n" + msg)


class StartFailure(Exception):
    pass


class MetaWearScanner():
    def __init__(self):
        try:
            self._mw_discover = importlib.import_module(
                'discover', 'pymetawear')
        except ImportError as e:
            _print_extra_dep_warning(e)
            raise ImportError(e)

    def get_nearby_devices(self, max_devices=None):
        metawears = set()
        retries = 0
        max_devices = 100 if max_devices is None else max_devices
        while retries < 5 and len(metawears) < max_devices:
            logger.info('Scanning metawear devices nearby...')
            try:
                retries += 1
                candidates = self._mw_discover.discover_devices(timeout=1)
                metawears |= set(
                    map(lambda d: d[0],
                        filter(lambda d: d[1] == 'MetaWear', candidates)))
            except ValueError as e:
                logger.error(str(e))
                continue
        return list(metawears)


class MetawearTimestampCorrector(object):
    def __init__(self, sr):
        self._current_nofix_ts = None
        self._current_noloss_ts = None
        self._current_withloss_ts = None
        self._sample_interval = 1.0 / sr
        self._real_world_offset = None
        self._last_real_world_ts = None

    def _get_current_nofix_ts_in_seconds(self, data):
        return data['epoch'] / 1000.0

    def _get_real_world_offset(self, data, current_real_world_ts):
        return current_real_world_ts - \
            self._get_current_nofix_ts_in_seconds(data)

    def _apply_no_fix(self, data):
        return self._get_current_nofix_ts_in_seconds(data) + self._real_world_offset

    def _apply_fix_noloss(self, data, previous_noloss_ts):
        if previous_noloss_ts == None:
            current_noloss_ts = self._apply_no_fix(data)
        else:
            current_noloss_ts = previous_noloss_ts + self._sample_interval
        return current_noloss_ts

    def _apply_fix_withloss(self, data, previous_withloss_ts, current_real_world_ts, previous_real_world_ts):
        if previous_withloss_ts == None:
            current_withloss_ts = self._apply_fix_noloss(
                data, previous_withloss_ts)
        elif current_real_world_ts - previous_real_world_ts <= 2 * self._sample_interval:
            current_withloss_ts = self._apply_fix_noloss(
                data, previous_withloss_ts)
        elif abs(previous_withloss_ts - self._get_current_nofix_ts_in_seconds(data)) < 2 * self._sample_interval:
            # if there is no loss occuring
            current_withloss_ts = self._apply_fix_noloss(
                data, previous_withloss_ts)
        else:
            # data loss occurs
            current_nofix_ts = self._apply_no_fix(data)
            current_noloss_ts = self._apply_fix_noloss(
                data, previous_withloss_ts)
            if current_nofix_ts - current_noloss_ts > 2 * self._sample_interval:
                # late for more than two intervals, renew timestamp
                current_withloss_ts = current_nofix_ts
            else:
                current_withloss_ts = current_noloss_ts
        return current_withloss_ts

    def correct(self, data, current_real_world_ts):
        if self._real_world_offset == None:
            self._real_world_offset = self._get_real_world_offset(
                data, current_real_world_ts)
        if self._last_real_world_ts == None:
            self._last_real_world_ts = current_real_world_ts
        self._current_nofix_ts = self._apply_no_fix(data)
        self._current_noloss_ts = self._apply_fix_noloss(
            data, self._current_noloss_ts)
        self._current_withloss_ts = self._apply_fix_withloss(
            data, self._current_withloss_ts, current_real_world_ts, self._last_real_world_ts)
        diff_rw = current_real_world_ts - self._last_real_world_ts
        self._last_real_world_ts = current_real_world_ts
        return self._current_nofix_ts, self._current_noloss_ts, self._current_withloss_ts, self._get_current_nofix_ts_in_seconds(data), current_real_world_ts


class MetaWearAccelDataGenerator(generator.Generator):
    def __init__(self, addr, sr, grange, max_retries=3, **kwargs):
        super().__init__(**kwargs)

        try:
            self._mw = importlib.import_module('metawear', 'mbientlab')
            self._mw_client = importlib.import_module('client', 'pymetawear')
        except ImportError as e:
            _print_extra_dep_warning(e)
            raise ImportError(e)

        self._addr = addr
        self._sr = sr
        self._grange = grange
        self._max_retries = max_retries

        self._device = None
        self._corrector = MetawearTimestampCorrector(sr)
        self._input_count = 0
        self._internal_buffer = queue.Queue()
        self._callback_started = False
        self._start_condition = threading.Condition(threading.Lock())
        self._setup_metawear()

    def run(self, values=None, src=None, context={}):
        if self._start_metawear():
            self._generate()
        else:
            raise StartFailure(
                'Device fails to start correctly, please call generate to retry')

    def stop(self):
        self._device.led.stop_and_clear()
        time.sleep(0.5)
        self._device.accelerometer.notifications(callback=None)
        time.sleep(1)
        self._device.disconnect()
        logger.info('Disconnected.')
        self._callback_started = False
        self._internal_buffer.queue.clear()
        super().stop()

    def get_device_name(self):
        model_code = self._mw.libmetawear.mbl_mw_metawearboard_get_model(
            self._device.mw.board)
        metawear_models = self._mw.cbindings.Model()
        model_names = list(
            filter(lambda attr: '__' not in attr, dir(metawear_models)))
        for name in model_names:
            if getattr(metawear_models, name) == model_code:
                return name
        return 'NA'

    def _generate(self):
        while self._callback_started:
            if self._stop:
                break
            try:
                data = self._internal_buffer.get(timeout=0.1)
                result = self._buffering(data)
                if result is not None:
                    self._result.put((result, self._context))
            except queue.Empty:
                continue

    def _start_metawear(self):
        self._input_count = 0

        def _callback(data):
            formatted = self._format_data_as_mhealth(data)
            self._internal_buffer.put(formatted)
            if self._input_count == 0:
                logger.info('Accelerometer callback starts running...')
                with self._start_condition:
                    self._callback_started = True
                    self._start_condition.notify_all()

            self._input_count = self._input_count + 1
            if self._input_count == self._sr:
                logger.debug('Received data for one second')
                self._input_count = 1

        def _start():
            logger.info('starting accelerometer module...')
            self._device.accelerometer.notifications(
                callback=_callback)
            with self._start_condition:
                logger.info("Waiting for accelerometer callback to start...")
                self._start_condition.wait(timeout=2)
            if self._callback_started:
                logger.info(
                    'Accelerometer callback started successfully.')
                pattern = self._device.led.load_preset_pattern(
                    'solid', repeat_count=0xff)
                self._device.led.write_pattern(pattern, 'g')
                self._device.led.play()
                return True
            else:
                logger.warning(
                    'Accelerometer callback did not start successfully.')
                return False
        return _start()

    def _setup_metawear(self):
        count_retry = 0
        while self._device is None:
            try:
                self._device = self._mw_client.MetaWearClient(
                    self._addr, connect=True, debug=False)
                self._device_name = self.get_device_name()
            except Exception as e:
                if count_retry == self._max_retries:
                    raise Exception(
                        "Max retry reaches, still cannot connect to device: " + self._addr)
                logger.error(str(e))
                logger.info(str(count_retry) +
                            ' retry connect to ' + self._addr)
                count_retry = count_retry + 1
                time.sleep(1)
        logger.info("New metawear connected: {0}".format(
            self._device))
        self._mw.libmetawear.mbl_mw_metawearboard_set_time_for_response(
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
            'NO_FIX': [pd.Timestamp.fromtimestamp(ts_set[3])],
            'CLOCK': [pd.Timestamp.fromtimestamp(ts_set[4])],
            'MAC_ADDRESS': [self._addr],
            'DEVICE_NAME': [self._device_name]
        }
        return pd.DataFrame.from_dict(formatted)

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
