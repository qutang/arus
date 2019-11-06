from arus.core.broadcaster import MhealthFileBroadcaster
from arus.core.stream import SensorFileStream
from arus.core.libs.mhealth_format.logging import display_start_and_stop_time
from arus.testing import load_test_data
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
    data, sr = load_test_data(file_type='actigraph',
                              file_num='single', exception_type='consistent_sr')
    streamer = SensorFileStream(
        data_source=data, sr=sr, window_size=3600, storage_format='actigraph', buffer_size=3600, name='test-actigraph-data')
    broadcaster = MhealthFileBroadcaster(
        name='test-actigraph-data-broadcaster')
    streamer.start(scheduler='thread')
    broadcaster.start(scheduler='thread', output_folder='./outputs/', pid='SPADES_12', file_type='sensor',
                      sensor_or_annotation_type='ActigraphGT9X',
                      data_type='AccelerationCalibrated',
                      version_code='NA',
                      sensor_or_annotator_id='TAS1E2XXXXX',
                      split_hours=True,
                      flat=True)
    for data in streamer.get_iterator():
        logging.info('Loaded {}'.format(
            display_start_and_stop_time(data, file_type='sensor')))
        broadcaster.send_data(data)
    broadcaster.send_data(None)
    streamer.stop()
    broadcaster.wait_to_finish()
    logging.info('finished')
