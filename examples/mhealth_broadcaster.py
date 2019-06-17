from arus.core.broadcaster import MhealthFileBroadcaster
from arus.core.stream import ActigraphFileStream
from arus.libs.mhealth_format.logging import display_start_and_stop_time
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
    streamer = ActigraphFileStream(
        data_source='/mnt/d/data/spades-2day/SPADES_12/OriginalRaw/Spades_12_dominant_ankle (2015-12-11)RAW.csv', sr=80, name='SPADES_12_DA')
    broadcaster = MhealthFileBroadcaster(name='SPADES_12_DA_writer')
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
