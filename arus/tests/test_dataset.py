from .. import dataset
from .. import env
from .. import segmentor
from .. import spades_lab as slab
from .. import feature_vector as fv
from .. import cli
import os
import shutil
import pandas as pd
from dataclasses import asdict
import pytest
from loguru import logger


class TestSpadesLab:
    def test_get_dataset_path(self):
        dataset_path = dataset.get_dataset_path('spades_lab')
        assert dataset_path == os.path.join(
            env.get_data_home(), 'spades_lab')
        assert os.path.exists(dataset_path)

    def test_load_dataset(self):
        dataset_dict = dataset.load_dataset('spades_lab')
        assert type(dataset_dict) == dict
        assert 'subjects' in dataset_dict.keys()
        assert 'processed' in dataset_dict.keys()


class TestSensorObj:
    @pytest.mark.parametrize("seg", [None, 'sliding_window'])
    @pytest.mark.parametrize("placement_parser", [None, slab.get_sensor_placement])
    @pytest.mark.parametrize("stream_id", [None, "test"])
    def test_get_stream(self, seg, placement_parser, stream_id):
        dataset_path = dataset.get_dataset_path('spades_lab')
        mh_ds = dataset.MHDataset(
            path=dataset_path, name='spades_lab', input_type=dataset.InputType.MHEALTH_FORMAT)
        mh_ds.set_placement_parser(placement_parser)
        if seg == 'sliding_window':
            seg = segmentor.SlidingWindowSegmentor(window_size=12)
        stream = mh_ds.subjects[0].sensors[0].get_stream(
            seg=seg, stream_id=stream_id, window_size=12)
        stream.start()
        count = 0
        for data, context in stream.get_result():
            logger.debug(count)
            count += 1
            assert data.shape[0] == 12 * 80
            assert data.shape[1] == 4
            if stream_id is not None:
                assert context['data_id'] == 'test'
            elif placement_parser is None:
                assert context['data_id'] == 'TAS1E23150066-AccelerationCalibrated-stream'
            else:
                assert context['data_id'] == 'DT'
            if count == 5:
                logger.debug('stop')
                stream.stop()

    def test_compute_features_on_the_fly(self):
        dataset_path = dataset.get_dataset_path('spades_lab')
        mh_ds = dataset.MHDataset(
            path=dataset_path, name='spades_lab', input_type=dataset.InputType.MHEALTH_FORMAT)
        mh_ds.set_placement_parser(slab.get_sensor_placement)
        stream = mh_ds.subjects[0].sensors[0].get_stream(
            seg=None, stream_id=None, window_size=12.8)
        stream.start()
        count = 0
        for data, context in stream.get_result():
            logger.debug(count)
            count += 1
            fv_df, fv_names = fv.inertial.single_triaxial(
                data, sr=80, st=context['start_time'], et=context['stop_time'], selected=['MEAN'])
            assert fv_df.iloc[0, 1].timestamp(
            ) == context['start_time'].timestamp()
            assert fv_df.iloc[0, 2].timestamp(
            ) == context['stop_time'].timestamp()
            assert len(fv_names) == 1
            assert fv_df.shape[0] == 1
            if count == 5:
                logger.debug('stop')
                stream.stop()


class TestMHDataset:
    def test_mh_dataset(self):
        dataset_path = dataset.get_dataset_path('spades_lab')
        mh_ds = dataset.MHDataset(
            path=dataset_path, name='spades_lab', input_type=dataset.InputType.MHEALTH_FORMAT)
        assert mh_ds.name == 'spades_lab'
        assert mh_ds.subjects[0].pid == 'SPADES_1'
        assert mh_ds.subjects[0].demography is None
        assert mh_ds.subjects[0].sensors[0].data_type == 'AccelerationCalibrated'
        assert mh_ds.subjects[0].annotations[0].annotation_type == 'SPADESInLab'
        assert len(mh_ds.subjects[0].sensors[0].paths) == 3
        assert mh_ds.subjects[0].sensors[0].start_time.timestamp(
        ) == pd.Timestamp('2015-09-24 14:22:48.013000').timestamp()


class TestSGDataset:
    def test_sg_dataset(self):
        spades_lab_path = dataset.get_dataset_path('spades_lab')
        dataset_path = os.path.join(
            spades_lab_path, 'SPADES_1', 'Derived', 'signaligner')
        if not os.path.exists(dataset_path):
            cli.convert_to_signaligner_both(spades_lab_path, 'SPADES_1', 80)
        sg_ds = dataset.SGDataset(
            path=dataset_path, name='spades_1_sg', sensor_folder='SPADES_1_2015092414_2015092417_sensors', label_folder='SPADES_1_2015092414_2015092417_labelsets')
        assert sg_ds.name == 'spades_1_sg'
        assert sg_ds.subjects[0].pid == 'spades_1_sg'
        assert sg_ds.subjects[0].demography is None
        assert sg_ds.subjects[0].sensors[0].data_type == 'SPADES_1_ActigraphGT9X_TAS1E23150066-AccelerationCalibrated_2015092414_2015092417.sensor'
        assert sg_ds.subjects[0].sensors[0].sid == 'TAS1E23150066'
        assert sg_ds.subjects[0].sensors[0].sr == 80
        assert sg_ds.subjects[0].sensors[0].grange == 8
        assert sg_ds.subjects[0].annotations[0].annotation_type == 'SPADES_1_diego-SPADESInLab_2015092414_2015092417.annotation'
        assert len(sg_ds.subjects[0].sensors[0].paths) == 1
