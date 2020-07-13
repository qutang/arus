from .. import generator as gr
from .. import segmentor as seg
from .. import synchronizer as sync
from .. import processor as pr
from .. import scheduler as sch
from .. import node
import numpy as np


def test_random_data_generator_operator():
    generator = gr.RandomAccelDataGenerator(sr=80, buffer_size=10)
    op = node.Node(generator, name='accel', t=node.Node.Type.INPUT)

    op.start()
    i = 0
    while True:
        data = next(op.produce())
        if data.signal == node.Node.Signal.WAIT:
            continue
        assert data.values.shape[0] == 10
        if i == 2:
            break
        i += 1
    op.stop()


def test_mhealth_file_generator_operator(spades_lab_data):
    mhealth_filepath = spades_lab_data['subjects']['SPADES_2']['sensors']['DW'][0]
    generator = gr.MhealthSensorFileGenerator(
        mhealth_filepath, buffer_size=10)

    op = node.Node(generator, name='SPADES_2-DW', t=node.Node.Type.INPUT)

    op.start()
    i = 0
    while True:
        data = next(op.produce())
        if data.signal == node.Node.Signal.WAIT:
            continue
        assert data.values.shape[0] == 10
        if i == 2:
            break
        i += 1
    op.stop()


def test_segmentor_operator(spades_lab_data):
    mhealth_filepath = spades_lab_data['subjects']['SPADES_2']['sensors']['DW'][0]
    generator = gr.MhealthSensorFileGenerator(
        mhealth_filepath, buffer_size=1800)
    generator.run()
    values, context = next(generator.get_result())
    generator.stop()
    segmentor = seg.SlidingWindowSegmentor(window_size=1)
    op = node.Node(segmentor, name='segmentor-1s')

    op.start()
    op.consume(node.Node.Pack(values=values, signal=node.Node.Signal.DATA,
                              context=context, src='SPADES_2-dw'))
    i = 0
    while True:
        data = next(op.produce())
        if data.signal == node.Node.Signal.WAIT:
            continue
        assert data.values.shape[0] == 80
        if i == 2:
            break
        i += 1
    op.stop()


def test_synchronizer_operator(spades_lab_data):
    dw_filepath = spades_lab_data['subjects']['SPADES_2']['sensors']['DW'][0]
    da_filepath = spades_lab_data['subjects']['SPADES_2']['sensors']['DA'][0]
    dw_generator = gr.MhealthSensorFileGenerator(
        dw_filepath, buffer_size=1800)
    da_generator = gr.MhealthSensorFileGenerator(
        da_filepath, buffer_size=1800)
    dw_generator.run()
    da_generator.run()
    dw_values, dw_context = next(dw_generator.get_result())
    da_values, da_context = next(da_generator.get_result())
    dw_context = {**dw_context, 'data_id': 'DW'}
    da_context = {**da_context, 'data_id': 'DA'}
    dw_generator.stop()
    da_generator.stop()

    segmentor = seg.SlidingWindowSegmentor(window_size=1)
    segmentor.run(
        dw_values, src=None, context=dw_context)
    seg_dw_values, dw_context = next(segmentor.get_result())
    segmentor = seg.SlidingWindowSegmentor(window_size=1)
    segmentor.run(
        da_values, src=None, context=da_context)
    seg_da_values, da_context = next(segmentor.get_result())

    synchronizer = sync.Synchronizer()
    synchronizer.add_sources(2)
    op = node.Node(synchronizer, name='sync-dw-da')
    op.start()
    op.consume(node.Node.Pack(values=seg_dw_values,
                              signal=node.Node.Signal.DATA, context=dw_context, src='dw'))
    op.consume(node.Node.Pack(values=seg_da_values,
                              signal=node.Node.Signal.DATA, context=da_context, src='da'))
    while True:
        data = next(op.produce())
        if data.signal == node.Node.Signal.WAIT:
            continue
        assert data.values[0].shape[0] == 80
        assert data.values[1].shape[0] == 80
        np.testing.assert_array_equal(data.context['data_ids'], ['DW', 'DA'])
        break
    op.stop()


def test_processor_operator(spades_lab_data):
    dw_filepath = spades_lab_data['subjects']['SPADES_2']['sensors']['DW'][0]
    dw_generator = gr.MhealthSensorFileGenerator(
        dw_filepath, buffer_size=1800)
    dw_generator.run()
    dw_values, dw_context = next(dw_generator.get_result())
    dw_context = {**dw_context, 'data_id': 'DW'}
    dw_generator.stop()

    def compute_mean(values, src=None, **context):
        import numpy as np
        data = values.values[:, 1:]
        result = np.mean(data, axis=0)
        return result, context

    processor = pr.Processor(compute_mean,
                             mode=sch.Scheduler.Mode.PROCESS,
                             scheme=sch.Scheduler.Scheme.SUBMIT_ORDER, max_workers=2)

    op = node.Node(processor, name='compute-mean')
    op.start()
    op.consume(node.Node.Pack(values=dw_values, signal=node.Node.Signal.DATA,
                              context=dw_context, src='dw'))
    while True:
        data = next(op.produce())
        if data.signal == node.Node.Signal.WAIT:
            continue
        assert len(data.values) == 3
        break
    op.stop()
