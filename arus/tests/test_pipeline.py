
from .. import stream2
from .. import generator
from .. import segmentor
from .. import synchronizer
from .. import processor
from .. import scheduler
from .. import pipeline
from .. import node
from .. import mh
from .. import developer
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_pipeline(spades_lab_data):
    def compute_mean(values_list, src=None, **context):
        import numpy as np
        import pandas as pd
        results = []
        for values, data_id in zip(values_list, context['data_ids']):
            data = values.values[:, 1:]
            col_names = [data_id + '_' +
                         name for name in ['MEAN_0', 'MEAN_1', 'MEAN_2']]
            result = pd.DataFrame(data=[np.mean(
                data, axis=0)], columns=col_names)
            results.append(result)

        result = pd.concat(results, axis=1)
        result.insert(0, 'START_TIME', context['start_time'])
        result.insert(1, 'STOP_TIME', context['stop_time'])
        return result, context
    dw_data = spades_lab_data['subjects']['SPADES_2']['sensors']['DW']
    da_data = spades_lab_data['subjects']['SPADES_2']['sensors']['DA']

    start_time = mh.get_session_start_time(
        'SPADES_2', spades_lab_data['meta']['root'], round_to='minute')

    dw_stream = stream2.Stream(
        generator.MhealthSensorFileGenerator(*dw_data, buffer_size=18000), segmentor.SlidingWindowSegmentor(window_size=2), name='dw-stream')
    dw_stream.set_essential_context(start_time=start_time, stream_id='DW')

    da_stream = stream2.Stream(
        generator.MhealthSensorFileGenerator(*da_data, buffer_size=18000), segmentor.SlidingWindowSegmentor(window_size=2), name='dw-stream')
    da_stream.set_essential_context(start_time=start_time, stream_id='DA')

    sync = synchronizer.Synchronizer()
    sync.add_sources(n=2)
    proc = processor.Processor(
        compute_mean, mode=scheduler.Scheduler.Mode.PROCESS, scheme=scheduler.Scheduler.Scheme.SUBMIT_ORDER, max_workers=10)

    pip = node.Node(op=pipeline.Pipeline(dw_stream, da_stream, synchronizer=sync,
                                         processor=proc, name='test-pipeline'),
                    t=node.Node.Type.INPUT, name='test-pipeline')
    return pip


def test_pipeline_lifecycle(sample_pipeline):
    sample_pipeline.start()
    results = []
    while True:
        pack = next(sample_pipeline.produce())
        if pack.signal == node.Node.Signal.DATA:
            if pack.values is not None:
                results.append(pack.values)
            if len(results) == 3 or pack.values is None:
                break
    sample_pipeline.stop()
    result = pd.concat(results, axis=0)
    np.testing.assert_array_equal(result.shape, [3, 8])
    sample_pipeline.get_op().shutdown()
