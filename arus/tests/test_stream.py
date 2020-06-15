
from .. import stream2
from .. import generator
from .. import segmentor
from .. import developer
from .. import node
import pytest


@pytest.fixture(params=['infinite', 'finite'])
def test_stream(request):
    if request.param == 'infinite':
        gr = generator.RandomAccelDataGenerator(50, buffer_size=50)
    else:
        gr = generator.RandomAccelDataGenerator(
            50, buffer_size=50, max_samples=300)
    seg = segmentor.SlidingWindowSegmentor(window_size=1)
    stream_op = stream2.Stream(gr, seg, name='{}-stream'.format(request.param))
    stream_op.set_context(ref_start_time=None, data_id='DW')
    stream = node.Node(
        op=stream_op,
        t=node.Node.Type.INPUT, name='{}-stream'.format(request.param))
    return stream


def test_stream_lifecycle(test_stream):
    test_stream.start()
    results = []
    while True:
        pack = next(test_stream.produce())
        if pack.signal == node.Node.Signal.DATA:
            if pack.values is not None:
                assert 'start_time' in pack.context
                assert 'data_id' in pack.context
                results.append(pack)
            if len(results) == 10 or pack.values is None:
                break
    if 'infinite' in test_stream._name:
        assert len(results) == 10
    else:
        assert len(results) == 6
    test_stream.stop()
