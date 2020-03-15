
from .. import stream2
from .. import generator
from .. import segmentor
from .. import developer
from .. import o
import pytest


@pytest.fixture(params=['infinite', 'finite'])
def test_stream(request):
    if request.param == 'infinite':
        gr = generator.RandomAccelDataGenerator(50, buffer_size=50)
    else:
        gr = generator.RandomAccelDataGenerator(
            50, buffer_size=50, max_samples=300)
    seg = segmentor.SlidingWindowSegmentor(window_size=1)
    return stream2.Stream(gr, seg, name='{}-stream'.format(request.param))


def test_stream_lifecycle(test_stream):
    developer.set_default_logging()
    test_stream.start()
    results = []
    for pack in test_stream.get_result():
        if pack.signal == o.O.Signal.DATA:
            if pack.values is not None:
                results.append(pack)
            if len(results) == 10 or pack.values is None:
                break
    if 'infinite' in test_stream._name:
        assert len(results) == 10
    else:
        assert len(results) == 6
    test_stream.stop()
