
from .. import stream
from .. import generator
from .. import segmentor
from .. import developer
import pytest


@pytest.fixture(params=['infinite', 'finite'])
def test_stream(request):
    if request.param == 'infinite':
        gr = generator.RandomAccelDataGenerator(50, buffer_size=50)
    else:
        gr = generator.RandomAccelDataGenerator(
            50, buffer_size=50, max_samples=300)
    seg = segmentor.SlidingWindowSegmentor(window_size=1)
    return stream.Stream(gr, seg, name='{}-stream'.format(request.param))


def test_stream_lifecycle(test_stream):
    developer.set_default_logging()
    assert test_stream.get_status() == stream.Stream.Status.NOT_START
    test_stream.start()
    assert test_stream.get_status() == stream.Stream.Status.RUN
    assert test_stream._loading_thread.isAlive()
    assert test_stream._segment_thread.isAlive()
    results = []
    for sample, st, et, prev_st, prev_et, name in test_stream.generate():
        results.append(sample)
        if len(results) == 10:
            break
    if 'infinite' in test_stream._name:
        assert len(results) == 10
    else:
        assert len(results) == 6
    test_stream.stop()
    assert test_stream.get_status() == stream.Stream.Status.NOT_START
    assert not test_stream._loading_thread.isAlive()
    assert not test_stream._segment_thread.isAlive()
