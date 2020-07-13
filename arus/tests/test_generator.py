
from .. import generator
from .. import moment
import numpy as np
import pandas as pd
import time


def test_mHealthSensorFileGenerator(spades_lab_data):
    sensor_files = [spades_lab_data['subjects']
                    ['SPADES_1']['sensors']['DW'][0]]
    sizes = []
    gen = generator.MhealthSensorFileGenerator(
        *sensor_files, buffer_size=18000)
    gen.run()
    for data, _ in gen.get_result():
        if data is None:
            break
        assert type(data) == pd.DataFrame
        sizes.append(data.shape[0])
    sizes = sizes[:-1]
    assert np.all(np.array(sizes) == 18000)

    sizes = []
    gen = generator.MhealthSensorFileGenerator(
        *sensor_files, buffer_size=18000)
    gen.run()
    while True:
        try:
            data, _ = next(gen.get_result())
            if data is None:
                break
            assert type(data) == pd.DataFrame
            sizes.append(data.shape[0])
        except StopIteration:
            break
    sizes = sizes[:-1]
    assert np.all(np.array(sizes) == 18000)


def test_mHealthAnnotationFileGenerator(spades_lab_data):
    annotation_files = spades_lab_data['subjects']['SPADES_1']['annotations']['SPADESInLab']
    sizes = []
    gen = generator.MhealthAnnotationFileGenerator(
        *annotation_files, buffer_size=5)
    gen.run()
    for data, _ in gen.get_result():
        if data is None:
            break
        assert type(data) == pd.DataFrame
        sizes.append(data.shape[0])
    sizes = sizes[:-1]
    assert np.all(np.array(sizes) == 5)

    sizes = []
    gen = generator.MhealthAnnotationFileGenerator(
        *annotation_files, buffer_size=5)
    gen.run()
    while True:
        try:
            data, _ = next(gen.get_result())
            if data is None:
                break
            assert type(data) == pd.DataFrame
            sizes.append(data.shape[0])
        except StopIteration:
            break
    sizes = sizes[:-1]
    assert np.all(np.array(sizes) == 5)


def test_RandomAccelDataGenerator():
    # default setting
    sr = 3600
    grange = 4
    start_time = None
    buffer_size = int(3600 / 2)
    sigma = 1
    max_samples = buffer_size * 2
    gen = generator.RandomAccelDataGenerator(
        sr=sr,
        grange=grange,
        st=start_time,
        buffer_size=buffer_size,
        sigma=sigma,
        max_samples=max_samples)
    gen.run()
    for data, _ in gen.get_result():
        if data is None:
            break
        mean_data = np.mean(data.values[:, 1:], axis=0)
        std_data = np.std(data.iloc[:, 1:].values, axis=0)
        duration = (data.iloc[-1, 0] - data.iloc[0, 0]) / \
            pd.Timedelta(1, unit='seconds')
        np.testing.assert_array_almost_equal(
            mean_data, np.array([0, 0, 0]), decimal=1)
        np.testing.assert_array_almost_equal(
            std_data, np.array([sigma, sigma, sigma]),  decimal=1)
        np.testing.assert_almost_equal(
            duration, buffer_size / sr, decimal=1)


def test_RandomAnnotationDataGenerator():
    # default setting
    duration_mu = 5
    duration_sigma = 1
    start_time = None
    num_mu = 3
    labels = ['Sitting', 'Standing', 'Lying']
    max_samples = 50

    durations = []
    rows = []
    gen = generator.RandomAnnotationDataGenerator(labels=labels,
                                                  duration_mu=duration_mu, duration_sigma=duration_sigma, st=start_time, num_mu=num_mu, max_samples=max_samples)
    gen.run()
    for data, _ in gen.get_result():
        if data is None:
            break
        durations += moment.Moment.get_durations(
            data['START_TIME'], data['STOP_TIME'], unit='s')
        rows.append(data.shape[0])
    duration_mean = np.mean(durations)
    rows_mean = np.mean(rows)
    np.testing.assert_almost_equal(duration_mean, duration_mu, decimal=0)
    np.testing.assert_almost_equal(rows_mean, num_mu, decimal=0)
