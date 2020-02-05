
from .. import generator
import numpy as np
import pandas as pd
import time


def test_mHealthSensorFileGenerator(spades_lab):
    sensor_files = spades_lab['subjects']['SPADES_1']['sensors']['DW']
    sizes = []
    gen = generator.MhealthSensorFileGenerator(
        *sensor_files, buffer_size=1800)
    gen_data = gen.generate()
    for data in gen_data:
        assert type(data) == pd.DataFrame
        sizes.append(data.shape[0])
    sizes = sizes[:-1]
    assert np.all(np.array(sizes) == 1800)

    sizes = []
    gen = generator.MhealthSensorFileGenerator(
        *sensor_files, buffer_size=1800)
    gen_data = gen.generate()
    while True:
        try:
            data = next(gen_data)
            assert type(data) == pd.DataFrame
            sizes.append(data.shape[0])
        except StopIteration:
            break
    sizes = sizes[:-1]
    assert np.all(np.array(sizes) == 1800)


def test_mHealthAnnotationFileGenerator(spades_lab):
    annotation_files = spades_lab['subjects']['SPADES_1']['annotations']['SPADESInLab']
    sizes = []
    gen = generator.MhealthAnnotationFileGenerator(
        *annotation_files, buffer_size=5)
    gen_data = gen.generate()
    for data in gen_data:
        assert type(data) == pd.DataFrame
        sizes.append(data.shape[0])
    sizes = sizes[:-1]
    assert np.all(np.array(sizes) == 5)

    sizes = []
    gen = generator.MhealthAnnotationFileGenerator(
        *annotation_files, buffer_size=5)
    gen_data = gen.generate()
    while True:
        try:
            data = next(gen_data)
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
    gen_data = gen.generate()
    for data in gen_data:
        mean_data = np.mean(data.values[:, 1:], axis=0)
        std_data = np.std(data.iloc[:, 1:].values, axis=0)
        duration = (data.iloc[-1, 0] - data.iloc[0, 0]) / \
            pd.Timedelta(1, unit='seconds')
        np.testing.assert_array_almost_equal(
            mean_data, np.array([0, 0, 0]), decimal=1)
        np.testing.assert_array_almost_equal(
            std_data, np.array([sigma, sigma, sigma]),  decimal=1)
        np.testing.assert_almost_equal(duration, buffer_size / sr, decimal=1)


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
    gen_data = gen.generate()
    for data in gen_data:
        durations += ((data['STOP_TIME'] - data['START_TIME']
                       )/pd.Timedelta(1, 'S')).values.tolist()
        rows.append(data.shape[0])
    duration_mean = np.mean(durations)
    rows_mean = np.mean(rows)
    np.testing.assert_almost_equal(duration_mean, duration_mu, decimal=0)
    np.testing.assert_almost_equal(rows_mean, num_mu, decimal=0)
