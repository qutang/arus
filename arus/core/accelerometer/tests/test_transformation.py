import numpy as np
from .. import transformation as tr


def test_vector_magnitude():
    # test with a single row of data
    X = np.array([[1., 1., 1.]])
    result = tr.vector_magnitude(X)
    assert np.allclose(result, np.sqrt(3), atol=0.001)
    # test with an array
    X = np.array([[1., 1., 1.], [1., 1., 1.]])
    result = tr.vector_magnitude(X)
    assert np.allclose(result, np.array(
        [[np.sqrt(3)], [np.sqrt(3)]]), atol=0.001)
    # test with NaN data
    X = np.array([[1., np.nan, 1.]])
    result = tr.vector_magnitude(X)
    assert np.isnan(result)


def test_flip_and_swap():
    # test on 1d data
    # test with flip
    X = np.array([[1., 1., 1.]])
    result = tr.flip_and_swap(X, x_flip='x', y_flip='y', z_flip='-z')
    assert np.allclose(result, np.array([[1., 1., -1.]]), atol=0.001)
    # test with swap
    X = np.array([[1., 2., 3.]])
    result = tr.flip_and_swap(X, x_flip='y', y_flip='x', z_flip='z')
    assert np.allclose(result, np.array([[2., 1., 3.]]), atol=0.001)
    # test with flip and swap
    X = np.array([[1., 2., 3.]])
    result = tr.flip_and_swap(X, x_flip='y', y_flip='-x', z_flip='z')
    assert np.allclose(result, np.array([[2., -1., 3.]]), atol=0.001)
    # test on 2d data
    X = np.array([[1., 1., 1.], [1, 2, 3]])
    result = tr.flip_and_swap(X, x_flip='y', y_flip='-x', z_flip='z')
    assert np.allclose(result, np.array(
        [[1., -1., 1.], [2., -1., 3.]]), atol=0.001)
