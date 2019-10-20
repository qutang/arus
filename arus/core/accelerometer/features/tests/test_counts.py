import numpy as np
from .. import counts


def test_enmo():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = counts.enmo(X)
    assert np.allclose(result[0], np.sqrt(3) - 1, atol=0.001)
    assert result[1] == 'ENMO'

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = counts.enmo(X)
    assert np.allclose(result[0], np.nan, atol=0.001, equal_nan=True)
    assert result[1] == 'ENMO'

    # test on an array
    X = np.array([[1., 1., 1., ], [1., 1., 1.]])
    result = counts.enmo(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[np.sqrt(3) - 1]]))
    assert result[1] == 'ENMO'

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = counts.enmo(X)
    np.testing.assert_array_equal(result[0], np.array([[np.sqrt(3) - 1]]))
    assert result[1] == 'ENMO'
