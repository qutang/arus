import numpy as np
import pytest

from .. import dataset as ds
from .. import feature_vector as fv
from .. import mhealth_format as mh


@pytest.fixture
def test_data(single_mhealth_sensor):
    data = single_mhealth_sensor.iloc[:1000, :]
    st = data.iloc[0, 0]
    et = data.iloc[-1, 0]
    sr = 80
    return data, st, et, sr


@ pytest.mark.parametrize('use_vm', [True, False])
def test_inertial_single_triaxial(test_data, use_vm):

    raw_df, st, et, sr = test_data

    fv_df, fv_names = fv.inertial.single_triaxial(
        raw_df, sr=sr, st=st, et=et, subwin_secs=2, ori_unit='rad', activation_threshold=0.2, use_vm=use_vm)

    feature_names = fv.inertial.assemble_fv_names(use_vm=use_vm)
    np.testing.assert_array_equal(fv_names, feature_names)

    assert fv_df.shape[0] == 1
    assert np.sum(fv_df.notna().values) + \
        (2 if use_vm else 3) == fv_df.shape[1]
