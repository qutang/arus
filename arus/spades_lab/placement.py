from .. import mhealth_format as mh

MAPPING = {
    'dominant wrist': mh.DOMINANT_WRIST,
    'non dominant wrist': mh.NON_DOMINANT_WRIST,
    'dominant ankle': mh.DOMINANT_ANKLE,
    'non dominant ankle': mh.NON_DOMINANT_ANKLE,
    'dominant hip': mh.DOMINANT_HIP,
    'non dominant hip': mh.NON_DOMINANT_HIP,
    'dominant thigh': mh.DOMINANT_THIGH
}


def get_sensor_placement(dataset_path, pid, sid):
    p = mh.get_sensor_placement(dataset_path, pid, sid)
    return MAPPING[p]
