from .. import mhealth_format as mh


def test_transform_class_category(spades_lab):
    class_category = spades_lab['meta']['class_category']
    input_category = 'FINEST_ACTIVITIES'
    output_category = 'MUSS_3_POSTURES'
    input_label = 'Lying on the back'
    assert mh.transform_class_category(
        input_label, class_category, input_category, output_category) == 'Lying'
