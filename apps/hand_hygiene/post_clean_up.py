from loguru import logger
import glob
import pandas as pd
import os
import arus


def convert_to_mhealth(root, pid):
    _convert_expert_annotations(root, pid)


def _read_side_annotation_file(filepath):
    raw_df = pd.read_csv(filepath, header=0,
                         infer_datetime_format=True, parse_dates=[0, 1])
    raw_df.insert(0, 'HEADER_TIME_STAMP', raw_df['START_TIME'])
    raw_df = raw_df.rename(columns={'PREDICTION': 'LABEL_NAME'})
    side_annot_df = raw_df.loc[:, ['HEADER_TIME_STAMP',
                                   'START_TIME', 'STOP_TIME', 'LABEL_NAME']]
    return side_annot_df


def _convert_expert_annotations(root, pid):
    logger.info(
        "Convert expert annotations to mhealth format for hand hygiene raw dataset")

    side_annotation_files = glob.glob(os.path.join(
        root, pid, "OriginalRaw", "*Side*annotation.csv"), recursive=True)

    if len(side_annotation_files) == 1:
        side_annot_df = _read_side_annotation_file(side_annotation_files[0])
    elif len(side_annotation_files) == 0:
        side_annot_df = None
    else:
        logger.warning(
            "Multiple expert annotated hand side information is found.")
        side_annot_df = _read_side_annotation_file(side_annotation_files[0])
    if side_annot_df is not None:
        writer = arus.mh.MhealthFileWriter(
            root, pid, hourly=True, date_folders=True)
        writer.set_for_annotation("HandHygieneSide", "Expert")
        writer.write_csv(side_annot_df, append=False, block=True)
    else:
        logger.warning(
            "No expert annotated hand side information is found, skip this task.")


if __name__ == "__main__":
    convert_to_mhealth('D:/datasets/hand_hygiene', 'P1-1')
