
DOMINANT_WRIST = 'DW'
NON_DOMINANT_WRIST = 'NDW'
DOMINANT_ANKLE = 'DA'
NON_DOMINANT_ANKLE = 'NDA'
DOMINANT_THIGH = 'DT'
NON_DOMINANT_THIGH = 'NDT'
DOMINANT_HIP = 'DH'
NON_DOMINANT_HIP = 'NDH'

TIMESTAMP_COL = 'HEADER_TIME_STAMP'
START_TIME_COL = 'START_TIME'
STOP_TIME_COL = 'STOP_TIME'
FEATURE_SET_TIMESTAMP_COLS = [TIMESTAMP_COL, START_TIME_COL, STOP_TIME_COL]
FEATURE_SET_PID_COL = 'PID'
FEATURE_SET_PLACEMENT_COL = 'PLACEMENT'
ANNOTATION_LABEL_COL = 'LABEL_NAME'

META_FOLDER = 'MetaCrossParticipants'
PROCESSED_FOLDER = 'DerivedCrossParticipants'
MASTER_FOLDER = "MasterSynced"
DERIVED_FOLDER = 'Derived'
SUBJECT_META_FOLDER = 'Meta'
SUBJECT_LOG_FOLDER = 'Logs'

META_LOCATION_MAPPING_FILENAME = 'location_mapping.csv'
META_SUBJECTS_FILENAME = 'subjects.csv'
META_CLASS_CATEGORY = 'class_category.csv'

FILE_TIMESTAMP_FORMAT = '%Y-%m-%d-%H-%M-%S-%f'

SENSOR_FILE_TYPE = 'sensor'
ANNOTATION_FILE_TYPE = 'annotation'
FEATURE_SET_FILE_TYPE = 'set'
FEATURE_FILE_TYPE = 'feature'
CLASS_FILE_TYPE = 'class'

FILE_TYPES = [SENSOR_FILE_TYPE, ANNOTATION_FILE_TYPE,
              FEATURE_SET_FILE_TYPE, FEATURE_FILE_TYPE, CLASS_FILE_TYPE]

MHEALTH_FLAT_FILEPATH_PATTERN = r'(\w+)[\/\\]{1}(?:(?:MasterSynced[\/\\]{1})|(?:Derived[\/\\]{1}(?:\w+[\/\\]{1})*))[0-9A-Za-z\-\.]+\.csv(\.gz)*'
MHEALTH_FILEPATH_PATTERN = r'(\w+)[\/\\]{1}(?:(?:MasterSynced[\/\\]{1})|(?:Derived[\/\\]{1}(?:\w+[\/\\]{1})*))\d{4}[\/\\]{1}\d{2}[\/\\]{1}\d{2}[\/\\]{1}\d{2}'
CAMELCASE_PATTERN = r'(?:[A-Z][A-Za-z0-9]+)+'
VERSIONCODE_PATTERN = r'(?:NA|[0-9x]+)'
SID_PATTERN = r'[A-Z0-9]+'
ANNOTATOR_PATTERN = r'[A-Za-z0-9]+'
FILE_TIMESTAMP_PATTERN = r'[0-9]{4}(?:\-[0-9]{2}){5}-[0-9]{3}-(?:P|M)[0-9]{4}'
FILE_EXTENSION_PATTERN = r'''
(?:sensor|event|log|annotation|feature|class|prediction|model|classmap)\.csv
'''
