
```bash
arus signaligner FOLDER PID [SR] [-t <file_type>] [--date_range=<date_range>] [--auto_range=<auto_range>] [--debug]
arus app APP_COMMAND FOLDER NAME [--app_version=<app_version>]
arus dataset DATASET_COMMAND DATASET_NAME [FOLDER] [OUTPUT_FOLDER] [--debug]
arus package PACK_COMMAND [NEW_VERSION] [--dev] [--release]
arus --help
arus --version
```

## Arguments

| Argument       | Description                                            |
|----------------|--------------------------------------------------------|
| `FOLDER`       | Dataset folder.                                        |
| `PID`          | Participant ID.                                        |
| `SR`           | Sampling rate in Hz.                                   |
| `APP_COMMAND`  | Sub commands for app command. Either "build" or "run". |
| `NAME`         | Name of the app.                                       |
| `PACK_COMMAND` | "release", "docs"                                      |
| `NEW_VERSION`  | "major", "minor", "patch" or number.                   |

## Options

* `-t <file_type>, --file_type=<file_type>`: File type: either "sensor" or "annotation". If omit, both are included.  
* `--date_range=<date_range>`: Date range. E.g., "--date_range 2020-06-01,2020-06-10", or "--date_range 2020-06-01," or "--date_range ,2020-06-10".  
* `--auto_range=<auto_range>`: Auto date freq. Default is "W-SUN", or weekly starting from Sunday.  
* `--app_version=<app_version>`: App version. If omit, default is the same as the version of arus package.  
* `-h, --help`: Show help message.  
* `-v, --version`: Program/app version.  