# 0.6.3+9000

## Added

### Console

- Add a console to replace independent scripts to provide utility commands for developers.
- Developer console now supports
    - Bump package version
    - Build website and app
    - Compress dataset

### Arus core

- Add dataset module for sample dataset loading and reproducing.
- Add env module for managing environment variables and paths related to the package.

### Arus models

- Support replacement and combination LOSO validation with plugin-in data for muss model.
- Support draw on an existing figure for plot_confusion_matrix.

### Arus demo app

- Support validating updated model for arus demo app.
- Support snapshot and restore application states for different PIDs.
- Support set custom PID when starting a new session.
- Support real-time data summary in stream window.
- Support thigh placement.
- Support select placements for training, testing, validating and data collection.
- Support select device addrs for different placements via a dropdown list.

## Improved

### Arus demo app

- Stability and better code structure.

## Fix

### Arus models

- Now muss model combines feature sets for each placement correctly when there are more than two placements involved.

### Arus demo app

- Now new model is correctly trained with new data of user-added activitiy labels.
- Now collecting and training new data requires only one activity label being selected.

# 0.6.2

## Improved

- Better flow control for model testing in arus demo app.

## Added

- New libs module for plotting helpers. Added `adjust_lightness` for color manipulation.
- New data collection pipeline for muss model. Support both active training scheme and passive data collection (no need for a model).
- Data collection functionality for arus demo app.
- Model updating functionality for arus demo app.
- Saving and restoring application state for arus demo app.

## Changed

- Adjust metawear stream retry interval to one second.

## Fix

- The first window will also delay in real time when setting `simulate_reality` as `True` in streams.
- Better multi-threading performance and flow control for pipeline.
- Resolve infinite waiting bug when stopping generator stream.
- Make sure stream is non-blocking when waiting for incoming data. This had a chance causing infinit loop when stopping the stream.
- Make sure pipeline is restarting process pool every time it starts in case stop and start got called multiple times.

# 0.6.1

## Build

- Freeze arus demo app using PyInstaller.
- Now `arus.plugins.metawear` is extra and the requirements should be installed with extra `metawear`.
- Now arus demo app is optional and the requirements should be installed with extra `demo`.

## Refactoring

- Modulize arus demo app.

# 0.6.0

## Features

- Add module for training, validating and testing with [MUSS](https://qutang.github.io/MUSS/) activity recognition model.
- Add a sophisticated demo app to show case model training, real-time testing, active learning and active training.

## Refactoring

- Organize module imports with better and more convenient APIs.

## Documentation

- Discard pdoc3, use sphinx instead.
- Add examples using `sphinx_gallery`.
