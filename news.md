# 0.6.1+9000

## Added

* New libs module for plotting helpers. Added `adjust_lightness` for color manipulation.

## Changed

* Adjust metawear stream retry interval to one second.

## Fix

* The first window will also delay in real time when setting `simulate_reality` as `True` in streams.
* Better multi-threading performance and flow control for pipeline.

# 0.6.1

## Build

* Freeze arus demo app using PyInstaller.
* Now `arus.plugins.metawear` is extra and the requirements should be installed with extra `metawear`.
* Now arus demo app is optional and the requirements should be installed with extra `demo`.

## Refactoring

* Modulize arus demo app.

# 0.6.0

## Features

* Add module for training, validating and testing with [MUSS](https://qutang.github.io/MUSS/) activity recognition model.
* Add a sophisticated demo app to show case model training, real-time testing, active learning and active training.

## Refactoring

* Organize module imports with better and more convenient APIs.

## Documentation

* Discard pdoc3, use sphinx instead.
* Add examples using `sphinx_gallery`.