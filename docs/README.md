# Introduction

__Arus__ python package provides a computation framework to manage and process ubiquitous sensory data for activity recognition.

[![PyPI version](https://badge.fury.io/py/arus.svg)](https://badge.fury.io/py/arus)
[![Downloads](https://pepy.tech/badge/arus)](https://pepy.tech/project/arus)
[![Build Status](https://github.com/qutang/arus/workflows/Continuous%20integration/badge.svg)](https://github.com/qutang/arus/actions)
[![codecov](https://codecov.io/gh/qutang/arus/branch/master/graph/badge.svg)](https://codecov.io/gh/qutang/arus)


## Prerequists

```bash
python >= 3.6
```

### Windows

For `arus[metawear]` extra, you need,

```bash
Visual Studio C++ SDK (v14.1)
Windows SDK (10.0.16299.0)
Windows SDK (10.0.17763.0)
```

### Ubuntu

For `arus[metawear]` extra, you need,

```bash
libbluetooth-dev
libboost-all-dev
bluez
```

## Installation

### Stable version

```bash
> pip install arus
```

### Development version

```bash
> pip install git+https://github.com/qutang/arus.git
```

## Extras

* `arus[metawear]`: This extra provides support for metawear devices.
    ```bash
    > pip install arus[metawear]
    ```
* `arus[demo]`: This extra provides support for a demo GUI program.
    ```bash
    > pip install arus[demo]
    ```

## For developer

### Prerequists

```bash
python >= 3.6
poetry >= 0.12.17
```

### Set up development environment

```bash
> git clone https://github.com/qutang/arus.git
> cd arus
> poetry install
```

### Run tests
```bash
> poetry run pytest
```