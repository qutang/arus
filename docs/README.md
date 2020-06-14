# Overview

__ARUS__ python package provides a computational and experimental framework to manage and process sensory data or wireless devices, to develop and run activity recognition algorithms on the data, and to create interactive programs using the algorithms and wireless devices.

This package is licensed under [GNU version 3](/LICENSE).

[![PyPI version](https://badge.fury.io/py/arus.svg)](https://badge.fury.io/py/arus)  
[![Downloads](https://pepy.tech/badge/arus)](https://pepy.tech/project/arus)  
[![deployment build](https://github.com/qutang/arus/workflows/deploy/badge.svg)](https://github.com/qutang/arus/actions?query=workflow%3Adeploy)  
[![unittest and build test](https://github.com/qutang/arus/workflows/unittest%20and%20build%20test/badge.svg)](https://github.com/qutang/arus/actions?query=workflow%3A%22unittest+and+build+test%22)  
[![codecov](https://codecov.io/gh/qutang/arus/branch/master/graph/badge.svg)](https://codecov.io/gh/qutang/arus)  

## Prerequists

```bash
python >= 3.6.1
```

```bash
# Need these SDKs to install arus[metawear] on Windows.
Visual Studio C++ SDK (v14.1)
Windows SDK (10.0.16299.0)
Windows SDK (10.0.17763.0)

# Need these packages to install arus[metawear] on Ubuntu or equivalent packages on other linux distributions.
libbluetooth-dev
libboost-all-dev
bluez
```

## Installation

```bash
> pip install arus
# Optionally, you may install plugins via pip extra syntax.
> pip install arus[metawear]
> pip install arus[demo]
```

## Optional components

`arus[metawear]`: This optional component installs dependency supports for streaming data from Bluetooth metawear sensors.

`arus[demo]`: This optional component installs dependency supports for running the demo app that demonstrates a real-time interactive activity recognition training and testing program.

## Get started for development

```bash
> git clone https://github.com/qutang/arus.git
> cd arus
> # Install poetry python package management tool https://python-poetry.org/docs/
> # On Linux
> curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
> # On windows powershell
> (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
> # Install package dependencies
> poetry install
> # Install optional component dependencies
> poetry install --extras "metawear demo"
> # Run unit tests
> poetry run pytest
```
