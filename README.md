## Overview

__arus__ python package provides a computation framework to manage and process ubiquitous sensory data for activity recognition.

[![PyPI version](https://badge.fury.io/py/arus.svg)](https://badge.fury.io/py/arus)
[![Downloads](https://pepy.tech/badge/arus)](https://pepy.tech/project/arus)
[![Build Status](https://github.com/qutang/arus/workflows/Continuous%20integration/badge.svg)](https://github.com/qutang/arus/actions)
[![codecov](https://codecov.io/gh/qutang/arus/branch/master/graph/badge.svg)](https://codecov.io/gh/qutang/arus)


## Get started

#### Prerequists

```bash
python >= 3.6
```

#### Installation

```bash
> pip install arus
```

or with `pipenv`

```bash
> pipenv install arus
```

or with `poetry`

```bash
> poetry add arus
```

### Plugins

`arus` packages can be extended to involve more file types, Bluetooth devices or network protocols. The following list includes official plugins that extend `arus`.

#### Official plugins

1. [`arus-stream-metawear`](https://qutang.github.io/arus-stream-metawear/): a `Stream` class that can be used to stream data acquired from a Metawear device (A wireless Bluetooth sensor) in real-time.
2. [`arus-components-hci`](https://qutang.github.io/arus-components-hci/): plugins that include streams, pipelines and broadcasters connected to some HCI interface, such as GUI, headphones and haptical devices.


### Development

#### Prerequists

```bash
python >= 3.6
poetry >= 0.12.17
```

#### Set up development environment

```bash
> git clone https://github.com/qutang/arus.git
> cd arus
> poetry install
```