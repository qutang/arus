"""Module includes classes that loads external data sources (e.g., file, network port, socket, user inputs and etc.) into a data queue using a separate thread.

# Usage of `arus.core.stream.SensorFileStream`

# On mhealth sensor files

```python
.. include:: ../../../examples/mhealth_stream.py
```

# On an Actigraph sensor file

```python
.. include:: ../../../examples/actigraph_stream.py
```

# On mhealth sensor files with real-time delay

```python
.. include:: ../../../examples/sensor_stream_simulated_reality.py
```

# Usage of `arus.core.stream.AnnotationFileStream`

```python
.. include:: ../../../examples/annotation_stream.py
```

Author: Qu Tang

Date: 2019-11-15

License: see LICENSE file
"""

from .base_stream import *
from .sensor_stream import *
from .generator_stream import *
from .annotation_stream import *
