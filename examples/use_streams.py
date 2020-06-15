
# %% [markdown]
# ## Import dependencies

# %%
import arus
from loguru import logger

# %% [markdown]
# ## stream with random data generator

# %%
arus.dev.set_default_logger()

gr = arus.generator.RandomAccelDataGenerator(
    50, buffer_size=25, max_samples=75)
seg = arus.segmentor.SlidingWindowSegmentor(window_size=1)

stream = arus.stream2.Stream(
    generator=gr, segmentor=seg, name='random-accel-stream')

stream.start()

i = 0
for data in stream.get_result():
    if data.signal == arus.Node.Signal.DATA:
        logger.info(data)
        if i == 3 or data.values is None:
            break
        i += 1

stream.stop()

# %% [markdown]
# ## stream with metawear generator

# %%
arus.dev.set_default_logger()

addrs = arus.plugins.metawear.MetaWearScanner().get_nearby_devices(max_devices=1)
gr = arus.plugins.metawear.MetaWearAccelDataGenerator(
    addrs[0], sr=50, grange=8, buffer_size=25)
seg = arus.segmentor.SlidingWindowSegmentor(window_size=1)

stream = arus.stream2.Stream(
    generator=gr, segmentor=seg, name='metawear-stream')

stream.start()

i = 0
for data in stream.get_result():
    if data.signal == arus.Node.Signal.DATA:
        logger.info(data)
        if i == 3 or data.values is None:
            break
        i += 1

stream.stop()
