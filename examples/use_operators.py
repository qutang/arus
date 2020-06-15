# %%

import arus
from loguru import logger
arus.dev.set_default_logger()

# %% [markdown]
# ## input operators

# %% [markdown]
# ### generator operator

# %%
generator = arus.generator.RandomAccelDataGenerator(sr=80, buffer_size=10)
op = arus.Node(generator, name='accel', t=arus.Node.Type.INPUT)

op.start()
i = 0
while True:
    data = next(op.produce())
    if data.signal == arus.Node.Signal.WAIT:
        continue
    logger.info(data)
    if i == 2:
        break
    i += 1
op.stop()

# %% [markdown]
# ### metawear generator operator

# %%
scanner = arus.plugins.metawear.MetaWearScanner()
addrs = scanner.get_nearby_devices(max_devices=1)
generator = arus.plugins.metawear.MetaWearAccelDataGenerator(
    addrs[0], sr=50, grange=8, buffer_size=10)
op = arus.Node(generator, name=addrs[0], t=arus.Node.Type.INPUT)

op.start()
i = 0
while True:
    data = next(op.produce())
    if data.signal == arus.Node.Signal.WAIT:
        continue
    logger.info(data)
    if i == 2:
        break
    i += 1
op.stop()

# %% [markdown]
# ### mhealth file generator operator

# %%
spades_lab = arus.dataset.load_dataset('spades_lab')
mhealth_filepath = spades_lab['subjects']['SPADES_2']['sensors']['DW'][0]
generator = arus.generator.MhealthSensorFileGenerator(
    mhealth_filepath, buffer_size=10)

op = arus.Node(generator, name='SPADES_2-DW', t=arus.Node.Type.INPUT)

op.start()
i = 0
while True:
    data = next(op.produce())
    if data.signal == arus.Node.Signal.WAIT:
        continue
    logger.info(data)
    if i == 2:
        break
    i += 1
op.stop()

# %% [markdown]
# ## pipe operators

# %% [markdown]
# ### segmentor operator

# %%
spades_lab = arus.dataset.load_dataset('spades_lab')
mhealth_filepath = spades_lab['subjects']['SPADES_2']['sensors']['DW'][0]
generator = arus.generator.MhealthSensorFileGenerator(
    mhealth_filepath, buffer_size=1800)
generator.run()
values, context = next(generator.get_result())
generator.stop()
segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=1)
op = arus.Node(segmentor, name='segmentor-1s')

op.start()
op.consume(arus.Node.Pack(values=values, signal=arus.Node.Signal.DATA,
                          context=context, src='SPADES_2-dw'))
i = 0
while True:
    data = next(op.produce())
    if data.signal == arus.Node.Signal.WAIT:
        continue
    logger.info(data)
    if i == 2:
        break
    i += 1
op.stop()


# %% [markdown]
# ### synchronizer operator


# %%
spades_lab = arus.dataset.load_dataset('spades_lab')
dw_filepath = spades_lab['subjects']['SPADES_2']['sensors']['DW'][0]
da_filepath = spades_lab['subjects']['SPADES_2']['sensors']['DA'][0]
dw_generator = arus.generator.MhealthSensorFileGenerator(
    dw_filepath, buffer_size=1800)
da_generator = arus.generator.MhealthSensorFileGenerator(
    da_filepath, buffer_size=1800)
dw_generator.run()
da_generator.run()
dw_values, dw_context = next(dw_generator.get_result())
da_values, da_context = next(da_generator.get_result())
dw_context = {**dw_context, 'data_id': 'DW'}
da_context = {**da_context, 'data_id': 'DA'}
dw_generator.stop()
da_generator.stop()

segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=1)
segmentor.run(
    dw_values, src=None, context=dw_context)
seg_dw_values, dw_context = next(segmentor.get_result())
segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=1)
segmentor.run(
    da_values, src=None, context=da_context)
seg_da_values, da_context = next(segmentor.get_result())

synchronizer = arus.synchronizer.Synchronizer()
synchronizer.add_sources(2)
op = arus.Node(synchronizer, name='sync-dw-da')
op.start()
op.consume(arus.Node.Pack(values=seg_dw_values,
                          signal=arus.Node.Signal.DATA, context=dw_context, src='dw'))
op.consume(arus.Node.Pack(values=seg_da_values,
                          signal=arus.Node.Signal.DATA, context=da_context, src='da'))
while True:
    data = next(op.produce())
    if data.signal == arus.Node.Signal.WAIT:
        continue
    logger.info(data)
    break
op.stop()


# %% [markdown]
# ### scheduler/processor operator


# %%
spades_lab = arus.dataset.load_dataset('spades_lab')
dw_filepath = spades_lab['subjects']['SPADES_2']['sensors']['DW'][0]
da_filepath = spades_lab['subjects']['SPADES_2']['sensors']['DA'][0]
dw_generator = arus.generator.MhealthSensorFileGenerator(
    dw_filepath, buffer_size=1800)
dw_generator.run()
dw_values, dw_context = next(dw_generator.get_result())
dw_context = {**dw_context, 'data_id': 'DW'}
dw_generator.stop()


def compute_mean(values, src=None, **context):
    import numpy as np
    data = values.values[:, 1:]
    result = np.mean(data, axis=0)
    return result, context


processor = arus.processor.Processor(compute_mean,
                                     mode=arus.Scheduler.Mode.THREAD,
                                     scheme=arus.Scheduler.Scheme.SUBMIT_ORDER, max_workers=2)

op = arus.Node(processor, name='compute-mean')
op.start()
op.consume(arus.Node.Pack(values=dw_values, signal=arus.Node.Signal.DATA,
                          context=dw_context, src='dw'))
while True:
    data = next(op.produce())
    if data.signal == arus.Node.Signal.WAIT:
        continue
    logger.info(data)
    break
op.stop()
