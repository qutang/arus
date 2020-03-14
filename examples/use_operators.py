# %%
import arus
import logging
arus.dev.set_default_logging()

# %% [markdown]
# ## input operators

# %% [markdown]
# ### generator operator

# %%
generator = arus.generator.RandomAccelDataGenerator(sr=80, buffer_size=10)
op = arus.O(generator, name='accel', t=arus.O.Type.INPUT)

op.start()
i = 0
while True:
    data = next(op.produce())
    if data.signal == arus.O.Signal.WAIT:
        continue
    logging.info(data)
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
op = arus.O(generator, name=addrs[0], t=arus.O.Type.INPUT)

op.start()
i = 0
while True:
    data = next(op.produce())
    if data.signal == arus.O.Signal.WAIT:
        continue
    logging.info(data)
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

op = arus.O(generator, name='SPADES_2-DW', t=arus.O.Type.INPUT)

op.start()
i = 0
while True:
    data = next(op.produce())
    if data.signal == arus.O.Signal.WAIT:
        continue
    logging.info(data)
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
values, context = next(generator.generate())
generator.stop()
segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=1)
op = arus.O(segmentor, name='segmentor-1s')

op.start()
op.consume(arus.O.Pack(values=values, signal=arus.O.Signal.DATA,
                       context=context, src='SPADES_2-dw'))
i = 0
while True:
    data = next(op.produce())
    if data.signal == arus.O.Signal.WAIT:
        continue
    logging.info(data)
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
dw_values, dw_context = next(dw_generator.generate())
da_values, da_context = next(da_generator.generate())
dw_context = {**dw_context, 'data_id': 'DW'}
da_context = {**da_context, 'data_id': 'DA'}
dw_generator.stop()
da_generator.stop()

segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=1)
seg_dw_values, dw_context = next(segmentor.generate(
    dw_values, src=None, context=dw_context))
segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=1)
seg_da_values, da_context = next(segmentor.generate(
    da_values, src=None, context=da_context))

synchronizer = arus.synchronizer.Synchronizer()
synchronizer.add_sources(2)
op = arus.O(synchronizer, name='sync-dw-da')
op.start()
op.consume(arus.O.Pack(values=seg_dw_values,
                       signal=arus.O.Signal.DATA, context=dw_context, src='dw'))
op.consume(arus.O.Pack(values=seg_da_values,
                       signal=arus.O.Signal.DATA, context=da_context, src='da'))
while True:
    data = next(op.produce())
    if data.signal == arus.O.Signal.WAIT:
        continue
    logging.info(data)
    break
op.stop()
