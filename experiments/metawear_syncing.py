# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import arus
import pandas as pd
import numpy as np
import logging
import time
arus.dev.set_default_logging()


# %%
# Get device addr
scanner = arus.plugins.metawear.MetaWearScanner()
addrs = scanner.get_nearby_devices(max_devices=3)
addrs = addrs[0:3]


# %%
# prepare streams
streams = []
ref_st = pd.Timestamp.now()
for addr in addrs:
    generator = arus.plugins.metawear.MetaWearAccelDataGenerator(
        addr, sr=50, grange=8, buffer_size=100)
    segmentor = arus.segmentor.SlidingWindowSegmentor(
        window_size=2)
    stream = arus.Stream(generator, segmentor,
                         name='metawear-stream-{}'.format(addr))
    streams.append(stream)


# %%
# start stream
results = {}
for stream in streams:
    stream.start(start_time=ref_st)
    results[stream._name] = []


# %%
done = [False] * len(streams)
while True:
    for stream in streams:
        for data, st, et, _, _, name in stream.generate():
            data['ST'] = st
            data['ET'] = et
            i = streams.index(stream)
            logging.info('data from: {}, {}'.format(
                name, i))
            results[name].append(data)
            if len(results[name]) == 20:
                done[i] = True
            break
    if np.all(done):
        break

for stream in streams:
    stream.stop()

combined = []
for name, result in results.items():
    name = name.replace(':', '')
    result = pd.concat(result, sort=False)
    combined.append(result)

combined = pd.concat(combined, sort=False)

# %% [markdown]
#  ## Analyze results
#  ### Check the first sample timestamp
#  1. **referenced start time**: The start time used as a reference to segment the incoming data. This value is set before setting up and start the streams.
#  2. **first sample timestamp**: The timestamp of the first sample received from each device. This timestamp has been synced with the computer clock when receiving the data.
#  3. **first sample device timestamp**: The timestamp of the first sample received from each device. This timestamp has not been synced with the computer clock when receiving the data.

# %%
first_sample = combined.groupby(by=['MAC_ADDRESS']).apply(
    lambda df: df[['HEADER_TIME_STAMP', 'NO_FIX']].iloc[0, :])
first_sample = first_sample.rename(
    columns={'HEADER_TIME_STAMP': 'first sample timestamp', 'NO_FIX': 'first sample device timestamp'})
first_sample = first_sample.sort_values(by=['first sample timestamp'])
first_sample['referenced start time'] = ref_st
first_sample['diff_ref (s)'] = arus.Moment.get_durations(
    first_sample['referenced start time'], first_sample['first sample timestamp'])
first_sample['diff_device (s)'] = arus.Moment.get_durations(
    first_sample['first sample device timestamp'],
    first_sample['first sample timestamp'])
first_sample.transpose()

# %% [markdown]
#  ### Check number of samples in each segments

# %%
count_segment = combined.groupby(
    by=['MAC_ADDRESS', 'ST']).size().to_frame(name='# of samples').reset_index(drop=False)
count_segment = count_segment.pivot(
    columns='MAC_ADDRESS', index='ST', values=['# of samples'])
count_segment.columns = sorted(addrs)
count_segment.reset_index(drop=False, inplace=True)
count_segment.plot(x='ST', kind='line', ylim=(0, 210))


# %%
