# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Developement of RT-MUSS algorithm
#
# RT-MUSS is an extension of MUSS algorithm. With the following changes,
#
# 1. Use a shorter window size (Evaluated in experiment 1)
# 2. Use a higher-level smoothing algorithm (Evaluated in experiment 2)
# %% [markdown]
# ## Performance of MUSS algorithm with data of different window sizes
#
# 1. The experiment will be done on triple-sensor (DW + DA + DT) model.
# 2. Metrics include F1-scores for each posture, inner- and inter-activity groups, and individual activities
# %% [markdown]
# ### By standard MUSS

# %%
# import modules
import pandas as pd
from arus.models import muss
from loguru import logger
import os
from sklearn import metrics
import numpy as np
import arus
import matplotlib.pyplot as plt

# %%

arus.dev.set_default_logger()

# constants
dataset_name = 'spades_lab'
approach = 'muss'
session_name = 'SPADESInLab'

# variables
placements = ['DW', 'DA', 'DT']
window_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
window_sizes = [2, 4, 6, 8, 10]

muss_model = muss.MUSSModel()
spades_lab = arus.dataset.load_dataset(dataset_name)

# %% [markdown]
# ### Different window sizes

# %%
# Process spades lab datasets with different window sizes


def prepare_dataset(window_sizes):
    dataset_paths = []
    for ws in window_sizes:
        output_path = os.path.join(arus.mh.get_processed_path(
            spades_lab['meta']['root']), 'muss_' + str(ws) + '.csv')
        dataset_paths.append(output_path)
        if os.path.exists(output_path):
            continue
        else:
            logger.info(
                'Processing spades lab using muss with window size {}'.format(ws))
            df = arus.dataset.process_mehealth_dataset(
                spades_lab, approach='muss', window_size=ws, sr=80)
            # cache data
            df.to_csv(output_path, index=False)
    return dataset_paths


dataset_paths = prepare_dataset(window_sizes)

# %%
# Load and prepare for validation


def prepare_validation(window_sizes, dataset_paths, spades_lab):
    d0 = pd.read_csv(spades_lab['processed']['muss'], parse_dates=[0, 1, 2])
    ds = [d0] + [pd.read_csv(path, parse_dates=[0, 1, 2])
                 for path in dataset_paths]
    window_sizes = [12.8] + window_sizes
    prepared_ds = []
    for d, ws in zip(ds, window_sizes):
        logger.info("Preparing dataset for window size: {}".format(ws))
        p_feature_sets = []
        for p in placements:
            cond = d['PLACEMENT'] == p
            p_feature_set = d.loc[cond, :]
            del p_feature_set['PLACEMENT']
            p_feature_sets.append(p_feature_set)
        logger.info('  merging placement features')
        prepared, feature_names = arus.ext.pandas.merge_all(
            *p_feature_sets,
            suffix_names=placements,
            suffix_cols=muss_model.get_feature_names(),
            on=arus.mh.FEATURE_SET_TIMESTAMP_COLS +
            [arus.mh.FEATURE_SET_PID_COL, 'CLASS_LABEL_' + session_name],
            how='inner',
            sort=False
        )
        prepared['POSTURES'] = arus.ext.pandas.fast_series_map(
            prepared['CLASS_LABEL_' + session_name],
            arus.mh.transform_class_category,
            class_category=spades_lab['meta']['class_category'],
            input_category='FINEST_ACTIVITIES',
            output_category='MUSS_3_POSTURES'
        )
        prepared['ACTIVITIES'] = arus.ext.pandas.fast_series_map(
            prepared['CLASS_LABEL_' + session_name],
            arus.mh.transform_class_category,
            class_category=spades_lab['meta']['class_category'],
            input_category='FINEST_ACTIVITIES',
            output_category='MUSS_22_ACTIVITIES'
        )
        logger.info('  filtering classes')
        prepared = arus.ext.pandas.filter_column(prepared,
                                                 'ACTIVITIES',
                                                 values_to_filter_out=['Unknown', 'Transition'])

        features = prepared[arus.mh.FEATURE_SET_TIMESTAMP_COLS +
                            feature_names + [arus.mh.FEATURE_SET_PID_COL]]
        classes = prepared[arus.mh.FEATURE_SET_TIMESTAMP_COLS +
                           ['ACTIVITIES', 'POSTURES'] + [arus.mh.FEATURE_SET_PID_COL]]

        prepared_ds.append((features, classes, feature_names, ws))
    return prepared_ds


prepared_ds = prepare_validation(window_sizes, dataset_paths, spades_lab)

# %%
# run validations for activities


def run_validation(prepared_ds, class_category):
    results = {}
    for prepared in prepared_ds:
        muss_model = muss.MUSSModel()
        logger.info('Validating for window size: {} and category: {}'.format(
            prepared[3], class_category))
        input_class_vec, output_class_vec, class_labels, acc = muss_model.validate_classifier(
            prepared[0], prepared[1], class_col=class_category, feature_names=prepared[2], placement_names=placements, group_col=arus.mh.FEATURE_SET_PID_COL)
        scores = metrics.f1_score(input_class_vec, output_class_vec,
                                  labels=class_labels, average=None)
        mean_score = np.mean(scores)
        if 'window_size' not in results:
            results['window_size'] = []
            results[class_category] = []
            for label in class_labels:
                results[class_category + '_' + label] = []
        results['window_size'].append(prepared[3])
        results[class_category].append(mean_score)
        for label, score in zip(class_labels, scores):
            results[class_category + '_' + label].append(score)
    result = pd.DataFrame.from_dict(results)
    return result


result1 = run_validation(
    prepared_ds, class_category='POSTURES')
result2 = run_validation(
    prepared_ds, class_category='ACTIVITIES')

result = result1.merge(result2, on=['window_size'], how='inner')
result = result.sort_values(by=['window_size'])
result.to_csv(os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'rt_muss.csv'), index=False)

# %%
result_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'rt_muss.csv')

results = pd.read_csv(result_path)
sit_scores = []
stand_scores = []
amb_scores = []
for col in results.columns:
    col_lower = col.lower()
    if 'sit' in col_lower:
        sit_scores.append(results[col].values)
    elif 'stand' in col_lower:
        stand_scores.append(results[col].values)
    elif 'walk' in col_lower:
        amb_scores.append(results[col].values)
    elif 'ly' in col_lower:
        results['ACTIVITIES_Lying'] = results[col].values
    elif 'run' in col_lower:
        results['ACTIVITIES_Run'] = results[col].values
    elif 'cycle' in col_lower:
        results['ACTIVITIES_Bike'] = results[col].values

results['ACTIVITIES_Sit'] = np.mean(np.array(sit_scores), axis=0)
results['ACTIVITIES_Stand'] = np.mean(np.array(stand_scores), axis=0)
results['ACTIVITIES_Amb'] = np.mean(np.array(amb_scores), axis=0)
# %%
# plot
line_styles = ['-'] * 4 + ['--'] * 7
ax = results.plot(x='window_size', y=['POSTURES', 'POSTURES_Lying',
                                      'POSTURES_Upright', 'POSTURES_Sitting', 'ACTIVITIES', 'ACTIVITIES_Sit', 'ACTIVITIES_Stand', 'ACTIVITIES_Amb', 'ACTIVITIES_Lying', 'ACTIVITIES_Run', 'ACTIVITIES_Bike'], kind='line', figsize=(8, 10), style=line_styles, fontsize=16)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          shadow=True, ncol=2)
fig_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'rt_muss_ws_details.png')
plt.tight_layout()
plt.savefig(fig_path, dpi=300)

ax = results.plot(x='window_size', y=[
                  'POSTURES', 'ACTIVITIES'], kind='line', figsize=(8, 8), style=line_styles, fontsize=16)
fig_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'rt_muss_ws.png')
plt.tight_layout()
plt.savefig(fig_path, dpi=300)

# %%
