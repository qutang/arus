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
from arus import dataset
import pandas as pd
from arus.models import muss
from arus import mhealth_format as mh
from arus import developer
import logging
import os

if __name__ == "__main__":

    # %%
    developer.set_default_logging()

    # %%
    # constants
    dataset_name = 'spades_lab'
    approach = 'muss'
    session_name = 'SPADESInLab'

    # %%
    # variables
    placements = ['DW', 'DA', 'DT']
    window_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    window_sizes = [2, 4, 6, 8, 10]

    # %%
    muss_model = muss.MUSSModel()
    spades_lab = dataset.load_dataset(dataset_name)

    # %% [markdown]
    # ### Different window sizes

    # %%
    # Process spades lab datasets with different window sizes
    for ws in window_sizes:
        logging.info(
            'Processing spades lab using muss with window size {}'.format(ws))
        df = dataset.process_mehealth_dataset(
            spades_lab, approach='muss', window_size=ws, sr=80)
        # cache data
        output_path = os.path.join(mh.get_processed_path(
            spades_lab['meta']['root']), 'muss_' + str(ws) + '.csv')
        df.to_csv(output_path, index=False)


# %%
# get and run validation on the dataset processed using standard
# MUSS algorithm (window size: 12.8s)
# d0 = pd.read_csv(spades_lab['processed']['muss'], parse_dates=[0, 1, 2])
# p_feature_sets = []
# for p in placements:
#     cond = d0['PLACEMENT'] == p
#     p_feature_set = d0.loc[cond, :]
#     del p_feature_set['PLACEMENT']
#     p_feature_sets.append(p_feature_set)
# d0_combined, feature_names = dataset.combine_multi_sensor_feature_sets(
#     *p_feature_sets,
#     placements=placements,
#     feature_names=muss_model.get_feature_names(),
#     fixed_cols=['PID', 'CLASS_LABEL_' + session_name]
# )


# # %%
# d0_filtered = dataset.filter_out_column_values(
#     d0_combined, selected_col='CLASS_LABEL_' + session_name, values_to_remove=['Unknown', 'Transition'])
# d0_features = dataset.select_feature_set_columns(
#     d0_filtered, selected_cols=feature_names, fixed_cols=['PID'])
# d0_classes = dataset.select_feature_set_columns(
#     d0_filtered, selected_cols=['CLASS_LABEL_' + session_name], fixed_cols=['PID'])
# d0_features.shape, d0_classes.shape


# # %%
# input_class_vec, output_class_vec, class_labels, acc = muss_model.validate_classifier(
#     d0_features, d0_classes, class_col='CLASS_LABEL_' + session_name, feature_names=feature_names, placement_names=placements, group_col='PID')
