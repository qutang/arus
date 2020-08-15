import matplotlib.colors as mc
import colorsys
from matplotlib import rcParams, rcParamsDefault
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def adjust_lightness(color, amount=0.5, return_format='rgb'):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    result = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    if return_format == 'rgb':
        return result
    elif return_format == 'hex':
        return mc.to_hex(result)
    else:
        return result


def set_for_publication():
    rcParams.update(rcParamsDefault)
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    rcParams['font.serif'] = ['Times New Roman']


def confusion_matrix(cm_df):
    rcParams.update(rcParamsDefault)
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style({
        'font.family': 'serif',
        'font.size': 10
    })
    g = sns.heatmap(cm_df, annot=True, cmap="Greys",
                    cbar=False, fmt='d', robust=True, linewidths=0.2)
    g.set(xlabel="Prediction", ylabel="Ground truth")
    plt.tight_layout()
    return fig


def learning_curve(lc_df, title='Learning curve', axes=None, ylim=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes = lc_df['train_sizes']
    train_scores_mean = lc_df['train_scores_means']
    train_scores_std = lc_df['train_scores_stds']
    test_scores_mean = lc_df['test_scores_means']
    test_scores_std = lc_df['test_scores_stds']
    fit_times_mean = lc_df['fit_times_means']
    fit_times_std = lc_df['fit_times_stds']

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return fig
