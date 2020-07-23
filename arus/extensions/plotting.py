import matplotlib.colors as mc
import colorsys
from matplotlib import rcParams, rcParamsDefault
from matplotlib import pyplot as plt
import seaborn as sns


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
