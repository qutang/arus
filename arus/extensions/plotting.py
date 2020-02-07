import matplotlib.colors as mc
import colorsys


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
