import numpy as np
from ..num import format_arr


def apply_over_subwins(X, func, subwins=None, subwin_samples=None, **kwargs):
    X = format_arr(X)
    if subwins is not None:
        # compute the length of each sub window if the number of sub windows is given.
        if subwins == 0:
            # if subwins is zero, treat as one
            subwins = 1
            win_length = X.shape[0]
        else:
            win_length = int(np.floor(X.shape[0] / subwins))
            if win_length == 0:
                # if subwins is zero, treat as one
                subwins = 1
                win_length = X.shape[0]
    elif subwin_samples is not None:
        # or if the number of samples of each sub window is provided, compute the number of sub windows.
        if subwin_samples == 0:
            subwins = 1
            win_length = X.shape[0]
        else:
            win_length = subwin_samples
            subwins = int(np.floor(X.shape[0] / subwin_samples))
            if subwins == 0:
                # if subwins is zero, treat as one
                subwins = 1
                win_length = X.shape[0]
    else:
        # or treat the entire input array as a single sub window
        subwins = 1
        win_length = X.shape[0]

    # Use the sub windows evenly reside in the middle of the bigger window
    start_index = np.ceil((X.shape[0] - subwins * win_length) / 2)

    result = []
    for i in range(subwins):
        indices = int(start_index) + np.array(range(
            i * win_length,
            (i + 1) * win_length
        ))
        subwin_X = X[indices, :]
        subwin_result = func(subwin_X, **kwargs)
        subwin_result = format_arr(subwin_result)
        result.append(subwin_result)
    # each row is the result from one sub window
    final_result = np.concatenate(result, axis=0)
    return final_result
