"""
Module of extension functions to be applied to pandas objects (e.g., DataFrame or Series)

Author: Qu Tang

Date: 2020-02-03

License: see LICENSE file
"""
import functools


def merge_all(*dfs, suffix_names, suffix_cols, **kwargs):
    def _append_suffix(df, suffix_name):
        new_cols = []
        for col in df.columns:
            if col in suffix_cols and suffix_name != '':
                col = col + '_' + suffix_name
            new_cols.append(col)
        df.columns = new_cols
        return df

    def _combine(left, right):
        left_df = _append_suffix(left[0], left[1])
        right_df = _append_suffix(right[0], right[1])
        merged = left_df.merge(
            right_df, **kwargs)
        return (merged, '')

    sequence = zip(dfs, suffix_names)
    if len(suffix_names) == 1:
        merged = _append_suffix(dfs[0], suffix_names[0])
    else:
        tuple_results = functools.reduce(_combine, sequence)
        merged = tuple_results[0]
    cols_with_suffixes = list(filter(lambda name: name.split('_')
                                     [-1] in suffix_names, merged.columns))
    return merged, cols_with_suffixes


def filter_column(df, col, values_to_filter_out=[]):
    # remove values
    is_valid_values = ~df[col].isin(values_to_filter_out).values
    filtered_df = df.loc[is_valid_values, :]
    return filtered_df