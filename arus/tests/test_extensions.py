import pandas as pd
import numpy as np
from .. import extensions


class TestPandas:
    def test_merge_all(self):
        df1 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],
                            'value': [1, 2, 3, 5], 'group': [1, 1, 2, 2]})

        df2 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],
                            'value': [5, 6, 7, 8], 'group': [1, 1, 2, 2]})

        df3 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],
                            'value': [9, 10, 11, 12], 'group': [1, 1, 2, 2]})
        dfs = [df1, df2, df3]
        merged, cols_with_suffixes = extensions.pandas.merge_all(*dfs, suffix_names=['DW', 'DA', 'DT'], suffix_cols=[
            'value'], on=['key', 'group'], how='inner', sort=False)

        np.testing.assert_array_equal(
            cols_with_suffixes, ['value_DW', 'value_DA', 'value_DT'])
        np.testing.assert_array_equal(
            merged[['key', 'group']], df1[['key', 'group']]
        )
        np.testing.assert_array_equal(
            set(merged.columns), set(['key', 'value_DW', 'value_DA', 'value_DT', 'group']))

        merged, cols_with_suffixes = extensions.pandas.merge_all(df1, suffix_names=['DW'], suffix_cols=[
            'value'], on=['key', 'group'], how='inner', sort=False)
        np.testing.assert_array_equal(merged.values, df1.values)
        np.testing.assert_array_equal(
            set(merged.columns), set(['key', 'value_DW', 'group']))

    def test_filter_column(self):
        df1 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],
                            'value': [1, 2, 3, 5], 'group': [1, 1, 2, 2]})
        filtered = extensions.pandas.filter_column(
            df1, col='key', values_to_filter_out=['foo'])
        np.testing.assert_array_equal(filtered['value'].values, [2, 3])
        np.testing.assert_array_equal(filtered['key'].values, ['bar', 'baz'])

        filtered = extensions.pandas.filter_column(
            df1, col='value', values_to_filter_out=[2, 5])
        np.testing.assert_array_equal(filtered['value'].values, [1, 3])
        np.testing.assert_array_equal(filtered['key'].values, ['foo', 'baz'])
