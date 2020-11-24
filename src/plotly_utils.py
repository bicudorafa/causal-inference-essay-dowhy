"""Importing Dependencies"""
import pandas as pd

def group_share_per_category(dataframe: pd.DataFrame, group_col: str, category_col: str):
    """
    Calculates the % share of a categorical variable in relation to a group (ie another categorical)
    :dataframe: original dataframe in a subject
    :param to_test_pd_series: pd series to be validated
    :returns is_contained: boolen value regarding if the unique elements are contained in the list
    """
    # enable a dynamic percentual share column naming
    pct_share_col_name = f'share_per_{group_col}'
    group_share_per_category_df = (
        dataframe
        .loc[:, [group_col, category_col]]
        .groupby([group_col, category_col])
        .agg(pct_share_col_name=(group_col, 'count'))
        .groupby(level=0) # it has a similar use case of partition by in sql
        .apply(lambda x: x / float(x.sum())) # then calculating the share by index
        .reset_index()
        .rename(columns={category_col:'categorical_value'})
        .assign(category_name = category_col)
    )
    return group_share_per_category_df
