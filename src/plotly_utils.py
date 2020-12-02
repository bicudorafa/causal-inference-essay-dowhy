"""Importing Dependencies"""
# %%
import pandas as pd

def group_share_per_category(dataframe: pd.DataFrame, group_col: str, category_col: str):
    """
    Calculates the % share of a categorical variable in relation to a group (ie another categorical)
    :dataframe: original dataframe in a subject level
    :param group_col: the name of the group column
    :param category_col: the name of the category_col column
    :returns group_share_per_category_df: df containing the % share each category has by group
    """
    group_share_per_category_df = (
        dataframe
        .loc[:, [group_col, category_col]]
        .groupby([group_col, category_col])
        .agg(count=(group_col, 'count'))
        .groupby(level=0) # it has a similar use case of partition by in sql
        .apply(lambda x: x / float(x.sum())) # then calculating the share by index
        .reset_index()
        .rename(columns={category_col:'category_value', 'count':'value_share_on_group'})
        .assign(category_name = category_col)
    )
    return group_share_per_category_df

def group_share_per_category_looper(
        dataframe: pd.DataFrame, group_col: str, category_col_list: list
    ):
    """
    Loops the group_share_per_category funct into a df based on a list of categories
    :dataframe: original dataframe in a subject level
    :param group_col: the name of the group column
    :param category_col_list: tthe list of category columns
    :returns agg_group_share_per_category_df: group_share_per_category concatened outputed dfs
    """
    agg_group_share_per_category_df = pd.DataFrame()
    for category_col in category_col_list:
        temp_df = group_share_per_category(dataframe, group_col, category_col)
        agg_group_share_per_category_df = pd.concat([agg_group_share_per_category_df, temp_df])
        del temp_df
    return agg_group_share_per_category_df
