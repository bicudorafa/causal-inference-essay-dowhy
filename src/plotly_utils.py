"""Importing Dependencies"""
import pandas as pd
import plotly.express as px

def group_share_per_category(dataframe: pd.DataFrame, group_col: str, category_col: str):
    """
    Calculates the % share of a categorical variable in relation to a group (ie another categorical)
    :param dataframe: original dataframe in a subject level
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
    :param dataframe: original dataframe in a subject level
    :param group_col: the name of the group column
    :param category_col_list: the list of category columns
    :returns agg_group_share_per_category_df: group_share_per_category concatened outputed dfs
    """
    agg_group_share_per_category_df = pd.DataFrame()
    for category_col in category_col_list:
        temp_df = group_share_per_category(dataframe, group_col, category_col)
        agg_group_share_per_category_df = pd.concat([agg_group_share_per_category_df, temp_df])
        del temp_df
    return agg_group_share_per_category_df

def prop_stacked_chart(
        share_per_category_df: pd.DataFrame, group_col: str, value_share_on_group_col:str,
        category_value_col:str, category_name_col:str
    ):
    """
    Plots a proportional bar chart by using plotly express bar plot interface along with the output
    from the group_share_per_category_looper function
    :param group_share_per_category_looper: manipulated df from group_share_per_category_looper
    :param group_col: the name of the group column
    :param category_name_col: column containing the variables names
    :param category_value_col: column containing the variables' values
    :param value_share_on_group_col: column containing the variables' share per group
    :returns fig: plotly object containing proportional bar chart
    """
    facet_col_wrap_value = len(share_per_category_df[category_name_col].unique())
    fig=px.bar(
        share_per_category_df, x=group_col, y=value_share_on_group_col, color=category_value_col,
        facet_col=category_name_col, facet_col_wrap=facet_col_wrap_value
    )
    # exclude variable names to avoid repetitiveness
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(yaxis=dict(title_text='Percentage per Sample', tickformat=".2%"))
    # exclude group names to avoid repetitiveness
    fig.update_xaxes(title_text='')
    # exclude scale to improve beuty
    fig.update(layout_coloraxis_showscale=False)

    return fig

def continuous_variables_to_boxplot_format(
        dataframe: pd.DataFrame, group_col: str, continuous_col_list: list
    ):
    """
    Melts a df into a group, variable name and value structure to be plotted as a boxplot
    :param dataframe: original dataframe in a subject level
    :param group_col: the name of the group column
    :param continuous_col_list: the list of continuous columns
    :returns melted_df: group_share_per_category concatened outputed dfs
    """
    melted_df = (
        dataframe
        .loc[:, ([group_col] + continuous_col_list)]
        .pipe(pd.melt, id_vars=[group_col], value_vars=continuous_col_list)
    )
    return melted_df

