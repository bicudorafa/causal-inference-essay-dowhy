"""Importing Dependencies"""
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

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

def adjusted_boxplot_per_groups(
        melted_df: pd.DataFrame, group_col: str, std_y="value", std_facet_row="variable"
    ):
    """
    Plots a boxplot chart by using plotly express interface and the output from the above function
    :param melted_df: manipulated df from continuous_variables_to_boxplot_format
    :param group_col: the name of the group column
    :param std_y: column containing the variables' values
    :param facet_row: column containing the variables' names
    :returns fig: plotly object containing boxplots
    """
    fig = px.box(
        melted_df, y=std_y, x=group_col,
        color=group_col, facet_row=std_facet_row
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text='')
    fig.update_yaxes(matches=None, title_text='')

    return fig

def to_distplot_per_groups_format(dataframe: pd.DataFrame, target_col: str, groups_col: str):
    """
    Returns a list of arrays and their names to be used at the distplot_per_groups
    :param dataframe: manipulated df from continuous_variables_to_boxplot_format
    :param target_col: the name of the group column
    :param groups_col: column containing the variables' values
    :returns to_plot_data, to_plot_labels: list of arrays and list of str
    """
    groups = dataframe[groups_col].unique()
    to_plot_data = []
    to_plot_labels = []

    for group in groups:
        to_plot_data.append(dataframe.loc[dataframe[groups_col] == group, target_col].values)
        to_plot_labels.append(str(group))

    return to_plot_data, to_plot_labels

def distplot_per_groups(dataframe: pd.DataFrame, target_col: str, groups_col: str):
    """
    Plots the distribution of a continuous variable aggrouped per a group/categorical variable
    :param dataframe: manipulated df from continuous_variables_to_boxplot_format
    :param target_col: the name of the group column
    :param groups_col: column containing the variables' values
    :returns fig: plotly object containing boxplots
    """
    to_plot_data, to_plot_labels = to_distplot_per_groups_format(dataframe, target_col, groups_col)
    fig = ff.create_distplot(to_plot_data, to_plot_labels, bin_size=.2)
    return fig
