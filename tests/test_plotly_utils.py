"""Importing Dependencies"""
import sys
import pytest
import pandas as pd
from plotly import graph_objs
# Env tests
try:
    sys.path.insert(1, './src')  # the type of path is string
    import plotly_utils as pu
except (ModuleNotFoundError, ImportError) as error_message:
    print("{} fileure".format(type(error_message)))

@pytest.fixture
def _mock_subject_level_df():
    """
    Simulate df at a subject level with continuous, categorial and group variables
    """
    subject_level_dict = {
        'subject':[str(i) for i in range(10)],
        'group_identifier':[True if i//5==0 else False for i in range(10)],
        'categorical_example_1':[i % 2 for i in range(10)],
        'categorical_example_2':[i % 2 for i in range(1,11)],
        'continuous_example_1': [i * 5 for i in range(10)],
        'continuous_example_2': [i * 6 for i in range(10)]
    }
    subject_level_df = pd.DataFrame(data=subject_level_dict)
    return subject_level_df

@pytest.mark.parametrize(
    "group_identifier, expected_result", [
        (False, 0.4)
    ]
)
def test_group_share_per_category(_mock_subject_level_df, group_identifier, expected_result):
    """Test a function to automate integration tests of unique elements per column"""
    to_test_pd_df = pu.group_share_per_category(
        _mock_subject_level_df,'group_identifier','categorical_example_1'
    )
    value_to_check = to_test_pd_df.loc[
        (to_test_pd_df.group_identifier == group_identifier)
        & (to_test_pd_df.category_value == 0)
        , 'value_share_on_group'
    ][0]
    assert value_to_check == expected_result

def test_group_share_per_category_looper(_mock_subject_level_df):
    """Straightforward tests to validate the looping is function is working properly"""
    test_category_col_list = ['categorical_example_1','categorical_example_2']
    to_test_pd_df = pu.group_share_per_category_looper(
        _mock_subject_level_df, 'group_identifier', category_col_list=test_category_col_list
    )
    assert type(to_test_pd_df) == pd.core.frame.DataFrame
    assert to_test_pd_df.empty == False

def test_prop_stacked_chart(_mock_subject_level_df):
    """Straightforward test to validate the plotly applicator is working properly"""
    test_category_col_list = ['categorical_example_1','categorical_example_2']
    to_test_pd_df = pu.group_share_per_category_looper(
        _mock_subject_level_df, 'group_identifier', category_col_list=test_category_col_list
    )
    test_fig = pu.prop_stacked_chart(
        to_test_pd_df, 'group_identifier', 'value_share_on_group',
        'category_value', 'category_name'
    )
    assert type(test_fig) == graph_objs._figure.Figure

def test_continuous_variables_to_boxplot_format(_mock_subject_level_df):
    """Straightforward tests to validate the melt operation is working properly"""
    test_continuous_col_list = ['continuous_example_1','continuous_example_2']
    to_test_boxplot_df = pu.continuous_variables_to_boxplot_format(
        _mock_subject_level_df, 'group_identifier', continuous_col_list=test_continuous_col_list
    )
    assert type(to_test_boxplot_df) == pd.core.frame.DataFrame
    assert to_test_boxplot_df.empty == False
