"""Importing Dependencies"""
# %%
import sys
import pytest
import pandas as pd
# Env tests
try:
    sys.path.insert(1, './src')  # the type of path is string
    import plotly_utils as pu
except (ModuleNotFoundError, ImportError) as error_message:
    print("{} fileure".format(type(error_message)))
#%%
@pytest.fixture
def _mock_subject_level_df():
    """
    Simulate df at a subject level with continuous, categorial and group variables
    """
    subject_level_dict = {
        'subject':[str(i) for i in range(10)],
        'group_identifier':[True if i//5==0 else False for i in range(10)],
        'categorical_example':[i % 2 for i in range(10)],
        'continuous_example': [i * 5 for i in range(10)]
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
        _mock_subject_level_df,'group_identifier','categorical_example'
    )
    value_to_check = to_test_pd_df.loc[
        (to_test_pd_df.group_identifier == group_identifier)
        & (to_test_pd_df.category_value == 0)
        , 'value_share_on_group'
    ][0]
    assert value_to_check == expected_result
