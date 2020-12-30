"""Importing Dependencies"""
from dowhy import CausalModel

def dowhy_quick_backdoor_estimator(
        dataframe, outcome, treatment,
        cofounders_list, method_name,
        populaton_of_interest='ate', view_model=False
    ):
    """
    Make a quick statistical assessment for the mean of 2 different samples (hypothesis test based)
    :param dataframe: original dataframe in a subject level
    :param group_col: the name of the group column
    :param category_col: the name of the category_col column
    :returns group_share_per_category_df: df containing the % share each category has by group
    """
    causal_model = CausalModel(
        data=dataframe, treatment=treatment,
        outcome=outcome, common_causes=cofounders_list
    )
    if view_model:
        causal_model.view_model(layout="dot")
    identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    causal_estimate = causal_model.estimate_effect(
        identified_estimand, method_name=method_name,
        target_units = populaton_of_interest#, confidence_intervals=True # not in this release
    )
    return causal_estimate.value
