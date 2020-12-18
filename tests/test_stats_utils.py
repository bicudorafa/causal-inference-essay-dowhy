"""Importing Dependencies"""
import sys
import pytest
import numpy as np
from numpy.random import seed
# Env tests
try:
    sys.path.insert(1, './src')  # the type of path is string
    import stats_utils as su
except (ModuleNotFoundError, ImportError) as error_message:
    print("{} fileure".format(type(error_message)))

@pytest.mark.parametrize(
    "add_mu, add_sigma, add_n, p_exp_result, power_exp_result", [
        (0, 0, 0, False, False) # obvious non rejected case
    ]
)

def test_mean_ttest_analyzer(add_mu, add_sigma, add_n, p_exp_result, power_exp_result):
    """Test different hypotehsis tests for mean comparisons"""
    # random seed to generate the same ficitonal samples
    seed(666)
    population_1 = np.random.normal(0, 1, 50)
    population_2 = np.random.normal(0 + add_mu, 1 + add_sigma, 50 + add_n)
    pvalue, power, _ = su.mean_ttest_analyzer(population_1, population_2)
     # reference p, power values
    p_assessment = pvalue < 0.1
    power_assessment = power > 0.6
    assert p_assessment == p_exp_result, "error on pvalue comparison"
    assert power_assessment == power_exp_result, "error on power comparison"
