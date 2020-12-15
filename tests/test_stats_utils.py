"""Importing Dependencies"""
import sys
import pytest
import numpy as np
# Env tests
try:
    sys.path.insert(1, './src')  # the type of path is string
    import stats_utils as su
except (ModuleNotFoundError, ImportError) as error_message:
    print("{} fileure".format(type(error_message)))

@pytest.mark.parametrize(
    "mu, sigma, sample_size", [
        (0.0, 0.1, 50)
    ]
)
def test_mean_ttest_analysis(_mu, sigma, sample_size):
    """"""
    # https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.normal.html
    pass
