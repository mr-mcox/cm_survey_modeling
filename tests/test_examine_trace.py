from helpers.examine_trace import label_trace
import numpy as np
from scipy.stats import norm
import pandas as pd
import pytest


@pytest.fixture
def national_trace():
    thresh = [0.2 for i in range(5)]
    mu = 3
    sigma = 5

    rec_row = (mu, sigma, *thresh)
    cols = 'mu sigma thresh__0  thresh__1  thresh__2  thresh__3  thresh__4'.split()

    trace_df = pd.DataFrame.from_records([rec_row], columns=cols)

    f_thresh = [-1 * np.inf] + [i + 0.5 for i in range(6)] + [np.inf]
    exp = norm.cdf(f_thresh[1:], mu, sigma) - \
        norm.cdf(f_thresh[:-1], mu, sigma)

    return {'trace_df': trace_df, 'ps': exp}


@pytest.fixture
def regional_trace():
    thresh = [0.2 for i in range(5)]
    b0_mu = 3
    sigma = 5

    reg_mu_delta_0 = 0
    reg_mu_delta_1 = 1
    ps = [0.15 for x in range(7)]
    rec_row = (b0_mu, sigma, reg_mu_delta_0, reg_mu_delta_1, *thresh, *ps, *ps)
    cols = """b0_mu sigma
    mu_reg__0 mu_reg__1
    thresh__0  thresh__1  thresh__2  thresh__3  thresh__4
    ps__0_0 ps__0_1 ps__0_2 ps__0_3 ps__0_4 ps__0_5 ps__0_6
    ps__1_0 ps__1_1 ps__1_2 ps__1_3 ps__1_4 ps__1_5 ps__1_6
    """.split()

    trace_df = pd.DataFrame.from_records(
        [rec_row, rec_row, rec_row], columns=cols)

    heads = [{'name': 'Region', 'values': ['Alabama', 'Atlanta']}]

    return {'heads': heads, 'trace': trace_df}


def test_overall_metrtics_df(regional_trace):
    res = label_trace(regional_trace['trace'], regional_trace['heads'])
    assert {'b0_mu', 'sigma', 'i', 'thresh__3'} <= set(res['overall'].columns)


def test_reg_metrtics_df(regional_trace):
    res = label_trace(regional_trace['trace'], regional_trace['heads'])
    reg_df = res[1]
    assert (reg_df.loc[reg_df.Region == 'Alabama', 'mu_reg'] == 0).all()
    assert (reg_df.loc[reg_df.Region == 'Atlanta', 'mu_reg'] == 1).all()
    assert (reg_df.loc[reg_df.Region == 'Atlanta', 'ps0'] == 0.15).all()

def test_reg_nets_df(regional_trace):
    res = label_trace(regional_trace['trace'], regional_trace['heads'])
    reg_df = res[1]
    assert (reg_df.loc[reg_df.Region == 'Alabama', 'net'] == -0.3).all()