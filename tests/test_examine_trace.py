from helpers.examine_trace import compute_ps, compute_net
import numpy as np
from scipy.stats import norm
import pandas as pd
import pytest


@pytest.fixture
def national_trace():
    thresh = [0.2 for i in range(5)]
    mu = 4
    sigma = 3

    rec_row = (mu, sigma, *thresh)
    cols = 'mu sigma thresh__0  thresh__1  thresh__2  thresh__3  thresh__4'.split()

    trace_df = pd.DataFrame.from_records([rec_row], columns=cols)

    f_thresh = [-1 * np.inf] + [i + 1.5 for i in range(6)] + [np.inf]
    exp = norm.cdf(f_thresh[1:], mu, sigma) - \
        norm.cdf(f_thresh[:-1], mu, sigma)

    return {'trace_df': trace_df, 'ps': exp}


@pytest.fixture
def regional_trace():
    thresh = [0.2 for i in range(5)]
    mu_0 = 4
    mu_1 = 5
    sigma_0 = 3
    sigma_1 = 4

    rec_row = (mu_0, mu_1, sigma_0, sigma_1, *thresh)
    cols = """mu__0_0 mu__0_1
    sigma__0_0 sigma__0_1
    thresh__0  thresh__1  thresh__2  thresh__3  thresh__4""".split()

    trace_df = pd.DataFrame.from_records([rec_row], columns=cols)

    f_thresh = [-1 * np.inf] + [i + 1.5 for i in range(6)] + [np.inf]
    exp_0 = norm.cdf(f_thresh[1:], mu_0, sigma_0) - \
        norm.cdf(f_thresh[:-1], mu_0, sigma_0)
    exp_1 = norm.cdf(f_thresh[1:], mu_1, sigma_1) - \
        norm.cdf(f_thresh[:-1], mu_1, sigma_1)

    return {'trace_df': trace_df, 'ps': {'0_0': exp_0, '0_1': exp_1}}


def test_trace_to_compute_ps(national_trace):
    trace_df = national_trace['trace_df']
    exp = national_trace['ps']

    res = compute_ps(trace_df)
    assert np.allclose(res[0], exp)


def test_trace_to_compute_net(national_trace):
    trace_df = national_trace['trace_df']
    ps = national_trace['ps']

    net = sum(ps[5:]) - sum(ps[:4])

    assert compute_net(trace_df)[0] == net


def test_trace_to_compute_ps_reg(regional_trace):
    trace_df = regional_trace['trace_df']
    exp = regional_trace['ps']

    res = compute_ps(trace_df)
    for k in exp.keys():
        assert np.allclose(res[k][0], exp[k])

def test_trace_to_compute_net_reg(regional_trace):
    trace_df = regional_trace['trace_df']
    exp = regional_trace['ps']

    res = compute_net(trace_df)
    for k in exp.keys():
        net = sum(exp[k][5:]) - sum(exp[k][:4])
        assert np.allclose(res[k][0], net)
