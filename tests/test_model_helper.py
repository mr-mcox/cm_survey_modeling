from model_helper import cdf, compute_ps, full_thresh
import numpy as np
from scipy.stats import norm
import theano
import theano.tensor as tt
import pytest


def test_cdf():
    xs = np.arange(5)
    mu = 4
    sigma = 3

    exp = norm.cdf(xs, mu, sigma)

    t_xs, t_mu, t_sigma = tt.vector(), tt.scalar(), tt.scalar()
    c = cdf(t_xs, t_mu, t_sigma)
    f_cdf = theano.function([t_xs, t_mu, t_sigma], c)

    actual = f_cdf(xs, mu, sigma)
    assert np.allclose(exp, actual)


def test_compute_ps():
    thresh = [i + 2.5 for i in range(4)]
    f_thresh = [-1 * np.inf] + [i + 1.5 for i in range(6)] + [np.inf]
    mu = 4
    sigma = 3
    exp = norm.cdf(f_thresh[1:], mu, sigma) - norm.cdf(f_thresh[:-1], mu, sigma)

    t_thresh, t_mu, t_sigma = tt.vector(), tt.scalar(), tt.scalar()
    ps = compute_ps(t_thresh, t_mu, t_sigma)
    f_ps = theano.function([t_thresh, t_mu, t_sigma], ps)

    actual = f_ps(thresh, mu, sigma)
    assert np.allclose(exp, actual)


def test_full_thresh():
    thresh = [2, 4.5, 4, 5]
    mod_thresh = [2, 4.5, 4.5, 5]
    exp = np.concatenate([[-1*np.inf, 1.5], mod_thresh, [6.5, np.inf]])

    t_thresh = tt.vector()
    ft = full_thresh(t_thresh)
    f_ft = theano.function([t_thresh], ft)

    actual = f_ft(thresh)
    assert np.allclose(exp, actual)
