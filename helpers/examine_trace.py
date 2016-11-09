import numpy as np
from scipy.stats import norm
import re


def ps_from_thresh(thresh, mu, sigma):
    lows = norm.cdf(thresh[:, :-1].T, mu, sigma).T
    highs = norm.cdf(thresh[:, 1:].T, mu, sigma).T
    return highs - lows


def compute_ps(df):
    ths = df.ix[:, df.columns.str.contains('thresh__')].as_matrix()
    ths_sum = np.add.accumulate(ths, axis=1)
    inner_thresh = ths_sum * 5 + 1.5
    thresh = np.insert(inner_thresh, 0, 1.5, axis=1)
    thresh = np.insert(thresh, 0, -1*np.inf, axis=1)
    thresh = np.append(
        thresh, np.inf * np.ones(thresh.shape[0]).reshape(-1, 1), axis=1)

    if 'mu' in df.columns:
        return ps_from_thresh(thresh, df.mu, df.sigma)
    else:
        mu_regs_cols = df.ix[:, df.columns.str.contains('mu_reg__')]
        labels = [re.search('mu_reg__([\d_]+)', c).group(1) for c in mu_regs_cols.columns]

        mu_regs = df.ix[:, df.columns.str.contains('mu_reg__')].as_matrix()

        num_runs = mu_regs.shape[0]
        reg_mu = mu_regs + df.b0_mu.as_matrix().reshape(num_runs, -1)

        out = dict()
        for i, label in enumerate(labels):
            ps = ps_from_thresh(
                thresh, reg_mu[:, i], df.sigma)
            assert (ps >= 0).all()
            out[label] = ps
        return out


def net_from_ps(ps):
    weak = ps[:, :4].sum(axis=1)
    strong = ps[:, 5:].sum(axis=1)
    return strong - weak


def compute_net(df):
    ps = compute_ps(df)

    if type(ps)is dict:
        out = dict()
        for k in ps.keys():
            out[k] = net_from_ps(ps[k])
        return out
    else:
        return net_from_ps(ps)
