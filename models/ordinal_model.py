import pandas as pd
import pymc3 as pm
import numpy as np
from os import path
import theano
import theano.tensor as tt

from helpers.model_helper import compute_ps


def load_data():
    with pd.HDFStore(path.join('inputs', 'responses.h5'))as store:
        df = pd.DataFrame()
        if 'responses' in store:
            df = store['responses']
        else:
            print("Loading CSV from file")
            df = pd.read_csv(
                path.join('inputs', 'cm_response_confidential.csv'))
            store['responses'] = df

    qc = 'CSI1'
    data = df.loc[(df.survey_code == '1617F8W') & (
        df.question_code == qc) & df.response.notnull()]

    return data


def run_national_model(data):

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=4, sd=3)
        sigma = pm.Uniform('sigma', lower=0.7, upper=70)
        thresh = pm.Dirichlet('thresh', a=np.ones(5))

        cat_p = compute_ps(thresh, mu, sigma)

        resp = data.response - 1
        results = pm.Categorical('results', p=cat_p, observed=resp)

    with model:
        step = pm.Metropolis()
        burn = pm.sample(2000, step=step)
        trace = pm.sample(5000, step=step, start=burn[-1])

    return trace


def run_regional_model(data,
                       progressbar=False, db_file=None, burn=2000, samp=5000):

    # Setup masks
    r_dum = data.Region.str.get_dummies()
    regs = r_dum.columns
    r_mtx = r_dum.as_matrix()
    num_reg = r_mtx.shape[1]

    heads = [{'name': 'Region', 'values': regs}]

    with pm.Model() as model:
        b0_mu = pm.Normal('b0_mu', mu=4, sd=3)
        sigma = pm.Uniform('sigma', lower=0.7, upper=70)
        thresh = pm.Dirichlet('thresh', a=np.ones(5))

        mu_reg = pm.Normal('mu_reg', mu=0, sd=3, shape=num_reg)

        reg_mu = b0_mu + mu_reg

        reg_range = tt.arange(num_reg)
        cat_ps, update = theano.scan(fn=lambda r_i: compute_ps(thresh, reg_mu[r_i], sigma),
                                     sequences=[reg_range])
        ps = pm.Deterministic('ps', cat_ps)

        cat_r = theano.dot(r_mtx, cat_ps)
        resp = data.response - 1
        results = pm.Categorical('results', p=cat_r, observed=resp)

    with model:
        db = None
        if db_file is not None:
            db = pm.backends.Text(db_file)
        step = pm.Metropolis()
        burn = pm.sample(burn, step=step, progressbar=progressbar)
        trace = pm.sample(
            samp, step=step, start=burn[-1], progressbar=progressbar, trace=db)

    return {'heads': heads, 'trace': trace}

if __name__ == '__main__':
    data = load_data()
    run_national_model(data, progressbar=True)
