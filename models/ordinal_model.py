import pandas as pd
import pymc3 as pm
import numpy as np
from os import path

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


def run_model(data):

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

if __name__ == '__main__':
    data = load_data()
    run_model(data)
