import pandas as pd
import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
import sys

sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from model_helper import compute_ps

with pd.HDFStore('../inputs/responses.h5') as store:
    df = pd.DataFrame()
    if 'responses' in store:
        df = store['responses']
    else:
        print("Loading CSV from file")
        df = pd.read_csv('../inputs/cm_response_confidential.csv')
        store['responses'] = df

data = df.loc[(df.survey_code == '1617F8W') & (df.question_code.isin(qc)) & df.response.notnull()]

q_dum = data.question_code.str.get_dummies()
qcodes = q_dum.columns
q_mtx = q_dum.as_matrix()

with pm.Model() as model:
    num_qc = q_mtx.shape[1]
    mu = pm.Normal('mu', mu=4, sd=3, shape=num_qc)
    sigma = pm.HalfCauchy('sigma', beta=1, shape=num_qc)
    thresh = pm.Dirichlet('thresh', a=np.ones(5), shape=(num_qc, 5))
    
    qc_range = tt.arange(num_qc)
    cat_ps, update = theano.scan(fn=lambda i: compute_ps(thresh[i], mu[i], sigma[i]),
                        sequences=[qc_range])

    resp = data.response - 1
    results = pm.Categorical('results', p=theano.dot(q_mtx,cat_ps), observed=resp)

with model:
    step = pm.Metropolis()
    trace = pm.sample(100, step=step)








