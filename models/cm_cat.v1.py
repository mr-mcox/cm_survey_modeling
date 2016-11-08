import pandas as pd
import pymc3 as pm
from pymc3.variational import advi
from pymc3.math import clip
import numpy as np
import theano
import theano.tensor as tt
from scipy.stats import norm
from scipy import optimize
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from model_helper import compute_ps
from examine_trace import compute_net

with pd.HDFStore('../inputs/responses.h5') as store:
    df = pd.DataFrame()
    if 'responses' in store:
        df = store['responses']
    else:
        print("Loading CSV from file")
        df = pd.read_csv('../inputs/cm_response_confidential.csv')
        store['responses'] = df

qc = 'CSI1'
data = df.loc[(df.survey_code == '1617F8W') & (df.question_code == qc) & df.response.notnull()]

with pm.Model() as model:
    mu = pm.Normal('mu', mu=4, sd=3)
    sigma = pm.HalfCauchy('sigma', beta=1)
    thresh = pm.Dirichlet('thresh', a=np.ones(5))
    
    cat_p = compute_ps(thresh, mu, sigma)

    resp = data.response - 1
    results = pm.Categorical('results', p=cat_p, observed=resp)

with model:
    step = pm.Metropolis()
    burn = pm.sample(2000, step=step)
    trace = pm.sample(5000, step=step, start=burn[-1])
    
sns.distplot(compute_net(pm.trace_to_dataframe(trace)))



