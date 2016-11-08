import pandas as pd
import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
import sys
import json

sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from model_helper import compute_ps

version = 3

with pd.HDFStore('../inputs/responses.h5') as store:
    df = pd.DataFrame()
    if 'responses' in store:
        df = store['responses']
    else:
        print("Loading CSV from file")
        df = pd.read_csv('../inputs/cm_response_confidential.csv')
        store['responses'] = df

qc = ['CSI1', 'CSI2']
reg_include = ['Alabama',
               'Appalachia',
               'Arkansas',
               'Baltimore',
               'Bay Area']
surveys = '1617F8W 1516F8W 1415F8W 1314F8W'.split() 

for survey in surveys:
    data = df.loc[(df.survey_code == survey) & 
        df.response.notnull()]
    
    q_dum = data.question_code.str.get_dummies()
    qcodes = q_dum.columns
    q_mtx = q_dum.as_matrix()
    
    r_dum = data.Region.str.get_dummies()
    regs = r_dum.columns
    r_mtx = r_dum.as_matrix()
    
    col_headers = {'questions':list(qcodes), 'regions':list(regs)}
    
    with pm.Model() as model:
        num_qc = q_mtx.shape[1]
        num_reg = r_mtx.shape[1]
        
        mu = pm.Normal('mu', mu=4, sd=3, shape=(num_qc, num_reg))
        sigma = pm.HalfCauchy('sigma', beta=1, shape=(num_qc, num_reg))
        thresh = pm.Dirichlet('thresh', a=np.ones(5), shape=(num_qc, 5))
        
        qc_range = tt.arange(num_qc)
        reg_range = tt.arange(num_reg)
        cat_ps, update = theano.scan(fn=lambda q_i,r_i: compute_ps(thresh[q_i], mu[q_i, r_i], sigma[q_i, r_i]),
                            sequences=[qc_range, reg_range])
    
        resp = data.response - 1
        
        cat_q = theano.dot(q_mtx,cat_ps)
        cat_q = cat_q.dimshuffle(0, 'x', 1)
        r_mtx3 = r_mtx.reshape(r_mtx.shape[0],r_mtx.shape[1],-1)
        cat_r =  r_mtx3 * cat_q
        cat_r_flat = cat_r.sum(axis=1)
        
        results = pm.Categorical('results', p=cat_r_flat, observed=resp)
    
    with model:
        db_name = '../traces/cm_cat_v{}_survey_{}'.format(version, survey)
        db = pm.backends.Text(db_name)
        step = pm.Metropolis()
        burn = pm.sample(2000, step=step)
        trace = pm.sample(5000, step=step, start=burn[-1], trace=db)
        
    with open('../traces/cm_cat_v{}_survey_{}_columns.json'.format(version, survey), 'w') as json_file:
        json.dump(col_headers, json_file)



