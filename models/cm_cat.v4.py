import pandas as pd
import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
import sys
import json

sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from model_helper import compute_ps

version = 4
min_regions = False

with pd.HDFStore('../inputs/responses.h5') as store:
    df = pd.DataFrame()
    if 'responses' in store:
        df = store['responses']
    else:
        print("Loading CSV from file")
        df = pd.read_csv('../inputs/cm_response_confidential.csv')
        store['responses'] = df

reg_include = ['Alabama',
               'Appalachia',
               'Arkansas',
               'Baltimore',
               'Bay Area']
survey_incl = '1617F8W 1516F8W 1415F8W 1314F8W'.split()
df = df.ix[df.survey_code.isin(survey_incl)]

# qcs = ['CSI9', 'Culture1']
# df = df.ix[df.question_code.isin(qcs)]

reg_mask = [True for x in range(len(df))]
if min_regions:
    version = '{}_min'.format(version)
    reg_mask = df.Region.isin(reg_include)

qs = df.question_code.unique()

for ques in qs:
    print('Modeling question {}'.format(ques))
    data = df.loc[(df.question_code == ques) &
                  reg_mask &
                  df.response.notnull()]

    s_dum = data.survey_code.str.get_dummies()
    surveys = s_dum.columns
    s_mtx = s_dum.as_matrix()

    r_dum = data.Region.str.get_dummies()
    regs = r_dum.columns
    r_mtx = r_dum.as_matrix()

    col_headers = {'questions': list(surveys), 'regions': list(regs)}

    with pm.Model() as model:
        num_svy = s_mtx.shape[1]
        num_reg = r_mtx.shape[1]

        mu = pm.Normal('mu', mu=4, sd=3, shape=(num_svy, num_reg))
        sigma = pm.HalfCauchy('sigma', beta=1, shape=(num_svy, num_reg))
        thresh = pm.Dirichlet('thresh', a=np.ones(5), shape=5)

        svy_range = tt.arange(num_svy)
        reg_range = tt.arange(num_reg)
        cat_ps, update = theano.scan(fn=lambda s_i, r_i: compute_ps(thresh, mu[s_i, r_i], sigma[s_i, r_i]),
                                     sequences=[svy_range, reg_range])

        resp = data.response - 1

        cat_s = theano.dot(s_mtx, cat_ps)
        cat_s = cat_s.dimshuffle(0, 'x', 1)
        r_mtx3 = r_mtx.reshape(r_mtx.shape[0], r_mtx.shape[1], -1)
        cat_r = r_mtx3 * cat_s
        cat_r_flat = cat_r.sum(axis=1)

        results = pm.Categorical('results', p=cat_r_flat, observed=resp)

    with model:
        db_name = '../traces/cm_cat_v{}_ques_{}'.format(version, ques)
        db = pm.backends.Text(db_name)
        step = pm.Metropolis()
        burn = pm.sample(2000, step=step)
        trace = pm.sample(5000, step=step, start=burn[-1], trace=db)

    with open('../traces/cm_cat_v{}_ques_{}_columns.json'.format(version, ques), 'w') as json_file:
        json.dump(col_headers, json_file)
