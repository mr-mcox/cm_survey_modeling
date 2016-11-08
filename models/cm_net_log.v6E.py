import sys
import os
import pandas as pd
import pymc3 as pm
from pymc3.variational import advi
import numpy as np
import theano
import json

version = '6E'

nets_file = os.path.join('..', 'inputs', 'processed', 'cm_nets.xlsx')
try:
    cm_nets = pd.read_excel(nets_file)
except FileNotFoundError:
    if '../code' not in sys.path:
        sys.path.append('../code')
    from model_data import ModelData
    # Import original data
    with pd.HDFStore('../inputs/responses.h5') as store:
        obs = pd.DataFrame()
        if 'responses' in store:
            obs = store['responses']
        else:
            print("Loading CSV from file")
            obs = pd.read_csv('../inputs/cm_response_confidential.csv')
    md = ModelData(obs)
    md.add_net()
    cm_nets = md.observations(row_filter={'question_code': 'Net'})
    cm_nets.to_excel(nets_file, index=False)

reg_include = ['Alabama',
               'Appalachia',
               'Arkansas',
               'Baltimore',
               'Bay Area']


cm_nets = cm_nets.ix[
    cm_nets.response.notnull() &
    cm_nets.Corps.notnull()]

cm_nets = cm_nets.ix[
    cm_nets.survey_code.str.startswith('14') |
    cm_nets.survey_code.str.startswith('15') |
    cm_nets.survey_code.str.startswith('16')
]

cm_nets = cm_nets.ix[cm_nets.survey_mod == 'F8W']

# cm_nets = cm_nets.ix[cm_nets.Region.isin(reg_include)]


def transform_resp(resp):
    if type(resp) is not float:
        resp[resp == 1] = 0.9999
        resp[resp == -1] = -0.9999
    resp_trans = resp / 2 + 0.5
    resp_log = np.log((1-resp_trans)/resp_trans)
    return resp_log

data = cm_nets.copy()

is_2y = np.array(data.Corps == '2nd year')

survey_dum = data.survey_code.str.get_dummies()
surveys = survey_dum.columns
survey_mtx = survey_dum.as_matrix()

reg_dum = data.Region.str.get_dummies()
regs = reg_dum.columns
reg_mtx = reg_dum.as_matrix()

# print(list(surveys))

col_headers = {'surveys': list(surveys), 'regs': list(regs)}

with pm.Model() as model:
    # survey_mu = pm.Normal(
    #     'survey_mu', mu=transform_resp(0.4), sd=5, shape=survey_mtx.shape[1])

    # reg_delta_dist_mu = pm.Normal('reg_delta_dist_mu', mu=0, sd=3, testval=0)
    # reg_delta_dist_sd = pm.Uniform('reg_delta_dist_sd', lower=0, upper=5, testval=2)

    sigma = pm.HalfCauchy('sigma', beta=10)

    # delta_2y = pm.Normal('delta_2y', mu=0, sd=3, testval=0.22)

    reg_delta = pm.Normal(
        'reg_delta', mu=0, sd=3, testval=0, shape=(reg_mtx.shape[1], survey_mtx.shape[1]))

    resp = transform_resp(data.response)

    resp_mu = (survey_mtx * theano.dot(reg_mtx, reg_delta)).sum(axis=1)

    responses = pm.Normal('responses',
                          mu=resp_mu,
                          sd=sigma,
                          observed=resp)

# print(col_headers)

with model:
    db_name = '../traces/net_log_v{}'.format(version)
    db = pm.backends.Text(db_name)
    # start = pm.find_MAP(fmin=optimize.fmin_powell)
    # start = pm.find_MAP()
    # print('found map:\n{}'.format(start))
    v_params = advi(n=50000)
    df_v_means = v_params.means
    print(df_v_means)
    step = pm.NUTS(scaling=np.power(model.dict_to_array(v_params.stds),2), is_cov=True)
    trace = pm.sample(10000, start=v_params.means, step=step, progressbar=True, trace=db)
    # trace = pm.sample(10000, progressbar=True,  trace=db)

with open('../traces/net_log_v{}_columns.json'.format(version), 'w') as json_file:
    json.dump(col_headers, json_file)
