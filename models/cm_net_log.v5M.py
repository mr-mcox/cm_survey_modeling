import sys
import os
import pandas as pd
import pymc3 as pm
import numpy as np
import theano
import yaml
from scipy import optimize

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

cm_nets = cm_nets.ix[
    cm_nets.prev_response.notnull() &
    cm_nets.response.notnull() &
    cm_nets.Corps.notnull()]

cm_nets = cm_nets.ix[
    cm_nets.survey_code.str.startswith('14') |
    cm_nets.survey_code.str.startswith('15') |
    cm_nets.survey_code.str.startswith('16')
]


def transform_resp(resp):
    resp = resp.copy()
    resp[resp == 1] = 0.9999
    resp[resp == -1] = -0.9999
    resp_trans = resp / 2 + 0.5
    resp_log = np.log((1-resp_trans)/resp_trans)
    return resp_log

col_headers = dict()


for sm in ['F8W', 'MYS', 'EYS']:

    data = cm_nets.ix[(cm_nets.survey_mod == sm)].copy()

    is_2y = np.array(data.Corps == '2nd year')

    survey_dum = data.survey_code.str.get_dummies()
    surveys = survey_dum.columns
    survey_mtx = survey_dum.as_matrix()

    reg_dum = data.Region.str.get_dummies()
    regs = reg_dum.columns
    reg_mtx = reg_dum.as_matrix()

    print(list(surveys))

    col_headers[sm] = {'surveys': list(surveys), 'regs': list(regs)}

    with pm.Model() as log_model:
        admin_alpha_mu = pm.Normal('admin_alpha_mu', mu=0, sd=3, testval=0)
        admin_alpha_sd = pm.Uniform(
            'admin_alpha_sd', lower=-10, upper=10, testval=1)
        admin_beta_mu = pm.Normal('admin_beta_mu', mu=0, sd=3, testval=0)
        admin_beta_sd = pm.Uniform(
            'admin_beta_sd', lower=-10, upper=10, testval=1)
        sigma = pm.HalfCauchy('sigma', beta=10)

        alpha_delta_2y = pm.Normal('alpha_delta_2y', mu=0, sd=3, testval=0)
        beta_delta_2y = pm.Normal('beta_delta_2y', mu=0, sd=3, testval=0)

        alpha = pm.Normal('alpha',
                          mu=admin_alpha_mu,
                          sd=admin_alpha_sd,
                          shape=survey_mtx.shape[1])
        beta = pm.Normal('beta',
                         mu=admin_beta_mu,
                         sd=admin_beta_sd,
                         shape=survey_mtx.shape[1])

        beta_delta_region = pm.Normal('beta_delta_region', mu=0, sd=3, testval=0, shape=reg_mtx.shape[1])
        alpha_delta_region = pm.Normal('alpha_delta_region', mu=0, sd=3, testval=0, shape=reg_mtx.shape[1])

        prev = transform_resp(data.prev_response)
        resp = transform_resp(data.response)

        alpha_v = theano.dot(survey_mtx, alpha) + \
            alpha_delta_2y * is_2y + theano.dot(reg_mtx, alpha_delta_region)
        beta_v = theano.dot(survey_mtx, beta) + \
            beta_delta_2y * is_2y + theano.dot(reg_mtx, beta_delta_region)

        responses = pm.Normal('responses',
                              mu=beta_v*prev+alpha_v,
                              sd=sigma,
                              observed=resp)

    with log_model:
        db_name = '../traces/net_log_v5M_survey_mod_{}'.format(sm)
        db = pm.backends.Text(db_name)
        start = pm.find_MAP(fmin=optimize.fmin_powell)
        print('found map')
        trace = pm.sample(10000, start=start, progressbar=True,  trace=db)
        # trace = pm.sample(10000, progressbar=True,  trace=db)

with open('../traces/net_log_v5M_columsn.yaml', 'w') as yml_file:
    yaml.dump(col_headers, yml_file)
