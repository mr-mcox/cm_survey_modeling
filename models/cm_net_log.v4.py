import sys
import os
import pandas as pd
import pymc3 as pm
import numpy as np

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


def transform_resp(resp):
    resp[resp == 1] = 0.9999
    resp[resp == -1] = -0.9999
    resp_trans = resp / 2 + 0.5
    resp_log = np.log((1-resp_trans)/resp_trans)
    return resp_log

excluded_surveys = ['0910MYS']

# for sm in ['F8W', 'MYS', 'EYS']:
for sm in ['MYS']:

    surveys = cm_nets.ix[(cm_nets.survey_mod == sm) &
                         (~cm_nets.survey_code.isin(excluded_surveys)),
                         'survey_code'].unique()

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

        num_groups = len(surveys)
        betas = [None for x in range(num_groups)]
        alphas = [None for x in range(num_groups)]
        responses = [None for x in range(num_groups)]

        for i, survey in enumerate(surveys):
            survey_mask = (cm_nets.survey_code == survey)
            prev = transform_resp(cm_nets.ix[survey_mask, 'prev_response'])
            resp = transform_resp(cm_nets.ix[survey_mask, 'response'])
            is_2y = np.array(cm_nets.ix[survey_mask, 'Corps'] == '2nd year')
            print('{}: prev len: {}, cur len: {}, num 2nd: {}'.format(
                survey, len(prev), len(resp), is_2y.sum()))

            alphas[i] = pm.Normal('a_{}'.format(survey),
                                  mu=admin_alpha_mu,
                                  sd=admin_alpha_sd)
            betas[i] = pm.Normal('b_{}'.format(survey),
                                 mu=admin_beta_mu,
                                 sd=admin_beta_sd)
            beta_v = betas[i] + (beta_delta_2y * is_2y)
            alpha_v = alphas[i] + (alpha_delta_2y * is_2y)
            responses[i] = pm.Normal('r_{}'.format(survey),
                                     mu=beta_v*prev+alpha_v,
                                     sd=sigma,
                                     observed=resp)

    with log_model:
        db_name = '../traces/net_log_v4_survey_mod_{}'.format(sm)
        db = pm.backends.Text(db_name)
        # start = pm.find_MAP()
        # print('found map')
        # trace = pm.sample(10000, start=start, progressbar=True,  trace=db)
        trace = pm.sample(10000, progressbar=True,  trace=db)
