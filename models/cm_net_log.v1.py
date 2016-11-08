import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt

nets_file = os.path.join('..', 'inputs', 'processed', 'cm_nets.xlsx')
try:
    cm_nets = pd.read_excel(nets_file)
except FileNotFoundError:
    if '../code' not in sys.path:
        sys.path.append('../code')
    from model_data import ModelData
    #Import original data
    with pd.HDFStore('../inputs/responses.h5') as store:
        obs = pd.DataFrame()
        if 'responses' in store:
            obs = store['responses']
        else:
            print("Loading CSV from file")
            obs = pd.read_csv('../inputs/cm_response_confidential.csv')
    md = ModelData(obs)
    md.add_net()
    cm_nets = md.observations(row_filter={'question_code':'Net'})
    cm_nets.to_excel(nets_file, index=False)

cm_nets = cm_nets.ix[cm_nets.prev_response.notnull() & cm_nets.response.notnull()]
    
num_admins = 6

s_groups =cm_nets.ix[(cm_nets.survey_seq>=1) & (cm_nets.survey_seq <= num_admins),
                    ['survey_code','survey_seq']].drop_duplicates().to_records(index=False)

with pm.Model() as log_model:
    admin_alpha_mu = pm.Normal('admin_alpha_mu', mu=0, sd=3, testval=0, shape=num_admins)
    admin_alpha_sd = pm.Uniform('admin_alpha_sd', lower=-10, upper=10, testval=1, shape=num_admins)
    admin_beta_mu = pm.Normal('admin_beta_mu', mu=0, sd=3, testval=0, shape=num_admins)
    admin_beta_sd = pm.Uniform('admin_beta_sd', lower=-10, upper=10, testval=1, shape=num_admins)
    sigma = pm.HalfCauchy('sigma', beta=10, shape=num_admins)
    
    num_groups = len(s_groups)
    betas = [None for x in range(num_groups)]
    alphas = [None for x in range(num_groups)]
    responses = [None for x in range(num_groups)]
    
    for i, (survey, seq) in enumerate(s_groups):
        seq_i = seq - 1
        prev = cm_nets.ix[(cm_nets.survey_code==survey) &
                          (cm_nets.survey_seq==seq),
                          'prev_response'
                         ]
        resp = cm_nets.ix[(cm_nets.survey_code==survey) &
                          (cm_nets.survey_seq==seq),
                          'response'
                         ]
        alphas[i] = pm.Normal('a_{}_{}'.format(survey,seq), 
                               mu=admin_alpha_mu[seq_i], 
                               sd=admin_alpha_sd[seq_i])
        betas[i] = pm.Normal('b_{}_{}'.format(survey,seq), 
                               mu=admin_beta_mu[seq_i], 
                               sd=admin_beta_sd[seq_i])
        responses[i] = pm.Normal('r_{}_{}'.format(survey,seq),
                                  mu=1/(1 + tt.exp(betas[i]*prev+alphas[i])),
                                  sd=sigma[seq_i],
                                  observed=resp)

with log_model:
    start = pm.find_MAP()
    print('found map')
    trace = pm.sample(300, start=start, progressbar=True)
