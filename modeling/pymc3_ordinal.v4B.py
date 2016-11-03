import sys
import pandas as pd
import numpy as np
import pymc3 as pm
from scipy.stats import norm
import theano.tensor as t
import theano

sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from model_data import ModelData

with pd.HDFStore('../inputs/responses.h5') as store:
    df = pd.DataFrame()
    if 'responses' in store:
        df = store['responses']
    else:
        print("Loading CSV from file")
        df = pd.read_csv('../inputs/cm_response_confidential.csv')

md = ModelData(df.loc[~df.survey_code.str.contains('1516')])
qs = ['Culture1']
num_levels = 7


@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dvector], otypes=[t.dvector])
def compute_category_p(mu, sigma, thresh):
    return norm.cdf(thresh[1:], mu, sigma) - norm.cdf(thresh[:-1], mu, sigma)

@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar, t.dscalar], otypes=[t.dvector])
def full_thresh(*t_in):
    t_a = list(t_in)
    t_b = [x if x < 6.5 else 6.5 for x in t_a]
    t_c = [x if x > 1.5 else 1.5 for x in t_a]
    thresh = np.maximum.accumulate(t_c)
    return np.concatenate([[-1*np.inf, 1.5], thresh, [6.5, np.inf]])

for question_code in qs:
    for survey_seq in range(1, 7):
        for prev_response in range(1, 8):
            input_data = md.observations(row_filter={
                                         'survey_seq': survey_seq,
                                         'prev_response': prev_response,
                                         'question_code': question_code},
                                         group_col='survey_code')
            num_surveys = len(input_data)
            print("Starting trace {} {} {}".format(
                question_code, survey_seq, prev_response))

            with pm.Model() as ordinal_model:
                master_mu = pm.Normal(
                    'master_mu', mu=prev_response, sd=1/(num_levels))
                survey_mu = pm.Normal(
                    'survey_mu', mu=master_mu, sd=1/(num_levels), shape=num_surveys)
                sigma = pm.Uniform(
                    'sigma', lower=num_levels/100, upper=num_levels*3)
                thresh = [
                    pm.Normal('thresh_{}'.format(i), i + 0.5, 1/2**2) for i in range(2, 6)]
                f_thresh = full_thresh(*thresh)
                category_p = [
                    compute_category_p(survey_mu[i], sigma, f_thresh) for i in range(num_surveys)]
                results = [pm.Categorical('results', p=category_p[i], observed=input_data[
                                          survey].response-1) for i, survey in enumerate(input_data.keys())]

            with ordinal_model:
                db_name = '../traces/qc_{}_survey_seq_{}_prev_resp_{}'.format(
                    question_code, survey_seq, prev_response)
                db = pm.backends.Text(db_name)
                step = pm.Metropolis()
                trace = pm.sample(3500, progressbar=True, step=step, trace=db)

            # Get rid of burned rows
            csv_file = '{}/chain-0.csv'.format(db_name)
            trace_df = pd.read_csv(csv_file)
            trace_df.iloc[2000:].to_csv(csv_file, index=False)
    
