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
# qs = ['CSI1']
qs = ['CSI1', 'CSI10', 'CSI12', 'CSI2', 'CSI3', 'CSI4', 'CSI5', 'CSI6',
      'CSI7', 'CSI8', 'Culture1']
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
        input_data = md.observations(row_filter={
                                     'survey_seq': survey_seq,
                                     'question_code': question_code})
        input_data = input_data[input_data.prev_response.notnull()]
        num_surveys = len(input_data.survey_code.unique())
        print("Starting trace {} {}".format(
            question_code, survey_seq))

        with pm.Model() as ordinal_model:
            level_mu = pm.Normal('level_mu', mu=[x for x in range(num_levels)], sd=1/(num_levels), shape=num_levels)
            cohort_mu = pm.Normal('cohort_mu', mu=level_mu, sd=1/(num_levels), shape=(num_surveys, num_levels))
            sigma = pm.Uniform('sigma', lower=num_levels/100, upper=num_levels*3, shape=num_levels)
            thresh = [
                pm.Normal('thresh_{}'.format(i), i + 0.5, 1/2**2) for i in range(2, 6)]
            f_thresh = full_thresh(*thresh)
            # category_p = list()
            # for i in range(num_surveys):
            #     level_p = list()
            #     for level in range(num_levels):
            #         level_p.append(compute_category_p(cohort_mu[i][level], sigma[level], f_thresh))
            #     category_p.append(level_p)
            category_p = [
                [compute_category_p(cohort_mu[i][level], sigma[level], f_thresh) for level in range(num_levels)] for i in range(num_surveys)]
            results = list()
            for survey_i, survey in enumerate(input_data.survey_code.unique()):
                for prev_response in input_data.prev_response.unique():
                    responses = input_data.ix[(input_data.survey_code==survey) & (input_data.prev_response==prev_response),'response'] -1
                    results.append(pm.Categorical('results_{}_{}'.format(survey,prev_response),
                                    p=category_p[survey_i][int(prev_response)-1], 
                                    observed=responses))

        with ordinal_model:
            db_name = '../traces/v5_qc_{}_survey_seq_{}'.format(
                question_code, survey_seq)
            db = pm.backends.Text(db_name)
            step = pm.Metropolis()
            trace = pm.sample(6000, progressbar=True, step=step, trace=db)
            # trace = pm.sample(6000, progressbar=True, step=step)
            # ordinal_model.profile(ordinal_model.logpt).summary()

        # Get rid of burned rows
        csv_file = '{}/chain-0.csv'.format(db_name)
        trace_df = pd.read_csv(csv_file)
        trace_df.iloc[3000:].to_csv(csv_file, index=False)




