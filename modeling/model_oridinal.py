import sys
import numpy as np
from scipy.stats import norm
from pymc import MCMC, Normal, Uniform, Categorical
import pymc


sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from surveyformat import add_dimensions, melt
from model_data import ModelData

md = ModelData('../inputs/cm_response_confidential.csv')
data = md.proportion_for_cut(prev_response=7, unit='survey_code', groups={'survey_seq': 1, 'question_code': 'CSI1'})
matrix = np.array(data).transpose()

num_levels = matrix.shape[1]
mu = Normal('mu', (1+ num_levels)/2, 1/(num_levels**2))
sigma = Uniform('sigma', num_levels/1000, num_levels*10)
thresh = [x + 1.5 for x in range(num_levels - 1)]

for i in range(1, len(thresh) -1):
    thresh[i] = Normal('thresh_{}'.format(i), i + 1.5, 1/2**2)

@pymc.deterministic
def category_p(mu=mu, sigma=sigma, thresh=thresh):
    p = list()
    p.append(norm.cdf(thresh[0], mu, 1/sigma**2))
    for i in range(1, len(thresh) - 2 ):
        p.append(max(0, norm.cdf(thresh[i], mu, 1/sigma**2) -
                    norm.cdf(thresh[i-1], mu, 1/sigma**2)))
    p.append(1 - norm.cdf(thresh[num_levels - 2], mu, 1/sigma**2))
    print(p)
    return p
    
p_by_question = Categorical('p_by_question', category_p, value=matrix, observed=True)
    
mcmc = MCMC([p_by_question, mu, sigma, thresh, category_p])
mcmc.sample(iter=100)


