import sys
import numpy as np
from scipy.stats import norm
from pymc import MCMC, Normal, Uniform, Multinomial, Categorical
import pymc
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from surveyformat import add_dimensions, melt
from model_data import ModelData

# md = ModelData('../inputs/cm_response_confidential.csv')
# data = md.proportion_for_cut(prev_response=7, unit='survey_code', groups={'survey_seq': 1, 'question_code': 'CSI1'})
# matrix = np.array(data).transpose()
matrix = np.array([[ 0.0010582 ,  0.01904762,  0.03915344,  0.02751323,  0.14285714,
         0.48359788,  0.28677249],
       [ 0.00125156,  0.01001252,  0.0350438 ,  0.04130163,  0.18523154,
         0.47934919,  0.24780976],
       [ 0.00175285,  0.00964067,  0.02804557,  0.028922  ,  0.13847502,
         0.47151621,  0.32164768],
       [ 0.00460829,  0.01382488,  0.04608295,  0.02488479,  0.17603687,
         0.44239631,  0.2921659 ],
       [ 0.00580833,  0.01645692,  0.03291384,  0.03872217,  0.16844143,
         0.37754114,  0.36011617],
       [ 0.00116009,  0.01276102,  0.02204176,  0.03944316,  0.12180974,
         0.4199536 ,  0.38283063]])
# matrix = matrix * 6000

num_levels = matrix.shape[1]
mu = Normal('mu', (1+ num_levels)/2, 1/(num_levels**2))
sigma = Uniform('sigma', num_levels/100, num_levels*3)
thresh = [x + 1.5 for x in range(num_levels - 1)]

# for i in range(1, len(thresh) -1):
#     thresh[i] = Normal('thresh_{}'.format(i), i + 1.5, 1/2**2)

@pymc.deterministic
def category_p( mu=mu, sigma=sigma, thresh=thresh):
    p = list()
    eff_sigma = 1/sigma**2
    
    fixed_thresh = [float(t) for t in thresh]
    for i in range(1, len(fixed_thresh)):
        last = fixed_thresh[i -1]
        if thresh[i] < last:
            fixed_thresh[i] = last
            
    p.append(norm.cdf(fixed_thresh[0], mu, eff_sigma))
    for i in range(1, len(fixed_thresh)):
        p.append(norm.cdf(float(fixed_thresh[i]), mu, eff_sigma) -
                    norm.cdf(float(fixed_thresh[i-1]), mu, eff_sigma))
    p.append(1 - norm.cdf(fixed_thresh[-1], mu, eff_sigma))
    # print('mu: {} sigma: {} fixed_thresh: {} p: {}'.format(mu, sigma, str(fixed_thresh), str(p)))
    return [p]
    
# p_by_question = Multinomial('p_by_question', n=matrix.shape[0], p=category_p, value=matrix[0], observed=True)
# test_values = [[1/7 for x in range(7)] for y in range(3)]
test_values = [1/7 for x in range(7)]
p_by_question = Categorical('p_by_question', p=category_p, value=test_values, observed=True)
    
m = MCMC([p_by_question, mu, sigma, thresh, category_p])
m.sample(iter=1000, burn=100)

sns.distplot(m.trace('mu')[:])
sns.distplot(m.trace('sigma')[:])
print(m.trace('category_p')[-5])











