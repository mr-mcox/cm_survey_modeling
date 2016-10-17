import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
from scipy.stats import norm
import theano.tensor as t
import theano
import scipy

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

num_levels = matrix.shape[1]
test_values=[
            np.random.choice(7, 6000, p=[ 0.0010582 ,  0.01904762,  0.03915344,  0.02751323,  0.14285714, 0.48359788,  0.28677249]),
            np.random.choice(7, 6000, p=[ 0.00125156,  0.01001252,  0.0350438 ,  0.04130163,  0.18523154, 0.47934919,  0.24780976]),
            ]


@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dvector],otypes=[t.dvector])
def compute_category_p(mu, sigma, thresh):
    return np.array([norm.cdf(thresh[i], mu, sigma) - norm.cdf(thresh[i-1], mu, sigma) for i in range(1, len(thresh))])
    
@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar, t.dscalar],otypes=[t.dvector])
def full_thresh(*t_in):
    t_a = list(t_in)
    t_b = [x if x < 6.5 else 6.5 for x in t_a]
    t_c = [x if x > 1.5 else 1.5 for x in t_a]
    thresh = np.maximum.accumulate(t_c)
    return np.concatenate([[-1*np.inf, 1.5], thresh, [6.5, np.inf]])

with pm.Model() as ordinal_model:
    mu = pm.Normal('mu', mu=(1+ num_levels)/2, sd=1/(num_levels), shape=2)
    sigma = pm.Uniform('sigma', lower=num_levels/100, upper=num_levels*3, shape=2)
    thresh = [pm.Normal('thresh_{}'.format(i), i + 0.5, 1/2**2) for i in range(2, 6)]
    f_thresh = full_thresh(*thresh)
    category_p = [compute_category_p(mu[i], sigma[i], f_thresh) for i in range(2)]
    results = [pm.Categorical('results_{}'.format(i), p=category_p[i], observed=test_values[i]) for i in range(2)]
    
with ordinal_model:
    # start_map = pm.find_MAP(model=ordinal_model, fmin=scipy.optimize.fmin_powell)
    step = pm.Metropolis()
    
    trace = pm.sample(4000, progressbar=True, step=step)
    
pm.traceplot(trace[2000:])

#ordinal_model.profile(ordinal_model.logpt).summary()













