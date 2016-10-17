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
test_values=np.random.choice(7, 1000, p=[ 0.0010582 ,  0.01904762,  0.03915344,  0.02751323,  0.14285714, 0.48359788,  0.28677249])
# test_values = [0, 2, 4, 3, 14, 50, 30]

@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dvector],otypes=[t.dvector])
def compute_category_p(mu, eff_sigma, full_thresh):
    # thresh = [x + 1.5 for x in range(num_levels - 1)]
    # full_thresh = np.array([-1*np.inf] + thresh + [np.inf])
    fixed_thresh = full_thresh
    for i in range(1, len(fixed_thresh) - 1):
        last = fixed_thresh[i -1]
        if full_thresh[i] < last:
            fixed_thresh[i] = last
        if full_thresh[i] > 6.5:
            fixed_thresh[i] = 6.5
    p = [norm.cdf(float(full_thresh[i]), mu, eff_sigma) -
                    norm.cdf(float(full_thresh[i-1]), mu, eff_sigma) for i in range(1, len(full_thresh))]
    # print(p)
    return np.array(p)
    
compute_category_p.grad = lambda *x: x[0]
    
@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar, t.dscalar],otypes=[t.dvector])
def full_thresh(*thresh):
    thresh = list(thresh)
    return np.array([-1*np.inf, 1.5] + thresh + [6.5, np.inf])
    
full_thresh.grad = lambda *x: x[0]
    
@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar, t.dscalar],otypes=[t.dscalar])
def bar_compute(mu, sigma, thresh1, thresh2):
    return norm.cdf(mu, eff_sigma, thresh2) - norm.cdf(mu, eff_sigma, thresh1)
    
bar_compute.grad = lambda *x: x[0]

with pm.Model() as ordinal_model:
    mu = pm.Normal('mu', mu=(1+ num_levels)/2, sd=1/(num_levels))
    sigma = pm.Uniform('sigma', lower=num_levels/100, upper=num_levels*3)
    # thresh = [x + 1.5 for x in range(num_levels - 1)]
    # full_thresh = np.array([-1*np.inf] + thresh + [np.inf])
    thresh = [pm.Normal('thresh_{}'.format(i), i + 0.5, 1/2**2) for i in range(2, 6)]
    f_thresh = full_thresh(*thresh)
    eff_sigma = sigma
    category_p = compute_category_p(mu, eff_sigma, f_thresh)
    # category_p = [norm.cdf(float(full_thresh[i]), mu, eff_sigma) -
    #                 norm.cdf(float(full_thresh[i-1]), mu, eff_sigma) for i in range(1, len(full_thresh))]
    # category_p= [1/7 for f in range(7)]
    # category_p = [bar_compute(mu, eff_sigma, full_thresh[i-1], full_thresh[i]) for i in range(1, len(full_thresh))]
    # category_p = [bar_compute(mu, eff_sigma, 1.0, 2.0) for i in range(1, len(full_thresh))]
    # category_p = pm.Deterministic('category_p', 
    #     [norm.cdf(float(full_thresh[i]), mu, eff_sigma) -
    #                 norm.cdf(float(full_thresh[i-1]), mu, eff_sigma) for i in range(1, len(full_thresh))]
    # )
    results = pm.Categorical('results', p=category_p, observed=test_values)
    
with ordinal_model:
    # category_p_print = theano.printing.Print('category_of_p')(category_p)
    trace = pm.sample(5)
    
pm.traceplot(trace)
# sns.distplot(trace.get_values('mu'))
# pm.summary(trace)

# map_estimate = pm.find_MAP(model=ordinal_model, fmin=scipy.optimize.fmin_powell)
# print(results.logp(mu=6.5,sigma=10.5, thresh_2=5.0, thresh_3=6, thresh_4=7,thresh_5=8,sigma_interval=2.6,value=test_values))
# print(results.logp(mu=3.5,sigma=0.01, thresh_2=2.5, thresh_3=3.5, thresh_4=4.5,thresh_5=5.5,sigma_interval=0.1,value=test_values))
# print(map_estimate)




