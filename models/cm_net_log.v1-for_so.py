import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt

# Some helper functions
def cdf(x, location=0, scale=1):
    epsilon = np.array(1e-32, dtype=theano.config.floatX)

    location = tt.cast(location, theano.config.floatX)
    scale = tt.cast(scale, theano.config.floatX)

    div = tt.sqrt(2 * scale ** 2 + epsilon)
    div = tt.cast(div, theano.config.floatX)

    erf_arg = (x - location) / div
    return .5 * (1 + tt.erf(erf_arg + epsilon))


def percent_to_thresh(idx, vect):
    return 5 * tt.sum(vect[:idx + 1]) + 1.5


def full_thresh(thresh):
    idxs = tt.arange(thresh.shape[0] - 1)
    thresh_mod, updates = theano.scan(fn=percent_to_thresh,
                                      sequences=[idxs],
                                      non_sequences=[thresh])
    return tt.concatenate([[-1 * np.inf, 1.5], thresh_mod, [6.5, np.inf]])


def compute_ps(thresh, location, scale):
    f_thresh = full_thresh(thresh)
    return cdf(f_thresh[1:], location, scale) - cdf(f_thresh[:-1], location, scale)

# Generate data
real_ps = [0.05, 0.05, 0.1, 0.1, 0.2, 0.3, 0.2]
data = np.random.choice(7, size=1000, p=real_ps)

# Run model
with pm.Model() as model:
    mu = pm.Normal('mu', mu=4, sd=3)
    sigma = pm.Uniform('sigma', lower=0.1, upper=70)
    thresh = pm.Dirichlet('thresh', a=np.ones(5))

    cat_p = compute_ps(thresh, mu, sigma)

    results = pm.Categorical('results', p=cat_p, observed=data)

with model:
    start = pm.find_MAP()
    trace = pm.sample(2000, start=start)
