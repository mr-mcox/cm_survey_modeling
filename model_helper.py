import theano
import theano.tensor as tt
import numpy as np
from theano.ifelse import ifelse

epsilon = np.array(1e-32, dtype=theano.config.floatX)
e_dif = np.array(1e-32, dtype=theano.config.floatX)


def cdf(x, location=0, scale=1):
    # Adapted from Breeze
    location = tt.cast(location, theano.config.floatX)
    scale = tt.cast(scale, theano.config.floatX)

    div = tt.sqrt(2 * scale ** 2 + epsilon)
    div = tt.cast(div, theano.config.floatX)

    erf_arg = (x - location) / div
    return .5 * (1 + tt.erf(erf_arg + epsilon))


def compute_ps(thresh, location, scale):
    f_thresh = full_thresh(thresh)
    return cdf(f_thresh[1:], location, scale) - cdf(f_thresh[:-1], location, scale)


def percent_to_thresh(idx, vect):
    return 5 * tt.sum(vect[:idx + 1]) + 1.5


def full_thresh(thresh):
    idxs = tt.arange(thresh.shape[0]-1)
    thresh_mod, updates = theano.scan(fn=percent_to_thresh,
                                      sequences=[idxs],
                                      non_sequences=[thresh])
    return tt.concatenate([[-1*np.inf, 1.5], thresh_mod, [6.5, np.inf]])
