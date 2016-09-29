from pymc import Multinomial, Dirichlet
import numpy as np
from pymc import MCMC

NUM_DRAWS = 100
NUM_SAMPLES = 2
TRUE_PROPS = [0.10, .20, .20, .15, .15, .10, .10]


def generate_data():
    data = []
    for i in range(NUM_SAMPLES):
        x = np.random.multinomial(NUM_DRAWS, TRUE_PROPS)
        data.append(x)
    return data

change_prob = Dirichlet('change_prob', theta=[1.0 for x in range(7)])

actual_results = [
    [160, 160, 160, 100, 140, 140, 90],
    [160, 160, 160, 100, 140, 140, 90],
]
num_draws = 100
# results = Multinomial(
#     'results', n=100, p=change_prob, value=generate_data(), observed=True)
results = Multinomial(
    'survey_results', n=sum(actual_results[0]), p=change_prob, value=actual_results, observed=True)

mcmc = MCMC([change_prob, results])
mcmc.sample(iter=100000, burn=10000, thin=100)
