import sys

new_py_dir = '/Users/mcox/Box Sync/experiments/survey_predictions/code'

if new_py_dir not in sys.path:
    sys.path.append(new_py_dir)

from modeling import basic_model
from pymc import MCMC
import matplotlib.pyplot as plt
import seaborn as sns

m = MCMC(basic_model)
