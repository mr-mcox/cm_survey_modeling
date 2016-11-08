import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json
from scipy.stats.mstats import mquantiles

def untransform_resp(t_resp):
    r_e = 1/(1+np.exp(t_resp))
    return (r_e - 0.5)*2
    
nets_file = os.path.join('..', 'inputs', 'processed', 'cm_nets.xlsx')
cm_nets = pd.read_excel(nets_file)

with open(os.path.join('..','traces','net_log_v6E_columns.json')) as col_file:
    heads = json.load(col_file)
    
chain = pd.read_csv(os.path.join('..','traces','net_log_v6E','chain-0.csv'))
chain['i'] = range(len(chain))
ch = pd.melt(chain, id_vars='i')

nat = ch.ix[ch.variable.str.contains('survey_mu')]
nat['s_num'] = ch.variable.str.extract('survey_mu__(\d)', expand=False)
nat_q = nat.groupby('variable').quantile([0.025,0.5,0.975])
nat_q['v_trans'] = nat_q.value.map(untransform_resp)

reg = ch.ix[ch.variable.str.contains('reg_delta')]
reg_q = nat.groupby('variable').quantile([0.025,0.5,0.975])
reg_q['v_trans'] = reg_q.value.map(untransform_resp)
