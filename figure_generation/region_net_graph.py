import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json
from scipy.stats import norm

version = '4_min'

q_df = list()
qs = 'CSI1 CSI2 CSI3 CSI4 CSI5 CSI6 CSI7 CSI8 CSI10 CSI12 Culture1'.split()
for question in qs:
    print('Building frame for {}'.format(question))
    question = question
    basename = 'cm_cat_v{}_ques_{}'.format(version, question)
    
    #Load headers
    with open('../traces/{}_columns.json'.format(basename)) as json_file:
        heads = json.load(json_file)
    
    #Load trace
    chain = pd.read_csv(os.path.join('..','traces','{}'.format(basename),'chain-0.csv'))
    chain['i'] = range(len(chain))
    ch = pd.melt(chain, id_vars='i')
    
    #Add region, survey, question code
    ch['question_code'] = question
    s_val = pd.to_numeric(ch.variable.str.extract('\w+__(\d+)_\d+',expand=False))
    ch['survey'] = s_val.map(lambda x: heads['questions'][int(x)] if not np.isnan(x) else None)
    r_val = pd.to_numeric(ch.variable.str.extract('\w+__\d+_(\d+)',expand=False))
    ch['region'] = r_val.map(lambda x: heads['regions'][int(x)] if not np.isnan(x) else None)
    
    
    #Construct thresholds
    ths = ch.ix[ch.variable.str.contains('thresh__')].pivot('i', 'variable', 'value').as_matrix()
    ths_sum = np.add.accumulate(ths, axis=1)
    inner_thresh = ths_sum * 5 + 1.5
    it2 = np.insert(inner_thresh,0,1.5,axis=1)
    it2 = np.insert(it2,0,-1*np.inf,axis=1)
    it2 = np.append(it2,np.inf * np.ones(it2.shape[0]).reshape(-1,1),axis=1)
    
    #Compute ps
    ch_w_idx = ch.set_index(['survey','region', 'i'])
    # assert (ch_w_idx.index.unique()).all()
    mus = ch_w_idx.ix[ch_w_idx.variable.str.contains('mu__'), 'value']
    sigmas = ch_w_idx.ix[ch_w_idx.variable.str.contains('sigma__'), 'value']
    
    it3 = np.repeat(it2, mus.shape[0] / it2.shape[0], axis=0)
    
    lows = norm.cdf(it3[:,:-1].T, mus, sigmas).T
    highs = norm.cdf(it3[:,1:].T, mus, sigmas).T
    ps = (highs-lows)
    
    weak = ps[:,:4].sum(axis=1)
    strong = ps[:,5:].sum(axis=1)
    
    q_df.append(pd.DataFrame({'weak': weak, 'strong': strong}, index=mus.index))

sum_q = q_df[0]

for df in q_df[1:]:
    sum_q = sum_q + df

mean_q = sum_q / len(q_df)
mean_q['net'] = mean_q.strong - mean_q.weak

qtl = mean_q.reset_index().groupby(['survey','region']).quantile([0.025, 0.5, 0.975])

surveys = '1314F8W 1415F8W 1516F8W 1617F8W'.split()
lvls = [0.025, 0.975]

reg = 'Alabama'
locs = [(s, reg, lvls[0]) for s in surveys] + [(s, reg, lvls[1]) for s in surveys[::-1]]
qtl_vals = np.array(qtl.loc[locs,'net']).reshape(2,len(surveys))

plt.fill_between(range(len(surveys)), qtl_vals[0], qtl_vals[1], alpha=0.15)
plt.show()


