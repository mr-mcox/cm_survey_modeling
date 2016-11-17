import pandas as pd
import json
from os import path
from helpers.examine_trace import label_trace
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with pd.HDFStore(path.join('..', 'inputs', 'processed', 'reg_influence.h5')) as store:
    oall = pd.DataFrame()
    regs = pd.DataFrame()
    
    if 'overall' in store:
        oall = store['overall']
        regs = store['region']
    else:
        with pd.HDFStore(path.join('..', 'inputs', 'responses.h5')) as resp_store:
            resp = pd.DataFrame()
            assert 'responses' in resp_store
            resp = resp_store['responses']
        
        qcs = resp.loc[(resp.survey_code == '1617F8W'), 'question_code'].unique()
        overalls = list()
        regionals = list()
        
        for qc in qcs:
            print('Processing {}'.format(qc))
            db_name = 'region_ordinal_qc_{}'.format(qc)
            db_file = path.join('..', 'traces', db_name)
            trace = pd.read_csv(path.join(db_file, 'chain-0.csv'))
            
            with open('{}.json'.format(db_file)) as json_file:
                heads = json.load(json_file)
                
            lt = label_trace(trace, heads)
            lt_oall = lt['overall']
            lt_oall['question_code'] = qc
            overalls.append(lt_oall)
            
            lt_reg = lt[1]
            lt_reg['question_code'] = qc
            regionals.append(lt_reg)
            
        oall = pd.concat(overalls)
        regs = pd.concat(regionals)
        
        store['overall'] = oall
        store['region'] = regs
        
nat_csi = oall.groupby('i')['nat_net'].mean().reset_index()
nat_csi_qs = nat_csi.loc[:,'nat_net'].quantile([0.05, 0.25, 0.75, 0.95])
b_width_l = nat_csi_qs[0.95] - nat_csi_qs[0.05]
b_width_s = nat_csi_qs[0.75] - nat_csi_qs[0.25]
    
reg_csi = regs.groupby(['Region','i'])['reg_net'].mean().reset_index()

reg_csi_qs = reg_csi.groupby('Region')['reg_net'].quantile([0.05, 0.25, 0.75, 0.95]).unstack(level=-1)
reg_csi_qs.sort_values(by=0.05, inplace=True)
reg_csi_qs['b_width_l'] = reg_csi_qs[0.95] - reg_csi_qs[0.05]
reg_csi_qs['b_width_s'] = reg_csi_qs[0.75] - reg_csi_qs[0.25]

fig, ax = plt.subplots(figsize=(10, 15))
fig.suptitle('CSI at 1617F8W by Region')

colors = sns.color_palette("Greys_r")
plt.barh(bottom=-0.5, width=b_width_l, left=nat_csi_qs[0.05], height=len(reg_csi_qs) + 0.5, color=colors[4], linewidth=0, label='90% credible National CSI')
plt.barh(bottom=-0.5, width=b_width_s, left=nat_csi_qs[0.25], height=len(reg_csi_qs) + 0.5, color=colors[3], linewidth=0, label='50% credible National CSI')

colors = sns.color_palette("Blues_r")
plt.barh(bottom=np.arange(len(reg_csi_qs))-0.5, width=reg_csi_qs.b_width_l, left=reg_csi_qs[0.05], color=colors[2], linewidth=0, label='90% credible CSI')
plt.barh(bottom=np.arange(len(reg_csi_qs))-0.5, width=reg_csi_qs.b_width_s, left=reg_csi_qs[0.25], color=colors[1], linewidth=0, label='50% credible CSI')
plt.ylim(ymin=-0.5, ymax=len(reg_csi_qs))
ax.yaxis.set_ticks(range(len(reg_csi_qs)))
ax.set_yticklabels(reg_csi_qs.index)
xvals = ax.get_yticks()
ax.set_xticklabels(['{:3.0f}%'.format(x*10) for x in xvals])
ax.legend(loc='lower right')
plt.show()


