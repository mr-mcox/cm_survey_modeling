import sys
import os
import random
import pandas as pd
# import pymc3 as pm
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon

sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from simulation import counts_from_distribution, compute_net
from surveyformat import add_dimensions, melt

#Import original data
with pd.HDFStore('inputs/responses.h5') as store:
    obs = pd.DataFrame()
    if 'responses' in store:
        obs = store['responses']
    else:
        print("Loading CSV from file")
        obs = pd.read_csv('inputs/cm_response_confidential.csv')

eis = obs[obs.survey_code=='15EIS']


qs = ['CSI1', 'CSI10', 'CSI12', 'CSI2', 'CSI3', 'CSI4', 'CSI5', 'CSI6',
      'CSI7', 'CSI8', 'Culture1']
# qs = ['CSI1']
eis_counts = {q:[sum(eis.ix[eis.question_code==q, 'response'] == r) for r in range(1,8)] for q in qs}

num_runs = 400
run_counts = list()
run_counts.append([eis_counts for x in range(num_runs)])
run_nets = list()
eis_net = compute_net([v for k, v in eis_counts.items()])
run_nets.append([eis_net for x in range(num_runs)])

for survey_seq in range(1,7):
    #Load traces
    traces = dict()
    for q in qs:
        traces[q] = list()
        for level in range(1, 8):
            traces[q].append(pd.read_csv('traces/qc_{}_survey_seq_{}_prev_resp_{}/chain-0.csv'.format(q, survey_seq, level)))
    #Do runs
    counts = list()
    nets = list()
    print('Doing runs for sequence {}'.format(survey_seq))
    for r in range(num_runs):
        count_by_q = dict()
        for q in qs:
            ps = list()
            for level in range(1, 8):
                trace = traces[q][level-1]
                rand_row = random.randrange(len(trace))
                mu = trace.get_value(rand_row, 'master_mu')
                sigma = trace.get_value(rand_row, 'sigma')
                thresh_t = [trace.get_value(rand_row, 'thresh_{}'.format(t)) for t in range(2, 6)]
                thresh = np.concatenate([[-1*np.inf, 1.5], thresh_t, [6.5, np.inf]])
                ps.append([norm.cdf(thresh[i], mu, sigma) -
                 norm.cdf(thresh[i-1], mu, sigma) for i in range(1, len(thresh))])
            count_by_q[q] = counts_from_distribution(run_counts[survey_seq-1][r][q], ps)
        counts.append(count_by_q)
        nets.append(compute_net([v for k, v in count_by_q.items()]))
    run_counts.append(counts)
    run_nets.append(nets)

#Format run_nets for plotting
format_nets = list()
for survey_seq, runs in enumerate(run_nets):
    df = pd.DataFrame({
        'run': [x for x in range(len(runs))],
        'value': runs,
    })
    df['survey_seq'] = survey_seq
    format_nets.append(df)
sim_df = pd.concat(format_nets)
sim_df.to_excel('./outputs/simulation_results.xlsx',index=False)

#Import historical observed data
all = pd.DataFrame()
store_path = 'inputs/store.h5'
if os.path.exists(store_path):
    with pd.HDFStore(store_path) as store:
        all = store['hist']
else:
    df = pd.read_excel('inputs/by_region_hist.xlsx')
    df = add_dimensions(df)
    df = melt(df)
    with pd.HDFStore(store_path) as store:
        store['hist'] = df
    all = df
    
#Make survey categorical
survey_dict = dict(zip(all.survey.unique(), all.survey_seq.unique()))
survey_sort = sorted(survey_dict.keys(), key=lambda x: survey_dict[x])
    
#Filter for CSI
csi = all.loc[all.variable=='CSI-Net']
# csi = all.loc[all.variable=='CSI1-Net']
csi_no_dup = csi.drop_duplicates(['survey','cohort','region'])
fig, ax = plt.subplots()

#Plot simulation range
#Create axes
bot = [np.percentile(sim_df.ix[sim_df.survey_seq==s,'value'],q=[0.5])[0] for s in range(7)]
top = [np.percentile(sim_df.ix[sim_df.survey_seq==s,'value'],q=[99.5])[0] for s in reversed(range(7))]
ys = bot + top
xs = [x for x in range(7)] + [x for x in reversed(range(7))]
points = np.array([xs, ys]).transpose()

ax.add_patch(Polygon(points, fill=True, color='orange'))

csi_in_group = csi_no_dup.ix[csi_no_dup.region.isnull()].sort_values('survey_seq')

for cohort in [2012,2013,2014,2015]:
    xvals = csi_in_group.loc[csi_in_group.cohort == cohort, 'survey_seq']
    yvals = csi_in_group.loc[csi_in_group.cohort == cohort, 'value']
    ax.plot(xvals, yvals, '-', label=cohort)

#Format axes
survey_seq_map = csi.loc[:,['survey_seq','survey']].drop_duplicates().set_index('survey_seq').to_dict()
ax.xaxis.set_ticks([k for k in survey_seq_map['survey'].keys()])
yvals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in yvals])
labels = [survey_seq_map['survey'][i] for i in ax.get_xticks().tolist()]
ax.set_xticklabels(labels)
ax.set_xlabel('Survey')
ax.set_ylabel('Net CSI')

ax.legend()

# plt.savefig('./outputs/simulation_results_one_q.png')
plt.savefig('./outputs/simulation_results.png')