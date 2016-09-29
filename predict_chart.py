import pandas as pd
import os
import sys
import matplotlib as ml
import matplotlib.pyplot as plt
from matplotlib import lines as mpl_lines
import seaborn as sns
import numpy as np
import random
sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from surveyformat import add_dimensions, melt

os.chdir('/Users/mcox/Box Sync/experiments/survey_predictions')

#Import data
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
    
def count_to_net(counts):
    total = sum(counts)
    strong = sum(counts[5:]) / total
    weak = sum(counts[:4]) / total
    return strong - weak
    
def sum_responses(resp_matrix):
    return np.sum(resp_matrix,axis=0)
    
base_p_seq = [(x+1)**2 for x in range(7)]
full_p_seq = base_p_seq + [x for x in reversed(base_p_seq[:-1])]
prob_matrix_count = [
    [x**((7-start)*2.8/7) for x in full_p_seq[start:start+7]]
    for start in range(6,-1,-1)]

def compute_prob_matrix(prob_matrix_count):
    # print("Before stack {}".format(prob_matrix_count))
    stacked = np.stack([np.array(a)/sum(a) for a in prob_matrix_count])
    # print("After stack {}".format(stacked))
    return stacked
    
def roughed_up_matrix(prob_matrix_count):
    roughed = [[x* random.uniform(0.5,2) for x in a] for a in prob_matrix_count]
    return compute_prob_matrix(roughed)
    
def simulate(num_runs):
    records = list()
    for run_i, run in enumerate(range(num_runs)):
        prob_matrix = roughed_up_matrix(prob_matrix_count)
        counts = [50, 150, 300, 400, 900, 2200, 2000]
        records.append((run_i, count_to_net(counts), 0))
        for seq in range(6):
            response_counts = list()
            for i, count in enumerate(counts):
                response_counts.append(np.random.multinomial(count, prob_matrix[i], 1)[0])
            counts = sum_responses(np.stack(response_counts))
            records.append((run_i, count_to_net(counts), seq+1))
    return pd.DataFrame.from_records(records, columns=['run', 'value', 'survey_seq'])
            
model_runs = simulate(50)
    
#Make survey categorical
survey_dict = dict(zip(all.survey.unique(), all.survey_seq.unique()))
survey_sort = sorted(survey_dict.keys(), key=lambda x: survey_dict[x])
# all.survey = all.survey.astype('category', categories = survey_sort, ordered=True)
    
#Filter for CSI
csi = all.loc[all.variable=='CSI-Net']
csi_no_dup = csi.drop_duplicates(['survey','cohort','region'])
# model_runs = csi_no_dup.loc[csi_no_dup.cohort==2014]
# model_runs.reset_index(inplace=True)
fig, ax = plt.subplots()

# sns.tsplot(model_runs, time='survey_seq', value='value', unit='run', err_style="unit_traces", ax=ax, color='orange')
sns.tsplot(model_runs, time='survey_seq', value='value', unit='run', ci=[99], ax=ax, color='orange')

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

plt.savefig('./outputs/simulation_demo_uncertainty_range.png')

plt.show()