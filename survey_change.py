import pandas as pd
import os
import sys
import matplotlib as ml
import matplotlib.pyplot as plt
from matplotlib import lines as mpl_lines
import seaborn as sns
sys.path.append('/Users/mcox/Box Sync/experiments/survey_predictions/code')

from surveyformat import add_dimensions, melt

os.chdir('/Users/mcox/Box Sync/experiments/survey_predictions')

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
all.survey = all.survey.astype('category', categories = survey_sort, ordered=True)
    
#Filter for CSI
csi = all.loc[all.variable=='CSI-Net']

#Obtain CSI for EIS
eis_csi = csi.loc[csi.survey == 'EIS-1st year']

#Compute change since EIS
csi = pd.merge(csi, eis_csi.ix[:,['region','cohort','value']], on=['region','cohort'])
csi.rename(columns={'value_x':'value','value_y':'eis_value'}, inplace=True)
csi['prev_survey_seq'] = csi.survey_seq - 1
csi = pd.merge(csi, csi.ix[:,['region','cohort','value', 'survey_seq']],
        left_on=['region','cohort','prev_survey_seq'],
        right_on=['region','cohort','survey_seq'])
csi.rename(columns={'value_x':'value','value_y':'prev_value'}, inplace=True)

csi['change_since_eis'] = csi['value'] - csi['eis_value']
csi['change_since_prev_value'] = csi['value'] - csi['prev_value']

#Graph changes since EIS for each region to F8W

nat = csi.loc[csi.region.isnull() & (csi.survey != 'EIS-1st year')]

g = sns.FacetGrid(nat, col='survey', col_order=survey_sort[1:])
g.map(plt.plot, 'cohort', 'change_since_prev_value', color=sns.xkcd_rgb["denim blue"])

#plt.plot(nat.cohort, nat.change_since_eis)
for ax in g.axes[0]:
    vals = ax.get_yticks()
    ax.xaxis.set_major_formatter(ml.ticker.FormatStrFormatter('%4d'))
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    xlim = ax.get_xlim()
    line = mpl_lines.Line2D(xlim, [0,0], color='grey')
    ax.add_line(line)
    
plt.show()

g.savefig('./outputs/for_national.png')

plt.clf()

reg = csi.loc[csi.region.notnull() & (csi.survey != 'EIS-1st year')]
sns.set_palette(sns.cubehelix_palette(len(csi.region.unique()), start=.5, rot=-.75))

g = sns.FacetGrid(reg, col='survey', col_order=survey_sort[1:], hue='region')
g.map(plt.plot, 'cohort', 'change_since_prev_value', alpha=0.1)

#plt.plot(nat.cohort, nat.change_since_eis)
for ax in g.axes[0]:
    ax.set_ylim([-0.5,0.5])
    vals = ax.get_yticks()
    ax.xaxis.set_major_formatter(ml.ticker.FormatStrFormatter('%4d'))
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    xlim = ax.get_xlim()
    line = mpl_lines.Line2D(xlim, [0,0], color='grey')
    ax.add_line(line)
    
plt.show()

g.savefig('./outputs/by_region.png')

print('done!')