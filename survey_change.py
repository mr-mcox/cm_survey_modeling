import pandas as pd
import os
import sys
import matplotlib as ml
import matplotlib.pyplot as plt
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
    
#Filter for CSI
csi = all.loc[all.variable=='CSI-Net']

#Obtain CSI for EIS
eis_csi = csi.loc[csi.survey == 'EIS-1st year']

#Compute change since EIS
csi = pd.merge(csi, eis_csi.ix[:,['region','cohort','value']], on=['region','cohort'])
csi = csi.rename(columns={'value_x':'value','value_y':'eis_value'})
csi['change_since_eis'] = csi['value'] - csi['eis_value']

#Graph changes since EIS for each region to F8W

nat = csi.loc[csi.region.isnull()]
fig, ax = plt.subplots()

#g = sns.FacetGrid(nat, col='survey', col_order='survey_seq')
g = sns.FacetGrid(nat, col='survey')
g.map(plt.plot, 'cohort', 'change_since_eis')

#plt.plot(nat.cohort, nat.change_since_eis)
ax.xaxis.set_major_formatter(ml.ticker.FormatStrFormatter('%4d'))
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])

plt.show()

print('done!')