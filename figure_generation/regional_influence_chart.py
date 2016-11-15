import pandas as pd
import json
from os import path
from helpers.examine_trace import label_trace
import seaborn as sns

with pd.HDFStore(path.join('..', 'inputs', 'responses.h5')) as store:
    resp = pd.DataFrame()
    assert 'responses' in store
    resp = store['responses']

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
    
reg_csi = regs.groupby(['Region','i'])['net'].mean().reset_index()


reg_csi_qs = reg_csi.groupby('Region')['net'].quantile(0.05)
reg_rank = reg_csi_qs.sort_values(ascending=False).index

g = sns.FacetGrid(reg_csi, row='Region', row_order=reg_rank, size=1.7, aspect=4)
g.map(sns.distplot, 'net', hist=False)
