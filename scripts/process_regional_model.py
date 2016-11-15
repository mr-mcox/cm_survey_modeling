import pandas as pd
import json
from os import path
from helpers.examine_trace import label_trace
import seaborn as sns

qc = 'CSI1'
db_name = 'region_ordinal_qc_{}'.format(qc)
db_file = path.join('..', 'traces', db_name)
trace = pd.read_csv(path.join(db_file, 'chain-0.csv'))

with open('{}.json'.format(db_file)) as json_file:
    heads = json.load(json_file)
    
lt = label_trace(trace, heads)
lt_r = lt[1]
