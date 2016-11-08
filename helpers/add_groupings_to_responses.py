import pandas as pd
import os

idx_cols = ['person_id', 'survey_code']
resp = pd.read_csv(os.path.join(
    '..', 'inputs', 'cm_response_only_confidential.csv')).set_index(idx_cols)
group = pd.read_csv(
    os.path.join('..', 'inputs', 'cm_groupings.csv')).set_index(idx_cols)

for gp in group.grouping_category.unique():
    resp[gp] = group.ix[group.grouping_category == gp, 'group_name']

resp.reset_index().to_csv(os.path.join(
    '..', 'inputs', 'cm_response_confidential.csv'), index=False)
