import pandas as pd
from os import path
from models.ordinal_model import run_regional_model
import json

with pd.HDFStore(path.join('..', 'inputs', 'responses.h5'))as store:
    df = pd.DataFrame()
    assert 'responses' in store
    df = store['responses']

data = df.loc[(df.survey_code == '1617F8W') & df.response.notnull()]

qcs = data.question_code.unique()

for qc in qcs:
    cur_data = data.ix[data.question_code == qc]
    db_name = 'region_ordinal_qc_{}'.format(qc)
    db_file = path.join('..', 'traces', db_name)
    print('Producing trace for {}'.format(qc))
    run = run_regional_model(cur_data, progressbar=True, db_file=db_file)

    with open('{}.json'.format(db_file), 'w') as json_file:
        json.dump(run['heads'], json_file)
