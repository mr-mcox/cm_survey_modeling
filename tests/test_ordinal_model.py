import pytest
from os import path
import pandas as pd
import numpy as np
import pymc3 as pm

from models.ordinal_model import run_national_model, run_regional_model
from helpers.examine_trace import compute_net

pytestmark = pytest.mark.model


@pytest.fixture
def surveys_data():
    with pd.HDFStore(path.join('..', 'inputs', 'responses.h5'))as store:
        df = pd.DataFrame()
        assert 'responses' in store
        df = store['responses']
    qc = 'CSI1'
    data = df.loc[(df.survey_code == '1617F8W') & (
        df.question_code == qc) & df.response.notnull()]

    return data.copy()


@pytest.fixture
def small_regional_run(surveys_data):
    data = surveys_data.ix[surveys_data.Region.isin(["Alabama", "Houston"])]

    return run_regional_model(data, burn=10, samp=10)


def test_national_model(surveys_data):
    data = surveys_data
    exp = ((data.response >= 6).sum() -
           (data.response < 5).sum())/len(data)
    trace = run_national_model(surveys_data)
    nets = compute_net(pm.trace_to_dataframe(trace))

    assert abs(np.median(nets) - exp) < 0.01


def test_regional_model(surveys_data):
    # 15% regions more than 5% difference
    # Mean diff by region is 3%
    # Median diff is 2%
    data = surveys_data
    few_reg = data.Region.unique()[:5]
    data = data.ix[data.Region.isin(few_reg)]
    data['net'] = 0
    data.ix[data.response >= 6, 'net'] = 1
    data.ix[data.response < 5, 'net'] = -1

    exp = data.groupby('Region').mean()

    result = run_regional_model(data)
    reg_list = result['heads'][0]['values']
    exp_by_reg = exp.loc[reg_list].net
    nets = compute_net(pm.trace_to_dataframe(result['trace']))

    net_r = {reg_list[i]: nets[
        str(i)] for i in range(len(reg_list))}

    recs = list()
    for reg, vals in net_r.items():
        recs.append((reg, (vals < exp.loc[reg, 'net']).sum()/len(vals)))

    ptile = pd.DataFrame.from_records(recs, columns=['region', 'ptile'])

    assert np.sum((ptile.ptile < 0.95) & (ptile.ptile > 0.05))/len(ptile) > 0.9


def test_ps_in_model(small_regional_run):
    trace = pm.trace_to_dataframe(small_regional_run['trace'])
    assert trace.columns.str.contains('ps__').any()


def test_json_out(small_regional_run):
    heads = small_regional_run['heads']
    assert heads[0]['name'] == 'Region'
    assert set(heads[0]['values']) == {'Houston', 'Alabama'}
