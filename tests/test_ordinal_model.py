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


def test_national_model(surveys_data):
    data = surveys_data
    exp = ((data.response >= 6).sum() -
           (data.response < 5).sum())/len(data)
    trace = run_national_model(surveys_data)
    nets = compute_net(pm.trace_to_dataframe(trace))

    assert abs(np.median(nets) - exp) < 0.01


@pytest.mark.current
def test_regional_model(surveys_data):
    # Regional model is ok, but performs poorly for small regions.
    # 20% regions more than 5% difference
    # Mean diff by region is 9%
    # Median diff is 2.8%
    data = surveys_data
    # few_reg = ['New York', 'Bay Area']
    # few_reg = data.Region.unique()[:10]
    # data = data.ix[data.Region.isin(few_reg)]
    data['net'] = 0
    data.ix[data.response >= 6, 'net'] = 1
    data.ix[data.response < 5, 'net'] = -1

    exp = data.groupby('Region').mean()

    result = run_regional_model(data)
    exp_by_reg = exp.loc[result['heads']['regs']].net
    nets = compute_net(pm.trace_to_dataframe(result['trace']))

    num_reg = len(result['heads']['regs'])
    med_by_run = [np.median(nets[str(k)]) for k in range(num_reg)]

    diff = np.absolute(exp_by_reg - med_by_run)

    assert np.sum(diff > 0.05)/len(diff) < 0.1
