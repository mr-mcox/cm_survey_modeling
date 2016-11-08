import pytest
from os import path
import pandas as pd
import numpy as np
import pymc3 as pm

from models.ordinal_model import run_model
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

    return data


@pytest.mark.current
def test_ordinal(surveys_data):
    data = surveys_data
    exp = ((data.response >= 6).sum() -
           (data.response < 5).sum())/len(data)
    trace = run_model(surveys_data)
    nets = compute_net(pm.trace_to_dataframe(trace))

    assert abs(np.median(nets) - exp) < 0.01
