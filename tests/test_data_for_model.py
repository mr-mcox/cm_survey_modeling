import pytest
import pandas as pd
from model_data import ModelData


@pytest.fixture
def md_with_dataframe():
    cols = 'person_id   question_code   survey_code response Corps'.split()
    recs = [(3554131, 'CSI1', '1415MYS', 5, '1st year'),
            (3490339, 'CSI1', '1415MYS', 6, '1st year'),
            (3557769, 'CSI1', '1415MYS', 3, '1st year'),
            (3013575, 'CSI1', '1415MYS', 5, '1st year'),
            (3554131, 'CSI1', '1415EYS', 6, '1st year'),
            (3490339, 'CSI1', '1415EYS', 6, '1st year'),
            (3557769, 'CSI1', '1415EYS', 6, '1st year'),
            (3013575, 'CSI1', '1415EYS', 5, '1st year'), ]
    md = ModelData()
    md.df = pd.DataFrame.from_records(recs, columns=cols)
    return md


def test_process_functional(md_with_dataframe):
    # Given enerate proportions for a cut
    md = md_with_dataframe
    res = md.assign_previous_response()
    assert res.ix[(res.person_id == 3554131) & (
        res.survey_code == '1415EYS'), 'prev_response'].iat[0] == 5


def test_assign_previous_response(md_with_dataframe):
    md = md_with_dataframe
    res = md.assign_previous_response()
    assert res.ix[(res.person_id == 3554131) & (
        res.survey_code == '1415EYS'), 'prev_response'].iat[0] == 5
    assert res.ix[(res.person_id == 3557769) & (
        res.survey_code == '1415EYS'), 'prev_response'].iat[0] == 3


def test_add_survey_seq(md_with_dataframe):
    md = md_with_dataframe
    res = md.add_survey_seq()
    assert (res.ix[res.survey_code == '1415EYS', 'survey_seq'] >
            res.ix[res.survey_code == '1415MYS', 'survey_seq']).all()

def test_add_next_survey_seq(md_with_dataframe):
    md = md_with_dataframe
    res = md.add_next_survey_seq()
    assert (res.survey_seq+1 == res.next_survey_seq).all()

