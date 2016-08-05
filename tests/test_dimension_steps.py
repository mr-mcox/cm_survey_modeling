import pandas as pd
import pytest
from surveyformat import remove_blank_corps, add_cohort, add_survey, add_survey_seq


def test_remove_rows_without_corps():
    df = pd.DataFrame({'Corps': ['1st year', None, '2nd year']})
    out = remove_blank_corps(df)
    assert out.Corps.isnull().sum() == 0


@pytest.fixture
def cohort_input():
    cols = ['Corps', 'survey_code']
    recs = [('1st year', '1516EYS'),
            ('1st year', '1415EYS'),
            ('1st year', '1415F8W'),
            ('1st year', '1011EIS'),
            ('2nd year', '1415EYS'),
            ('2nd year', '1415F8W'),
            ('1st year', '1516F8W'),
            ('1st year', '1011R0'),
            ]
    df = pd.DataFrame.from_records(recs, columns=cols)
    return df

def test_add_cohort_case_1(cohort_input):
    out = add_cohort(cohort_input)

    assert (out.ix[(out.Corps == '1st year') & (
        out.survey_code == '1516EYS'), 'cohort'] == 2015).all()

def test_add_cohort_case_2(cohort_input):
    out = add_cohort(cohort_input)

    assert (out.ix[(out.Corps == '1st year') & (
        out.survey_code == '1415EYS'), 'cohort'] == 2014).all()

def test_add_cohort_case_3(cohort_input):
    out = add_cohort(cohort_input)

    assert (out.ix[(out.Corps == '2nd year') & (
        out.survey_code == '1415EYS'), 'cohort'] == 2013).all()

def test_add_survey_case_1(cohort_input):
    out = add_survey(cohort_input)
    assert (out.ix[(out.Corps == '1st year') & out.survey_code.str.contains('EYS$'),'survey'] == 'EYS-1st year').all()

def test_add_survey_case_2(cohort_input):
    out = add_survey(cohort_input)
    assert (out.ix[(out.Corps == '2nd year') & out.survey_code.str.contains('EYS$'),'survey'] == 'EYS-2nd year').all()

def test_add_survey_case_3(cohort_input):
    out = add_survey(cohort_input)
    assert (out.ix[(out.Corps == '1st year') & out.survey_code.str.contains('F8W$'),'survey'] == 'F8W-1st year').all()

def test_add_survey_case_4(cohort_input):
    out = add_survey(cohort_input)
    assert (out.ix[(out.Corps == '1st year') & out.survey_code.str.contains('R0$'),'survey'] == 'F8W-1st year').all()

def test_add_survey_seq(cohort_input):
    out = add_survey_seq(cohort_input)
    assert out.ix[out.survey == 'EYS-1st year', 'survey_seq'].min() > out.ix[out.survey == 'F8W-1st year', 'survey_seq'].max()

def test_add_survey_seq_case_2(cohort_input):
    out = add_survey_seq(cohort_input)
    assert out.ix[out.survey == 'EYS-2nd year', 'survey_seq'].min() > out.ix[out.survey == 'EYS-1st year', 'survey_seq'].max()

def test_survey_seq_complete_seq(cohort_input):
    out = add_survey_seq(cohort_input)
    assert set(out.survey_seq.unique()) == set(range(len(out.survey.unique())))
