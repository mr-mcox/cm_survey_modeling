import pandas as pd
import pytest
from surveyformat import remove_blank_corps, add_cohort


def test_remove_rows_without_corps():
    df = pd.DataFrame({'Corps': ['1st year', None, '2nd year']})
    out = remove_blank_corps(df)
    assert out.Corps.isnull().sum() == 0


@pytest.fixture
def cohort_input():
    cols = ['Corps', 'survey_code']
    recs = [('1st year', '1516EYS'),
            ('1st year', '1415EYS'),
            ('2nd year', '1415EYS')]
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