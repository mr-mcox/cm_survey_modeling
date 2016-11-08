import pandas as pd
import pytest
# from helpers.surveyformat import melt

@pytest.fixture
def melt_input():
    col = ['Region', 'Corps', 'survey_code', 'CALI',
           'CSI', 'cohort', 'survey_seq', 'survey']
    recs = [
        ('Alabama', '1st year', '1415EYS', 0.7, 0.5, 2014, 1, 'EYS-1st year'),
        ('Alabama', '1st year', '1415F8W', 0.7, 0.5, 2014, 1, 'F8W-1st year'),
    ]
    df = pd.DataFrame.from_records(recs, columns=col)
    df.survey = df.survey.astype('category', categories=['F8W-1st year', 'EYS-1st year'], ordered=True)
    return df

# def test_melt(melt_input):
#     out = melt(melt_input)
#     assert set(out.columns) == {'region','cohort','survey', 'survey_seq', 'variable', 'value'}

# def test_melt_variable_column(melt_input):
#     out = melt(melt_input)
#     assert list(out.variable.unique()) == ['CALI', 'CSI']

# def test_preserve_survey_category(melt_input):
#     out = melt(melt_input)
#     assert out.survey.dtype == 'category'