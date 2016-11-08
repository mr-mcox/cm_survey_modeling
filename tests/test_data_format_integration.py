import pandas as pd
from helpers.surveyformat import add_dimensions


def test_integration():
    #Read in data
    df = pd.read_excel('./tests/data/input.xlsx')

    # Add dimensions
    out = add_dimensions(df)

    # There are no rows without a corps
    assert out.Corps.isnull().sum() == 0

    # Cohort computed
    # All rows with Corps = 1st year and survey_code starting with 14 have a
    # cohort of 2014
    assert (out.ix[(out.Corps == '1st year') & (
        out.survey_code.str.contains('^14')), 'cohort'] == 2014).all()
    # All rows with Corps = 2nd year and survey_code starting with 15 have a
    # cohort of 2014
    assert (out.ix[(out.Corps == '2nd year') & (
        out.survey_code.str.contains('^15')), 'cohort'] == 2014).all()

    # Survey is a concatenation of last three letters of survey code and Corps
    assert (out.survey == out.survey_code.str.extract(
        '(\w{3})$', expand=False) + '-' + out.Corps).all()

    # Survey sequence is in proper order
    # EYS-1st year has a higher sequence number than MYS-1st year
    assert out.ix[out.survey == 'EYS-1st year',
                  'survey_seq'].min() > out.ix[out.survey == 'MYS-1st year', 'survey_seq'].max()
    # EYS-2nd year has a higher sequence number than EYS-1st year
    assert out.ix[out.survey == 'EYS-2nd year',
                  'survey_seq'].min() > out.ix[out.survey == 'EYS-1st year', 'survey_seq'].max()

    # Melt data frame
    # melt_out = melt(out)

    # # Only expected dimensions included
    # assert set(melt_out.columns) == {
    #     'region', 'cohort', 'survey', 'survey_seq', 'variable', 'value'}

    # # Measure includes both question types
    # assert set(melt_out.variable.unique().tolist()) >= {'CALI-% strong', 'CALI-% weak'}

