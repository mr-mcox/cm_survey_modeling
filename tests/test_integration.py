import pandas as pd
from surveyformat import add_dimensions, melt


def test_integration():
    #Read in data
    df = pd.read_excel('./tests/data/input.xlsx')

    # Add dimensions
    out = add_dimensions(df)

    # There are no rows without a corps
    assert out.Corps.isnull().sum() == 0

    # Cohort computed
    # All rows with Corps = 1st year and survey_code starting with 14 have a cohort of 2014
    assert (out.ix[(out.Corps == '1st year') & (
        out.survey_code.str.contains('^14')), 'cohort'] == 2014).all()
    # All rows with Corps = 2nd year and survey_code starting with 15 have a
    # cohort of 2014
    assert (out.ix[(out.Corps == '2nd year') & (
        out.survey_code.str.contains('^15')), 'cohort'] == 2014).all()

    # Survey is a concatenation of last three letters of survey code and Corps
    assert (out.survey == out.survey_code.str.extract('(\w{3})$', expand=False) + '-' + out.Corps).all()

    # Survey sequence is in proper order
    # EYS-1st year has a higher sequence number than MYS-1st year
    # EYS-2nd year has a higher sequence number than EYS-1st year

    # Melt data frame
    melt_out = melt(out)

    # Only expected dimensions included
    #assert set(melt_out.columns) == {'region','cohort','survey', 'survey_sequence', 'measure', 'value'}

    # Measure includes both question types
    #melt_out.measure.unique().tolist() == ['CALI-% strong', 'CALI-% weak']
