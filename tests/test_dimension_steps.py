import pandas as pd
from surveyformat import remove_blank_corps

def test_remove_rows_without_corps():
    df = pd.DataFrame({'Corps':['1st year',None,'2nd year']})
    out = remove_blank_corps(df)
    assert out.Corps.isnull().sum() == 0