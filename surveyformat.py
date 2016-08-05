import pandas as pd


def add_dimensions(df):
    out = remove_blank_corps(df)
    return out


def melt(df):
    return df


def remove_blank_corps(df):
    return df[df.Corps.notnull()]


def add_cohort(df):
    out = df.copy()

    year_digits = out.survey_code.str.extract('^(\d\d)', expand=False)
    cohort_str = '20' + year_digits

    out['cohort'] = pd.to_numeric(cohort_str)
    out.ix[out.Corps=='2nd year','cohort'] = out.ix[out.Corps=='2nd year','cohort'] - 1
    return out
