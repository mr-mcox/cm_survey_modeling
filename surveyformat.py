import pandas as pd


def add_dimensions(df):
    out = remove_blank_corps(df)
    out = add_cohort(out)
    out = add_survey(out)
    out = add_survey_seq(out)
    return out


def melt(df):
    df2 = df.rename(columns={'Region':'region'})
    id_vars = ['region','cohort','survey','survey_seq']
    cols = list(df2.columns)
    survey_code_idx = cols.index('survey_code') + 1
    pre_survey_code_cols = cols[:survey_code_idx]
    for col in set(pre_survey_code_cols) - set(id_vars):
        del df2[col]
    out = pd.melt(df2, id_vars=id_vars)
    return out


def remove_blank_corps(df):
    return df[df.Corps.notnull()]


def add_cohort(df):
    out = df.copy()

    year_digits = out.survey_code.str.extract('^(\d\d)', expand=False)
    cohort_str = '20' + year_digits

    out['cohort'] = pd.to_numeric(cohort_str)
    out.ix[out.Corps == '2nd year', 'cohort'] = out.ix[
        out.Corps == '2nd year', 'cohort'] - 1
    return out


def add_survey(df):
    out = df.copy()
    out['survey_mod'] = out.survey_code.str.replace('R0', 'F8W')
    out['survey_mod'] = out.survey_mod.str.extract('(\w{3})$', expand=False)
    out['survey'] = out.survey_mod + '-' + out.Corps
    return out


def add_survey_seq(df):
    out = df.copy()
    if 'survey' not in out.columns:
        out = add_survey(out)

    fy_surveys = out.ix[out.Corps == '1st year', 'survey_mod'].unique()
    seq = dict(zip(reversed(fy_surveys), range(len(fy_surveys))))
    out['survey_seq'] = out.survey_mod.map(seq)

    sy_surveys = out.ix[out.Corps == '2nd year', 'survey_mod'].unique()
    seq_nums = [x + len(fy_surveys) for x in range(len(sy_surveys))]
    seq = dict(zip(reversed(sy_surveys), seq_nums))
    sy = out.Corps == '2nd year'
    out.ix[sy, 'survey_seq'] = out.ix[sy, 'survey_mod'].map(seq)

    return out
