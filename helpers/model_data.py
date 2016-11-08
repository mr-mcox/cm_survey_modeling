import pandas as pd


def add_survey(df):
    out = df.copy()
    out['survey_mod'] = out.survey_code.str.replace('R0', 'F8W')
    out['survey_mod'] = out.survey_mod.str.extract('(\w{3})$', expand=False)
    out['survey'] = out.survey_mod + '-' + out.Corps
    return out


def format_survey_seq(df):
    out = df.copy()
    if 'survey' not in out.columns:
        out = add_survey(out)

    survey_order = 'EIS F8W MYS EYS'.split()

    fy_survey_sel = out.ix[out.Corps == '1st year', 'survey_mod'].unique()
    fy_survey_mod = [s for s in survey_order if s in fy_survey_sel]
    fy_nums = [x for x in range(len(fy_survey_mod))]
    seq = dict(zip(fy_survey_mod, fy_nums))
    out['survey_seq'] = out.survey_mod.map(seq)

    sy_survey_sel = out.ix[out.Corps == '2nd year', 'survey_mod'].unique()
    sy_survey_mod = [s for s in survey_order if s in sy_survey_sel]
    sy_nums = [x + len(fy_survey_mod) for x in range(len(sy_survey_mod))]
    seq = dict(zip(sy_survey_mod, sy_nums))
    sy = out.Corps == '2nd year'
    out.ix[sy, 'survey_seq'] = out.ix[sy, 'survey_mod'].map(seq)

    survey_sort = out.sort_values('survey_seq')['survey'].unique()

    out.survey = out.survey.astype(
        'category', categories=survey_sort, ordered=True)

    return out


class ModelData(object):

    """docstring for ModelData"""

    def __init__(self, df=None):
        if type(df) is str:
            self.df = pd.read_csv(df)
        else:
            self.df = df

        self._prop_calcs = dict()

    def assign_previous_response(self):
        if 'next_survey_seq' not in self.df.columns:
            self.add_next_survey_seq()
        dedup_df = self.df.drop_duplicates(
            ['person_id', 'question_code', 'survey_seq'], keep='last')
        df = dedup_df.set_index(['person_id', 'survey_seq', 'question_code'])
        p_df = dedup_df.set_index(
            ['person_id', 'next_survey_seq', 'question_code'])
        df['prev_response'] = p_df.response
        df.reset_index(inplace=True)
        self.df = df
        return df

    def add_survey_seq(self):
        return format_survey_seq(self.df)

    def add_next_survey_seq(self):
        if 'survey_seq' not in self.df.columns:
            self.df = self.add_survey_seq()
        self.df['next_survey_seq'] = self.df.survey_seq + 1
        return self.df

    def add_net(self):
        df = self.df.copy()

        idx_cols = ['person_id', 'survey_code']
        orig_cms = df.drop_duplicates(idx_cols)
        orig_cms.set_index(idx_cols, inplace=True)

        net_codes = [-1 for x in range(4)] + [0, 1, 1]
        net_map = dict(zip([x+1 for x in range(7)], net_codes))
        df['net_num'] = df.response.map(net_map)
        cm_net = df.groupby(idx_cols).mean()

        orig_cms['response'] = cm_net.net_num
        orig_cms['question_code'] = 'Net'

        self.df = pd.concat([self.df, orig_cms.reset_index()])
