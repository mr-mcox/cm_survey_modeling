from surveyformat import add_survey_seq as format_survey_seq
import pandas as pd


class ModelData(object):

    """docstring for ModelData"""

    def __init__(self, df=None):
        if type(df) is str:
            self.df = pd.read_csv(df)
        else:
            self.df = df
        self._prop_calcs = dict()

    def assign_previous_response(self):
        if 'survey_seq' not in self.df.columns:
            self.add_next_survey_seq()
        df = self.df.set_index(['person_id', 'survey_seq'])
        p_df = self.df.set_index(['person_id', 'next_survey_seq'])
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

    def count_by_prev_response(self, cut_cols=None):
        if 'survey_seq' not in self.df.columns:
            self.assign_previous_response()

        df = self.df

        cols = list()
        if cut_cols is None:
            cols = [c for c in df.columns if c != 'person_id']
        else:
            cols = cut_cols + ['response', 'prev_response']

        size_s = df.groupby(cols).size()

        out = pd.DataFrame({'num': size_s}, index=size_s.index).reset_index()
        return out

    def calculate_proportions(self, input_df):
        df = input_df.copy()
        cols_for_groups = [
            c for c in df.columns if c not in ['num', 'response']]
        sum_s = df.groupby(cols_for_groups).sum()['num']
        df.set_index(cols_for_groups,  inplace=True)
        df['total_num'] = sum_s
        df['percent'] = df.num / df.total_num
        return df.reset_index()

    def proportion_for_cut(self, prev_response, unit, groups=dict()):
        cut_cols = [k for k in groups.keys()] + [unit]
        cut_cols_tup = tuple(cut_cols)
        if cut_cols_tup not in self._prop_calcs:
            counts = self.count_by_prev_response(cut_cols=cut_cols)
            calc_props = self.calculate_proportions(counts)
            self._prop_calcs[cut_cols_tup] = calc_props
        df = self._prop_calcs[cut_cols_tup]

        full_mask = [True] * len(df)
        for key, value in groups.items():
            full_mask = full_mask & (df[key] == value)
        df = df.ix[full_mask]
        df.set_index('response', inplace=True)
        df_pr = df.ix[df.prev_response == prev_response, 'percent']
        out_list = list()
        for i in range(1, 8):
            val = 0
            if i in df_pr.index:
                val = df_pr.get(i)
            out_list.append(val)

        return out_list
