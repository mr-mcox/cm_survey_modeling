from surveyformat import add_survey_seq as format_survey_seq

class ModelData(object):

    """docstring for ModelData"""

    def __init__(self, df=None):
        self.df = df

    def assign_previous_response(self):
        if 'survey_seq' not in self.df.columns:
            self.add_next_survey_seq()
        df = self.df.set_index(['person_id', 'survey_seq'])
        p_df = self.df.set_index(['person_id', 'next_survey_seq'])
        df['prev_response'] = p_df.response
        df.reset_index(inplace=True)
        print(df)
        return df

    def add_survey_seq(self):
        return format_survey_seq(self.df)

    def add_next_survey_seq(self):
        if 'survey_seq' not in self.df.columns:
            self.df = self.add_survey_seq()
        self.df['next_survey_seq'] = self.df.survey_seq + 1
        return self.df
