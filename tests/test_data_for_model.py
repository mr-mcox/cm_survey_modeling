import pytest
import pandas as pd
from model_data import ModelData


@pytest.fixture
def md_with_dataframe():
    cols = 'person_id   question_code   survey_code response Corps'.split()
    recs = [(3554131, 'CSI1', '1415MYS', 5, '1st year'),
            (3490339, 'CSI1', '1415MYS', 3, '1st year'),
            (3557769, 'CSI1', '1415MYS', 3, '1st year'),
            (3013575, 'CSI1', '1415MYS', 5, '1st year'),
            (3554131, 'CSI1', '1415EYS', 6, '1st year'),
            (3490339, 'CSI1', '1415EYS', 6, '1st year'),
            (3557769, 'CSI1', '1415EYS', 6, '1st year'),
            (3013575, 'CSI1', '1415EYS', 5, '1st year'), ]
    md = ModelData()
    md.df = pd.DataFrame.from_records(recs, columns=cols)
    return md


@pytest.fixture
def md_with_duplicates_df():
    cols = 'person_id   question_code   survey_code response Corps'.split()
    recs = [(3554131, 'CSI1', '1415MYS', 5, '1st year'),
            (3490339, 'CSI1', '1415MYS', 3, '1st year'),
            (3557769, 'CSI1', '1415MYS', 3, '1st year'),
            (3013575, 'CSI1', '1415MYS', 5, '1st year'),
            (3554131, 'CSI1', '1415EYS', 4, '1st year'),
            (3554131, 'CSI1', '1415EYS', 6, '1st year'),
            (3490339, 'CSI1', '1415EYS', 6, '1st year'),
            (3557769, 'CSI1', '1415EYS', 6, '1st year'),
            (3013575, 'CSI1', '1415EYS', 5, '1st year'), ]
    md = ModelData()
    md.df = pd.DataFrame.from_records(recs, columns=cols)
    return md


@pytest.fixture
def md_with_dataframe_region(md_with_dataframe):
    md = md_with_dataframe
    cm_region = pd.DataFrame.from_records([
        (3554131, 'Atlanta'),
        (3490339, 'Atlanta'),
        (3557769, 'Chicago'),
        (3013575, 'Chicago'),
    ], columns=['person_id', 'Region'])
    md.df = pd.merge(md.df, cm_region)
    return md


@pytest.fixture
def md_multiple_qs():
    cols = 'person_id   question_code   survey_code response Corps'.split()
    recs = [(3554131, 'CSI1', '1415MYS', 1, '1st year'),
            (3490339, 'CSI1', '1415MYS', 5, '1st year'),
            (3557769, 'CSI1', '1415MYS', 7, '1st year'),
            (3013575, 'CSI1', '1415MYS', 5, '1st year'),
            (3554131, 'CSI1', '1415MYS', 6, '1st year'),
            (3490339, 'CSI1', '1415MYS', 6, '1st year'),
            (3557769, 'CSI1', '1415MYS', 6, '1st year'),
            (3013575, 'CSI1', '1415MYS', 5, '1st year'), ]
    md = ModelData()
    md.df = pd.DataFrame.from_records(recs, columns=cols)
    return md


def test_process_functional(md_with_dataframe):
    # Given enerate proportions for a cut
    md = md_with_dataframe
    res = md.assign_previous_response()
    assert res.ix[(res.person_id == 3554131) & (
        res.survey_code == '1415EYS'), 'prev_response'].iat[0] == 5

    # Group responses
    gres = md.count_by_prev_response()
    assert gres.ix[
        (gres.prev_response == 3) & (gres.response == 6), 'num'].iat[0] == 2

    # Calculate responsses
    res = md.calculate_proportions(gres)
    assert res.ix[
        (res.prev_response == 5) & (res.response == 6), 'percent'].iat[0] == 0.5


def test_assign_previous_response(md_with_dataframe):
    md = md_with_dataframe
    res = md.assign_previous_response()
    assert res.ix[(res.person_id == 3554131) & (
        res.survey_code == '1415EYS'), 'prev_response'].iat[0] == 5
    assert res.ix[(res.person_id == 3557769) & (
        res.survey_code == '1415EYS'), 'prev_response'].iat[0] == 3


def test_add_survey_seq(md_with_dataframe):
    md = md_with_dataframe
    res = md.add_survey_seq()
    assert (res.ix[res.survey_code == '1415EYS', 'survey_seq'].min() >
            res.ix[res.survey_code == '1415MYS', 'survey_seq'].max())


def test_add_next_survey_seq(md_with_dataframe):
    md = md_with_dataframe
    res = md.add_next_survey_seq()
    assert (res.survey_seq + 1 == res.next_survey_seq).all()


def test_count_by_prev_response(md_with_dataframe):
    md = md_with_dataframe
    res = md.count_by_prev_response()
    assert res.ix[
        (res.prev_response == 3) & (res.response == 6), 'num'].iat[0] == 2


def test_count_by_prev_response_specify_cut(md_with_dataframe_region):
    md = md_with_dataframe_region
    res = md.count_by_prev_response(cut_cols=['survey_code'])
    assert res.ix[
        (res.prev_response == 3) & (res.response == 6), 'num'].iat[0] == 2


def test_proportion_for_cut(md_with_dataframe):
    md = md_with_dataframe
    res = md.proportion_for_cut(prev_response=5, unit='survey_code')
    assert res == [0, 0, 0, 0, 0.5, 0.5, 0]


def test_proportion_for_cut_v2(md_with_dataframe_region):
    md = md_with_dataframe_region
    res = md.proportion_for_cut(prev_response=5, unit='survey_code')
    assert res == [0, 0, 0, 0, 0.5, 0.5, 0]


def test_proportion_for_cut_v3(md_with_dataframe_region):
    md = md_with_dataframe_region
    res = md.proportion_for_cut(
        prev_response=5, unit='survey_code', groups={'Region': 'Atlanta'})
    assert res == [0, 0, 0, 0, 0, 1, 0]


def test_duplicates_bug(md_with_duplicates_df):
    md = md_with_duplicates_df
    res = md.proportion_for_cut(prev_response=5, unit='survey_code')
    assert res == [0, 0, 0, 0, 0.5, 0.5, 0]


def test_run_type(md_with_dataframe_region):
    md = md_with_dataframe_region
    resp = md.proportion_for_cut(prev_response=5, unit='survey_code', groups={
                                 'survey_seq': 1, 'question_code': 'CSI1'})


def test_observations_with_filter(md_with_dataframe_region):
    md = md_with_dataframe_region
    res = md.observations(
        row_filter={'survey_code': '1415EYS', 'prev_response': 5})
    assert (res.survey_code == '1415EYS').all()
    assert (res.prev_response == 5).all()


def test_observations_with_filter_group(md_with_dataframe_region):
    md = md_with_dataframe_region
    res = md.observations(
        row_filter={'survey_code': '1415EYS', 'prev_response': 5}, group_col='Region')
    assert (res['Chicago'].Region == 'Chicago').all()

# def test_observations_with_filter_group_two_level(md_with_dataframe_region):
#     md = md_with_dataframe_region
#     res = md.observations(row_filter={'survey_code': '1415EYS'}, group_col=['Region','prev_response'])
#     assert (res['Chicago'][5].Region == 'Chicago').all()
#     assert (res['Chicago'][5].prev_response == 5).all()


def test_compute_net(md_multiple_qs):
    md = md_multiple_qs
    md.add_net()
    res = md.df.ix[md.df.question_code == 'Net']
    pids = [3554131, 3490339, 3557769, 3013575]
    exp_net = [0, 0.5, 1, 0]
    assert (res.set_index('person_id').loc[pids, 'response'] == exp_net).all()
    assert (res.Corps == '1st year').all()
