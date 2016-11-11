import pandas as pd


def label_trace(trace, heads):
    thresh_cols = trace.columns.str.contains('thresh__\d')
    numbered_cols = trace.columns.str.contains('__\d')
    numbered_cols = numbered_cols & (~thresh_cols)

    overall_df = trace.loc[:, ~numbered_cols]
    overall_df['i'] = range(len(overall_df))

    numbered_df = trace.loc[:, numbered_cols]
    numbered_df['i'] = range(len(numbered_df))
    mlt = pd.melt(numbered_df, id_vars='i')
    mlt['idx_num'] = pd.to_numeric(mlt.variable.str.extract('__(\d+)'))
    mlt[heads[0]['name']] = mlt.idx_num.map(lambda x: heads[0]['values'][x])
    mlt['var_name'] = mlt.variable.str.extract('([\w_]+)__')

    # Rework ps
    mlt_ps = mlt.ix[mlt.var_name == 'ps']
    mlt_ps['var_name'] = mlt_ps['var_name'] + \
        mlt_ps.variable.str.extract('_(\d)$')
    mlt.ix[mlt_ps.index, 'var_name'] = mlt_ps['var_name']

    idx_cols = ['i', heads[0]['name'], 'var_name']
    stacked = mlt.set_index(idx_cols)['value']
    one_df = stacked.unstack(level=-1)

    # Add net
    ps_cols = one_df.columns.str.contains('ps\d')
    all_ps = all_ps = one_df.loc[:, ps_cols].as_matrix()
    one_df['net'] = all_ps[:, 5:].sum(axis=1) - all_ps[:, :4].sum(axis=1)

    one_df.reset_index(inplace=True)

    return {'overall': overall_df, 1: one_df}
