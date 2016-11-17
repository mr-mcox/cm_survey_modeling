import pandas as pd


def label_trace(trace, heads):
    trace['i'] = range(len(trace))
    mlt = pd.melt(trace, id_vars='i')

    # Reformat thresh cols
    thresh_num = mlt.variable.str.extract('thresh__(\d)', expand=False)
    mlt.ix[thresh_num.notnull(), 'variable'] = 'thresh' + thresh_num

    # Reformat ps
    ps_base = mlt.variable.str.extract('(.*ps)_', expand=False)
    ps_num = mlt.variable.str.extract('.*ps_.*(\d)$', expand=False)
    ps_rest = mlt.variable.str.extract('.*ps(__.*)_\d$', expand=False)
    ps_rest.fillna('', inplace=True)

    mlt.ix[ps_base.notnull(), 'variable'] = ps_base + ps_num + ps_rest

    # Extract variable name
    mlt['var_name'] = mlt.variable.str.extract('([\w_]+)__', expand=False)
    mlt.ix[mlt.var_name.isnull(), 'var_name'] = mlt.variable

    has_dbl_under = mlt.variable.str.contains('__')

    overall_mlt = mlt.loc[~has_dbl_under].set_index(['i', 'var_name'])['value']
    overall_stack = overall_mlt.unstack(level=-1)
    add_net(overall_stack)

    overall_df = overall_stack.reset_index()

    mlt_1_var = mlt.loc[has_dbl_under]

    mlt_1_var['idx_num'] = pd.to_numeric(
        mlt_1_var.variable.str.extract('__(\d+)'))
    mlt_1_var[heads[0]['name']] = mlt_1_var.idx_num.map(
        lambda x: heads[0]['values'][x])
    mlt_1_var['var_name'] = mlt_1_var.variable.str.extract('([\w_]+)__')

    idx_cols = ['i', heads[0]['name'], 'var_name']
    stacked = mlt_1_var.set_index(idx_cols)['value']
    one_df = stacked.unstack(level=-1)

    add_net(one_df)

    one_df.reset_index(inplace=True)

    return {'overall': overall_df, 1: one_df}


def add_net(df):
    id_vars = df.index.names
    p_unstk = df.reset_index()
    p_mlt = pd.melt(p_unstk, id_vars=id_vars)
    p_rows = p_mlt[p_mlt.var_name.str.contains('ps\d')]
    p_base = p_rows.var_name.str.extract('(.*)ps\d')
    p_base_name = p_base.iloc[0]
    p_rows.set_index(id_vars + ['var_name']).unstack(level=-1)
    all_ps = p_rows.set_index(
        id_vars + ['var_name']).unstack(level=-1).as_matrix()
    net_col = p_base_name + 'net'
    df[net_col] = all_ps[:, 5:].sum(axis=1) - all_ps[:, :4].sum(axis=1)
