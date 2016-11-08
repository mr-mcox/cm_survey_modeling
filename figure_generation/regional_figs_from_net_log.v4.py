import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import mquantiles

# Import traces
traces = {s: pd.read_csv(os.path.join(
    '..', 'traces', 'net_log_v4_survey_mod_{}'.format(s), 'chain-0.csv')) for s in ['F8W', 'MYS', 'EYS']}

# Import nets
nets_file = os.path.join('..', 'inputs', 'processed', 'cm_nets.xlsx')
nets = pd.read_excel(nets_file)

# Add cohort
nets['cohort'] = pd.to_numeric(
    nets.survey_code.str.extract('^(\d{2})', expand=False)) + 2000
nets.ix[nets.Corps == '2nd year', 'cohort'] = nets.ix[
    nets.Corps == '2nd year', 'cohort'] - 1

# Add SY
start_yr = pd.to_numeric(
    nets.survey_code.str.extract('^(\d{2})', expand=False))
nets['sy'] = start_yr.map(lambda y: '{}-{} SY'.format(y, y+1))


def compute_next_resp(responses, is_2y, alpha, beta, sigma, a_2y_delta, b_2y_delta):
    beta_v = beta + (is_2y + b_2y_delta)
    alpha_v = alpha + (is_2y + a_2y_delta)
    new_resp_mu = beta_v * responses + alpha_v
    new_resp = np.random.normal(new_resp_mu, sigma)
    return new_resp


def transform_resp(resp):
    resp[resp == 1] = 0.9999
    resp[resp == -1] = -0.9999
    resp_trans = resp / 2 + 0.5
    resp_log = np.log((1-resp_trans)/resp_trans)
    return resp_log


def untransform_resp(t_resp):
    r_e = 1/(1+np.exp(t_resp))
    return (r_e - 0.5)*2

# Compute parameters for each survey seq
params = dict()
burn_len = 500
trace_len = 0

for sm in ['F8W', 'MYS', 'EYS']:
    s_trace = traces[sm][burn_len:]
    alphas = np.random.normal(s_trace.admin_alpha_mu, s_trace.admin_alpha_sd)
    betas = np.random.normal(s_trace.admin_beta_mu, s_trace.admin_beta_sd)
    sigmas = s_trace.sigma.tolist()
    alpha_delta_2ys = s_trace.alpha_delta_2y.tolist()
    beta_delta_2ys = s_trace.beta_delta_2y.tolist()
    param_df = pd.DataFrame({'alphas': alphas,
                             'betas': betas,
                             'sigmas': sigmas,
                             'alpha_delta_2ys': alpha_delta_2ys,
                             'beta_delta_2ys': beta_delta_2ys,
                             })
    params[sm] = param_df
    trace_len = len(param_df)


def simulate_run(init, is_2y, survey_mods):
    sim_net = list()
    sim_net.append([np.mean(init) for i in range(trace_len)])
    init_trans = transform_resp(init)
    last_resp = np.array(init_trans) * np.ones((trace_len, 1))

    for sm in survey_mods:
        c_param = params[sm]
        last_resp = [compute_next_resp(last_resp[i],
                                       is_2y,
                                       c_param.get_value(i, 'alphas'),
                                       c_param.get_value(i, 'betas'),
                                       c_param.get_value(i, 'sigmas'),
                                       c_param.get_value(i, 'alpha_delta_2ys'),
                                       c_param.get_value(i, 'beta_delta_2ys'),
                                       ) for i in range(trace_len)]
        sim_net.append(np.mean(untransform_resp(last_resp), axis=1))
    return sim_net


def plot_projections(data, region_name):
    init = data.ix[(data.survey_code == '1617F8W'), 'response']
    is_2y = (data.ix[(data.survey_code == '1617F8W'), 'Corps'] == '2nd year')
    surveys = ['MYS', 'EYS']
    sim = simulate_run(init, is_2y, surveys)

    # Plot
    fig, ax = plt.subplots()
    title = '16-17 School Year Survey Projections for {}'.format(region_name)
    fig.suptitle(title)
    reg_nets = data
    actual = reg_nets.groupby(['survey_mod', 'sy']).mean().reset_index().set_index('survey_mod')
    c_pal = sns.color_palette()

    for sy in ['13-14 SY', '14-15 SY', '15-16 SY', '16-17 SY']:
        xlabels = ['EIS', 'F8W', 'MYS', 'EYS']
        xvals = [i for i in range( len(xlabels)) ]
        yvals = actual.loc[actual.sy == sy, 'response'][xlabels]
        ax.plot(xvals, yvals, '-', label=sy)

    qs = mquantiles(sim, [0.025, 0.975], axis=1).T
    plt.fill_between(np.arange(3) + 1, *qs, alpha=0.15,
                     color=c_pal[3], label='70% credible')
    qs = mquantiles(sim, [0.15, 0.85], axis=1).T
    plt.fill_between(np.arange(3) + 1, *qs, alpha=0.35,
                     color=c_pal[3], label='95% credible')

    ax.xaxis.set_ticks(np.arange(4))
    ax.set_xticklabels(xlabels)

    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in yvals])
    ax.legend(loc='lower left')

    out_file = os.path.join(
        '..', 'outputs', 'figures', 'projections_from_net_log_v4', 'Projection for {}.png'.format(region_name))

    # plt.show()

    plt.savefig(out_file)

    plt.close()


plot_projections(nets, 'National')

for region in nets.ix[nets.survey_code == '1617F8W', 'Region'].unique():
    plot_projections(nets[nets.Region == region], region)

