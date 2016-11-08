import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import mquantiles

# Import traces
traces = {s: pd.read_csv(os.path.join(
    '..', 'traces', 'net_log_v3_survey_seq_{}'.format(s), 'chain-0.csv')) for s in range(1, 7)}

# Import nets
nets_file = os.path.join('..', 'inputs', 'processed', 'cm_nets.xlsx')
nets = pd.read_excel(nets_file)

# Add cohort
nets['cohort'] = pd.to_numeric(
    nets.survey_code.str.extract('^(\d{2})', expand=False)) + 2000
nets.ix[nets.Corps == '2nd year', 'cohort'] = nets.ix[
    nets.Corps == '2nd year', 'cohort'] - 1


def compute_next_resp(responses, alpha, beta, sigma):
    new_resp_mu = beta * responses + alpha
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
burn_len = 300
trace_len = 0

for seq in range(1, 7):
    s_trace = traces[seq][burn_len:]
    alphas = np.random.normal(s_trace.admin_alpha_mu, s_trace.admin_alpha_sd)
    betas = np.random.normal(s_trace.admin_beta_mu, s_trace.admin_beta_sd)
    sigmas = s_trace.sigma.tolist()
    param_df = pd.DataFrame({'alphas': alphas,
                             'betas': betas,
                             'sigmas': sigmas})
    params[seq] = param_df
    trace_len = len(param_df)


def simulate_seq(init, start_seq, end_seq):
    sim_net = list()
    sim_net.append([np.mean(init) for i in range(trace_len)])
    init_trans = transform_resp(init)
    last_resp = np.array(init_trans) * np.ones((trace_len, 1))

    for seq in range(start_seq, end_seq + 1):
        c_param = params[seq]
        last_resp = [compute_next_resp(last_resp[i],
                                       c_param.get_value(i, 'alphas'),
                                       c_param.get_value(i, 'betas'),
                                       c_param.get_value(i, 'sigmas')) for i in range(trace_len)]
        sim_net.append(np.mean(untransform_resp(last_resp), axis=1))
    return sim_net


def plot_projections(data, region_name):
    init_1y = data.ix[(data.cohort == 2016) &
                      (data.survey_code == '1617F8W'), 'response']
    init_2y = data.ix[(data.cohort == 2015) &
                      (data.survey_code == '1617F8W'), 'response']
    sim_1y = simulate_seq(init_1y, 2, 3)
    sim_2y = simulate_seq(init_2y, 5, 6)

    # Plot
    fig, ax = plt.subplots()
    title = '16-17 School Year Survey Projections for {}'.format(region_name)
    fig.suptitle(title)
    reg_nets = data
    actual = reg_nets.groupby(['survey_seq', 'cohort']).mean().reset_index()
    survey_seq_list = data.ix[
        data.Corps.notnull(), ['survey_seq', 'survey_mod', 'Corps']].drop_duplicates()
    survey_seq_list[
        'survey_label'] = survey_seq_list.survey_mod + '-' + survey_seq_list.Corps
    survey_seq_map = survey_seq_list.set_index('survey_seq').to_dict()
    c_pal = sns.color_palette()

    for cohort in [2013, 2014, 2015, 2016]:
        xvals = actual.loc[actual.cohort == cohort, 'survey_seq']
        yvals = actual.loc[actual.cohort == cohort, 'response']
        ax.plot(xvals, yvals, '-', label=cohort)

    qs = mquantiles(sim_1y, [0.025, 0.975], axis=1).T
    plt.fill_between(np.arange(3) + 1, *qs, alpha=0.15,
                     color=c_pal[3])
    qs = mquantiles(sim_1y, [0.15, 0.85], axis=1).T
    plt.fill_between(np.arange(3) + 1, *qs, alpha=0.35,
                     color=c_pal[3])

    qs = mquantiles(sim_2y, [0.025, 0.975], axis=1).T
    plt.fill_between(np.arange(3) + 4, *qs, alpha=0.15,
                     color=c_pal[2], label='70% credible')
    qs = mquantiles(sim_2y, [0.15, 0.85], axis=1).T
    plt.fill_between(np.arange(3) + 4, *qs, alpha=0.35,
                     color=c_pal[2], label='95% credible')

    ax.xaxis.set_ticks([k for k in survey_seq_map['survey_mod'].keys()])
    labels = [survey_seq_map['survey_label'][i]
              for i in ax.get_xticks().tolist()]
    ax.set_xticklabels(labels)
    ymin, ymax = plt.ylim()
    if ymax > 1:
        plt.ylim(ymax=1.01)

    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in yvals])
    ax.legend(loc='lower left')

    out_file = os.path.join(
        '..', 'outputs', 'figures', 'projections_from_net_log_v3', 'Projection for {}.png'.format(region_name))

    plt.savefig(out_file)

    plt.close()


plot_projections(nets, 'National')

for region in nets.ix[nets.survey_code == '1617F8W', 'Region'].unique():
    plot_projections(nets[nets.Region == region], region)
