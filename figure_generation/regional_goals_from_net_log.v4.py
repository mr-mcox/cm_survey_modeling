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


def net_quantiles(data, reg_goal):
    recs = list()
    for region in data.Region.unique():
        init = data.ix[(data.Region == region), 'response']
        is_2y = (data.ix[(data.Region == region), 'Corps'] == '2nd year')
        surveys = ['MYS', 'EYS']
        sim = simulate_run(init, is_2y, surveys)
        qs = mquantiles(sim, [0.025, 0.975], axis=1).T
        lower = qs[0][2]
        upper = qs[1][2]
        goal = reg_goal[region]
        g_ptile = (sim[2] > goal).mean() if not np.isnan(goal) else None
        g_diff = goal - upper if not np.isnan(goal) else None
        rec = (region, lower, upper, goal, g_ptile, g_diff)
        print(rec)
        recs.append(rec)
    return pd.DataFrame.from_records(recs, columns=['region', 'lower', 'upper', 'goal','g_ptile', 'g_diff']).set_index('region')

goal_file = os.path.join('..', 'inputs', 'region_goals.xlsx')
rg = pd.read_excel(goal_file).set_index('Region')
rg.Goal = rg.Goal.map(lambda g: g / 100 if g > 1 else g)
reg_goals = rg.to_dict()['Goal']
q_df = net_quantiles(nets.ix[nets.survey_code == '1617F8W'], reg_goals)

g_sort = q_df[q_df.goal.notnull()].sort_values('g_ptile')
ebar = np.array([g_sort.goal - g_sort.lower, g_sort.upper - g_sort.goal ])
fig, ax = plt.subplots(figsize=(10,10))
plt.errorbar(g_sort.goal, np.arange(len(g_sort)), xerr=ebar, fmt='o', label='EYS goal')
# plt.plot(g_sort.goal, np.arange(len(g_sort)), 'o')
plt.yticks(np.arange( len(g_sort) ), g_sort.index)
plt.ylim(-1, len(g_sort))
xvals = ax.get_xticks()
ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in xvals])
ax.legend()
fig.suptitle('Regional CSI goals compared to CSI projections')
# plt.show()

out_file = os.path.join(
        '..', 'outputs', 'figures', 'regional_goals_vs_projections.png')
plt.savefig(out_file)
