import os
import numpy as np
from scipy import stats as spstats
from scipy.spatial import distance
import torch
from tqdm import tqdm
import pickle

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

# sbi
import sbi.utils as utils
from sbi.inference.base import infer

# Neuron
from neuron import h

# remove top and right axis from plots
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
log_filename = ''


def calculate_summary_statistics(x):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """

    # initialise array of spike counts
    n_mom = 4
    n_summary = 7 + 2

    # n_summary = np.minimum(n_summary, n_mom + 3)
    t_on = 0
    t_off = x.shape[1]
    dt = h.dt

    t = torch.linspace(0, dt * x.shape[1], x.shape[1])

    returnVec = torch.zeros((x.shape[0], n_summary))
    for index in range(x.shape[0]):
        # initialise array of spike counts
        # v = np.copy(x[index, :])
        v = x[index, :].clone().detach().cpu().numpy()

        # put everything to -10 that is below -10 or has negative slope
        ind = np.where(v < -10)
        v[ind] = -10
        ind = np.where(np.diff(v) < 0)
        v[ind] = -10

        # remaining negative slopes are at spike peaks
        ind = np.where(np.diff(v) < 0)
        spike_times = np.array(t.cpu().numpy())[ind]
        spike_times_stim = spike_times[(spike_times > t_on) & (spike_times < t_off)]

        # number of spikes
        if spike_times_stim.shape[0] > 0:
            spike_times_stim = spike_times_stim[
                np.append(1, np.diff(spike_times_stim)) > 0.5
                ]

        # resting potential and std
        # rest_pot = np.mean(x[index, :][t < t_on])
        # rest_pot_std = np.std(x[index, :][int(0.9 * t_on / dt): int(t_on / dt)])
        rest_pot = torch.mean(x[index, :])
        rest_pot_std = torch.std(x[index, :])

        # moments
        std_pw = torch.pow(
            torch.std(x[index, :][(t > t_on) & (t < t_off)]), torch.linspace(3, n_mom, n_mom - 2)
        )
        std_pw = torch.cat((torch.ones(1), std_pw))

        if sum(std_pw == 0) == 0:
            moments = (
                    spstats.moment(
                        x[index, :][(t > t_on) & (t < t_off)], torch.linspace(2, n_mom, n_mom - 1)
                    )
                    / std_pw
            )
            moments[torch.isnan(moments)] = 0
        else:
            moments = torch.zeros(torch.linspace(2, n_mom, n_mom - 1).shape)

        returnVec[index, 0] = torch.tensor(spike_times_stim.shape[0])
        returnVec[index, 1] = rest_pot
        returnVec[index, 2] = rest_pot_std
        returnVec[index, 3] = torch.mean(x[index, :][(t > t_on) & (t < t_off)])
        returnVec[index, 4:7] = moments
        returnVec[index, 7] = pearson([x[index, :], WT_voltage[index, :]]).cpu().item()  # Pearson correlation
        returnVec[index, 8] = distance.hamming(x[index, :], WT_voltage[index, :])  # Hamming

    return returnVec.view(-1)


def pearson(z):
    """Pearson correlation coefficient of the two 1-D arrays in z = (x, y),
    computed with scipy.stats.pearsonr."""
    x, y = z
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def pearsonMatrix(x):
    ans = torch.zeros(n_shape[0])
    for i in range(n_shape[0]):
        ans[i] = pearson([x[i, :], WT_voltage[i, :]])
    return ans


def runMT(param):
    x, y, z = param

    h.initMT()
    # change the values of the model in Neuron
    for c in range(h.MuTcell.__len__()):
        h.MuTcell[c].isoma.gl_ichanR859C1 = x
        h.MuTcell[c].isoma.gkfbar_ichanR859C1 = y
        h.MuTcell[c].isoma.gnatbar_ichanR859C1 = z
    #  Run the model
    h.runMuTsim()

    mut_voltage = None
    vec_size = h.data_vecs_Mut.__len__()
    if vec_size > 0:
        mut_voltage = torch.zeros(n_shape, dtype=torch.float)
        for y in range(n_shape[0]):
            mut_voltage[y, :] = torch.as_tensor(h.data_vecs_Mut[y].as_numpy())

    return mut_voltage


def runHH(param):
    x, y, z = param

    h.initSemiHH_default()
    # change the values of the model in Neuron
    for c in range(h.MuTcell.__len__()):
        h.semiHH_Default_cell[c].isoma.gl_hh = x
        h.semiHH_Default_cell[c].isoma.gkbar_hh = y
        h.semiHH_Default_cell[c].isoma.gnabar_hh = z
    #  Run the model
    h.runSemiHH_defaultsim()

    out_voltage = None
    vec_size = h.data_vecs_semiHH_Default.__len__()
    if vec_size > 0:
        out_voltage = torch.zeros(n_shape, dtype=torch.float)
        for y in range(n_shape[0]):
            out_voltage[y, :] = torch.as_tensor(h.data_vecs_semiHH_Default[y].as_numpy())

    return out_voltage


def runHHR859C1(param):
    x, y, z = param

    h.initSemiHH_default()
    # change the values of the model in Neuron
    for c in range(h.data_vecs_semiHH_R859C1.__len__()):
        h.semiHH_R859C1_cell[c].isoma.gl_hh = x
        h.semiHH_R859C1_cell[c].isoma.gkbar_hh = y
        h.semiHH_R859C1_cell[c].isoma.gnabar_hh = z
    #  Run the model
    h.runSemiHH_defaultsim()

    out_voltage = None
    vec_size = h.semiHH_R859C1_cell.__len__()
    if vec_size > 0:
        out_voltage = torch.zeros(n_shape, dtype=torch.float)
        for y in range(n_shape[0]):
            out_voltage[y, :] = torch.as_tensor(h.data_vecs_semiHH_R859C1[y].as_numpy())

    return out_voltage


def test_parameters(params):
    if params.dim() == 1:
        params = params.unsqueeze(0)
    n_tests = params.shape[0]

    Error = torch.zeros((n_tests, 315))
    for iter in range(n_tests):
        Error[iter, :] = calculate_summary_statistics(
            # runMT(params[iter, :])
            runHH(params[iter, :])
            # runHHR859C1(params[iter, :])
        )

    with open(log_filename + '.csv', 'a') as f_log:
        temp = torch.cat([params, Error], dim=1).cpu().numpy()
        np.savetxt(f_log, temp, delimiter=',')

    return Error


def plots(data1, data2):
    # Dividing figure
    x, y = [7, 5]
    fig, ax = plt.subplots(x, y, sharex=True, sharey=True, figsize=(50.0, 30.0))

    t = np.linspace(0, data1.shape[1] - 1, data1.shape[1])

    # Drawing images
    current = 20
    current_add = 10
    current_counter = 0
    for xi in range(x):
        for yi in range(y):
            ax[xi, yi].title.set_text(f'Injection current {current + current_add * current_counter}pA')
            ax[xi, yi].plot(t, data1[current_counter, :], lw=2, label='observation')
            ax[xi, yi].plot(t, data2[current_counter, :], '--', lw=2, label='posterior sample')

            if yi == 0:
                ax[xi, yi].set(ylabel='mV')
            if xi == x - 1:
                ax[xi, yi].set(xlabel='Millisecond')

            current_counter += 1

    plt.subplots_adjust(hspace=0.3)

    plt.savefig('volts.png')
    plt.draw()
    plt.pause(0.5)

    return [fig, ax]


def update_plots(data1, data2, graph):
    [fig, ax] = graph

    t = np.linspace(0, data1.shape[1] - 1, data1.shape[1])

    # Drawing images
    current = 20
    current_add = 10
    current_counter = 0
    for xi in range(len(ax)):
        for yi in range(len(ax[0])):
            ax[xi, yi].cla()
            ax[xi, yi].title.set_text(f'Injection current {current + current_add * current_counter}pA')
            ax[xi, yi].plot(t, data1[current_counter, :], lw=2, label='observation')
            ax[xi, yi].plot(t, data2[current_counter, :], '--', lw=2, label='posterior sample')

            if yi == 0:
                ax[xi, yi].set(ylabel='mV')
            if xi == len(ax) - 1:
                ax[xi, yi].set(xlabel='Millisecond')

            current_counter += 1

    plt.subplots_adjust(hspace=0.3)
    plt.draw()
    plt.pause(0.5)


def plots_single(data1, data2):
    # Dividing figure
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10.0, 6.0))

    t = np.linspace(0, data1.shape[1] - 1, data1.shape[1])

    for dp, _ in enumerate(data1):
        ax.cla()
        # Drawing images
        ax.plot(t, data1[dp], lw=2, label='observation')
        ax.plot(t, data2[dp], '--', lw=2, label='posterior sample')

        plt.savefig('fig-' + str(dp) + '.png')


def plots_overlay(data1, data2, pics):
    # Dividing figure
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(20.0, 12.0))

    t = np.linspace(0, data1.shape[1] - 1, data1.shape[1])
    current = 20
    current_add = 10
    ax.set(ylabel='mV')
    ax.set(xlabel='Millisecond')

    legent = []
    for dp in pics:
        # Drawing images
        ax.plot(t, data1[dp], lw=2)
        legent.append(f'WT with current input {current + current_add * dp}pA')
        ax.plot(t, data2[dp], '--', lw=2)
        legent.append(f'MT with current input {current + current_add * dp}pA')
        ax.legend(legent)
        plt.savefig('fig-' + str(dp) + '.png')


def scatter3d(x, y, z, cs, colorsMap='gist_ncar'):
    # def scatter3d(x, y, z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=num_n)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, marker=".", s=5, c=scalarMap.to_rgba(cs), alpha=0.2, edgecolors=None)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    ax.set_xlabel('gl')
    ax.set_ylabel('gK')
    ax.set_zlabel('gNa')

    plt.show()


# -------
# Observed data
h.load_file("stdrun.hoc")  # for run control
h.load_file("neuron.hoc")  # run the model
h.runWTsim()
h.initMT()
h.runMuTsim()
num_n = h.data_vecs_WT.__len__()
if num_n > 0:
    n_shape = [num_n, h.data_vecs_WT[0].__len__()]
    WT_voltage = torch.zeros(n_shape, dtype=torch.float)
    MT_voltage = torch.zeros(n_shape, dtype=torch.float)
    for y in range(n_shape[0]):
        WT_voltage[y, :] = torch.as_tensor(h.data_vecs_WT[y].as_numpy(), dtype=torch.float)
        MT_voltage[y, :] = torch.as_tensor(h.data_vecs_Mut[y].as_numpy(), dtype=torch.float)

    observation_summary_statistics_WT = calculate_summary_statistics(WT_voltage)
    observation_summary_statistics_MT = calculate_summary_statistics(MT_voltage)


#  ---------- parmeters ----------------
#  ----- init
# define Prior over model parameters
# ----------- HH ranges
#             gl,  gK, gNa
# prior_min = [0.1, 26., 65.]  # fields[0], fields[1], ....
# prior_max = [0.5, 49., 260.]  # fields[0], fields[1], ....
# ----------------
#
#             gl,  gK, gNa
prior_min = [0.00001, 0.001, 0.01]  # fields[0], fields[1], ....
prior_max = [0.001, 0.1, 1.0]  # fields[0], fields[1], ....
num_posterior_draw = 10000
num_simulations = 250000
data = {}

# Draw a sample from the posterior and convert to numpy for plotting.
observation_summary_statistics = calculate_summary_statistics(WT_voltage)
log_filename = 'posterior-log-SNPE-HH'
#
# --------------------
# methods = ['SNPE', 'SNLE', 'SNRE']
methods = ['SNPE']
for m in methods:
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                        high=torch.as_tensor(prior_max))
    # infer_obj = infer(test_parameters, prior, method=m, num_simulations=num_simulations, num_workers=1)
    data[m + '-posterior'] = infer(test_parameters, prior, method=m, num_simulations=num_simulations, num_workers=1)
    with open(log_filename + '.pkl', "wb") as f:
        pickle.dump(data, f)

    data[m + '-posterior_sample-WT'] = data[m + '-posterior'].sample(
        sample_shape=(num_posterior_draw,), x=observation_summary_statistics_WT).squeeze().cpu().numpy()
    with open(log_filename + '.pkl', "wb") as f:
        pickle.dump(data, f)

    data[m + '-posterior_sample-MT'] = data[m + '-posterior'].sample(
        sample_shape=(num_posterior_draw,), x=observation_summary_statistics_MT).squeeze().cpu().numpy()
    with open(log_filename + '.pkl', "wb") as f:
        pickle.dump(data, f)
#
# #  -----search the max in the log_posterior_sample--------------------
# # load_filename = "./Results/3/SNPE-log-posterior_sample"
#
# with open(log_filename + ".pkl", 'rb') as f:
#     posterior_sample = pickle.load(f)
#
# subText = '-posterior_sample'
# data = {}
#
# for key in posterior_sample.keys():
#     if key.count(subText) > 0:
#         p_s = posterior_sample[key]
#         name = key.split(subText)[0]
#         data[name] = {}
#
#         data[name]['pearson'] = np.zeros((p_s.shape[0]))
#         data[name]['max_pearson'] = 0
#
#         for i, d in enumerate(tqdm(p_s)):
#             ansMT = runMT(d)
#             temp = np.sum(pearsonMatrix(ansMT).cpu().numpy())
#             data[name]['pearson'][i] = temp
#             if temp > data[name]['max_pearson']:
#                 data[name]['max_pearson'] = temp
#                 data[name]['max_posterior_sample'] = d
#                 data[name]['max_ansMT'] = ansMT.cpu().numpy()
#
#         with open(log_filename + "-MAX.pkl", 'wb') as f:
#             pickle.dump(data, f)


# # ---------------plot the search results--------------------------
# files = [
# # "./Results/1/SNPE-log.npy",
# ]
# observation_summary_statistics_WT = observation_summary_statistics_WT.cpu().numpy()
# for f in files:
#     result = list()
#     with open(f, "rt") as tf:
#         for l in tqdm(tf):
#             l = np.array(l.split(','), dtype=np.float)
#
#
#             result.append([
#                 l[0], l[1], l[2],
#                 l[3+7::9].sum(),
#                 l[3+8::9].sum(),
#                 np.sum(np.abs(observation_summary_statistics_WT[0::9] - l[3 + 0::9]))
#                            ])
#     result = np.array(result, dtype=np.float)
#
#     tqdm.write(f'Filename: {f}')
#
#     max_pearson_index = np.where(result[:, 3] == max(result[:, 3]))
#     min_hamming_index = np.where(result[:, 4] == min(result[:, 4]))
#     mini_spike_count_index = np.where(result[:, 5] == min(result[:, 5]))
#
#     for max_cor in max_pearson_index:
#         tqdm.write(f'Max Pearson coordinates: {result[max_cor,0:4]}')
#     tqdm.write('')
#
#     for min_cor in min_hamming_index:
#         tqdm.write(f'Min Hamming coordinates: {result[min_cor, 0:3], result[min_cor, 4]}')
#     tqdm.write('')
#
#     for min_cor in mini_spike_count_index:
#         tqdm.write(f'Min spike count coordinates: {result[min_cor, 0:3], result[min_cor, 5]}')
#     tqdm.write('')
#
#     tqdm.write(f'Max Pearson: {np.max(result[:, 3])}')
#     tqdm.write(f'Min Hamming: {np.min(result[:, 4])}')
#     tqdm.write(f'Min spike count: {np.min(result[:, 5])}')
#     tqdm.write('')
#
#     n, b = np.histogram(result[:, 3])
#     tqdm.write('Pearson histogram')
#     tqdm.write(str(n))
#     tqdm.write(str(np.round(b, 2)))
#     tqdm.write('')
#
#     n, b = np.histogram(result[:, 4])
#     tqdm.write('Hamming histogram')
#     tqdm.write(str(n))
#     tqdm.write(str(np.round(b, 2)))
#     tqdm.write('')
#
#     n, b = np.histogram(result[:, 5])
#     tqdm.write('Min spike count histogram')
#     tqdm.write(str(n))
#     tqdm.write(str(np.round(b, 2)))
#     tqdm.write('')
#
#     scatter3d(result[:, 0], result[:, 1], result[:, 2], result[:, 3])
#     scatter3d(result[:, 0], result[:, 1], result[:, 2], result[:, 4])
#     scatter3d(result[:, 0], result[:, 1], result[:, 2], result[:, 5])
#
#     tqdm.write('end plotting')

#     # posterior_pearson = np.load(f)
#     # pearson_sum = np.sum(posterior_pearson[:, list(range(2 + 8, posterior_pearson.shape[1], 8))], 1)
#     # scatter3d(posterior_pearson[:, 0], posterior_pearson[:, 1], posterior_pearson[:, 2], pearson_sum[:])
#     # n, b = np.histogram(pearson_sum)
#     # tqdm.write(f'Filename: {f}')
#     # tqdm.write(f'Max: {np.max(pearson_sum)}')
#     # tqdm.write(str(n))
#     # tqdm.write(str(np.round(b, 2)))
#

# # ------------------- plot max graph -----------------------
# load_filename = "./Results/SBI1/posterior-log-SNLE-MAX.pkl"
# with open(load_filename, 'rb') as f:
#     Max = pickle.load(f)
#
# # plots_single(WT_voltage, Max['max_ansMT'][0])
# # plots(WT_voltage, Max['max_ansMT'][0])
# # print(f'0 = {Max["max_pearson"][0]:10.4} - {Max["max_posterior_sample"][0]} \n')
#
# for k in Max.keys():
#     plots_single(WT_voltage, Max[k]['max_ansMT'])
#     plots(WT_voltage, Max[k]['max_ansMT'])
#     print(f'0 = {Max[k]["max_pearson"]:10.4} - {Max[k]["max_posterior_sample"]} \n')

# #  ---------plot log_posterior_sample----------------
# key = 'SNPE'
# load_filename = "./Results/SBI2/posterior-log-" + key
#
# with open(load_filename + ".pkl", 'rb') as f:
#     posterior_sample = pickle.load(f)
# posterior_sample = posterior_sample[key + '-posterior_sample']
#
# with open(load_filename + "-MAX.pkl", 'rb') as f:
#     posterior_pearson = pickle.load(f)
# posterior_pearson = posterior_pearson[key]['pearson']
#
# scatter3d(posterior_sample[:, 0],
#           posterior_sample[:, 1],
#           posterior_sample[:, 2],
#           posterior_pearson)


# --- overlay graphs ------
# v = runMT([0.00051143, 0.05419266, 0.33151907])
# plots_overlay(WT_voltage, v, [0,1,2,3,4,5,6,7,8,9,10,11,12])
# plots(WT_voltage, v)
print('Done')
