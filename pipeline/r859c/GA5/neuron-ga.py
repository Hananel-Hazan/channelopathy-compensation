import os
import sys
import numpy as np
import scipy
from scipy import stats as spstats
from scipy.spatial import distance
from tqdm import tqdm
import datetime

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

# deap
from deap import creator, base, tools, algorithms
import random

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

    t = np.linspace(0, dt * x.shape[1], x.shape[1])

    returnVec = np.zeros((x.shape[0], n_summary))
    for index in range(x.shape[0]):
        # initialise array of spike counts
        # v = np.copy(x[index, :])
        v = x[index, :].copy()

        # put everything to -10 that is below -10 or has negative slope
        ind = np.where(v < -10)
        v[ind] = -10
        ind = np.where(np.diff(v) < 0)
        v[ind] = -10

        # remaining negative slopes are at spike peaks
        ind = np.where(np.diff(v) < 0)
        spike_times = np.array(t)[ind]
        spike_times_stim = spike_times[(spike_times > t_on) & (spike_times < t_off)]

        # number of spikes
        if spike_times_stim.shape[0] > 0:
            spike_times_stim = spike_times_stim[
                np.append(1, np.diff(spike_times_stim)) > 0.5
                ]

        # resting potential and std
        # rest_pot = np.mean(x[index, :][t < t_on])
        # rest_pot_std = np.std(x[index, :][int(0.9 * t_on / dt): int(t_on / dt)])
        #
        # if np.isnan(rest_pot_std):
        #     rest_pot_std = 0
        # if np.isnan(rest_pot):
        #     rest_pot = 0
        rest_pot = np.mean(x[index, :])
        rest_pot_std = np.std(x[index, :])

        # moments
        std_pw = np.power(
            np.std(x[index, :][(t > t_on) & (t < t_off)]), np.linspace(3, n_mom, n_mom - 2)
        )
        std_pw = np.concatenate((np.ones(1), std_pw))

        if sum(std_pw == 0) == 0:
            moments = (
                    spstats.moment(
                        x[index, :][(t > t_on) & (t < t_off)], np.linspace(2, n_mom, n_mom - 1)
                    )
                    / std_pw
            )
            moments[np.isnan(moments)] = 0
        else:
            moments = np.zeros(np.linspace(2, n_mom, n_mom - 1).shape)

        # concatenation of summary statistics
        # sum_stats_vec = np.concatenate(
        #     (
        #         np.array([spike_times_stim.shape[0]]),
        #         np.array(
        #             [rest_pot, rest_pot_std, np.mean(x[index, :][(t > t_on) & (t < t_off)])]
        #         ),
        #         moments,
        #     )
        # )
        # returnVec[index, :n_summary] = sum_stats_vec

        returnVec[index, 0] = spike_times_stim.shape[0]
        returnVec[index, 1] = rest_pot
        returnVec[index, 2] = rest_pot_std
        returnVec[index, 3] = np.mean(x[index, :][(t > t_on) & (t < t_off)])
        returnVec[index, 4:7] = moments
        returnVec[index, 7] = pearson([x[index, :], WT_voltage[index, :]])  # Pearson correlation
        returnVec[index, 8] = distance.hamming(x[index, :], WT_voltage[index, :])  # Hamming

    return returnVec.reshape(-1)


def pearson(z):
    """Pearson correlation coefficient of the two 1-D arrays in z = (x, y),
    computed with scipy.stats.pearsonr."""
    x, y = z

    val = scipy.stats.pearsonr(x, y)

    r_val = val[0]  # + (0 if val[1] > 1 else 1 - val[1])
    return r_val


def pearsonMatrix(x):
    ans = np.zeros(n_shape[0])
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
        mut_voltage = np.zeros(n_shape, dtype=np.float)
        for y in range(n_shape[0]):
            mut_voltage[y, :] = np.asarray(h.data_vecs_Mut[y])

    return mut_voltage


def hammingMatrix(x):
    ans = np.zeros(n_shape[0])
    for i in range(n_shape[0]):
        ans[i] = distance.hamming(x[i, :], WT_voltage[i, :])
    return ans


def test_parameters(params):
    mut_voltage = runMT(params)
    Error = calculate_summary_statistics(mut_voltage).cpu().numpy()

    # with open(log_filename + '.csv', 'a') as f_log:
    #     temp = np.concatenate((params, Error))
    #     np.savetxt(f_log, temp, delimiter=',')

    return Error


def plots(data1, data2):
    # Dividing figure
    fig, (
        (ax11, ax12, ax13, ax14, ax15),
        (ax21, ax22, ax23, ax24, ax25),
        (ax31, ax32, ax33, ax34, ax35),
        (ax41, ax42, ax43, ax44, ax45),
        (ax51, ax52, ax53, ax54, ax55),
        (ax61, ax62, ax63, ax64, ax65),
        (ax71, ax72, ax73, ax74, ax75),
    ) = plt.subplots(7, 5, sharex=True, sharey=True, figsize=(10.0, 6.0))

    t = np.linspace(0, data1.shape[1] - 1, data1.shape[1])
    d1 = data1
    d2 = data2

    # Drawing images
    ax11.plot(t, d1[0, :], lw=2, label='observation')
    ax11.plot(t, d2[0, :], '--', lw=2, label='posterior sample')
    ax12.plot(t, d1[1, :], lw=2, label='observation')
    ax12.plot(t, d2[1, :], '--', lw=2, label='posterior sample')
    ax13.plot(t, d1[2, :], lw=2, label='observation')
    ax13.plot(t, d2[2, :], '--', lw=2, label='posterior sample')
    ax14.plot(t, d1[3, :], lw=2, label='observation')
    ax14.plot(t, d2[3, :], '--', lw=2, label='posterior sample')
    ax15.plot(t, d1[4, :], lw=2, label='observation')
    ax15.plot(t, d2[4, :], '--', lw=2, label='posterior sample')

    ax21.plot(t, d1[5, :], lw=2, label='observation')
    ax21.plot(t, d2[5, :], '--', lw=2, label='posterior sample')
    ax22.plot(t, d1[6, :], lw=2, label='observation')
    ax22.plot(t, d2[6, :], '--', lw=2, label='posterior sample')
    ax23.plot(t, d1[7, :], lw=2, label='observation')
    ax23.plot(t, d2[7, :], '--', lw=2, label='posterior sample')
    ax24.plot(t, d1[8, :], lw=2, label='observation')
    ax24.plot(t, d2[8, :], '--', lw=2, label='posterior sample')
    ax25.plot(t, d1[9, :], lw=2, label='observation')
    ax25.plot(t, d2[9, :], '--', lw=2, label='posterior sample')

    ax31.plot(t, d1[10, :], lw=2, label='observation')
    ax31.plot(t, d2[10, :], '--', lw=2, label='posterior sample')
    ax32.plot(t, d1[11, :], lw=2, label='observation')
    ax32.plot(t, d2[11, :], '--', lw=2, label='posterior sample')
    ax33.plot(t, d1[12, :], lw=2, label='observation')
    ax33.plot(t, d2[12, :], '--', lw=2, label='posterior sample')
    ax34.plot(t, d1[13, :], lw=2, label='observation')
    ax34.plot(t, d2[13, :], '--', lw=2, label='posterior sample')
    ax35.plot(t, d1[14, :], lw=2, label='observation')
    ax35.plot(t, d2[14, :], '--', lw=2, label='posterior sample')

    ax41.plot(t, d1[15, :], lw=2, label='observation')
    ax41.plot(t, d2[15, :], '--', lw=2, label='posterior sample')
    ax42.plot(t, d1[16, :], lw=2, label='observation')
    ax42.plot(t, d2[16, :], '--', lw=2, label='posterior sample')
    ax43.plot(t, d1[17, :], lw=2, label='observation')
    ax43.plot(t, d2[17, :], '--', lw=2, label='posterior sample')
    ax44.plot(t, d1[18, :], lw=2, label='observation')
    ax44.plot(t, d2[18, :], '--', lw=2, label='posterior sample')
    ax45.plot(t, d1[19, :], lw=2, label='observation')
    ax45.plot(t, d2[19, :], '--', lw=2, label='posterior sample')

    ax51.plot(t, d1[20, :], lw=2, label='observation')
    ax51.plot(t, d2[20, :], '--', lw=2, label='posterior sample')
    ax52.plot(t, d1[21, :], lw=2, label='observation')
    ax52.plot(t, d2[21, :], '--', lw=2, label='posterior sample')
    ax53.plot(t, d1[22, :], lw=2, label='observation')
    ax53.plot(t, d2[22, :], '--', lw=2, label='posterior sample')
    ax54.plot(t, d1[23, :], lw=2, label='observation')
    ax54.plot(t, d2[23, :], '--', lw=2, label='posterior sample')
    ax55.plot(t, d1[24, :], lw=2, label='observation')
    ax55.plot(t, d2[24, :], '--', lw=2, label='posterior sample')

    ax61.plot(t, d1[25, :], lw=2, label='observation')
    ax61.plot(t, d2[25, :], '--', lw=2, label='posterior sample')
    ax62.plot(t, d1[26, :], lw=2, label='observation')
    ax62.plot(t, d2[26, :], '--', lw=2, label='posterior sample')
    ax63.plot(t, d1[27, :], lw=2, label='observation')
    ax63.plot(t, d2[27, :], '--', lw=2, label='posterior sample')
    ax64.plot(t, d1[28, :], lw=2, label='observation')
    ax64.plot(t, d2[28, :], '--', lw=2, label='posterior sample')
    ax65.plot(t, d1[29, :], lw=2, label='observation')
    ax65.plot(t, d2[29, :], '--', lw=2, label='posterior sample')

    ax71.plot(t, d1[30, :], lw=2, label='observation')
    ax71.plot(t, d2[30, :], '--', lw=2, label='posterior sample')
    ax72.plot(t, d1[31, :], lw=2, label='observation')
    ax72.plot(t, d2[31, :], '--', lw=2, label='posterior sample')
    ax73.plot(t, d1[32, :], lw=2, label='observation')
    ax73.plot(t, d2[32, :], '--', lw=2, label='posterior sample')
    ax74.plot(t, d1[33, :], lw=2, label='observation')
    ax74.plot(t, d2[33, :], '--', lw=2, label='posterior sample')
    ax75.plot(t, d1[34, :], lw=2, label='observation')
    ax75.plot(t, d2[34, :], '--', lw=2, label='posterior sample')

    plt.draw()
    plt.pause(0.5)

    return [fig, ax11, ax12, ax13, ax14, ax15, ax21, ax22, ax23, ax24, ax25, ax31, ax32, ax33, ax34, ax35,
            ax41, ax42, ax43, ax44, ax45, ax51, ax52, ax53, ax54, ax55, ax61, ax62, ax63, ax64, ax65,
            ax71, ax72, ax73, ax74, ax75]


def update_plots(data1, data2, ax):
    [fig, ax11, ax12, ax13, ax14, ax15, ax21, ax22, ax23, ax24, ax25, ax31, ax32, ax33, ax34, ax35,
     ax41, ax42, ax43, ax44, ax45, ax51, ax52, ax53, ax54, ax55, ax61, ax62, ax63, ax64, ax65,
     ax71, ax72, ax73, ax74, ax75] = ax

    ax11.cla(), ax12.cla(), ax13.cla(), ax14.cla(), ax15.cla(), ax21.cla(), ax22.cla(), ax23.cla(), ax24.cla(), ax25.cla(),
    ax31.cla(), ax32.cla(), ax33.cla(), ax34.cla(), ax35.cla(), ax41.cla(), ax42.cla(), ax43.cla(), ax44.cla(), ax45.cla(),
    ax51.cla(), ax52.cla(), ax53.cla(), ax54.cla(), ax55.cla(), ax61.cla(), ax62.cla(), ax63.cla(), ax64.cla(), ax65.cla(),
    ax71.cla(), ax72.cla(), ax73.cla(), ax74.cla(), ax75.cla(),

    t = np.linspace(0, data1.shape[1] - 1, data1.shape[1])
    d1 = data1
    d2 = data2

    # Drawing images
    ax11.plot(t, d1[0, :], lw=2, label='observation')
    ax11.plot(t, d2[0, :], '--', lw=2, label='posterior sample')
    ax12.plot(t, d1[1, :], lw=2, label='observation')
    ax12.plot(t, d2[1, :], '--', lw=2, label='posterior sample')
    ax13.plot(t, d1[2, :], lw=2, label='observation')
    ax13.plot(t, d2[2, :], '--', lw=2, label='posterior sample')
    ax14.plot(t, d1[3, :], lw=2, label='observation')
    ax14.plot(t, d2[3, :], '--', lw=2, label='posterior sample')
    ax15.plot(t, d1[4, :], lw=2, label='observation')
    ax15.plot(t, d2[4, :], '--', lw=2, label='posterior sample')

    ax21.plot(t, d1[5, :], lw=2, label='observation')
    ax21.plot(t, d2[5, :], '--', lw=2, label='posterior sample')
    ax22.plot(t, d1[6, :], lw=2, label='observation')
    ax22.plot(t, d2[6, :], '--', lw=2, label='posterior sample')
    ax23.plot(t, d1[7, :], lw=2, label='observation')
    ax23.plot(t, d2[7, :], '--', lw=2, label='posterior sample')
    ax24.plot(t, d1[8, :], lw=2, label='observation')
    ax24.plot(t, d2[8, :], '--', lw=2, label='posterior sample')
    ax25.plot(t, d1[9, :], lw=2, label='observation')
    ax25.plot(t, d2[9, :], '--', lw=2, label='posterior sample')

    ax31.plot(t, d1[10, :], lw=2, label='observation')
    ax31.plot(t, d2[10, :], '--', lw=2, label='posterior sample')
    ax32.plot(t, d1[11, :], lw=2, label='observation')
    ax32.plot(t, d2[11, :], '--', lw=2, label='posterior sample')
    ax33.plot(t, d1[12, :], lw=2, label='observation')
    ax33.plot(t, d2[12, :], '--', lw=2, label='posterior sample')
    ax34.plot(t, d1[13, :], lw=2, label='observation')
    ax34.plot(t, d2[13, :], '--', lw=2, label='posterior sample')
    ax35.plot(t, d1[14, :], lw=2, label='observation')
    ax35.plot(t, d2[14, :], '--', lw=2, label='posterior sample')

    ax41.plot(t, d1[15, :], lw=2, label='observation')
    ax41.plot(t, d2[15, :], '--', lw=2, label='posterior sample')
    ax42.plot(t, d1[16, :], lw=2, label='observation')
    ax42.plot(t, d2[16, :], '--', lw=2, label='posterior sample')
    ax43.plot(t, d1[17, :], lw=2, label='observation')
    ax43.plot(t, d2[17, :], '--', lw=2, label='posterior sample')
    ax44.plot(t, d1[18, :], lw=2, label='observation')
    ax44.plot(t, d2[18, :], '--', lw=2, label='posterior sample')
    ax45.plot(t, d1[19, :], lw=2, label='observation')
    ax45.plot(t, d2[19, :], '--', lw=2, label='posterior sample')

    ax51.plot(t, d1[20, :], lw=2, label='observation')
    ax51.plot(t, d2[20, :], '--', lw=2, label='posterior sample')
    ax52.plot(t, d1[21, :], lw=2, label='observation')
    ax52.plot(t, d2[21, :], '--', lw=2, label='posterior sample')
    ax53.plot(t, d1[22, :], lw=2, label='observation')
    ax53.plot(t, d2[22, :], '--', lw=2, label='posterior sample')
    ax54.plot(t, d1[23, :], lw=2, label='observation')
    ax54.plot(t, d2[23, :], '--', lw=2, label='posterior sample')
    ax55.plot(t, d1[24, :], lw=2, label='observation')
    ax55.plot(t, d2[24, :], '--', lw=2, label='posterior sample')

    ax61.plot(t, d1[25, :], lw=2, label='observation')
    ax61.plot(t, d2[25, :], '--', lw=2, label='posterior sample')
    ax62.plot(t, d1[26, :], lw=2, label='observation')
    ax62.plot(t, d2[26, :], '--', lw=2, label='posterior sample')
    ax63.plot(t, d1[27, :], lw=2, label='observation')
    ax63.plot(t, d2[27, :], '--', lw=2, label='posterior sample')
    ax64.plot(t, d1[28, :], lw=2, label='observation')
    ax64.plot(t, d2[28, :], '--', lw=2, label='posterior sample')
    ax65.plot(t, d1[29, :], lw=2, label='observation')
    ax65.plot(t, d2[29, :], '--', lw=2, label='posterior sample')

    ax71.plot(t, d1[30, :], lw=2, label='observation')
    ax71.plot(t, d2[30, :], '--', lw=2, label='posterior sample')
    ax72.plot(t, d1[31, :], lw=2, label='observation')
    ax72.plot(t, d2[31, :], '--', lw=2, label='posterior sample')
    ax73.plot(t, d1[32, :], lw=2, label='observation')
    ax73.plot(t, d2[32, :], '--', lw=2, label='posterior sample')
    ax74.plot(t, d1[33, :], lw=2, label='observation')
    ax74.plot(t, d2[33, :], '--', lw=2, label='posterior sample')
    ax75.plot(t, d1[34, :], lw=2, label='observation')
    ax75.plot(t, d2[34, :], '--', lw=2, label='posterior sample')

    plt.draw()
    plt.pause(0.001)


def scatter3d(x, y, z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=num_n)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, marker=".", s=5, c=scalarMap.to_rgba(cs), alpha=0.5, edgecolors=None)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    ax.set_xlabel('gl')
    ax.set_ylabel('gK')
    ax.set_zlabel('gNa')

    plt.show()


# -------
# Observed data
h.load_file("stdrun.hoc")  # for run control
# h.load_file("APthreshold WT-2005.hoc")  # run the model
h.load_file("neuron.hoc")  # run the model
h.runWTsim()

num_n = h.data_vecs_WT.__len__()
if num_n > 0:
    n_shape = [num_n, h.data_vecs_WT[0].__len__()]
    WT_voltage = np.zeros(n_shape, dtype=np.float)
    for y in range(n_shape[0]):
        WT_voltage[y, :] = np.array(h.data_vecs_WT[y], dtype=np.float)

    observation_summary_statistics_WT = calculate_summary_statistics(WT_voltage)

# -----------------------------
# param = [0.01, 0.01, 0.1]
# v = runMT(param)
# ax = plots(WT_voltage, v)
# count = 0
# while True:
#     p = random.randint(0, 2)
#     param[p] += 0.01 * random.uniform(-1 ,1)
#     param[p] = abs(param[p])
#     v = runMT(param)
#     update_plots(WT_voltage, v, ax)
#     pearson_score = pearsonMatrix(v)
#     print(f'{count} = {sum(pearson_score):8.5} -> [{param[0]:3.5}, {param[1]:3.5}, {param[2]:3.5}]')
#     count += 1

#  ---------- parmeters ----------------
TESTS = 10000
Generations = 100000
POPULATION_SIZE = 3
log_filename = str(datetime.datetime.now()).replace(':', '.')
# random.seed(64)
# -----------------------------


from deap import algorithms, base, creator, tools

creator.create("FitnessTarget", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessTarget)


# Attribute generator
def rnd():
    return np.random.random()


toolbox = base.Toolbox()
toolbox.register("myFunc", rnd)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.myFunc, n=POPULATION_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    individual = [individual]
    r1 = np.zeros(len(individual))
    r2 = np.zeros(len(individual))
    r3 = np.zeros(len(individual))
    for i, vec in enumerate(individual):

        if 0 > vec[0] or vec[0] > 0.001:
            return np.iinfo(np.int32).max,
            # return 0, np.iinfo(np.int32).max,
        if 0.001 > vec[1] or vec[1] > 0.1:
            return np.iinfo(np.int32).max,
            # return 0, np.iinfo(np.int32).max,
        if 0.01 > vec[2] or vec[2] > 1:
            return np.iinfo(np.int32).max,
            # return 0, np.iinfo(np.int32).max,
        v = runMT(vec)
        # if (v == np.nan).sum() + (v == np.inf).sum():
        #     return np.inf,
        
        r = calculate_summary_statistics(v)
        r1[i] = r[7::9].sum()  # Pearson
        r2[i] = r[8::9].sum()  # Hamming
        r3[i] = np.sum(np.abs(observation_summary_statistics_WT[0::9] - r[0::9]))  # Spikes      

        # update_plots(WT_voltage, v, ax)
        # print(f'{score:8.5} -> [{vec[0]:3.5}, {vec[1]:3.5}, {vec[2]:3.5}]')
        with open(log_filename + '.csv', 'a') as f_log:
            f_log.write(f'{r1[i]},{r2[i]},{r3[i]},{vec[0]}, {vec[1]}, {vec[2]} \n')
        sys.stdout.flush()

    # r1 = np.sum(r1)
    # r2 = np.sum(r2)

    # return r1, r2,
    return r3,



def cxSet(ind1, ind2):
    """
    Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    for i in range(ind1.__len__()):
        if random.random() > 0.5:
            t = ind1[i]
            ind1[i] = ind2[i]
            ind1[i] = t

    return ind1, ind2


toolbox.register("evaluate", evalOneMax)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", cxSet)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=int(TESTS/2))


def main():
    pop = toolbox.population(n=TESTS)
    hof = tools.HallOfFame(10)  # , similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=Generations,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    pop, log, hof = main()
    print(f'Hall of Fame - {hof}')
    print('Done')
