import datetime
import argparse
import numpy as np
from scipy import stats
from dtaidistance import dtw  # computing Dynamic Time Warping (DTW) Distance Measure
from tqdm import tqdm
import pickle
import bz2
import sys
import os
import multiprocessing

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.ticker import PercentFormatter

# Neuron
from neuron import h

# remove top and right axis from plots
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
log_filename = ''

crateria_cell_type = ['WT', 'MT']
test_crateria_type = ['Spike Count', 'Spike Width avg', 'Spike Width std', 'Dynamic Time Warping']


def compute_DTW(x):
    t_mt = 0
    t_wt = 0
    for indx, v in enumerate(x):
        t_mt += np.abs(dtw.distance_fast(MT_voltage[indx], v, use_pruning=True))
        t_wt += np.abs(dtw.distance_fast(WT_voltage[indx], v, use_pruning=True))
    return t_wt, t_mt


def find_spikes(x):
    t = np.linspace(0, h.dt * x.shape[1], x.shape[1])
    v = x.copy()
    threshold = -10

    new_v = np.zeros(v.shape)
    # put everything to -10 that is below -10 or has negative slope
    ind = np.where(v < threshold)
    v[ind] = -10
    # put everything above -10 to be +10 for easyer spike count.
    ind = np.where(v > threshold)
    v[ind] = 10

    # mark when spike begin
    ind = np.where(np.diff(v) < 0)
    new_v[ind] = -1

    # mark when spike end
    ind = np.where(np.diff(v) > 0)
    new_v[ind] = 1

    # remaining negative slopes are at spike peaks
    ind = np.where(np.diff(new_v) != 0)
    ind = ((new_v != 0).sum(axis=1), ind[0], t[ind[1]])

    return ind


def calculate_spike_properties(x):
    spike_indies = find_spikes(x)
    return_value = {}

    # count the number of spikes. (if the spike as been cutoff, round the number to be floor
    return_value[test_crateria_type[0]] = np.array(spike_indies[0] / 2, dtype=np.int)

    # calculate spike width
    spike_width_index = spike_indies[1][::2]
    spike_width = np.diff(spike_indies[2])[::2]

    if len(spike_width_index) > len(spike_width):
        spike_width_index = spike_width_index[:-1]

    spike_width_avg = np.zeros(x.shape[0])
    spike_width_std = np.zeros(x.shape[0])
    for test in range(x.shape[0]):
        sp_ind = np.where((spike_width_index == test) == 1)
        if sp_ind[0].shape[0] > 0:
            spike_width_avg[test] = np.mean(spike_width[sp_ind])
            spike_width_std[test] = np.std(spike_width[sp_ind])

    return_value[test_crateria_type[1]] = np.copy(spike_width_avg)
    return_value[test_crateria_type[2]] = np.copy(spike_width_std)
    return_value[test_crateria_type[3]] = compute_DTW(x)

    return return_value


def calculate_summary_statistics(x):
    return_value = {}
    spike_properties = calculate_spike_properties(x)
    spike_properties_keys = list(spike_properties.keys())
    return_value[spike_properties_keys[0]] = spike_properties[spike_properties_keys[0]]
    return_value[spike_properties_keys[1]] = spike_properties[spike_properties_keys[1]]
    return_value[spike_properties_keys[2]] = spike_properties[spike_properties_keys[2]]

    return_value[spike_properties_keys[0] + ' WT delta'] = np.sum(
        np.abs(observation_summary_statistics_WT[test_crateria_type[0]] - spike_properties[spike_properties_keys[0]])
    )
    return_value[spike_properties_keys[1] + ' WT delta'] = np.sum(
        np.abs(observation_summary_statistics_WT[test_crateria_type[1]] - spike_properties[spike_properties_keys[1]])
    )
    return_value[spike_properties_keys[2] + ' WT delta'] = np.sum(
        np.abs(observation_summary_statistics_WT[test_crateria_type[2]] - spike_properties[spike_properties_keys[2]])
    )
    return_value[spike_properties_keys[0] + ' MT delta'] = np.sum(
        np.abs(observation_summary_statistics_MT[test_crateria_type[0]] - spike_properties[spike_properties_keys[0]])
    )
    return_value[spike_properties_keys[1] + ' MT delta'] = np.sum(
        np.abs(observation_summary_statistics_WT[test_crateria_type[1]] - spike_properties[spike_properties_keys[1]])
    )
    return_value[spike_properties_keys[2] + ' MT delta'] = np.sum(
        np.abs(observation_summary_statistics_WT[test_crateria_type[2]] - spike_properties[spike_properties_keys[2]])
    )

    return_value[spike_properties_keys[3] + ' WT'] = spike_properties[spike_properties_keys[3]][0]  # DTW WT
    return_value[spike_properties_keys[3] + ' MT'] = spike_properties[spike_properties_keys[3]][1]  # DTW MT

    return return_value


def pearsonMatrix(x):
    ans = np.zeros(n_shape[0])
    for i in range(n_shape[0]):
        ans[i] = stats.pearsonr([x[i, :], WT_voltage[i, :]])
    return ans


def runMT(param):
    x, y, z = param

    h.initMT()
    # change the values of the model in Neuron
    for c in range(n_shape[0]):
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
            mut_voltage[y, :] = h.data_vecs_Mut[y].as_numpy()

    return mut_voltage


def runWT(param):
    x, y, z = param

    h.initWT()
    # change the values of the model in Neuron
    for c in range(n_shape[0]):
        h.WTcell[c].isoma.gl_ichanWT2005 = x
        h.WTcell[c].isoma.gkfbar_ichanWT2005 = y
        h.WTcell[c].isoma.gnatbar_ichanWT2005 = z
    #  Run the model
    h.runWTsim()

    mut_voltage = None
    vec_size = h.data_vecs_WT.__len__()
    if vec_size > 0:
        mut_voltage = np.zeros(n_shape, dtype=np.float)
        for y in range(n_shape[0]):
            mut_voltage[y, :] = h.data_vecs_WT[y].as_numpy()

    return mut_voltage


def test_parameters(p_min, p_max, num, crateria_type):
    path = './db/'
    log_filename = str(datetime.datetime.now()).replace(':', '.')

    it = 0
    # pbar = tqdm()
    # pbar.refresh()
    while True:
        with bz2.BZ2File(path + log_filename + '-' + crateria_type[0] + '_stats.pkl', 'ab') as ct0_log:
            with bz2.BZ2File(path + log_filename + '-' + crateria_type[1] + '_stats.pkl', 'ab') as ct1_log:
                drow_lots = np.random.uniform(p_min, p_max)
                if sum(drow_lots < 0) > 0:
                    continue
                WT_stat = calculate_summary_statistics(runWT(drow_lots))
                MT_stat = calculate_summary_statistics(runMT(drow_lots))

                if WT_stat['Spike Count'].sum() > 0:
                    WT_stat['gl,  gK, gNa'] = drow_lots
                    pickle.dump(WT_stat, ct0_log)

                if MT_stat['Spike Count'].sum() > 0:
                    MT_stat['gl,  gK, gNa'] = drow_lots
                    pickle.dump(MT_stat, ct1_log)
        # pbar.update()
        it += 1
        if it > num:
            break


def plots(data1, data2, filename=None, path=None):
    # Dividing figure
    x, y = [7, 5]
    fig, ax = plt.subplots(x, y, sharex=True, sharey=True,
                           # figsize=(50.0, 30.0)
                           figsize=(25, 15)
                           )

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
    plt.tight_layout()
    if filename is not None:
        if path is not None:
            plt.savefig(path + '/' + filename + '.png')
        else:
            plt.savefig(filename + '.png')
        plt.close()
    else:
        plt.savefig('volt.png')
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


def scatter3d(x, y, z, cs, colorsMap='gist_ncar', title=None, filename=None, path=None, science=True):
    # def scatter3d(x, y, z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = mpl.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize=(12, 7))
    ax = Axes3D(fig)
    ax.scatter(x, y, z, marker=".", s=5, c=scalarMap.to_rgba(cs), alpha=0.2, edgecolors=None)
    scalarMap.set_array(cs)
    cbar = fig.colorbar(scalarMap)
    if science:
        cbar.set_label('Delta in ' + title.split('test')[1])  # , rotation=270)
    else:
        cbar.set_label('similarity to Wild-Type in %')  # , rotation=270)
    ax.set_xlabel('gl')
    ax.set_ylabel('gK')
    ax.set_zlabel('gNa')
    if title is not None:
        fig.suptitle(title)
    if filename is not None:
        if path is not None:
            plt.savefig(path + '/3D - ' + filename + '.png')
        else:
            plt.savefig('3D - ' + filename + '.png')
        plt.close()
    else:
        plt.draw()
        plt.pause(0.5)


def Four_scatter3d(figs, filename=None, path=None, animation=False, science=True):  # colorsMap='jet'
    # Dividing figure
    x, y = [2, 2]
    fig = plt.figure(figsize=(25, 15))
    fig.subplots_adjust(hspace=0.3)
    axs = list()
    counter = 0
    for xi in range(x):
        for yi in range(y):
            axs.append(fig.add_subplot(int(str(x) + str(y) + str(counter + 1)), projection='3d'))
            counter += 1

    # Drawing images
    counter = 0
    for ax in axs:
        colorsMap = figs[counter][4]
        cs = figs[counter][3]
        title = figs[counter][5]
        mi = max(min(cs), 0)
        if science:
            ma = max(cs)
        else:
            ma = min(max(cs), 100)
        if abs(mi - ma) < 3:
            mi = ma - 3
        cm = plt.get_cmap(colorsMap)
        cNorm = mpl.colors.Normalize(vmin=mi, vmax=ma)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        # ax = fig.add_subplot(int(str(x) + str(y) + str(counter + 1)), projection='3d')
        ax.scatter(figs[counter][0],
                   figs[counter][1],
                   figs[counter][2],
                   marker=".", s=5, c=scalarMap.to_rgba(cs), alpha=0.2, edgecolors=None)
        scalarMap.set_array(cs)
        cbar = plt.colorbar(scalarMap, ax=ax)
        if science:
            cbar.set_label('Delta in ' + title.split('\n')[0])
        else:
            cbar.set_label('similarity to ' + title.split('\n')[1].split('to ')[1] + ' in %')
        ax.set(xlabel='gl')
        ax.set(ylabel='gK')
        ax.set(zlabel='gNa')
        if title is not None:
            ax.title.set_text(title)
        counter += 1

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    ax = axs

    if animation:
        ##### TO CREATE A SERIES OF PICTURES

        def make_views(ax, angles, path, filename, elevation=None, prefix='_', **kwargs):
            """
            Makes jpeg pictures of the given 3d ax, with different angles.
            Args:
                ax (3D axis): te ax
                angles (list): the list of angles (in degree) under which to
                               take the picture.
                width,height (float): size, in inches, of the output images.
                prefix (str): prefix for the files created.

            Returns: the list of files created (for later removal)
            """

            files = []
            # ax.figure.set_size_inches(width, height)

            for i, angle in enumerate(angles):
                for a_ in ax:
                    a_.view_init(elev=elevation, azim=angle)
                fname = path + filename + '%s%03d.jpeg' % (prefix, i)
                # ax.figure.savefig(fname)
                plt.savefig(fname)
                files.append(fname)
            plt.close()

            return files

        ##### TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION

        def make_movie(files, path, filename, fps=10, bitrate=1800, **kwargs):
            """
            Uses mencoder, produces a .mp4/.ogv/... movie from a list of
            picture files.
            """
            output = path + filename
            output_name, output_ext = os.path.splitext(filename)
            command = {'.mp4': 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                                 -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                               % (",".join(files), fps, output_name, bitrate)}

            command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s' % (output_name, fps, output)

            print
            command[output_ext]
            output_ext = os.path.splitext(output)[1]
            os.system(command[output_ext])

        def make_gif(files, path, filename, delay=100, repeat=True, **kwargs):
            """
            Uses imageMagick to produce an animated .gif from a list of
            picture files.
            """

            loop = -1 if repeat else 0
            os.system('convert -delay %d -loop %d %s %s'
                      % (delay, loop, " ".join(files), path + filename))

        def make_strip(files, path, filename, **kwargs):
            """
            Uses imageMagick to produce a .jpeg strip from a list of
            picture files.
            """

            os.system('montage -tile 1x -geometry +0+0 %s %s' % (" ".join(files), path + filename))

        def rotanimate(ax, angles, path, filename, **kwargs):
            """
            Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
            a 3D ax

            Args:
                ax (3D axis): the ax containing the plot of interest
                angles (list): the list of angles (in degree) under which to
                               show the plot.
                output : name of the output file. The extension determines the
                         kind of animation used.
                **kwargs:
                    - width : in inches
                    - heigth: in inches
                    - framerate : frames per second
                    - delay : delay between frames in milliseconds
                    - repeat : True or False (.gif only)
            """
            files = make_views(ax, angles, path=path, filename=filename)
            # output_ext = os.path.splitext(filename)[1]
            # D = {'.mp4': make_movie,
            #      '.ogv': make_movie,
            #      '.gif': make_gif,
            #      '.jpeg': make_strip,
            #      '.png': make_strip}
            #
            # D[output_ext](files, output + '.mp4', **kwargs)
            #
            # for f in files:
            #     os.remove(f)

        angles = np.linspace(0, 360, 21)[:-1]  # A list of 20 angles between 0 and 360

        if path is None:
            path = './'

        # # create an animated gif (20ms between frames)
        # rotanimate(ax, angles, output=path + '/3D - ' + filename + '.gif', delay=20)

        # create a movie with 10 frames per seconds and 'quality' 2000
        rotanimate(ax, angles, path=path, filename='/3D - ' + filename, fps=10, bitrate=2000)

        # # create an ogv movie
        # rotanimate(ax, angles, output=path + '/3D - ' + filename + '.ogv', fps=10)
    else:
        if filename is not None:
            if path is not None:
                plt.savefig(path + '/' + '3D - ' + filename + '.png')
            else:
                plt.savefig('3D - ' + filename + '.png')
            plt.close()
        else:
            plt.draw()
            plt.pause(0.5)

    return fig


def Histogram(data1, data2, path, testType, ind, science=True, all=False):
    # Spike Count
    if all:
        data_size = 0
        for i, data in enumerate(data1):
            _, _, plot = plt.hist(data[:, 3], bins=100, weights=(np.ones(data[:, 3].shape[0]) / data[:, 3].shape[0]))
            data_size += data[:, 3].shape[0]
    else:
        _, _, plot = plt.hist(data1[:, 3], bins=100, weights=(np.ones(data1[:, 3].shape[0]) / data1[:, 3].shape[0]))

    # add Original MT score
    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    if science:
        t_coordinate = observation_summary_statistics_MT['Spike Count WT delta']
        plt.text(min_xlim * 1.001, max_ylim * 0.95, 'Mutant similarity to wild type: {:.2f}'.format(t_coordinate))
    else:
        t_coordinate = 0
        plt.text(min_xlim * 1.001, max_ylim * 0.95, 'Mutant similarity to wild type: {:.2f}%'.format(t_coordinate))
    plt.axvline(t_coordinate, color='r', linestyle='dashed', linewidth=1)

    if all:
        plt.title(f'Similarity Distribution of Spike Count\nTotal {data_size}')
    else:
        plt.title(f'Similarity Distribution of Spike Count\nTotal {data1[:, 3].shape[0]}')

                                                                                      
    if science:
        plt.xlabel("Spike Count similarity to wild-type")
    else:
        plt.xlabel("similarity to wild-type in %")
    plt.ylabel("% of models")

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(fname=path + '/' + f'{ind}- histogram - Similarity Distribution of {testType} - Test Spike Count', )
    plt.close()

    # Dynamic Time Warping WT
    if all:
        data_size = 0
        for i, data in enumerate(data2):
            _, _, plot = plt.hist(data[:, 3], bins=100, weights=(np.ones(data[:, 3].shape[0]) / data[:, 3].shape[0]))
            data_size += data[:, 3].shape[0]
    else:
        _, _, plot = plt.hist(data2[:, 3], bins=100, weights=(np.ones(data2[:, 3].shape[0]) / data2[:, 3].shape[0]))

    # add Original MT score
    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    if science:
        t_coordinate = observation_summary_statistics_MT['Dynamic Time Warping WT']
        plt.text(min_xlim * 1.001, max_ylim * 0.95, 'Mutant similarity to wild type: {:.2f}'.format(t_coordinate))
    else:
        t_coordinate = 0
        plt.text(min_xlim * 1.001, max_ylim * 0.95, 'Mutant similarity to wild type: {:.2f}%'.format(t_coordinate))
    plt.axvline(t_coordinate, color='r', linestyle='dashed', linewidth=1)

    if all:
        plt.title(f'Similarity Distribution of Dynamic Time Warping\nTotal {data_size}')
    else:
        plt.title(f'Similarity Distribution of Dynamic Time Warping\nTotal {data2[:, 3].shape[0]}')

    if science:
        plt.xlabel("Dynamic Time Warping similarity to wild-type")
    else:
        plt.xlabel("similarity to wild-type in %")
    plt.ylabel("% of models")

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(
        fname=path + '/' + f'{ind}- histogram - Similarity Distribution of {testType} - Test Dynamic Time Warping', )
    plt.close()



# -------
# Observed data
h.load_file("stdrun.hoc")  # for run control
h.load_file("neuron.hoc")  # run the model
h.initWT()
h.runWTsim()
h.initMT()
h.runMuTsim()
num_n = h.data_vecs_WT.__len__()
if num_n > 0:
    n_shape = [num_n, h.data_vecs_WT[0].__len__()]
    observation_summary_statistics_WT = {}
    for i in test_crateria_type:
        observation_summary_statistics_WT[i] = 0

    observation_summary_statistics_MT = observation_summary_statistics_WT.copy()

    WT_voltage = np.zeros(n_shape, dtype=np.float)
    MT_voltage = np.zeros(n_shape, dtype=np.float)
    for y in range(n_shape[0]):
        WT_voltage[y, :] = h.data_vecs_WT[y].as_numpy()
        MT_voltage[y, :] = h.data_vecs_Mut[y].as_numpy()

    observation_summary_statistics_WT = calculate_summary_statistics(WT_voltage)
    observation_summary_statistics_MT = calculate_summary_statistics(MT_voltage)
    # not redundant: the previous evaluation is needed to re-evaluate the score
    observation_summary_statistics_WT = calculate_summary_statistics(WT_voltage)
    observation_summary_statistics_MT = calculate_summary_statistics(MT_voltage)

Orginal_WT_param = [
    h.WTcell[0].isoma.gl_ichanWT2005,
    h.WTcell[0].isoma.gkfbar_ichanWT2005,
    h.WTcell[0].isoma.gnatbar_ichanWT2005
]
Orginal_MT_param = [
    h.MuTcell[0].isoma.gl_ichanR859C1,
    h.MuTcell[0].isoma.gkfbar_ichanR859C1,
    h.MuTcell[0].isoma.gnatbar_ichanR859C1
]

# ----------- Main --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--parts', nargs='+', type=int, default=[1, 2, 3, 4, 5])

args = parser.parse_args()

Prog_parts = args.parts

for Prog_part in Prog_parts:
    if Prog_part == 1:
        # ---------- parmeters ----------------
        # define Prior over model parameters
        # # ----------- HH ranges
        # #             gl,  gK, gNa
        # prior_min = [0.1, 26., 65.]  # fields[0], fields[1], ....
        # prior_max = [0.5, 49., 260.]  # fields[0], fields[1], ....
        # # ----------------

        # ---------Draw a sample from the posterior and convert to numpy for plotting.-----------
        #             gl,  gK, gNa
        # parm_min = [0.0001, 0.001, 0.01]  # fields[0], fields[1], ....
        # parm_max = [0.001, 0.01, 0.1]  # fields[0], fields[1], ....
        parm_min = [Orginal_MT_param[0] / 50,
                    Orginal_MT_param[1] / 50,
                    Orginal_MT_param[2] / 50]  # fields[0], fields[1], ....
        parm_max = [Orginal_MT_param[0] * 50,
                    Orginal_MT_param[1] * 50,
                    Orginal_MT_param[2] * 50]  # fields[0], fields[1], ....
        num_of_tests = 500000000
        path = '.'

        # creath path
        path = os.path.join(path, 'db') + '/'
        if os.path.exists(path) is not True:
            os.mkdir(path)

        # num_process = 3
        # P = list()
        # for p in range(num_process):
        #     P.append(multiprocessing.Process(target=test_parameters, args=(parm_min, parm_max, num_of_tests, crateria_cell_type,)))
        #     P[-1].start()
        # for p in range(num_process):
        #     P[p].join()

        test_parameters(parm_min, parm_max, num_of_tests, crateria_cell_type)

    if Prog_part == 2:
        # -------summing and saving data ---------------------
        import glob

        path = './db/'
        file = 'sumary_explor_result.100K.pbz2'
        files = glob.glob(path + "*_stats.pkl")
        if len(files) == 0:
            print(' No files found!!!')
            continue
        num_of_candidates = 100000

        result_Matrix = {}
        result_Matrix_Max = {}

        for f in tqdm(files):
            # monitoring
            tqdm.write(f'Filename: {f}')
            for k1 in result_Matrix.keys():
                for k2 in result_Matrix[k1].keys():
                    for k3 in result_Matrix[k1][k2].keys():
                        tqdm.write(
                            f'Size of type {k1} at test {k2} type {k3} is '
                            f'{len(result_Matrix[k1][k2][k3])}')
            sys.stdout.flush()

            rf = bz2.BZ2File(f, "rb")
            k2 = ''
            if f.__contains__('MT'):
                k2 = 'MT'
            elif f.__contains__('WT'):
                k2 = 'WT'

            while 1:
                try:
                    l_2 = pickle.load(rf)
                    for key in l_2.keys():
                        k3 = ''
                        if key.__contains__('MT'):
                            k3 = 'MT'
                        elif key.__contains__('WT'):
                            k3 = 'WT'

                        if k3 == '':
                            continue

                        k1 = key.split(' ' + k3)[0]

                        if len(result_Matrix.keys()) == 0 or result_Matrix.keys().__contains__(k1) is False:
                            result_Matrix[k1] = {}
                            result_Matrix_Max[k1] = {}
                        if len(result_Matrix[k1].keys()) == 0 or result_Matrix[k1].keys().__contains__(k2) is False:
                            result_Matrix[k1][k2] = {}
                            result_Matrix_Max[k1][k2] = {}
                        if len(result_Matrix[k1][k2].keys()) == 0 or result_Matrix[k1][k2].keys().__contains__(
                                k3) is False:
                            result_Matrix[k1][k2][k3] = list()
                            result_Matrix_Max[k1][k2][k3] = np.array([0, 0, 0, 0])
                            # adding original points to the dataset
                            t_k1 = k1 + ' ' + k3 + ' delta'
                            if k1.__contains__(test_crateria_type[3]):
                                t_k1 = test_crateria_type[3] + ' ' + k3
                            t_observe = observation_summary_statistics_WT
                            t_coordinate = Orginal_WT_param
                            if k2 == 'MT':
                                t_observe = observation_summary_statistics_MT
                                t_coordinate = Orginal_MT_param
                            result_Matrix[k1][k2][k3].append([
                                t_coordinate[0],
                                t_coordinate[1],
                                t_coordinate[2],
                                t_observe[t_k1],
                            ])

                        result_Matrix[k1][k2][k3].append([
                            l_2['gl,  gK, gNa'][0], l_2['gl,  gK, gNa'][1], l_2['gl,  gK, gNa'][2], l_2[key]
                        ])

                except EOFError:
                    break

            for i_k1, k1 in enumerate(result_Matrix.keys()):
                for i_k3, k3 in enumerate(result_Matrix[k1][k2].keys()):
                    result_Matrix[k1][k2][k3] = np.array(result_Matrix[k1][k2][k3], dtype=np.float)
                    # eliminate duplication
                    result_Matrix[k1][k2][k3] = np.unique(result_Matrix[k1][k2][k3], axis=0)
                    # sorting
                    result_Matrix[k1][k2][k3] = result_Matrix[k1][k2][k3][result_Matrix[k1][k2][k3][:, 3].argsort()]
                    if result_Matrix_Max[k1][k2][k3][3] < result_Matrix[k1][k2][k3][-1, 3]:
                        result_Matrix_Max[k1][k2][k3] = result_Matrix[k1][k2][k3][-1, :].copy()
                    result_Matrix[k1][k2][k3] = result_Matrix[k1][k2][k3][:num_of_candidates, :]
                    result_Matrix[k1][k2][k3] = list(result_Matrix[k1][k2][k3])

        # finish procedures and adding MAX value
        for k1 in result_Matrix.keys():
            for k2 in result_Matrix[k1].keys():
                for k3 in result_Matrix[k1][k2].keys():
                    # add the max value
                    result_Matrix[k1][k2][k3].append(result_Matrix_Max[k1][k2][k3])
                    result_Matrix[k1][k2][k3] = np.array(result_Matrix[k1][k2][k3], dtype=np.float)

        # check if there are previus files that can be added to the dataset
        db_files = glob.glob(path + '../' + "*.pbz2")
        for db_file in db_files:
            with bz2.BZ2File(db_file, 'r') as f:
                db = pickle.load(f)
            for k1 in db.keys():
                for k2 in db[k1].keys():
                    for k3 in db[k1][k2].keys():
                        if db[k1][k2][k3].shape[1] != 0:
                            result_Matrix[k1][k2][k3] = \
                                np.concatenate((result_Matrix[k1][k2][k3], db[k1][k2][k3]), axis=0)
                            # eliminate duplication
                            result_Matrix[k1][k2][k3] = np.unique(result_Matrix[k1][k2][k3], axis=0)
                            # sort final result
                            result_Matrix[k1][k2][k3] = \
                                result_Matrix[k1][k2][k3][result_Matrix[k1][k2][k3][:, 3].argsort()]
                            # finding MAX value
                            Max = result_Matrix[k1][k2][k3][-1, :].copy()
                            result_Matrix[k1][k2][k3] = result_Matrix[k1][k2][k3][:num_of_candidates + 1, :]
                            result_Matrix[k1][k2][k3][-1, :] = Max

        with bz2.BZ2File(path + '../' + file, 'w') as f:
            pickle.dump(result_Matrix, f)

    if Prog_part == 3:
        # -------plot scatter graph ---------------------
        path = './'
        f_num = '.100K'

        file = 'sumary_explor_result' + f_num + '.pbz2'
        with bz2.BZ2File(path + file, 'r') as f:
            db = pickle.load(f)

        path = os.path.join(path, '3D Scatter Plots') + '/'
        if os.path.exists(path) is not True:
            os.mkdir(path)

        slice = [100, 80, 60, 40, 30, 20, 10, 5, 2, 1, -1]
        for science in [True, False]:
            for s in tqdm(slice):
                new_path = os.path.join(path, str(s) if s > 0 else 'Min Only') + '/'
                if os.path.exists(new_path) is not True:
                    os.mkdir(new_path)
                if science:
                    new_path = os.path.join(new_path, 'Untouched') + '/'
                    if os.path.exists(new_path) is not True:
                        os.mkdir(new_path)

                f_num = '.' + str(s)
                s *= 1000

                for k1 in db.keys():
                    figs = list()
                    for k2 in db[k1].keys():
                        for k3 in db[k1][k2].keys():
                            if db[k1][k2][k3].shape[1] != 0:
                                if s > 0:
                                    figs.append([
                                        db[k1][k2][k3][:s - 1, 0],
                                        db[k1][k2][k3][:s - 1, 1],
                                        db[k1][k2][k3][:s - 1, 2],
                                        db[k1][k2][k3][:s - 1, 3] if science else
                                        (1 - db[k1][k2][k3][:s - 1, 3] / db[k1][k2][k3][-1, 3]) * 100,
                                        'jet',
                                        k1 + "\n" + k2 + " distance close to " + k3 + "\n" + str(s) + " Samples"
                                    ])
                                else:
                                    # plot the minimal score only!
                                    t_min_score = np.where(db[k1][k2][k3][:, 3] == min(db[k1][k2][k3][:, 3]))[0]
                                    if t_min_score.shape[0] == 0:
                                        continue
                                    figs.append([
                                        db[k1][k2][k3][t_min_score, 0],
                                        db[k1][k2][k3][t_min_score, 1],
                                        db[k1][k2][k3][t_min_score, 2],
                                        db[k1][k2][k3][t_min_score, 3] if science else
                                        (1 - db[k1][k2][k3][t_min_score, 3] / db[k1][k2][k3][-1, 3]) * 100,
                                        'jet',
                                        k1 + "\n" + k2 + " distance close to " + k3 + "\n" + str(s) + " Samples"
                                    ])
                            else:
                                figs.append([[0], [0], [0], [0], 'jet',
                                             k1 + " cell - " + k2 + " distance close to " + k3 + " - " + str(
                                                 s) + " Samples"
                                             ])
                    Four_scatter3d(figs, filename=k1, path=new_path, animation=True, science=science)
                    # Four_scatter3d(figs, filename=None)

    if Prog_part == 4:
        # -------plots Voltage Curve fot only the best ---------------------
        path = './'
        file = 'sumary_explor_result.100K.pbz2'
        voltPath = 'VoltPlots'
        with bz2.BZ2File(path + file, 'r') as f:
            db = pickle.load(f)

        voltPath = os.path.join(path, voltPath) + '/'
        if os.path.exists(voltPath) is not True:
            os.mkdir(voltPath)

        for i_k1, k1 in enumerate(db.keys()):
            voltPathII = os.path.join(voltPath, k1) + '/'
            if os.path.exists(voltPathII) is not True:
                os.mkdir(voltPathII)

            for k2 in db[k1].keys():
                voltPathIII = os.path.join(voltPathII, f'{k2}') + '/'
                if os.path.exists(voltPathIII) is not True:
                    os.mkdir(voltPathIII)

                for k3 in db[k1][k2].keys():
                    voltPath4 = os.path.join(voltPathIII, f'{k3}') + '/'
                    if os.path.exists(voltPath4) is not True:
                        os.mkdir(voltPath4)

                    # plot and save
                    n, _, plot = plt.hist(
                        # (1 - db[k1][k2][k3][:-1, 3] / db[k1][k2][k3][-1, 3]) * 100,
                        db[k1][k2][k3][:-1, 3],
                        bins=100, weights=(np.ones(db[k1][k2][k3][:-1, 3].shape[0]) / db[k1][k2][k3][:-1, 3].shape[0]))
                    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
                    plt.title(f'Search distribution\nTotal {db[k1][k2][k3][:-1, 3].shape[0]}')
                    if k3 == crateria_cell_type[0]:
                        plt.xlabel("similarity to wild-type")
                    elif k3 == crateria_cell_type[1]:
                        plt.xlabel("similarity to mutant-type")
                    plt.ylabel("% of models")
                    cmap = plt.get_cmap('jet')
                    # plt.yscale('log')
                    for i, p in enumerate(plot):
                        p.set_facecolor(cmap(i / len(plot)))

                    plt.savefig(fname=voltPathII + '/' + f'Histogram of all {k2}-{k3} ', )
                    plt.close()

                    # run histogram for processing precentile
                    n, b = np.histogram(db[k1][k2][k3][:-1, 3], bins=100)
                    for cont, bb in enumerate(b):
                        if cont >= b.shape[0] - 1:
                            continue
                        else:
                            l = np.where((db[k1][k2][k3][:, 3] >= b[cont]) *
                                         (db[k1][k2][k3][:, 3] <= b[cont + 1]))[0]

                        if l.shape[0] == 0:
                            continue
                        else:
                            l = l[0]

                        coordinates = [db[k1][k2][k3][l, 0], db[k1][k2][k3][l, 1], db[k1][k2][k3][l, 2]]
                        filename = f'{k2}-{k3} cell - {k1} distance #{cont} - ' \
                                   f'{coordinates[0]}, {coordinates[1]}, {coordinates[2]} - ' \
                                   f'Score = {db[k1][k2][k3][l, 3]}'
                        if k2 == 'MT':
                            if k3 == 'MT':
                                plots(MT_voltage, runMT(coordinates), filename=filename, path=voltPath4)
                            else:
                                plots(WT_voltage, runMT(coordinates), filename=filename, path=voltPath4)
                        else:
                            if k3 == 'MT':
                                plots(MT_voltage, runWT(coordinates), filename=filename, path=voltPath4)
                            else:
                                plots(WT_voltage, runWT(coordinates), filename=filename, path=voltPath4)
                        print(filename)
                        plt.close()

    if Prog_part == 5:
        # -------check the ratio between MT-MT to MT-WT -> only the best ---------------------
        path_org = './'
        file = 'sumary_explor_result.100K.pbz2'
        num_of_candidates = 500
        num_of_processes = 140
        with bz2.BZ2File(path_org + file, 'r') as f:
            db = pickle.load(f)

        path = os.path.join(path_org, 'Mutant Fix') + '/'
        if os.path.exists(path) is not True:
            os.mkdir(path)

        '''
        calculate delta between all WT that behave like WT to a MT that behave like WT
        Assuming that the DB is ordered from minimum on top and max on bottom
        '''
        score_result = {}
        for k1 in db.keys():  # messurment category
            if k1.__contains__('std') or k1.__contains__('avg'):
                continue
            if not (k1 == test_crateria_type[0] or k1 == test_crateria_type[3]):
                tqdm.write("Error: Crateria not found!!!")
                break
            tqdm.write(f'Working on {k1}')
            score_result[k1] = {}
            
            
            # assumes the list is sorted
            MT_WT_candidates = np.where(db[k1]['MT']['WT'][:, 3] == min(db[k1]['MT']['WT'][:, 3]))[0]
            if k1 == test_crateria_type[0]:
                if num_of_candidates > MT_WT_candidates.shape[0]:
                    MT_WT_candidates = np.array(range(num_of_candidates))
                elif num_of_candidates < MT_WT_candidates.shape[0]:
                    MT_WT_candidates = np.array(range(num_of_candidates))
            elif k1 == test_crateria_type[3]:
                if num_of_candidates > MT_WT_candidates.shape[0]:
                    MT_WT_candidates = np.array(range(num_of_candidates))
                # elif num_of_candidates < MT_WT_candidates.shape[0]:
                #     MT_WT_candidates = np.array(range(num_of_candidates))

            WT_WT_candidates = np.where(db[k1]['WT']['WT'][:, 3] == min(db[k1]['WT']['WT'][:, 3]))[0]
            # if k1 == test_crateria_type[0]:
                # if num_of_candidates > WT_WT_candidates.shape[0]:
                    # WT_WT_candidates = np.array(range(num_of_candidates))
                # elif num_of_candidates < WT_WT_candidates.shape[0]:
                    # WT_WT_candidates = np.array(range(num_of_candidates))
            # el
            if k1 == test_crateria_type[3]:
                if num_of_candidates > WT_WT_candidates.shape[0]:
                    WT_WT_candidates = np.array(range(num_of_candidates))
                # elif num_of_candidates < WT_WT_candidates.shape[0]:
                #     WT_WT_candidates = np.array(range(num_of_candidates))
    
            MT_MT_candidates = np.where(db[k1]['MT']['MT'][:, 3] == min(db[k1]['MT']['MT'][:, 3]))[0]
            # if k1 == test_crateria_type[0]:
                # if num_of_candidates > MT_MT_candidates.shape[0]:
                    # MT_MT_candidates = np.array(range(num_of_candidates))
                # elif num_of_candidates < MT_MT_candidates.shape[0]:
                    # MT_MT_candidates = np.array(range(num_of_candidates))
            # el
            if k1 == test_crateria_type[3]:
                if num_of_candidates > MT_MT_candidates.shape[0]:
                    MT_MT_candidates = np.array(range(num_of_candidates))
                # elif num_of_candidates < MT_MT_candidates.shape[0]:
                #     MT_MT_candidates = np.array(range(num_of_candidates))

            if MT_MT_candidates.shape[0] == 0:
                tqdm.write(f'Category {k1} is empty')
                continue

            tqdm.write(f'Number of WT - WT candidates: {WT_WT_candidates.shape[0]} '
                       f"with value {list(np.unique(db[k1]['WT']['WT'][WT_WT_candidates, 3]))}")
            tqdm.write(f'Number of MT - WT candidates: {MT_WT_candidates.shape[0]} '
                       f"with value {list(np.unique(db[k1]['MT']['WT'][MT_WT_candidates, 3]))}")
            tqdm.write(f'Number of MT - MT candidates: {MT_MT_candidates.shape[0]} '
                       f"with value {list(np.unique(db[k1]['MT']['MT'][MT_MT_candidates, 3]))}")
            score_result[k1]['Number of WT-WT candidates'] = db[k1]['WT']['WT'][WT_WT_candidates, 3]
            score_result[k1]['Number of MT-WT candidates'] = db[k1]['MT']['WT'][MT_WT_candidates, 3]
            score_result[k1]['Number of MT-MT candidates'] = db[k1]['MT']['MT'][MT_MT_candidates, 3]

            delta = list()
            t_delta_centeroids = np.mean(db[k1]['MT']['WT'][MT_WT_candidates, 0:3], axis=0) - \
                                 np.mean(db[k1]['MT']['WT'][WT_WT_candidates, 0:3], axis=0)
            for i, d_point_from in enumerate(MT_WT_candidates):  # data point
                t_delta = np.zeros((WT_WT_candidates.shape[0], 4))
                t_delta[:, :3] = db[k1]['MT']['WT'][d_point_from, 0:3] - db[k1]['WT']['WT'][WT_WT_candidates, 0:3]
                # calculate Euclidean distance
                t_delta[:, 3] = np.linalg.norm(
                    (db[k1]['MT']['WT'][d_point_from, 0:3]) -
                    db[k1]['WT']['WT'][WT_WT_candidates, 0:3],
                    axis=1,
                )
                # sort final result
                t_delta = t_delta[t_delta[:, 3].argsort()]
                # take all 
                delta.extend(list(t_delta[:, :3]))
                # take only the minimum euqlidian points
                # t_unique = np.unique(t_delta[:, 3])[:num_of_candidates]
                # for t_u in t_unique:
                    # delta.extend(list(np.reshape(t_delta[np.where(t_delta[:, 3] == t_u), :3], (-1, 3))))

            delta = np.array(delta, dtype=np.float)

            '''
            check every minimal MT neuron that have parameter that is close to the performance of original MT behavioral
            if adding or removing the delta found above improve it score
            '''
            t_param_list = list()
            for d_point_from in range(MT_MT_candidates.shape[0]):  # data point
                t_param = db[k1]['MT']['MT'][MT_MT_candidates[d_point_from], 0:3] + delta
                t_p = np.where(np.sum(t_param < 0, axis=1) == 0)
                t_param_list.extend(t_param[t_p])

                t_param = db[k1]['MT']['MT'][MT_MT_candidates[d_point_from], 0:3] - delta
                t_p = np.where(np.sum(t_param < 0, axis=1) == 0)
                t_param_list.extend(t_param[t_p])

            pbar = tqdm()
            pbar.reset(total=len(t_param_list))
            pbar.refresh()


            def check_parm(param):
                score = calculate_summary_statistics(runMT(param))
                # coordinates
                readline = [param[0], param[1], param[2], ]
                # retreave comperession with WT
                readline.append(score[test_crateria_type[0] + ' WT delta'])
                readline.append(score['Dynamic Time Warping WT'])
                pbar.update()
                return readline


            pool = multiprocessing.Pool(processes=num_of_processes)
            score_result[k1]['score'] = pool.map(check_parm, t_param_list)

            score_result[k1]['score'] = np.array(score_result[k1]['score'], dtype=np.float)

        filetemp = file.split('.')[:-1]
        file = ''
        for ft in filetemp:
            file += ft
        file += '.mutationFix.pbz2'

        with bz2.BZ2File(path + file, 'w') as f:
            pickle.dump(score_result, f)

        tqdm.write('Saving graphs....')

        # # load saved mutation results
        # filetemp = file.split('.')[:-1]
        # file = ''
        # for ft in filetemp:
        #     file += ft
        # file += '.mutationFix.pbz2'
        # with bz2.BZ2File(path + file, 'r') as f:
        #     score_result = pickle.load(f)

        for k1 in score_result.keys():  # messurment category
            for science in [True, False]:
                t_path = os.path.join(path, k1) + '/'
                if os.path.exists(t_path) is not True:
                    os.mkdir(t_path)

                if science:
                    t_path = os.path.join(t_path, 'Untouched') + '/'
                    if os.path.exists(t_path) is not True:
                        os.mkdir(t_path)

                t_points_spike_count = score_result[k1]['score'][:, :4].copy()
                t_points_DTW = np.delete(score_result[k1]['score'].copy(), 3, axis=1)
                if not science:
                    # change the scale to percents
                    t_points_spike_count[:, 3] = (1 - t_points_spike_count[:, 3] /
                                                  observation_summary_statistics_MT['Spike Count WT delta'].sum()) * 100
                    t_points_DTW[:, 3] = (1 - t_points_DTW[:, 3] /
                                          observation_summary_statistics_MT['Dynamic Time Warping WT']) * 100
                    # remove smaller then -100%
                    t_points_spike_count = t_points_spike_count[np.where(t_points_spike_count[:, 3] >= -100)]
                    t_points_DTW = t_points_DTW[np.where(t_points_DTW[:, 3] >= -100)]

                scatter3d(
                    t_points_spike_count[:, 0], t_points_spike_count[:, 1], t_points_spike_count[:, 2],
                    t_points_spike_count[:, 3],
                    title=f'Fixed Mutation - source {k1}\ntest Spike Count',
                    filename=f'Fixed Mutation - source {k1} - test Spike Count',
                    path=t_path,
                    science=science
                )
                plt.close()
                scatter3d(
                    t_points_DTW[:, 0], t_points_DTW[:, 1], t_points_DTW[:, 2], t_points_DTW[:, 3],
                    title=f'Fixed Mutation - source {k1}\ntest Dynamic Time Warping',
                    filename=f'Fixed Mutation - source {k1} - test Dynamic Time Warping',
                    path=t_path,
                    science=science
                )
                plt.close()

                # plot histogram
                Histogram(t_points_spike_count, t_points_DTW, t_path, k1, science=science)

            # plot voltage curve 2 examples from every bin
            t_path2 = os.path.join(t_path, 'APs') + '/'
            if os.path.exists(t_path2) is not True:
                os.mkdir(t_path2)
            n, b = np.histogram(score_result[k1]['score'][:, 3], bins=100)
            for ex_bi, ex_b in enumerate(b):
                if ex_bi < n.shape[0] and n[ex_bi] > 0:
                    if ex_bi < b.shape[0] - 1:
                        clist = np.where(
                            (score_result[k1]['score'][:, 3] >= b[ex_bi]) *
                            (score_result[k1]['score'][:, 3] <= b[ex_bi + 1]))
                    else:
                        clist = np.where(score_result[k1]['score'][:, 3] <= ex_b)

                    for i, l in enumerate(clist[0]):
                        if i > 1:
                            break
                        filename = f'{k1} group, b#{ex_bi} ' \
                                   f"#{i} score={score_result[k1]['score'][l, 3]}"
                        plots(WT_voltage, runMT(score_result[k1]['score'][l, 0:3]), filename=filename, path=t_path2)

    if Prog_part == 6:
        # -------check the ratio between MT-MT to MT-WT -> only the best ---------------------
        path_org = './'
        file = 'sumary_explor_result.100K.pbz2'
        num_of_candidates = 70
        num_of_processes = 90
        with bz2.BZ2File(path_org + file, 'r') as f:
            db = pickle.load(f)

        path = os.path.join(path_org, 'Mutant Fix - 6') + '/'
        if os.path.exists(path) is not True:
            os.mkdir(path)

        '''
        calculate delta between all WT that behave like WT to a MT that behave like WT
        Assuming that the DB is ordered from minimum on top and max on bottom
        '''
        score_result = {}
        for k1 in db.keys():  # messurment category
            if k1.__contains__('std') or k1.__contains__('avg'):
                continue
            if not (k1 == test_crateria_type[0] or k1 == test_crateria_type[3]):
                tqdm.write("Error: Crateria not found!!!")
                break
            tqdm.write(f'Working on {k1}')
            score_result[k1] = {}

            # assumes the list is sorted
            MT_WT_candidates = np.where(db[k1]['MT']['WT'][:, 3] == min(db[k1]['MT']['WT'][:, 3]))[0]
            if k1 == test_crateria_type[0]:
                if num_of_candidates > MT_WT_candidates.shape[0]:
                    MT_WT_candidates = np.array(range(num_of_candidates))
                elif num_of_candidates < MT_WT_candidates.shape[0]:
                    MT_WT_candidates = np.array(range(num_of_candidates * 15))
            elif k1 == test_crateria_type[3]:
                if num_of_candidates > MT_WT_candidates.shape[0]:
                    MT_WT_candidates = np.array(range(num_of_candidates))
                elif num_of_candidates < MT_WT_candidates.shape[0]:
                    MT_WT_candidates = np.array(range(num_of_candidates))

            MT_MT_candidates = np.where(db[k1]['MT']['MT'][:, 3] == min(db[k1]['MT']['MT'][:, 3]))[0]
            # if k1 == test_crateria_type[0]:
                # if num_of_candidates > MT_MT_candidates.shape[0]:
                    # MT_MT_candidates = np.array(range(num_of_candidates))
                # elif num_of_candidates < MT_MT_candidates.shape[0]:
                    # MT_MT_candidates = np.array(range(num_of_candidates))
            # el
            if k1 == test_crateria_type[3]:
                if num_of_candidates > MT_MT_candidates.shape[0]:
                    MT_MT_candidates = np.array(range(num_of_candidates))
                elif num_of_candidates < MT_MT_candidates.shape[0]:
                    MT_MT_candidates = np.array(range(num_of_candidates))

            if MT_MT_candidates.shape[0] == 0:
                tqdm.write(f'Category {k1} is empty')
                continue

            tqdm.write(f'Number of MT - WT candidates: {MT_WT_candidates.shape[0]} '
                       f"with value {list(np.unique(db[k1]['MT']['WT'][MT_WT_candidates, 3]))}")
            tqdm.write(f'Number of MT - MT candidates: {MT_MT_candidates.shape[0]} '
                       f"with value {list(np.unique(db[k1]['MT']['MT'][MT_MT_candidates, 3]))}")
            score_result[k1]['Number of MT-WT candidates'] = db[k1]['MT']['WT'][MT_WT_candidates, 3]
            score_result[k1]['Number of MT-MT candidates'] = db[k1]['MT']['MT'][MT_MT_candidates, 3]

            delta = list()
            t_delta_centeroids = np.mean(db[k1]['MT']['WT'][MT_WT_candidates, 0:3], axis=0) - \
                                 np.mean(db[k1]['MT']['WT'][MT_MT_candidates, 0:3], axis=0)
            for i, d_point_from in enumerate(MT_WT_candidates):  # data point
                t_delta = np.zeros((MT_MT_candidates.shape[0], 4))
                t_delta[:, :3] = db[k1]['MT']['WT'][d_point_from, 0:3] - db[k1]['WT']['WT'][MT_MT_candidates, 0:3]
                # # calculate Euclidean distance
                # t_delta[:, 3] = np.linalg.norm(
                #     (db[k1]['MT']['WT'][d_point_from, 0:3]) -
                #     db[k1]['WT']['WT'][MT_MT_candidates, 0:3],
                #     axis=1,
                # )
                # # sort final result
                # t_delta = t_delta[t_delta[:, 3].argsort()]
                # t_unique = np.unique(t_delta[:, 3])[:num_of_candidates]
                # for t_u in t_unique:
                #     delta.extend(list(np.reshape(t_delta[np.where(t_delta[:, 3] == t_u), :3], (-1, 3))))

                # dont use Euclidean distance
                delta.extend(list(t_delta[:, :3]))

            delta = np.array(delta, dtype=np.float)

            '''
            check every minimal MT neuron that have parameter that is close to the performance of original MT behavioral
            if adding or removing the delta found above improve it score
            '''
            t_param_list = list()
            for d_point_from in range(MT_MT_candidates.shape[0]):  # data point
                t_param = np.zeros((delta.shape[0], 4))
                t_param[:, 3] = d_point_from
                t_param[:, :3] = db[k1]['MT']['MT'][MT_MT_candidates[d_point_from], 0:3] + delta
                t_p = np.where(np.sum(t_param < 0, axis=1) == 0)
                t_param_list.extend(t_param[t_p])

                t_param = np.zeros((delta.shape[0], 4))
                t_param[:, 3] = d_point_from
                t_param[:, :3] = db[k1]['MT']['MT'][MT_MT_candidates[d_point_from], 0:3] - delta
                t_p = np.where(np.sum(t_param < 0, axis=1) == 0)
                t_param_list.extend(t_param[t_p])

            pbar = tqdm()
            pbar.reset(total=len(t_param_list))
            pbar.refresh()


            def check_parm(param):
                score = calculate_summary_statistics(runMT([param[0], param[1], param[2]]))
                # coordinates
                readline = [param[0], param[1], param[2], ]
                # retreave comperession with WT
                readline.append(param[3])
                readline.append(score[test_crateria_type[0] + ' WT delta'])
                readline.append(score['Dynamic Time Warping WT'])
                pbar.update()
                return readline


            pool = multiprocessing.Pool(processes=num_of_processes)
            score_result[k1]['score'] = pool.map(check_parm, t_param_list)

            score_result[k1]['score'] = np.array(score_result[k1]['score'], dtype=np.float)

            # rearange the
            t_list = {}
            t_avg = []
            t_std = []
            for i in range(1 + int(max(score_result[k1]['score'][:, 3]))):
                t_p = np.where(score_result[k1]['score'][:, 3] == i)
                t_list[i] = np.delete(score_result[k1]['score'][t_p], 3, 1)
                t_avg.append([np.mean(t_list[i][:, 3]), np.mean(t_list[i][:, 4])])
                t_std.append([np.std(t_list[i][:, 3]), np.std(t_list[i][:, 4])])
            score_result[k1]['score'] = {}
            score_result[k1]['score']['candidates'] = t_list
            score_result[k1]['score']['mean'] = np.array(t_avg)
            score_result[k1]['score']['std'] = np.array(t_std)

        filetemp = file.split('.')[:-1]
        file = ''
        for ft in filetemp:
            file += ft
        file += '.mutationFix.pbz2'

        with bz2.BZ2File(path + file, 'w') as f:
            pickle.dump(score_result, f)

        tqdm.write('Saving graphs....')

        # # load saved mutation results
        # filetemp = file.split('.')[:-1]
        # file = ''
        # for ft in filetemp:
        #     file += ft
        # file += '.mutationFix.pbz2'
        # with bz2.BZ2File(path + file, 'r') as f:
        #     score_result = pickle.load(f)

        for k1 in score_result.keys():  # messurment category
            t_path = os.path.join(path, k1) + '/'
            if os.path.exists(t_path) is not True:
                os.mkdir(t_path)
                
            with open(t_path + 'report.txt', 'wt') as f:
                for k1 in score_result.keys():
                    f.write(f'Number of MT - WT candidates: {MT_WT_candidates.shape[0]} '
                            f"with value {list(np.unique(db[k1]['MT']['WT'][MT_WT_candidates, 3]))} \n")
                    f.write(f'Number of MT - MT candidates: {MT_MT_candidates.shape[0]} '
                            f"with value {list(np.unique(db[k1]['MT']['MT'][MT_MT_candidates, 3]))} \n")
                    f.write(f'------- \n')

            for science in [False, True]:
                if science:
                    t_path = os.path.join(t_path, 'Untouched') + '/'
                    if os.path.exists(t_path) is not True:
                        os.mkdir(t_path)
                t_all_points_spike_count = list()
                t_all_points_DTW = list()
                for ind in range(score_result[k1]['score']['mean'].shape[0]):
                    t_points_spike_count = score_result[k1]['score']['candidates'][ind][:, :4].copy()
                    t_points_DTW = np.delete(score_result[k1]['score']['candidates'][ind].copy(), 3, axis=1)
                    if not science:
                        # change the scale to percents
                        t_points_spike_count[:, 3] = (1 - t_points_spike_count[:, 3] /
                                                      observation_summary_statistics_MT[
                                                          'Spike Count WT delta'].sum()) * 100
                        t_points_DTW[:, 3] = (1 - t_points_DTW[:, 3] /
                                              observation_summary_statistics_MT['Dynamic Time Warping WT']) * 100
                        # remove smaller then -100%
                        t_points_spike_count = t_points_spike_count[np.where(t_points_spike_count[:, 3] >= -100)]
                        t_points_DTW = t_points_DTW[np.where(t_points_DTW[:, 3] >= -100)]

                    t_all_points_DTW.append(t_points_DTW)
                    t_all_points_spike_count.append(t_points_spike_count)

                    scatter3d(
                        t_points_spike_count[:, 0], t_points_spike_count[:, 1], t_points_spike_count[:, 2],
                        t_points_spike_count[:, 3],
                        title=f'{ind}. Fixed Mutation - source {k1}\ntest Spike Count',
                        filename=f'{ind}. Fixed Mutation - source {k1} - test Spike Count',
                        path=t_path,
                        science=science
                    )
                    plt.close()
                    scatter3d(
                        t_points_DTW[:, 0], t_points_DTW[:, 1], t_points_DTW[:, 2], t_points_DTW[:, 3],
                        title=f'{ind}. Fixed Mutation - source {k1}\ntest Dynamic Time Warping',
                        filename=f'{ind}. Fixed Mutation - source {k1} - test Dynamic Time Warping',
                        path=t_path,
                        science=science
                    )
                    plt.close()

                    # plot histogram
                    Histogram(t_points_spike_count, t_points_DTW, t_path, k1, ind, science=science)
                Histogram(t_all_points_spike_count, t_all_points_DTW, t_path, k1, 'all', science=science, all=True)

                # # plot voltage curve 2 examples from every bin
                # t_path2 = os.path.join(t_path, 'APs') + '/'
                # if os.path.exists(t_path2) is not True:
                #     os.mkdir(t_path2)
                # n, b = np.histogram(score_result[k1]['score'][:, 3], bins=100)
                # for ex_bi, ex_b in enumerate(b):
                #     if ex_bi < n.shape[0] and n[ex_bi] > 0:
                #         if ex_bi < b.shape[0] - 1:
                #             clist = np.where(
                #                 (score_result[k1]['score'][:, 3] >= b[ex_bi]) *
                #                 (score_result[k1]['score'][:, 3] <= b[ex_bi + 1]))
                #         else:
                #             clist = np.where(score_result[k1]['score'][:, 3] <= ex_b)
                #
                #         for i, l in enumerate(clist[0]):
                #             if i > 1:
                #                 break
                #             filename = f'{k1} group, b#{ex_bi} ' \
                #                        f"#{i} score={score_result[k1]['score'][l, 3]}"
                #             # plots(WT_voltage, runMT(score_result[k1]['score'][l, 0:3]), filename=filename, path=t_path2)

    print(f' Program part {Prog_part} - Done')

print('All Done.')
