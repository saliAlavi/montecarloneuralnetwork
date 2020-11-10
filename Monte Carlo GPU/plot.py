import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot(weights, values, ax, num=1000):
    # weights = np.array(np.random.rand(1000) * 1)
    # weights = np.array(np.random.rand(1000) * 20)
    indices = values.argsort()
    values = values[indices]
    weights = weights[indices]
    # print(values)
    bins = np.linspace(values.min(), values.max(), num)
    mid_bins = 0.5 * (bins[:-1] + bins[1:])
    batch_indx = np.asarray([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

    # batches = [values[(values >= lbound) & (values < hbound)] for lbound, hbound in batch_indx]
    weights = [weights[(values >= lbound) & (values < hbound)] for lbound, hbound in batch_indx]
    # s_weights = [np.sum(w_b) / len(w_b) if len(w_b)>0 else 0 for w_b in weights]
    s_weights = [np.sum(w_b) if np.sum(w_b) > 0 else 1e-16 for w_b in weights]
    s_weights = np.asarray(s_weights)
    # cs = np.cumsum(indices)

    # width = 1
    # clist = [(0, "red"), (s_weights.min(), "red"), (s_weights.max(), "blue"), (1, "blue")]
    # rvb = mcolors.LinearSegmentedColormap.from_list("", clist)

    # ax.bar(mid_bins, s_weights, width, color=rvb(s_weights))
    # print(mid_bins.shape, s_weights.shape)
    ax.plot(mid_bins, np.log10(s_weights))
    return ax


def plot_stokes(e_x_ampl, e_x_freq, e_y_ampl, e_y_freq, ax, num=1000, stokes='S_0'):
    indices = e_x_freq.argsort()
    values = e_x_freq[indices]
    weights = e_x_ampl[indices]
    bins = np.linspace(values.min(), values.max(), num)
    mid_bins = 0.5 * (bins[:-1] + bins[1:])
    batch_indx = np.asarray([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

    weights = [weights[(values >= lbound) & (values < hbound)] for lbound, hbound in batch_indx]
    s_weights = [np.sum(w_b) if np.sum(w_b) > 0 else 1e-16 for w_b in weights]
    e_x_weighted = np.asarray(s_weights)

    indices = e_y_freq.argsort()
    values = e_y_freq[indices]
    weights = e_y_ampl[indices]
    # bins = np.linspace(values.min(), values.max(), num)
    # mid_bins = 0.5 * (bins[:-1] + bins[1:])
    batch_indx = np.asarray([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

    weights = [weights[(values >= lbound) & (values < hbound)] for lbound, hbound in batch_indx]
    s_weights = [np.sum(w_b) if np.sum(w_b) > 0 else 1e-16 for w_b in weights]
    e_y_weighted = np.asarray(s_weights)

    if stokes == 'S_0':
        stoke_param = np.power(e_x_weighted, 2) + np.power(e_y_weighted, 2)
    elif stokes == 'S_1':
        stoke_param = np.power(e_x_weighted, 2) - np.power(e_y_weighted, 2)
    elif stokes=='S_2':
        stoke_param=2*np.real(e_x_weighted*np.conj(e_y_weighted))
    elif stokes=='S_3':
        stoke_param=-2*np.imag(e_x_weighted*np.conj(e_y_weighted))
    else:
        raise Exception(f'Incorrect stokes parameter \'{stokes}\'')

    stoke_param[stoke_param==0]=1e-32
    ax.plot(mid_bins, np.log10(stoke_param))
    return ax, stoke_param, mid_bins
