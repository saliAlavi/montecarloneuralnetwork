from numba.pycc import CC
from photon_xy import *
import numpy as np
from numba import typed, types
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plot import *
from numpy import genfromtxt
from numba.cuda.random import init_xoroshiro128p_states
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib import cm
from scipy import optimize
from tqdm.notebook import trange, tqdm
import time
import os



def fit_func(x, const, a, b, c, d):
    return np.power(x, 3) * d + np.power(x, 2) * c + np.power(x, 2) * b + x * a + const


def make_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


gridDim = 1
blockDim = 256
dim = gridDim * blockDim
all_adj_dist = []
all_collected = []
all_amps = []
all_modes = []
mode = 1.0
iters = 3 * 16
iters = 20
NNxy = 500
NNr = int(NNxy / 2)
NNz = int(NNxy / 2)
# degree_divs = 64
degree_divs = 64
n_steps = 20
n_sims = 1
timers = []
dataset_path = 'data/Linear 7'

np.random.seed(4)
s1 = genfromtxt(f's1_pm_{degree_divs}.txt', delimiter=',')
s2 = genfromtxt(f's2_pm_{degree_divs}.txt', delimiter=',')
m11 = genfromtxt(f'm11_pm_{degree_divs}.txt', delimiter=',')
m12 = genfromtxt(f'm12_pm_{degree_divs}.txt', delimiter=',')
temp = 1j * s1[:, 2]
temp += s1[:, 1]
s1 = temp
temp = 1j * s2[:, 2]
temp += s2[:, 1]
s2 = temp
m11 = m11[:, 1]
m12 = m12[:, 1]

s1 = np.ascontiguousarray(s1)
s2 = np.ascontiguousarray(s2)
m11 = np.ascontiguousarray(m11)
m12 = np.ascontiguousarray(m12)
cuda.pinned(s1)
cuda.pinned(s2)
cuda.pinned(m11)
cuda.pinned(m12)

co_xy_all = np.zeros((dim, NNxy, NNxy), dtype=np.float32)
reflection_all = np.zeros((dim), dtype=np.float32)
zstokes_all = np.zeros((dim, NNz, 3), dtype=np.float32)

ctr = 0
ctr_trad = 0
ctr_cross = 0

make_dir(dataset_path)

make_dir(os.path.join(dataset_path, 'reflection'))
make_dir(os.path.join(dataset_path, 'zstokes'))

incident_degrees = [10, 50, 80, 85, 89]
for i in range(n_sims):
    n = 1.33
    print(f'####SIMULATION {i + 1}######')
    # incident_degree=incident_degrees[i]
    incident_degree = 0
    for j in trange(iters, desc='1st loop'):
        time1 = time.time()
        mode = i % 3
        mode = 3
        random_nums = np.random.rand(dim, n_steps * 20)
        d_jones = cuda.to_device(np.zeros((dim, 4), dtype=np.complex64))
        d_jones_partial = cuda.to_device(np.zeros((dim, 4), dtype=np.complex64))
        d_U = cuda.to_device(np.zeros((dim, 3), dtype=np.float32))
        d_W = cuda.to_device(np.zeros(dim, dtype=np.float32))
        d_mu_as = cuda.to_device(np.ones(dim, dtype=np.float32) * 0)
        d_mu_ss = cuda.to_device(np.ones(dim, dtype=np.float32) * np.float(100))
        d_scat_events = cuda.to_device(np.zeros(dim, dtype=np.float32))
        # d_position = cuda.to_device(np.zeros((NNxy, 3), dtype=np.float32))
        d_random_nums = cuda.to_device(random_nums)
        d_reflection = cuda.to_device(np.zeros((dim, 2), dtype=np.float32))
        d_zstokes = cuda.to_device(np.zeros((dim, NNz, 3), dtype=np.float32))  # co\incoh cross\cross
        d_co_xy = cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
        seed = j

        process_steps_linear[gridDim, blockDim](seed, incident_degree, n, d_reflection, d_zstokes, d_random_nums, d_U,
                                                d_W, d_jones, d_mu_as, d_mu_ss, d_scat_events, d_jones_partial, s1, s2,
                                                m11, m12, d_co_xy)

        # position = d_position.copy_to_host()

        reflections = d_reflection.copy_to_host()
        zstokes = d_zstokes.copy_to_host()
        co_xy = d_co_xy.copy_to_host()

        co_xy_all += co_xy
        reflections[reflections[:, 0] == 0] = 1
        reflection_all += reflections[:, 1] / reflections[:, 0]
        zstokes_all[:, :, 0] += zstokes[:, :, 0]  # / reflections[:, 0, np.newaxis]
        zstokes_all[:, :, 1] += zstokes[:, :, 1]  # / reflections[:, 0, np.newaxis]
        zstokes_all[:, :, 2] += zstokes[:, :, 2]  # / reflections[:, 0, np.newaxis]

        time2 = time.time()
        timers.append(time2 - time1)