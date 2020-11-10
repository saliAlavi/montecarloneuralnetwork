from numba.pycc import CC
from photon_circular import *
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



def fit_func(x, const, a, b, c, d):
    return np.power(x, 3) * d + np.power(x, 2) * c + np.power(x, 2) * b + x * a + const


gridDim = 1
blockDim = 1
dim = gridDim * blockDim
all_adj_dist = []
all_collected = []
all_amps = []
all_modes = []
mode = 1.0
iters = 3 * 16
iters = 512
NNxy = 50
NNr = int(NNxy / 2)
NNz = int(NNxy / 2)
degree_divs = 64
n_steps = 100

np.random.seed(1)
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
co_rz_all = np.zeros((dim, NNr, NNz), dtype=np.float32)
co_rz_trad_all = np.zeros((dim, NNr, NNz), dtype=np.float32)
incoh_cross_xy_all = np.zeros((dim, NNxy, NNxy), dtype=np.float32)
incoh_cross_rz_all = np.zeros((dim, NNr, NNz), dtype=np.float32)
cross_xy_all = np.zeros((dim, NNxy, NNxy), dtype=np.float32)
cross_rz_all = np.zeros((dim, NNr, NNz), dtype=np.float32)
i_stokes_rz_all = np.zeros((dim, NNr, NNz), dtype=np.float32)
q_stokes_rz_all = np.zeros((dim, NNr, NNz), dtype=np.float32)
u_stokes_rz_all = np.zeros((dim, NNr, NNz), dtype=np.float32)
v_stokes_rz_all = np.zeros((dim, NNr, NNz), dtype=np.float32)

ctr = 0
ctr_trad = 0
ctr_cross = 0
# os.environ['NUMBA_DEBUG']='1'

for i in trange(iters, desc='1st loop'):
    mode = i % 3
    mode = 3
    random_nums = np.random.rand(dim, n_steps * 10)

    d_modes = cuda.to_device(np.ones(dim, dtype=np.float32) * mode)

    d_jones = cuda.to_device(np.zeros((dim, 4), dtype=np.complex64))
    d_jones_partial = cuda.to_device(np.zeros((dim, 4), dtype=np.complex64))
    d_U = cuda.to_device(np.zeros((dim, 3), dtype=np.float32))
    d_W = cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_mu_as = cuda.to_device(np.ones(dim, dtype=np.float32) * 0)
    d_mu_ss = cuda.to_device(np.ones(dim, dtype=np.float32) * np.float(60))
    d_scat_events = cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_co_xy = cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_cross_xy = cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_incoh_cross_xy = cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_co_rz = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_cross_rz = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_incoh_cross_rz = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_co_xy_trad = cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_cross_xy_trad = cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_incoh_cross_xy_trad = cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_co_rz_trad = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_cross_rz_trad = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_incoh_cross_rz_trad = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_position = cuda.to_device(np.zeros((NNxy, 3), dtype=np.float32))
    d_i_stokes_rz = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_q_stokes_rz = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_u_stokes_rz = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_v_stokes_rz = cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_random_nums = cuda.to_device(random_nums)

    rng_states = 1
    seed = i

    process_steps_circ[gridDim, blockDim](seed, rng_states, d_modes, d_random_nums, d_U, d_W, d_jones, d_mu_as,
                                          d_mu_ss, d_scat_events, d_jones_partial, d_co_xy, d_cross_xy,
                                          d_incoh_cross_xy, d_co_rz, d_cross_rz,
                                          d_incoh_cross_rz, d_co_xy_trad, d_cross_xy_trad, d_incoh_cross_xy_trad,
                                          d_co_rz_trad, d_cross_rz_trad,
                                          d_incoh_cross_rz_trad, d_position, s1, s2, m11, m12, d_i_stokes_rz,
                                          d_q_stokes_rz, d_u_stokes_rz, d_v_stokes_rz)

    co_xy = d_co_xy.copy_to_host()
    cross_xy = d_cross_xy.copy_to_host()
    incoh_cross_xy = d_incoh_cross_xy.copy_to_host()
    co_rz = d_co_rz.copy_to_host()
    cross_rz = d_cross_rz.copy_to_host()
    incoh_cross_rz = d_incoh_cross_rz.copy_to_host()
    co_xy_trad = d_co_xy_trad.copy_to_host()
    cross_xy_trad = d_cross_xy_trad.copy_to_host()
    incoh_cross_xy_trad = d_incoh_cross_xy_trad.copy_to_host()
    co_rz_trad = d_co_rz_trad.copy_to_host()
    cross_rz_trad = d_cross_rz_trad.copy_to_host()
    incoh_cross_rz_trad = d_incoh_cross_rz_trad.copy_to_host()
    position = d_position.copy_to_host()
    i_stokes_rz = d_i_stokes_rz.copy_to_host()
    q_stokes_rz = d_q_stokes_rz.copy_to_host()
    u_stokes_rz = d_u_stokes_rz.copy_to_host()
    v_stokes_rz = d_v_stokes_rz.copy_to_host()

    i_stokes_rz_all += i_stokes_rz
    q_stokes_rz_all += q_stokes_rz
    u_stokes_rz_all += u_stokes_rz
    v_stokes_rz_all += v_stokes_rz

    if not (np.isnan(co_rz.max())):
        co_rz_all += co_rz
        co_xy_all += co_xy

        ctr += 1
        print(ctr, 'partial', co_rz.max())
    if not (np.isnan(co_rz_trad.max())):
        co_rz_trad_all += co_rz_trad
        ctr_trad += 1
        print(ctr_trad, 'trad', co_rz_trad.max())

    if not (np.isnan(incoh_cross_xy_all.max())):
        incoh_cross_xy_all += incoh_cross_xy

        incoh_cross_rz_all += incoh_cross_rz
        ctr_cross += 1
    if not (np.isnan(cross_xy_all.max())):
        cross_xy_all += cross_xy
        cross_rz_all += cross_rz
