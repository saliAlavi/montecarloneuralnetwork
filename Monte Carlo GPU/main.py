from numba.pycc import CC
from photon import *
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
iters =64
NNxy = 200
NNr = 100
NNz = 100
degree_divs = 64
n_steps=100

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
ctr = 0
ctr_trad = 0
# os.environ['NUMBA_DEBUG']='1'
# print(cuda.gpus)
# polarization mode 0):nonpolarized 1:p-polarized 2:s-polarized
for i in range(iters):
    mode = i % 3
    mode =3
    random_nums=np.random.rand(dim, n_steps * 10)

    d_amplitudes =        cuda.to_device(np.ones(dim, dtype=np.float32))
    d_steps =             cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_lengths =           cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_direction_theta =   cuda.to_device(np.ones(dim, dtype=np.float32) * (np.pi / 2))
    d_maxZs =             cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_position_x =        cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_position_y =        cuda.to_device(np.ones(dim, dtype=np.float32) * (1e-8))
    d_polarization =      cuda.to_device(np.ones(dim, dtype=np.float32))
    d_adjusted_dist =     cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_collected =         cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_modes =             cuda.to_device(np.ones(dim, dtype=np.float32) * mode)

    d_jones =             cuda.to_device(np.zeros((dim, 4), dtype=np.complex64))
    d_jones_partial =     cuda.to_device(np.zeros((dim, 4), dtype=np.complex64))
    d_U =                 cuda.to_device(np.zeros((dim, 3), dtype=np.float32))
    d_W =                 cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_mu_as =             cuda.to_device(np.ones(dim, dtype=np.float32) * 0)
    d_mu_ss =             cuda.to_device(np.ones(dim, dtype=np.float32) * np.float(1000))
    d_scat_events =       cuda.to_device(np.zeros(dim, dtype=np.float32))
    d_co_xy =             cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_cross_xy =          cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_incoh_cross_xy =    cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_co_rz =             cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_cross_rz =          cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_incoh_cross_rz =    cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_co_xy_trad =        cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_cross_xy_trad =     cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_incoh_cross_xy_trad=cuda.to_device(np.zeros((dim, NNxy, NNxy), dtype=np.float32))
    d_co_rz_trad =        cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_cross_rz_trad =     cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_incoh_cross_rz_trad=cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_position =          cuda.to_device(np.zeros((NNxy, 3), dtype=np.float32))
    d_i_stokes_rz=        cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_q_stokes_rz =       cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_u_stokes_rz =       cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_v_stokes_rz =       cuda.to_device(np.zeros((dim, NNr, NNz), dtype=np.float32))
    d_random_nums =       cuda.to_device(random_nums)
    # rng_states = create_xoroshiro128p_states(dim, seed=1)
    rng_states=1
    seed = i
    # print(rng_states)
    # rng_states = init_xoroshiro128p_states(gridDim * blockDim + i + int(mode), seed=int(mode + 1 + i))

    process_steps[gridDim, blockDim](seed, d_amplitudes, d_direction_theta, d_position_x, d_position_y, d_polarization, d_steps,
                                     d_lengths, d_maxZs, rng_states, d_adjusted_dist, d_collected, d_modes,d_random_nums, d_U, d_W, d_jones, d_mu_as,
                                     d_mu_ss,
                                     d_scat_events, d_jones_partial, d_co_xy, d_cross_xy, d_incoh_cross_xy, d_co_rz, d_cross_rz,
                                     d_incoh_cross_rz,
                                     d_co_xy_trad, d_cross_xy_trad, d_incoh_cross_xy_trad, d_co_rz_trad, d_cross_rz_trad,
                                     d_incoh_cross_rz_trad, d_position, s1, s2, m11, m12,d_i_stokes_rz,d_q_stokes_rz,d_u_stokes_rz,d_v_stokes_rz)

        # print(random_nums)
    # nb.cuda.profile_stop()

    #amplitudes=d_amplitudes.copy_to_host()
    # d_steps.copy_to_host()
    # d_lengths.copy_to_host()
    # d_direction_theta.copy_to_host()
    # d_maxZs.copy_to_host()
    # d_position_x.copy_to_host()
    # d_position_y.copy_to_host()
    # d_polarization.copy_to_host()
    # d_adjusted_dist.copy_to_host()
    # d_collected.copy_to_host()
    # d_modes.copy_to_host()
    # d_jones.copy_to_host()
    # d_jones_partial.copy_to_host()
    # d_U.copy_to_host()
    # d_W.copy_to_host()
    # d_mu_as.copy_to_host()
    # d_mu_ss.copy_to_host()
    # d_scat_events.copy_to_host()

    # co_xy=d_co_xy.copy_to_host()
    # cross_xy=d_cross_xy.copy_to_host()
    # incoh_cross_xy=d_incoh_cross_xy.copy_to_host()
    # co_rz=d_co_rz.copy_to_host()
    # cross_rz=d_cross_rz.copy_to_host()
    # incoh_cross_rz=d_incoh_cross_rz.copy_to_host()
    # co_xy_trad=d_co_xy_trad.copy_to_host()
    # cross_xy_trad=d_cross_xy_trad.copy_to_host()
    # incoh_cross_xy_trad=d_incoh_cross_xy_trad.copy_to_host()
    # co_rz_trad=d_co_rz_trad.copy_to_host()
    # cross_rz_trad=d_cross_rz_trad.copy_to_host()
    # incoh_cross_rz_trad=d_incoh_cross_rz_trad.copy_to_host()
    # position=d_position.copy_to_host()





    print(i)
    # if not (np.isnan(co_rz.max())):
    #     co_rz_all += co_rz
    #     co_xy_all += co_xy
    #
    #     ctr += 1
    #     print(ctr, 'partial', co_rz.max())
    # if not (np.isnan(co_rz_trad.max())):
    #     co_rz_trad_all += co_rz_trad
    #     ctr_trad += 1
    #     print(ctr_trad, 'trad', co_rz_trad.max())

# grid_size = NNr
# x = np.sum(co_xy_all, axis=0)
#
# #x = x.squeeze()
# # x[0,0]=1
# # x=x*255
# mean = np.mean(x)
# std = np.std(x)
# x = (x - mean) / std
# print(x.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X, Y = np.meshgrid(range(grid_size), range(grid_size))  # `plot_surface` expects `x` and `y` data to be 2D
# ax.plot_surface(X, Y, x)
# ax.set_zlim(0, 1)
#
# plt.show()
# x_clipped = np.clip(x, 0, 1)
# plt.imshow(x_clipped)
# plt.show()
#
# pr = np.sum(x, axis=1)
# plt.plot(pr)
# plt.show()
#
# pz=np.sum(x,axis=0)
# #pz = x
# x_data = np.array(list(range(len(pz[:-8]))))
# params, params_covariance = optimize.curve_fit(fit_func, x_data, pz[:-3], p0=[2, 2, 2, 2, 2])
# plt.plot(x_data, fit_func(x_data, params[0], params[1], params[2], params[3], params[4]), label='Fitted function')
# plt.plot(pz)
# plt.show()


# print(jones[0])
# plt.show()
# all_adj_dist = np.asarray(all_adj_dist)
# all_collected = np.asarray(all_collected)
# all_amps = np.asarray(all_amps)
# all_adj_dist = all_adj_dist.reshape([-1])
# all_collected = all_collected.reshape([-1])
# all_amps = all_amps.reshape([-1])
# all_modes = np.asarray(all_modes)
# all_modes = all_modes.reshape([-1])
#
# position_y_cs = np.cumsum(position_y)
# weighted = position_y * amplitudes
# weighted_cs = np.cumsum(weighted)
#
# # print(all_collected.shape)
# fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# fig.tight_layout()
# photon_type = ['Non Polarized', 'P-Polarized', 'S-Polarized']
# # print(all_modes ==0)
# # print(len(all_adj_dist[(all_collected > 0) & (all_modes==0)] ))
# # print(len(all_adj_dist[(all_collected > 0) & (all_modes==1)] ))
# # print(len(all_adj_dist[(all_collected > 0) & (all_modes==2)] ))
# print(len(all_adj_dist[(all_collected > 0) & (all_modes == 2)]) / len(all_adj_dist[all_modes == 2]))
# # print(len(axes[1]))
# for i, ax in enumerate(axes[0]):
#     if i == 3:
#         break
#     # n, bins, patches=ax.hist(all_adj_dist[(all_collected > 0) & (all_modes==i)], bins=500, density=True, log=True, weights=all_amps[(all_collected > 0) & (all_modes==i)] / (iters * dim / 3))
#     ax = plot(all_amps[(all_collected > 0) & (all_modes == i)], all_adj_dist[(all_collected > 0) & (all_modes == i)],
#               ax)
#     ax.set_title(photon_type[i])
#     # print(len(n))
#     ax.set_xlim([0, 1.2e-3])
#     # bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     # scale values to interval [0,1]
#     # col = bin_centers - min(bin_centers)
#     # col /= max(col)
#     # cm = plt.cm.get_cmap('RdYlBu_r')
#     # for c, p in zip(col, patches):
#     #     plt.setp(p, 'facecolor', cm(c))
#
# stoke_vectors = ['S_0', 'S_1', 'S_2', 'S_3']
# for i, stoke_vector in enumerate(stoke_vectors):
#     axes[1][i], s_0, frequency_bins = plot_stokes(all_amps[(all_collected > 0) & (all_modes == 1)],
#                                                   all_adj_dist[(all_collected > 0) & (all_modes == 1)],
#                                                   all_amps[(all_collected > 0) & (all_modes == 2)],
#                                                   all_adj_dist[(all_collected > 0) & (all_modes == 2)], axes[1][i],
#                                                   stokes=stoke_vector)
#     axes[1][i].set_xlim([0, 1.2e-3])
#     axes[1][i].set_title(stoke_vectors[i])
# # n, bins, patches=ax.hist(all_amps[all_collected>0],bins=1000, density=False, log=True)
# # ax.scatter(list(range(dim)),position_y)
# # ax.set_ylim([0, 20])
# # ax.scatter(y_array, np.ones_like(y_array), marker="|",color='r')
# for ax in axes:
#     for a in ax:
#         for layer in y_array:
#             a.axvline(layer, color='r', linestyle='dashed', linewidth=1)
# # plt.scatter(list(range(dim)),direction_theta)
# # plt.yscale('log', nonposy='clip')
# # plt.show()
# fig.show(axes)
