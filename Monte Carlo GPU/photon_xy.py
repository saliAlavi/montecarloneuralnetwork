import numpy as np
from numba import jitclass  # import the decorator
from numba import int32, float32  # import the types
from numba import cuda
import numba as nb
import cmath
from numba import typed, types
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states, init_xoroshiro128p_states

g_ddivs = 64
g_ddivs = 64
nnxy=500
nnr=250
nnz=250

# debug=True
@cuda.jit(device=True, inline=True )
def polarized_photon_linear(seed, incident_degree, n, reflection, zstokes, random_nums, U, W, jones, mu_as,
                            mu_ss, s1, s2, m11, m12, scat_events, jones_partial, co_xy,thread_idx):

    NNxy = nnxy
    NNr = nnr
    NNz = nnz
    dr = 10e-5
    dx = 10e-5
    dy = 10e-5
    dz = 10e-5
    collection_rad = NNxy / 2 * dx
    slabsize = 1.00000

    degree_divs = g_ddivs

    jones[thread_idx, 0] = 1
    jones[thread_idx, 1] = 1
    jones[thread_idx, 2] = 0
    jones[thread_idx, 3] = 0

    jones_partial[thread_idx, 0] = 1
    jones_partial[thread_idx, 1] = 1
    jones_partial[thread_idx, 2] = 0
    jones_partial[thread_idx, 3] = 0

    dd = incident_degree
    deg = (dd) * cmath.pi / 180
    deg_s = (dd) * cmath.pi / 180
    # U[thread_idx, 0] = cmath.sin(deg).real
    # U[thread_idx, 1] = 0
    # U[thread_idx, 2] = cmath.cos(deg).real
    U[thread_idx, 0] = 0
    U[thread_idx, 1] = 0
    U[thread_idx, 2] = 1

    x = cuda.local.array(1, dtype=nb.float32)
    y = cuda.local.array(1, dtype=nb.float32)
    z = cuda.local.array(1, dtype=nb.float32)
    # x[0] = positions[thread_idx, 0]  # UPDATE in the end
    # y[0] = positions[thread_idx, 1]  # UPDATE in the end
    # z[0] = positions[thread_idx, 2]  # UPDATE in the end
    x[0]=0
    y[0]=0
    z[0]=0
    mu_a = mu_as[thread_idx]
    mu_s = mu_ss[thread_idx]
    scat_events[thread_idx] = 0  # UPDATE in the end
    scat_event = scat_events[thread_idx]
    W[thread_idx] = 1
    W_local = W[thread_idx]  # UPDATE in the end

    S0 = cuda.local.array(4, dtype=float32)
    S = cuda.local.array(4, dtype=float32)
    U_new = cuda.local.array(3, dtype=float32)
    U_prev = cuda.local.array(3, dtype=float32)
    U_epar = cuda.local.array(3, dtype=float32)
    B0 = cuda.local.array(3, dtype=float32)

    outEV = cuda.local.array(1, nb.complex128)
    outEQ = cuda.local.array(1, nb.complex128)
    outEI = cuda.local.array(1, nb.complex128)
    coherence = cuda.local.array(1, nb.float32)

    max_depth = cuda.local.array(1, dtype=float32)
    I = cuda.local.array(1, dtype=nb.complex64)
    rand = cuda.local.array(1, float32)
    s = cuda.local.array(1, dtype=float32)

    theta_scat = cuda.local.array(1, nb.float32)
    phi_scat = cuda.local.array(1, nb.float32)

    temp = cuda.local.array(1, dtype=nb.float32)
    temp6 = cuda.local.array(1, dtype=nb.float32)

    Epar = cuda.local.array(1, dtype=nb.complex64)
    Eper = cuda.local.array(1, dtype=nb.complex64)
    Efor = cuda.local.array(1, dtype=nb.complex64)
    Erev = cuda.local.array(1, dtype=nb.complex64)
    max_depth[0] = 0
    B0[0] = 1
    B0[1] = 0
    B0[2] = 0
    n_o = n
    n_e = n
    # n_e=ne
    lambda_0 = 632.8e-9
    chi = 0
    steps = 50
    acc_dist = 0

    reflection[thread_idx, 0]=0
    reflection[thread_idx, 1]=0

    albedo = mu_s / (mu_s + mu_a)

    S0[0] = 1
    S0[1] = 1
    S0[2] = 0
    S0[3] = 0

    for i in range(4):
        S[i] = S0[i]

    for step in range(steps):
        rand[0] = random_nums[step + 0]
        s[0] = -cmath.log(rand[0]).real / (mu_a + mu_s)
        acc_dist += s[0]
        if step > 0:
            pass
            # positions[thread_idx, 0] = x[0]
            # positions[thread_idx, 1] = y[0]
            # positions[thread_idx, 2] = z[0]

        if scat_event == 0:

            U_new[0] = 0
            U_new[1] = 0
            U_new[2] = 1
            U_epar[0] = 1
            U_epar[1] = 0
            U_epar[2] = 0

            birefringence_rot(s[0], chi, lambda_0, n_o, n_e, jones, thread_idx, B0, U_new, U_epar, random_nums, step + 1)

            EtoQ(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            S[1] = temp6[0]
            EtoI(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            S[1] /= temp6[0]

            EtoU(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            S[2] = temp6[0]
            EtoI(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            S[2] /= temp6[0]

        x[0] += U[thread_idx, 0] * s[0]
        y[0] += U[thread_idx, 1] * s[0]
        z[0] += U[thread_idx, 2] * s[0]
        scat_event += 1
        # print('z', step, z[0], U[thread_idx, 2])

        if z[0].real > max_depth[0]:
            max_depth[0] = z[0].real
        W_local = W_local * albedo  # partial photon

        if z[0].real >= 0:
            theta_partial = abs(cmath.pi - deg_s - cmath.acos(U[thread_idx, 2]).real)
            itheta_deg = int(theta_partial * (degree_divs - 1) / cmath.pi)

            if not U[thread_idx, 0] == 0:
                phi_partial = cmath.atan(U[thread_idx, 1] / U[thread_idx, 0])
            else:
                phi_partial = cmath.pi / 2
            # todo normalize I
            I_sum = 0
            for ctr in range(degree_divs):
                I_sum += m11[ctr] + m12[ctr] * (
                            S[1] * cmath.cos(2 * phi_partial).real + S[2] * cmath.sin(2 * phi_partial).real) / S[0]
            # I_sum *= 180 / degree_divs
            I[0] = m11[itheta_deg] + m12[itheta_deg] * (
                        S[1] * cmath.cos(2 * phi_partial).real + S[2] * cmath.sin(2 * phi_partial).real) / S[0]
            I[0] /= I_sum
            I[0] = (W_local * I[0] * (cmath.exp(-(mu_a + mu_s).real * z[0].real) / cmath.cos(theta_partial)).real).real
            backscatter_intensity = 2 * 2 * I[0]
            reflection[thread_idx,0]+=1
            reflection[thread_idx, 1]+=I[0].real
            if 1:

                jones_partial[thread_idx, 0] = jones[thread_idx, 0] * s2[itheta_deg]
                jones_partial[thread_idx, 1] = jones[thread_idx, 1] * s2[itheta_deg]
                jones_partial[thread_idx, 2] = jones[thread_idx, 2] * s1[itheta_deg]
                jones_partial[thread_idx, 3] = jones[thread_idx, 3] * s1[itheta_deg]

                U_new[0] = cmath.sin(deg_s).real
                U_new[1] = 0
                U_new[2] = -cmath.cos(deg_s).real
                if U[thread_idx, 2] == 1 or U[thread_idx, 2] == -1:
                    pass
                else:
                    temp1 = U[thread_idx, 0] * U_new[0] + U[thread_idx, 1] * U_new[1] + U[thread_idx, 2] * U_new[2]
                    U_epar[0] = U_new[0] - (U[thread_idx, 0] * temp1).real
                    U_epar[1] = U_new[1] - (U[thread_idx, 1] * temp1).real
                    U_epar[2] = U_new[2] - (U[thread_idx, 2] * temp1).real
                    temp2 = cmath.sqrt(U_epar[0] * U_epar[0] + U_epar[1] * U_epar[1] + U_epar[2] * U_epar[2]).real
                    U_epar[0] = U_epar[0] / temp2.real
                    U_epar[1] = U_epar[1] / temp2.real
                    U_epar[2] = U_epar[2] / temp2.real

                if U[thread_idx, 0] == 0 and U[thread_idx, 1] > 0:
                    phi_partial = cmath.pi / 2
                elif U[thread_idx, 0] == 0 and U[thread_idx, 1] < 0:
                    phi_partial = -cmath.pi / 2
                elif U[thread_idx, 0] == 0 and U[thread_idx, 1] == 0:
                    phi_partial = 0
                else:
                    phi_partial = cmath.atan(U[thread_idx, 1] / U[thread_idx, 0]).real
                # todo check birefringance
                rotate_phi(jones_partial, phi_partial, thread_idx)

                EtoQ(jones_partial[thread_idx, 0], jones_partial[thread_idx, 2], outEQ)
                EtoI(jones_partial[thread_idx, 0], jones_partial[thread_idx, 2], outEI)
                temp3 = outEQ[0] / outEI[0]
                # todo times I partial ?
                # co = (W_local * (1 + temp3)).real
                # cross = (W_local* (2 - temp3)).real
                co =  (W_local * (1+ S[1]/S[0] )).real
                cross = (W_local * (1 - S[1]/S[0])).real
                # if not cross==0:
                #     print('cross',cross)
                if acc_dist > dz * (NNz-1):
                    acc_iz=NNz-1
                else:
                    acc_iz=int(acc_dist/dz)
                if acc_iz >NNz-1:
                    acc_iz=NNz-1
                if scat_event == 1:

                    zstokes[thread_idx,0,0]+=co
                    zstokes[thread_idx, 0, 1] += 1
                    zstokes[thread_idx, acc_iz, 2] +=W_local*((S[1] * S[1] + S[2] * S[2] + S[3] * S[3]) / (S[0] * S[0]))
                    co_xy[thread_idx, 0,0, 0] += S[0].real
                    co_xy[thread_idx, 0,0, 1] += S[1]
                    co_xy[thread_idx, 0,0, 2] += S[2]
                    co_xy[thread_idx, 0,0, 3] += S[3]
                    # print(step)
                else:
                    Efor[0] = jones_partial[thread_idx, 2]
                    Erev[0] = -jones_partial[thread_idx, 1]

                    get_degree_coherence(Efor[0], Erev[0], coherence)
                    ir = int(cmath.sqrt(pow(x[0], 2) + pow(y[0], 2)).real / dr)

                    if ir > NNr - 1:
                        ir = NNr - 1
                    # todo check iz
                    iz = int(max_depth[0] / dz)
                    iz = int(z[0] / dz)
                    if iz > NNz - 1:
                        iz = NNz - 1
                    if iz < 0:
                        iz = 0

                    if x[0] >= -collection_rad:
                        ix = int(abs(x[0] + collection_rad) / dx)
                    elif x[0] < -collection_rad:
                        ix = 0

                    if y[0] >= -collection_rad:
                        iy = int(abs(y[0] + collection_rad) / dy)
                    elif y[0] < -collection_rad:
                        iy = 0

                    if ix > NNxy - 1:
                        ix = NNxy - 1
                    if iy > NNxy - 1:
                        iy = NNxy - 1

                    # i_stokes_rz[thread_idx, ir, iz] += S[0]
                    # q_stokes_rz[thread_idx, ir, iz] += S[1]
                    # u_stokes_rz[thread_idx, ir, iz] += S[2]
                    # v_stokes_rz[thread_idx, ir, iz] += S[3]

                    # zstokes[thread_idx,acc_iz,0]+=co
                    # zstokes[thread_idx, acc_iz, 1] += cross
                    # zstokes[thread_idx, acc_iz, 2] += cross * coherence[0]

                    # incoh_cross_rz[thread_idx, ir, iz] += cross
                    # co_rz[thread_idx, ir, iz] += co
                    # cross_rz[thread_idx, ir, iz] += cross * coherence[0]
                    #
                    # incoh_cross_xy[thread_idx, iy, ix] += cross
                    # print('x',ix,'y',iy)
                    zstokes[thread_idx, acc_iz, 0] += co
                    zstokes[thread_idx, acc_iz, 1] += 1
                    zstokes[thread_idx, acc_iz, 2] += W_local*((S[1] * S[1] + S[2] * S[2] + S[3] * S[3]) / (S[0] * S[0]))
                    temp_m11 = cuda.local.array(1,dtype=nb.float64)
                    get_m12(jones_partial[thread_idx],temp_m11)
                    temp_m = cuda.local.array(16,dtype=nb.float64)
                    co_xy[thread_idx, iy, ix, 0] += temp_m[0]+1
                    co_xy[thread_idx, iy, ix, 1] += temp_m[1]+1
                    co_xy[thread_idx, iy, ix, 2] += temp_m[2]+1
                    co_xy[thread_idx, iy, ix, 3] += 1
                    # cross_xy[thread_idx, iy, ix] += cross * coherence[0]

        # Collect

        # if abs(U[thread_idx, 2]) < 1e-12:
        #     degree = 90
        # else:
        #     degree = abs(cmath.atan(
        #         cmath.sqrt(U[thread_idx, 0] * U[thread_idx, 0] + U[thread_idx, 1] * U[thread_idx, 1]).real / U[
        #             thread_idx, 2]).real * 180 / cmath.pi)
        # if z[0] < 0 and degree < 20:
        #     x[0] -= U[thread_idx, 0] * s[0]
        #     y[0] -= U[thread_idx, 1] * s[0]
        #     z[0] -= U[thread_idx, 2] * s[0]
        #     scat_event -= 1
        #
        #     for i in range(3):
        #         U[thread_idx, i] = U_prev[i]
        #
        #     for i in range(4):
        #         jones[thread_idx, i] = jones_partial[thread_idx, i]
        #
        #     theta_partial = cmath.acos(-U[thread_idx, 2]).real
        #     itheta_deg = int(theta_partial / cmath.pi * degree_divs)
        #     phi_partial = 0
        #
        #     jones[thread_idx, 0] = jones[thread_idx, 0] * s2[itheta_deg]
        #     jones[thread_idx, 1] = jones[thread_idx, 1] * s2[itheta_deg]
        #     jones[thread_idx, 2] = jones[thread_idx, 2] * s1[itheta_deg]
        #     jones[thread_idx, 3] = jones[thread_idx, 3] * s1[itheta_deg]
        #
        #     U_new[0] = 0
        #     U_new[1] = 0
        #     U_new[2] = -1
        #
        #     if U[thread_idx, 2] == 1 or U[thread_idx, 2] == -1:
        #         pass
        #     else:
        #         temp2 = U[thread_idx, 0] * U_new[0] + U[thread_idx, 1] * U_new[1] + U[thread_idx, 2] * U_new[2]
        #         U_epar[0] = U_new[0] - U[thread_idx, 0] * temp2
        #         U_epar[1] = U_new[1] - U[thread_idx, 1] * temp2
        #         U_epar[2] = U_new[2] - U[thread_idx, 2] * temp2
        #         temp2 = cmath.sqrt(pow(U_epar[0], 2) + pow(U_epar[1], 2) + pow(U_epar[2], 2)).real
        #         U_epar[0] = U_epar[0] / temp2
        #         U_epar[1] = U_epar[1] / temp2
        #         U_epar[2] = U_epar[2] / temp2
        #
        #     if U[thread_idx, 0] == 0 and U[thread_idx, 1] > 0:
        #         phi = cmath.pi / 2
        #     elif U[thread_idx, 0] == 0 and U[thread_idx, 1] < 0:
        #         phi = -cmath.pi / 2
        #     elif U[thread_idx, 0] == 0 and U[thread_idx, 1] == 0:
        #         phi = 0
        #     elif cmath.atan(U[thread_idx, 1] / U[thread_idx, 0]).real >= 0:
        #         if U[thread_idx, 0] >= 0:
        #             phi = cmath.atan(U[thread_idx, 1] / U[thread_idx, 0])
        #         elif U[thread_idx, 0] < 0:
        #             phi = cmath.pi + cmath.atan(U[thread_idx, 1] / U[thread_idx, 0]).real
        #     else:
        #         if U[thread_idx, 0] >= 0:
        #             phi = cmath.atan(U[thread_idx, 1] / U[thread_idx, 0]).real
        #         else:
        #             phi = cmath.pi + cmath.atan(U[thread_idx, 1] / U[thread_idx, 0]).real
        #
        #     rotate_phi(jones, phi, thread_idx)
        #
        #     temp4 = cuda.local.array(1, dtype=nb.float32)
        #     EtoQ(jones_partial[thread_idx, 0], jones_partial[thread_idx, 2], outEQ)
        #     EtoI(jones_partial[thread_idx, 0], jones_partial[thread_idx, 2], outEI)
        #     temp3 = outEQ[0] / outEI[0]
        #     # todo times I partial ?
        #     co = (I[0] * (1 + temp3)).real
        #     cross = (I[0] * (1 - temp3)).real
        #
        #     if scat_event == 1:
        #         pass
        #         # i_stokes_rz[thread_idx, 0, 0] += S[0]
        #         # q_stokes_rz[thread_idx, 0, 0] += S[1]
        #         # u_stokes_rz[thread_idx, 0, 0] += S[2]
        #         # v_stokes_rz[thread_idx, 0, 0] += S[3]
        #
        #         # co_rz_trad[thread_idx, NNr - 1, NNz - 1] += co
        #         # incoh_cross_rz_trad[thread_idx, NNr - 1, NNz - 1] += cross
        #         # co_xy_trad[thread_idx, 0, 0] += co
        #         # incoh_cross_xy_trad[thread_idx, 0, 0] += cross
        #
        #     else:
        #         Efor[0] = jones_partial[thread_idx, 2]
        #         Erev[0] = -jones_partial[thread_idx, 1]
        #
        #         get_degree_coherence(Efor[0], Erev[0], coherence)
        #
        #         ir = int(cmath.sqrt(pow(x[0], 2) + pow(y[0], 2)).real / dr)
        #
        #         if ir > NNr - 1:
        #             ir = NNr - 1
        #
        #         iz = int(max_depth[0] / dz)
        #         if iz > NNz - 1:
        #             iz = NNz - 1
        #
        #         if x[0] >= -collection_rad:
        #             ix = int(abs(x[0] + collection_rad) / dx)
        #         elif x[0] < -collection_rad:
        #             ix = 0
        #
        #         if y[0] >= -collection_rad:
        #             iy = int(abs(y[0] + collection_rad) / dy)
        #         elif y[0] < -collection_rad:
        #             iy = 0
        #
        #         if ix > NNxy - 1:
        #             ix = NNxy - 1
        #         if iy > NNxy - 1:
        #             iy = NNxy - 1

                # i_stokes_rz[thread_idx, ir, iz] += S[0]
                # q_stokes_rz[thread_idx, ir, iz] += S[1]
                # u_stokes_rz[thread_idx, ir, iz] += S[2]
                # v_stokes_rz[thread_idx, ir, iz] += S[3]

                # incoh_cross_rz_trad[thread_idx, ir, iz] += cross
                # co_rz_trad[thread_idx, ir, iz] += co
                # cross_rz_trad[thread_idx, ir, iz] += cross * coherence[0]
                #
                # incoh_cross_xy_trad[thread_idx, iy, ix] += cross
                # co_xy_trad[thread_idx, iy, ix] += co
                # cross_xy_trad[thread_idx, iy, ix] += cross * coherence[0]
            # break
        # todo kill photon

        if z[0] < 0 or z[0] > slabsize :
            if step < steps - 1:
                ss_photons = steps - 1 - step
                # co_rz[thread_idx, NNr - 2, NNz - 2] += ss_photons * I[0].real
                # incoh_cross_rz[thread_idx, NNr - 2, NNz - 2] += ss_photons * I[0].real
                # co_xy[thread_idx, NNxy - 2, NNxy - 2] += ss_photons * I[0].real
                # incoh_cross_xy[thread_idx, NNxy - 2, NNxy - 2] += ss_photons * I[0].real
            break  # kill photon

        # todo if still alive
        if 1:

            for i in range(4):
                jones_partial[i] = jones[thread_idx, i]

            for i in range(3):
                U_prev[i] = U[thread_idx, i]
            random_by_pf(S, m11, m12, theta_scat, phi_scat, random_nums, step + 2)

            update_u(U, theta_scat[0], phi_scat[0], U_new, thread_idx)
            rotate_phi(jones, phi_scat[0], thread_idx)
            itheta_deg = int(theta_scat[0] / cmath.pi * (degree_divs - 1))

            jones[thread_idx, 0] = jones[thread_idx, 0] * s2[itheta_deg]
            jones[thread_idx, 1] = jones[thread_idx, 1] * s2[itheta_deg]
            jones[thread_idx, 2] = jones[thread_idx, 2] * s1[itheta_deg]
            jones[thread_idx, 3] = jones[thread_idx, 3] * s1[itheta_deg]
            phi = phi_scat[0]

            if theta_scat[0].real == 0:
                if abs(U[thread_idx, 2]) == 1:
                    pass
                else:
                    U_epar[0] = -U[thread_idx, 0] * U[thread_idx, 2]
                    U_epar[1] = -U[thread_idx, 1] * U[thread_idx, 2]
                    U_epar[2] = U[thread_idx, 0] * U[thread_idx, 0] + U[thread_idx, 1] * U[thread_idx, 1]
                    temp5 = cmath.sqrt(pow(U_epar[0], 2) + pow(U_epar[1], 2) + pow(U_epar[2], 2)).real

                    U_epar[0] = U_epar[0] / temp5
                    U_epar[1] = U_epar[1] / temp5
                    U_epar[2] = U_epar[2] / temp5

                U_epar[0] = U_epar[0] * cmath.cos(phi).real + (
                        U[thread_idx, 1] * U_epar[2] - U[thread_idx, 2] * U_epar[1]) * cmath.sin(phi).real
                U_epar[1] = U_epar[1] * cmath.cos(phi).real + (
                        U[thread_idx, 2] * U_epar[0] - U[thread_idx, 0] * U_epar[2]) * cmath.sin(phi).real
                U_epar[2] = U_epar[2] * cmath.cos(phi).real + (
                        U[thread_idx, 0] * U_epar[1] - U[thread_idx, 1] * U_epar[0]) * cmath.sin(phi).real
            else:
                temp5 = U[thread_idx, 0] * U_new[0] + U[thread_idx, 1] * U_new[1] + U[thread_idx, 2] * U_new[2]
                U_epar[0] = U_new[0] - U[thread_idx, 0] * temp5
                U_epar[1] = U_new[1] - U[thread_idx, 1] * temp5
                U_epar[2] = U_new[2] - U[thread_idx, 2] * temp5
                temp5 = cmath.sqrt(pow(U_epar[0], 2) + pow(U_epar[1], 2) + pow(U_epar[2], 2)).real

                U_epar[0] = U_epar[0] / temp5
                U_epar[1] = U_epar[1] / temp5
                U_epar[2] = U_epar[2] / temp5

            birefringence_rot(s[0], chi, lambda_0, n_o, n_e, jones, thread_idx, B0, U_new, U_epar,
                              random_nums, step + 5)

            costheta = cmath.cos(theta_scat[0]).real
            temp5 = (cmath.sqrt(1 - costheta * costheta) * cmath.sqrt(1 - U_new[2] * U_new[2])).real
            if temp5 == 0:
                cosi = 0
            else:

                if (phi.real > cmath.pi) and (phi.real < 2 * cmath.pi):
                    cosi = (U_new[2] * costheta - U[thread_idx, 2]) / temp5
                else:
                    cosi = -(U_new[2] * costheta - U[thread_idx, 2]) / temp5
                if cosi > 1:
                    cosi = 1
                if cosi < -1:
                    cosi = -1

            sini = cmath.sqrt(1 - cosi * cosi).real

            jones[thread_idx, 0] = cosi * jones[thread_idx, 0] - sini * jones[thread_idx, 2]
            jones[thread_idx, 1] = cosi * jones[thread_idx, 1] - sini * jones[thread_idx, 3]
            jones[thread_idx, 2] = sini * jones[thread_idx, 0] + cosi * jones[thread_idx, 2]
            jones[thread_idx, 3] = sini * jones[thread_idx, 1] + cosi * jones[thread_idx, 3]

            for i in range(3):
                U_prev[i] = U[thread_idx, i]
                U[thread_idx, i] = U_new[i]

            temp5 = pow(jones[thread_idx, 0].real, 2) + pow(jones[thread_idx, 1].real, 2) + pow(
                jones[thread_idx, 2].real, 2) + pow(jones[thread_idx, 3].real, 2) + pow(
                jones[thread_idx, 0].imag, 2) + pow(jones[thread_idx, 1].imag, 2) + pow(jones[thread_idx, 2].imag,
                                                                                        2) + pow(
                jones[thread_idx, 3].imag, 2)
            temp5 = cmath.sqrt(temp5).real
            for i in range(4):
                jones[thread_idx, i] /= temp5
            # EtoQ(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            # S[1] = temp6[0]
            # EtoI(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            # S[1] /= temp6[0]
            #
            # EtoU(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            # S[2] = temp6[0]
            # EtoI(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            # S[2] /= temp6[0]
            update_stokes(S,jones[thread_idx])

            # todo kill photon with russian roulette


# consumes 2 seeds
@cuda.jit(device=True, inline=True)
def random_by_pf(stokes, s11, s12, theta_out, phi_out, random_nums, seed):
    ddivs = g_ddivs
    pdivs = 100
    pf = cuda.local.array((ddivs, ddivs), dtype=nb.float32)
    pf_theta = cuda.local.array(ddivs, dtype=nb.float32)
    pf_phi = cuda.local.array(ddivs, dtype=nb.float32)
    p_theta_cs = cuda.local.array(ddivs, dtype=nb.float32)
    p_phi_cs = cuda.local.array(ddivs, dtype=nb.float32)
    inverse_p_theta = cuda.local.array(pdivs, dtype=nb.float32)
    inverse_p_phi = cuda.local.array(pdivs, dtype=nb.float32)

    for i in range(ddivs):
        for j in range(ddivs):
            pf[i, j] = (s11[i] * stokes[0] + s12[i] * (
                    stokes[1] * cmath.cos(2 * j * 2 * cmath.pi / ddivs) + stokes[2] * cmath.sin(
                2 * j * 2 * cmath.pi / ddivs))).real
    for i in range(ddivs):
        pf_theta[i] = 0
        pf_phi[i] = 0
    for i in range(ddivs):
        for j in range(ddivs):
            pf_theta[i] += pf[i, j]
            pf_phi[j] += pf[i, j]

    pf_theta_sum = 0
    pf_phi_sum = 0
    for i in range(ddivs):
        pf_theta_sum += pf_theta[i]
        pf_phi_sum += pf_phi[i]

    for i in range(ddivs):
        pf_theta[i] /= pf_theta_sum
        pf_phi[i] /= pf_phi_sum

    p_theta_cs[0] = pf_theta[0]
    p_phi_cs[0] = pf_phi[0]
    for i in range(1, ddivs):
        p_theta_cs[i] = p_theta_cs[i - 1] + pf_theta[i]
        p_phi_cs[i] = p_phi_cs[i - 1] + pf_phi[i]
        if p_theta_cs[i] > 1:
            p_theta_cs[i] = 1
        if p_phi_cs[i] > 1:
            p_phi_cs[i] = 1

    for i in range(pdivs):
        inverse_p_theta[i] = 0
        inverse_p_phi[i] = 0

    inverse_p_theta[int(p_theta_cs[0] * (pdivs - 1))] = 0
    inverse_p_phi[int(p_phi_cs[0] * (pdivs - 1))] = 0
    for i in range(1, ddivs):
        idx = int(p_theta_cs[i] * (pdivs - 1))
        j = i - 1
        while idx == int(p_theta_cs[j] * (pdivs - 1)):
            j -= 1
        inverse_p_theta[idx] = j + 1

    for i in range(1, ddivs):
        idx = int(p_theta_cs[i] * (pdivs - 1))
        j = i - 1
        while idx == int(p_theta_cs[j] * (pdivs - 1)):
            j -= 1
        inverse_p_phi[idx] = j + 1

    for i in range(pdivs):
        el = inverse_p_theta[i]
        if el == 0 and i > 0:
            inverse_p_theta[i] = inverse_p_theta[i - 1]

    for i in range(pdivs):
        el = inverse_p_phi[i]
        if el == 0 and i > 0:
            inverse_p_phi[i] = inverse_p_phi[i - 1]

    theta_out[0] = inverse_p_theta[int(random_nums[seed + 1] * (pdivs - 1))] * cmath.pi / ddivs
    phi_out[0] = inverse_p_phi[int(random_nums[seed + 2] * (pdivs - 1))] * 2 * cmath.pi / ddivs

    # phi_out[0] = inverse_p_phi[10] * 2 * cmath.pi / ddivs
    # phi_out[0]=1


@cuda.jit(device=True, inline=True)
def get_degree_coherence(Efor, Erev, out):
    out[0] = 2 * (Efor.real * Erev.real + Efor.imag * Erev.imag) / (
            Efor.real * Efor.real + Efor.imag * Efor.imag + Erev.real * Erev.real + Erev.imag * Erev.imag)


@cuda.jit(device=True, inline=True)
def EtoI(Epar, Eper, out):
    out[0] = Epar.real * Epar.real + Epar.imag * Epar.imag + Eper.real * Eper.real + Eper.imag * Eper.imag
    # print('I',out[0].real)


@cuda.jit(device=True, inline=True)
def EtoQ(Epar, Eper, out):
    out[0] = Epar.real * Epar.real + Epar.imag * Epar.imag - Eper.real * Eper.real - Eper.imag * Eper.imag


@cuda.jit(device=True, inline=True)
def EtoU(Epar, Eper, out):
    out[0] = 2 * (Epar.real * Eper.real + Epar.imag * Eper.imag)


@cuda.jit(device=True, inline=True)
def EtoV(Epar, Eper, out):
    out[0] = 2 * (Epar.imag * Eper.real - Epar.real * Eper.imag)

@cuda.jit(device=True, inline=True)
def update_stokes(stokes, jones):
    stokes_temp=cuda.local.array(4,dtype=float32)
    temps = cuda.local.array(4,dtype=float32)

    temps[0]=0.5*(jones[0].real*jones[0].real+jones[0].imag*jones[0].imag   +jones[1].real*jones[1].real+jones[1].imag*jones[1].imag   +jones[2].real*jones[2].real+jones[2].imag*jones[2].imag   +jones[3].real*jones[3].real+jones[3].imag*jones[3].imag)
    temps[1]=0.5*(jones[0].real*jones[0].real+jones[0].imag*jones[0].imag   +jones[1].real*jones[1].real+jones[1].imag*jones[1].imag   -(jones[2].real*jones[2].real+jones[2].imag*jones[2].imag   +jones[3].real*jones[3].real+jones[3].imag*jones[3].imag))
    temps[2]= ((jones[0])*(jones[2].real-1j*jones[2].imag) +(jones[1])*(jones[3].real-1j*jones[3].imag)).real
    temps[3]=-((jones[0])*(jones[2].real-1j*jones[2].imag) +(jones[1])*(jones[3].real-1j*jones[3].imag)).imag

    stokes_temp[0] = stokes[0]*temps[0]+stokes[1]*temps[1]+stokes[2]*temps[2]+stokes[3]*temps[3]
    ######################
    temps[0] = 0.5 * ((jones[0].real * jones[0].real + jones[0].imag * jones[0].imag) - (jones[1].real * jones[1].real + jones[1].imag * jones[1].imag) + (jones[2].real * jones[2].real + jones[2].imag * jones[2].imag) - (jones[3].real *jones[3].real + jones[3].imag * jones[3].imag))
    temps[1] = 0.5 * ((jones[0].real * jones[0].real + jones[0].imag * jones[0].imag) - (jones[1].real * jones[1].real + jones[1].imag * jones[1].imag) - (jones[2].real * jones[2].real + jones[2].imag * jones[2].imag) + (jones[3].real * jones[3].real + jones[3].imag * jones[3].imag))
    temps[2] =  ((jones[0]) * (jones[2].real - 1j*jones[2].imag) - (jones[1]) * (jones[3].real - 1j*jones[3].imag)).real
    temps[3] = -((jones[0]) * (jones[2].real - 1j*jones[2].imag) - (jones[1]) * (jones[3].real - 1j*jones[3].imag)).imag

    stokes_temp[1] = stokes[0] * temps[0] + stokes[1] * temps[1] + stokes[2] * temps[2] + stokes[3] * temps[3]
    ####################
    temps[0]=((jones[0])*(jones[1].real-1j*jones[1].imag) + (jones[2])*(jones[3].real-1j*jones[3].imag)).real
    temps[1] = ((jones[0]) * (jones[1].real - 1j * jones[1].imag) - (jones[2]) * (jones[3].real - 1j * jones[3].imag)).real
    temps[2] = ((jones[0]) * (jones[3].real - 1j * jones[3].imag) + (jones[1]) * (jones[2].real - 1j * jones[2].imag)).real
    temps[3] =-((jones[0]) * (jones[3].real - 1j * jones[3].imag) + (jones[1]) * (jones[2].real - 1j * jones[2].imag)).imag

    stokes_temp[2] = stokes[0] * temps[0] + stokes[1] * temps[1] + stokes[2] * temps[2] + stokes[3] * temps[3]
    ####################
    temps[0] = -((jones[1]) * (jones[0].real - 1j * jones[0].imag) + (jones[3]) * (jones[2].real - 1j * jones[2].imag)).imag
    temps[1] = -((jones[1]) * (jones[0].real - 1j * jones[0].imag) - (jones[3]) * (jones[2].real - 1j * jones[2].imag)).imag
    temps[2] = -((jones[2]) * (jones[3].real - 1j * jones[3].imag) - (jones[0]) * (jones[3].real - 1j * jones[3].imag)).imag
    temps[3] = -((jones[2]) * (jones[3].real - 1j * jones[3].imag) - (jones[0]) * (jones[3].real - 1j * jones[3].imag)).real

    stokes_temp[3] = stokes[0] * temps[0] + stokes[1] * temps[1] + stokes[2] * temps[2] + stokes[3] * temps[3]

    for i in range(4):
        stokes[i]=stokes_temp[i]

@cuda.jit(device=True, inline=True)
def get_m12(jones,out):
    out[0] = 0.5 * (
                jones[0].real * jones[0].real + jones[0].imag * jones[0].imag + jones[1].real * jones[1].real + jones[1].imag * jones[1].imag - (jones[2].real * jones[2].real + jones[2].imag * jones[2].imag + jones[3].real * jones[
                        3].real + jones[3].imag * jones[3].imag))

@cuda.jit(device=True, inline=True)
def get_m(jones, out):
    out[0] = 0.5 * (
            jones[0].real * jones[0].real + jones[0].imag * jones[0].imag + jones[1].real * jones[1].real + jones[
        1].imag * jones[1].imag - (
                        jones[2].real * jones[2].real + jones[2].imag * jones[2].imag + jones[3].real * jones[
                    3].real + jones[3].imag * jones[3].imag))

    out[1] = 0.5 * (
            jones[0].real * jones[0].real + jones[0].imag * jones[0].imag + jones[1].real * jones[1].real + jones[
        1].imag * jones[1].imag - (
                    jones[2].real * jones[2].real + jones[2].imag * jones[2].imag + jones[3].real * jones[
                3].real + jones[3].imag * jones[3].imag))
    out[2] = ((jones[0]) * (jones[2].real - 1j * jones[2].imag) + (jones[1]) * (
            jones[3].real - 1j * jones[3].imag)).real
    out[3] = -((jones[0]) * (jones[2].real - 1j * jones[2].imag) + (jones[1]) * (
            jones[3].real - 1j * jones[3].imag)).imag

    ######################
    out[4] = 0.5 * ((jones[0].real * jones[0].real + jones[0].imag * jones[0].imag) - (
            jones[1].real * jones[1].real + jones[1].imag * jones[1].imag) + (
                            jones[2].real * jones[2].real + jones[2].imag * jones[2].imag) - (
                            jones[3].real * jones[3].real + jones[3].imag * jones[3].imag))
    out[5] = 0.5 * ((jones[0].real * jones[0].real + jones[0].imag * jones[0].imag) - (
            jones[1].real * jones[1].real + jones[1].imag * jones[1].imag) - (
                            jones[2].real * jones[2].real + jones[2].imag * jones[2].imag) + (
                            jones[3].real * jones[3].real + jones[3].imag * jones[3].imag))
    out[6] = ((jones[0]) * (jones[2].real - 1j * jones[2].imag) - (jones[1]) * (
            jones[3].real - 1j * jones[3].imag)).real
    out[7] = -((jones[0]) * (jones[2].real - 1j * jones[2].imag) - (jones[1]) * (
            jones[3].real - 1j * jones[3].imag)).imag

    ####################
    out[8] = ((jones[0]) * (jones[1].real - 1j * jones[1].imag) + (jones[2]) * (
            jones[3].real - 1j * jones[3].imag)).real
    out[9] = ((jones[0]) * (jones[1].real - 1j * jones[1].imag) - (jones[2]) * (
            jones[3].real - 1j * jones[3].imag)).real
    out[10] = ((jones[0]) * (jones[3].real - 1j * jones[3].imag) + (jones[1]) * (
            jones[2].real - 1j * jones[2].imag)).real
    out[11] = -((jones[0]) * (jones[3].real - 1j * jones[3].imag) + (jones[1]) * (
            jones[2].real - 1j * jones[2].imag)).imag

    ####################
    out[12] = -((jones[1]) * (jones[0].real - 1j * jones[0].imag) + (jones[3]) * (
            jones[2].real - 1j * jones[2].imag)).imag
    out[13] = -((jones[1]) * (jones[0].real - 1j * jones[0].imag) - (jones[3]) * (
            jones[2].real - 1j * jones[2].imag)).imag
    out[14] = -((jones[2]) * (jones[3].real - 1j * jones[3].imag) - (jones[0]) * (
            jones[3].real - 1j * jones[3].imag)).imag
    out[15] = -((jones[2]) * (jones[3].real - 1j * jones[3].imag) - (jones[0]) * (
            jones[3].real - 1j * jones[3].imag)).real

@cuda.jit(device=True, inline=True)
def rotate_phi(jones, phi, thread_idx):
    cosphi = cmath.cos(phi).real
    sinphi = cmath.sin(phi).real
    jones[thread_idx, 0] = jones[thread_idx, 0] * cosphi + jones[thread_idx, 2] * sinphi
    jones[thread_idx, 1] = jones[thread_idx, 1] * cosphi + jones[thread_idx, 3] * sinphi
    jones[thread_idx, 2] = -jones[thread_idx, 0] * sinphi + jones[thread_idx, 2] * cosphi
    jones[thread_idx, 3] = -jones[thread_idx, 1] * sinphi + jones[thread_idx, 3] * cosphi


@cuda.jit(device=True, inline=True)
def get_rand(rand, rng_states, seed):
    rand[0] = xoroshiro128p_uniform_float32(rng_states, seed)
    i = 0
    while rand[0] == 0:
        rand[0] = xoroshiro128p_uniform_float32(rng_states, seed + i)
        i += 1


@cuda.jit(device=True, inline=True)
def update_u(U, theta, phi, U_new, thread_idx):
    ux = U[thread_idx, 0]
    uy = U[thread_idx, 1]
    uz = U[thread_idx, 2]

    costheta = cmath.cos(theta).real
    sintheta = cmath.sqrt(1.0 - costheta * costheta).real
    cosphi = cmath.cos(phi).real
    if (phi < cmath.pi):
        sinphi = cmath.sqrt(1.0 - cosphi * cosphi).real
    else:
        sinphi = -cmath.sqrt(1.0 - cosphi * cosphi).real
    sign = 1 if uz >= 0 else -1
    if (1 - abs(uz) <= 1.0E-12):
        uxx = sintheta * cosphi
        uyy = sintheta * sinphi
        uzz = costheta * sign
    else:
        temp = cmath.sqrt(1.0 - uz * uz).real
        uxx = sintheta * (ux * uz * cosphi - uy * sinphi) / temp + ux * costheta
        uyy = sintheta * (uy * uz * cosphi + ux * sinphi) / temp + uy * costheta
        uzz = -sintheta * cosphi * temp + uz * costheta
    temp = cmath.sqrt(pow(uxx, 2) + pow(uyy, 2) + pow(uzz, 2)).real
    uxx = uxx / temp
    uyy = uyy / temp
    uzz = uzz / temp
    U_new[0] = uxx
    U_new[1] = uyy
    U_new[2] = uzz


# consumes 2 seeds
# todo check this function for bugs
@cuda.jit(device=True, inline=True)
def birefringence_rot(s, chi, lambda_0, n_o, n_e, jones, thread_idx, B, U, U_epar, rand_nums, seed):
    if n_o != n_e or chi != 0:
        if B[0] == 2.0 and B[1] == 2.0 and B[2] == 2.0:

            rand_num = rand_nums[seed]
            theta = cmath.pi * rand_num
            rand_num = rand_nums[seed + 1]
            beta0 = cmath.pi * 2 * rand_num
            B[0] = cmath.sin(theta).real * cmath.cos(beta0).real
            B[1] = cmath.sin(theta).real * cmath.sin(beta0).real
            B[2] = cmath.cos(theta).real

        theta1 = cuda.local.array(1, float32)
        get_theta(B, U, theta1)
        Bp = cuda.local.array(3, float32)
        temp = B[0] * U[0] + B[1] * U[1] + B[2] * U[2]
        Bp[0] = B[0] - U[0] * temp
        Bp[1] = B[1] - U[1] * temp
        Bp[2] = B[2] - U[2] * temp
        temp = cmath.sqrt(Bp[0] * Bp[0] + Bp[1] * Bp[1] + Bp[2] * Bp[2]).real
        if temp != 0:
            Bp[0] /= temp
            Bp[1] /= temp
            Bp[2] /= temp
        beta = cuda.local.array(1, float32)
        get_beta(U_epar, Bp, U, beta)

        # temp = n_e * cmath.cos(theta1[0]).real * n_e * cmath.cos(theta1[0]).real
        delta_n = n_o * n_e / cmath.sqrt(
            n_e * cmath.cos(theta1[0]).real * n_e * cmath.cos(theta1[0]).real + n_o * cmath.sin(
                theta1[0]).real * n_o * cmath.sin(theta1[0]).real).real - n_o
        g_o = cmath.pi * delta_n / (lambda_0 * 1e4)

        # n_mat = np.array([g_o.j, -g_o.j, -chi, chi])
        n_mat = cuda.local.array(4, nb.complex64)
        n_mat[0] = g_o * 1j
        n_mat[1] = - g_o * 1j
        n_mat[2] = -chi
        n_mat[3] = chi
        Q_n = cmath.sqrt(n_mat[0] * n_mat[0] - n_mat[3] * n_mat[3])

        m = cuda.local.array(4, nb.complex64)
        m_new = cuda.local.array(4, nb.complex64)
        if Q_n != 0:
            # todo check which to keep
            # m[0] = (n_mat[0] - n_mat[1]) * cmath.sinh(Q_n * s) / (2 * Q_n) + cmath.cosh(Q_n * s)
            # m[1] = (n_mat[0] - n_mat[1])  * cmath.sinh(Q_n * s) / (2 * Q_n) + cmath.cosh(Q_n * s)
            # m[2] = (n_mat[2]) * cmath.sinh(Q_n * s) / (Q_n)
            # m[3] = -m[2]

            m[0] = cmath.exp((n_mat[0] + n_mat[1]) / 2 * s) * (
                        (n_mat[0] - n_mat[1]) * cmath.sinh(Q_n * s) / (2 * Q_n) + cmath.cosh(Q_n * s))
            m[1] = cmath.exp((n_mat[0] + n_mat[1]) / 2 * s) * (
                        (n_mat[0] - n_mat[1]) * cmath.sinh(Q_n * s) / (2 * Q_n) + cmath.cosh(Q_n * s))
            m[2] = cmath.exp((n_mat[0] + n_mat[1]) / 2 * s) * ((n_mat[2]) * cmath.sinh(Q_n * s) / (Q_n))
            m[3] = cmath.exp((n_mat[0] + n_mat[1]) / 2 * s) * (-m[2])

            m_new[0] = m[0] * cmath.cos(beta[0]) + m[2] * cmath.sin(beta[0])
            m_new[1] = m[1] * cmath.cos(beta[0]) + m[3] * cmath.sin(beta[0])
            m_new[2] = -m[0] * cmath.sin(beta[0]) + m[2] * cmath.cos(beta[0])
            m_new[3] = -m[1] * cmath.sin(beta[0]) + m[3] * cmath.cos(beta[0])

            jones[thread_idx, 0] = jones[thread_idx, 0] * m_new[0] + jones[thread_idx, 2] * m_new[3]
            jones[thread_idx, 1] = jones[thread_idx, 1] * m_new[0] + jones[thread_idx, 3] * m_new[3]
            jones[thread_idx, 2] = jones[thread_idx, 0] * m_new[2] + jones[thread_idx, 2] * m_new[1]
            jones[thread_idx, 3] = jones[thread_idx, 1] * m_new[2] + jones[thread_idx, 3] * m_new[1]


@cuda.jit(device=True, inline=True)
def get_theta(a, b, theta):
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    cross = cmath.sqrt((a[1] * b[2] - a[2] * b[1]) * (a[1] * b[2] - a[2] * b[1]) + (a[2] * b[0] - a[0] * b[2]) * (
            a[2] * b[0] - a[0] * b[2]) + (a[0] * b[1] - a[1] * b[0]) * (a[0] * b[1] - a[1] * b[0]))
    if dot > 1:
        dot = 1
    if dot < -1:
        dot = -1

    cross = cross.real
    if cross > 1:
        cross = 1
    if cross < -1:
        cross = -1

    if dot == 0 and cross >= 0:
        theta[0] = cmath.pi / 2
    elif dot == 0 and cross < 0:
        theta[0] = -cmath.pi / 2
    else:
        theta[0] = cmath.atan(cross / dot).real


@cuda.jit(device=True, inline=True)
def get_beta(U_epar, Bp, U, beta):
    theta = cuda.local.array(1, float32)
    get_theta(U, Bp, theta)

    direction = (U_epar[1] * Bp[2] - U_epar[2] * Bp[1]) * U[0] + (U_epar[2] * Bp[0] - U_epar[0] * Bp[2]) * U[1] + (
            U_epar[0] * Bp[1] - U_epar[1] * Bp[0]) * U[2]  # (U_epar X Bp).U

    if theta[0] < cmath.pi / 2:
        if direction > 0:
            beta[0] = theta[0]
        else:
            beta[0] = -theta[0]

    else:
        theta[0] = cmath.pi - theta[0]
        if direction > 0:
            beta[0] = -theta[0]
        else:
            beta[0] = theta[0]


@cuda.jit(inline=False)
def process_steps_linear(seed, incident_degree, n, reflection, zstokes, random_nums, U, W, jones, mu_as, mu_ss, scat_events, jones_partial, s1, s2, m11, m12,co_xy
                         ):
    threadIdx = cuda.threadIdx
    blockIdx = cuda.blockIdx
    blockDim = cuda.blockDim
    thread_idx = blockDim.x * blockIdx.x + threadIdx.x


    polarized_photon_linear(seed, incident_degree, n,reflection, zstokes, random_nums[thread_idx], U, W,
                            jones, mu_as, mu_ss, s1, s2, m11, m12,
                            scat_events, jones_partial,
                            co_xy,thread_idx)
