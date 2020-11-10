import numpy as np
from numba import jitclass  # import the decorator
from numba import int32, float32  # import the types
from numba import cuda
import numba as nb
import cmath
from numba import typed, types
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states, init_xoroshiro128p_states
import math
import operator

NNxy = 100
NNr = 50
NNz = 50
dr = 0.00009980039920159681
dx = 0.00009990009990009990
dy = 0.00009990009990009990
dz = 0.00009980039920159681
collection_rad = 0.05000
slabsize = 1.00000

g_ddivs=64

y_0 = 2e-5
y_1 = 3e-4
y_2 = 15e-5
y_3 = 2e-4
y_4 = 2e-4
y_5 = 3e-4

y_0_t = y_0
y_1_t = y_0 + y_1
y_2_t = y_1_t + y_2
y_3_t = y_2_t + y_2
y_4_t = y_3_t + y_3
y_5_t = y_4_t + y_4

y_array = [y_0_t, y_1_t, y_2_t, y_3_t, y_4_t, y_5_t]



# debug=True
@cuda.jit(device=True, inline=True,debug=True)
def polarized_photon(seed, random_nums,U, W, jones, mu_as, mu_ss, positions, s1, s2, m11, m12, scat_events, jones_partial, co_xy,
                     cross_xy, incoh_cross_xy, co_rz, cross_rz, incoh_cross_rz, co_xy_trad, cross_xy_trad,
                     incoh_cross_xy_trad, co_rz_trad, cross_rz_trad, incoh_cross_rz_trad, i_stokes_rz,q_stokes_rz,u_stokes_rz,v_stokes_rz,collected, rng_states,
                     thread_idx):
    # NNxy = 100
    # NNr = 50
    # NNz = 50
    NNxy = len(cross_xy[0, 0, :])
    NNr = len(cross_rz[0, :, 0])
    NNz = len(cross_rz[0, 0, :])
    dr = 4e-4
    dx = 4e-4
    dy = 4e-4
    dz = 4e-4
    collection_rad = NNxy / 2 * dx
    slabsize = 1.00000
    n_steps=100

    degree_divs = g_ddivs

    jones[thread_idx, 0] = 1
    jones[thread_idx, 1] = 0
    jones[thread_idx, 2] = 0
    jones[thread_idx, 3] = 1

    jones_partial[thread_idx, 0] = 1
    jones_partial[thread_idx, 1] = 0
    jones_partial[thread_idx, 2] = 0
    jones_partial[thread_idx, 3] = 1

    U[thread_idx, 0] = 0
    U[thread_idx, 1] = 0
    U[thread_idx, 2] = 1

    x = cuda.local.array(1, dtype=nb.float32)
    y = cuda.local.array(1, dtype=nb.float32)
    z = cuda.local.array(1, dtype=nb.float32)
    x[0] = positions[thread_idx, 0]  # UPDATE in the end
    y[0] = positions[thread_idx, 1]  # UPDATE in the end
    z[0] = positions[thread_idx, 2]  # UPDATE in the end

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
    max_depth[0] = 0
    B0[0] = 1
    B0[1] = 0
    B0[2] = 0
    n_o = 1.33
    n_e = 1.33
    lambda_0 = 632.8e-9
    chi = 0.0+5.36e-5
    # jones_partial=cuda.local.array(4)
    # jones_partial[0]=jones_partial_in[thread_idx,0]
    # jones_partial=jones_partial_in[thread_idx]
    albedo = mu_s / (mu_s + mu_a)
    # rng_state_gpu = cuda.local.array(5, dtype=float32)
    # init_xoroshiro128p_states()
    # r=np.random.rand()
    S0[0] = 1
    S0[1] = 1
    S0[2] = 0
    S0[3] = 0

    for i in range(4):
        S[i] = S0[i]

    for step in range(n_steps):

        #get_rand(rand, rng_states, thread_idx + step + nb.int16(scat_event)+seed)
        rand[0]=random_nums[step+0]
        s[0] = -cmath.log(rand[0]).real / (mu_a + mu_s)
        # print('s',s[0])
        if step > 0:
            positions[thread_idx, 0] = x[0]
            positions[thread_idx, 1] = y[0]
            positions[thread_idx, 2] = z[0]

        if scat_event == 0:
            # print('scat event')
            U_new[0] = 0
            U_new[1] = 0
            U_new[2] = 1
            U_epar[0] = 1
            U_epar[1] = 0
            U_epar[2] = 0

            birefringence_rot(s[0], chi, lambda_0, n_o, n_e, jones, rng_states, thread_idx, B0, U_new, U_epar,random_nums,step+1)

            # print('jones after 1st bifringe', jones[thread_idx, 0].real, jones[thread_idx, 0].imag)
            EtoQ(jones[thread_idx, 0], jones[thread_idx, 2], temp)
            S[1] = temp[0]
            EtoI(jones[thread_idx, 0], jones[thread_idx, 2], temp)
            S[1] /= temp[0]

            EtoU(jones[thread_idx, 0], jones[thread_idx, 2], temp)
            S[2] = temp[0]
            EtoI(jones[thread_idx, 0], jones[thread_idx, 2], temp)
            S[2] /= temp[0]
            # print('S2', S[2].real, S[2].imag)

        x[0] += U[thread_idx, 0] * s[0]
        y[0] += U[thread_idx, 1] * s[0]
        z[0] += U[thread_idx, 2] * s[0]
        scat_event += 1
        print('z',step, z[0], U[thread_idx, 2] )

        if z[0].real > max_depth[0]:
            max_depth[0] = z[0].real
        W_local = W_local *albedo       # partial photon
        # print('here')
        if z[0].real >= 0:
            theta_partial = cmath.acos(-U[thread_idx, 2]).real
            itheta_deg = int(theta_partial * (degree_divs - 1) / cmath.pi)
            if itheta_deg>degree_divs - 1:
                print('err')
                break
            phi_partial = 0
            #todo normalize I
            I[0] = m11[itheta_deg] + m12[itheta_deg] * (S[1] * cmath.cos(2 * phi_partial).real) / S[0]
            I[0] = (W_local * I[0] * (cmath.exp(-(mu_a + mu_s).real * z[0].real)).real).real
            backscatter_intensity = 2 * 2 * I[0]

            if 1:  # abs(I) > 1e-12:
                # print('itheta_deg',itheta_deg)
                # print('s2[itheta_deg]',s2[itheta_deg].real)
                jones_partial[thread_idx, 0] = jones[thread_idx, 0] * s2[itheta_deg]
                jones_partial[thread_idx, 1] = jones[thread_idx, 1] * s2[itheta_deg]
                jones_partial[thread_idx, 2] = jones[thread_idx, 2] * s1[itheta_deg]
                jones_partial[thread_idx, 3] = jones[thread_idx, 3] * s1[itheta_deg]

                U_new[0] = 0
                U_new[1] = 0
                U_new[2] = -1
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

                # pass on U_epar[0]!=U_epar[0]
                if U[thread_idx, 0] == 0 and U[thread_idx, 1] > 0:
                    phi_partial = cmath.pi / 2
                elif U[thread_idx, 0] == 0 and U[thread_idx, 1] < 0:
                    phi_partial = -cmath.pi / 2
                elif U[thread_idx, 0] == 0 and U[thread_idx, 1] == 0:
                    phi_partial = 0
                else:
                    phi_partial = cmath.atan(U[thread_idx, 1] / U[thread_idx, 0]).real
                #todo check birefringance
                rotate_phi(jones_partial, phi_partial, thread_idx)
                # print('test')
                # get polarization measurments

                EtoQ(jones_partial[thread_idx, 0], jones_partial[thread_idx, 2], outEQ)
                EtoI(jones_partial[thread_idx, 0], jones_partial[thread_idx, 2], outEI)
                # print('jones0',jones_partial[thread_idx, 0].real)
                # print('jones2', jones_partial[thread_idx, 2].real)
                # print('I',outEI[0].real)
                if outEI[0]==0:
                    print('err I')
                    break
                temp3 = outEQ[0] / outEI[0]
                # print('test')
                #todo times I partial ?
                co = (I[0] * (1 + temp3)).real
                cross = (I[0] * (1 - temp3)).real
                # print('co', co)
                if scat_event == 1:
                    co_rz[thread_idx, NNr - 1, NNz - 1] += co
                    incoh_cross_rz[thread_idx, NNr - 1, NNz - 1] += cross
                    co_xy[thread_idx, 0, 0] += co
                    incoh_cross_xy[thread_idx, 0, 0] += cross
                else:
                    Efor = jones_partial[thread_idx, 2]
                    Erev = -jones_partial[thread_idx, 1]

                    get_degree_coherence(Efor, Erev, coherence)
                    ir = int(cmath.sqrt(pow(x[0], 2) + pow(y[0], 2)).real / dr)
                    # print('x',x[0], 'y',y[0],'z',z[0],'ir',ir, 'dr',dr)
                    if ir > NNr - 1:
                        ir = NNr - 1
                    #todo check iz
                    iz = int(max_depth[0] / dz)
                    iz=int(z[0]/dz)
                    if iz > NNz - 1:
                        iz = NNz -1
                    if iz <0:
                        iz=0

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
                    # print(ix)
                    # print('x', x[0], 'y', y[0], 'z', z[0], 'ir', ir, 'ix',ix,'iy',iy,'dr', dr)
                    i_stokes_rz[thread_idx, ir, iz]+= S[0]
                    q_stokes_rz[thread_idx, ir, iz] += S[1]
                    u_stokes_rz[thread_idx, ir, iz] += S[2]
                    v_stokes_rz[thread_idx, ir, iz] += S[3]

                    incoh_cross_rz[thread_idx, ir, iz] += cross
                    co_rz[thread_idx, ir, iz] += co
                    cross_rz[thread_idx, ir, iz] += cross * coherence[0]

                    incoh_cross_xy[thread_idx, iy, ix] += cross
                    co_xy[thread_idx, iy, ix] += co
                    cross_xy[thread_idx, iy, ix] += cross * coherence[0]

        # print('here')
        # Collect

        # if z[0] < 0:
        #     deg = abs(-cmath.pi - cmath.atan(
        #         -cmath.sqrt(U[thread_idx, 0] * U[thread_idx, 0] + U[thread_idx, 1] * U[thread_idx, 1]).real / U[
        #             thread_idx, 2]).real)
        #
        #     # print('neg', z[0], 'deg', deg,'iteration',i)
        if U[thread_idx,2] <1e-12:
            degree=cmath.pi/2*180
        else:
            degree = abs(cmath.atan(cmath.sqrt(U[thread_idx, 0] * U[thread_idx, 0] + U[thread_idx, 1]*U[thread_idx, 1]).real/U[thread_idx, 2]).real*180/cmath.pi)
        if z[0] < 0 and degree < 10:
            # print('collect')
            x[0] -= U[thread_idx, 0] * s[0]
            y[0] -= U[thread_idx, 1] * s[0]
            z[0] -= U[thread_idx, 2] * s[0]
            scat_event -= 1

            for i in range(3):
                U[thread_idx, i] = U_prev[i]

            for i in range(4):
                jones[thread_idx, i] = jones_partial[thread_idx, i]

            theta_partial = cmath.acos(-U[thread_idx, 2]).real
            itheta_deg = int(theta_partial / cmath.pi * degree_divs)
            phi_partial = 0

            jones[thread_idx, 0] = jones[thread_idx, 0] * s2[itheta_deg]
            jones[thread_idx, 1] = jones[thread_idx, 1] * s2[itheta_deg]
            jones[thread_idx, 2] = jones[thread_idx, 2] * s1[itheta_deg]
            jones[thread_idx, 3] = jones[thread_idx, 3] * s1[itheta_deg]

            U_new[0] = 0
            U_new[1] = 0
            U_new[2] = -1

            if U[thread_idx, 2] == 1 or U[thread_idx, 2] == -1:
                pass
            else:
                temp2 = U[thread_idx, 0] * U_new[0] + U[thread_idx, 1] * U_new[1] + U[thread_idx, 2] * U_new[2]
                U_epar[0] = U_new[0] - U[thread_idx, 0] * temp2
                U_epar[1] = U_new[1] - U[thread_idx, 1] * temp2
                U_epar[2] = U_new[2] - U[thread_idx, 2] * temp2
                temp2 = cmath.sqrt(pow(U_epar[0], 2) + pow(U_epar[1], 2) + pow(U_epar[2], 2)).real
                U_epar[0] = U_epar[0] / temp2
                U_epar[1] = U_epar[1] / temp2
                U_epar[2] = U_epar[2] / temp2

            phi = cmath.atan(U[thread_idx, 1] / U[thread_idx, 0])
            rotate_phi(jones, phi, thread_idx)

            temp4 = cuda.local.array(1, dtype=nb.float32)

            EtoQ(jones[thread_idx, 0], jones[thread_idx, 2], temp4)
            S[1] = temp4[0]
            EtoQ(jones[thread_idx, 0], jones[thread_idx, 2], temp4)
            S[1] /= temp4[0]

            EtoU(jones[thread_idx, 0], jones[thread_idx, 2], temp4)
            S[2] = temp4[0]
            EtoQ(jones[thread_idx, 0], jones[thread_idx, 2], temp4)
            S[2] /= temp4[0]
            EtoQ(jones_partial[thread_idx, 0], jones_partial[thread_idx, 2], temp4)
            temp5 = temp4[0]
            EtoI(jones_partial[thread_idx, 0], jones_partial[thread_idx, 2], temp4)
            temp5 /= temp4[0]
            co = (I[0] * (1 + temp5)).real
            cross = (I[0] * (1 - temp5)).real
            # print(co.real)
            if scat_event == 1:
                i_stokes_rz[thread_idx, 0, 0] += S[0]
                q_stokes_rz[thread_idx, 0, 0] += S[1]
                u_stokes_rz[thread_idx, 0, 0] += S[2]
                v_stokes_rz[thread_idx, 0, 0] += S[3]

                co_rz_trad[thread_idx, NNr - 1, NNz - 1] += co
                incoh_cross_rz_trad[thread_idx, NNr - 1, NNz - 1] += cross
                co_xy_trad[thread_idx, 0, 0] += co
                incoh_cross_xy_trad[thread_idx, 0, 0] += cross

            else:
                Efor = jones_partial[thread_idx, 2]
                Erev = -jones_partial[thread_idx, 1]
                coherence = cuda.local.array(1, nb.float32)
                get_degree_coherence(Efor, Erev, coherence)
                ir = int(cmath.sqrt(pow(x[0], 2) + pow(y[0], 2)).real / dr)

                if ir > NNr - 1:
                    ir = NNr - 1

                iz = int(max_depth[0] / dz)
                if iz > NNz - 1:
                    iz = NNz - 1

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
                # print('ir', ir, 'iz', iz, 'cross', cross)
                i_stokes_rz[thread_idx, ir, iz] += S[0]
                q_stokes_rz[thread_idx, ir, iz] += S[1]
                u_stokes_rz[thread_idx, ir, iz] += S[2]
                v_stokes_rz[thread_idx, ir, iz] += S[3]

                incoh_cross_rz_trad[thread_idx, ir, iz] += cross
                co_rz_trad[thread_idx, ir, iz] += co
                cross_rz_trad[thread_idx, ir, iz] += cross * coherence[0]

                incoh_cross_xy_trad[thread_idx, iy, ix] += cross
                co_xy_trad[thread_idx, iy, ix] += co
                cross_xy_trad[thread_idx, iy, ix] += cross * coherence[0]
            break
        # todo kill photon

        if z[0] < 0 or z[0] > slabsize:
            #print('kill')
            break  # kill photon

        # todo if still alive
        if 1:
            # print('move another time')
            for i in range(4):
                jones_partial[i] = jones[thread_idx, i]

            for i in range(3):
                U_prev[i] = U[thread_idx, i]

            # get_rand(rand, rng_states, thread_idx + 3 + int(scat_event))
            #
            # # rejection method
            # itheta_deg = int(rand[0] * degree_divs) - 1
            #
            # get_rand(rand, rng_states, thread_idx + 4 + int(scat_event))
            # rand[0] = 2 * (rand[0] - 0.5)
            # phi = rand[0] * cmath.pi
            #
            # I[0] = m11[itheta_deg] + m12[itheta_deg] * (S[1] * cmath.cos(2 * phi) + S[2] * cmath.sin(2 * phi)) / S[0]
            # # print('last if I', I[0].real, I[0].imag)
            # while 0:  # todo choose random number uniformly distributed between 0 and I0
            #     pass  # choose randomly
            # print(len(random_nums))
            random_by_pf(S, m11, m12, theta_scat, phi_scat, rng_states, random_nums,step+2)
            # theta_scat[0]=random_nums[step]*cmath.pi
            # phi_scat[0]=random_nums[step+1]*2*cmath.pi

            # if theta_scat[0]>cmath.pi:
            #     print('theta big', theta_scat[0])
            # if theta_scat[0]<0:
            #     print('theta neg')
            # if phi_scat[0]>2*cmath.pi:
            #     print('phi big', phi_scat[0])
            # if phi_scat[0]<0:
            #     print('phi neg')
            # theta = itheta_deg * np.pi / degree_divs
            # print('theta', theta_scat[0])
            update_u(U, theta_scat[0], phi_scat[0], U_new, thread_idx)
            rotate_phi(jones, phi_scat[0], thread_idx)

            itheta_deg = int(theta_scat[0] / cmath.pi * (degree_divs - 1))
            # print('itheta',itheta_deg)
            # if itheta_deg>degree_divs - 1:
            #     break
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

            birefringence_rot(s[0], chi, lambda_0, n_o, n_e, jones, rng_states, thread_idx, B0, U_new, U_epar, random_nums,step+5)

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
                jones[thread_idx, 0].imag, 2) + pow(jones[thread_idx, 1].imag, 2) + pow(jones[thread_idx, 2].imag, 2) + pow( jones[thread_idx, 3].imag, 2)
            temp5=cmath.sqrt(temp5).real
            for i in range(4):
                jones[thread_idx, i] /= temp5



            EtoQ(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            S[1] = temp6[0]
            EtoI(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            S[1] /= temp6[0]

            EtoU(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            S[2] = temp6[0]
            EtoI(jones[thread_idx, 0], jones[thread_idx, 2], temp6)
            S[2] /= temp6[0]

            # todo kill photon with russian roulette

    # W[thread_idx] = W_local
    # positions[thread_idx, 0] = x[0]
    # positions[thread_idx, 1] = y[0]
    # positions[thread_idx, 2] = z[0]

#consumes 2 seeds
@cuda.jit(device=True, inline=True,debug=True)
def random_by_pf(stokes, s11, s12, theta_out, phi_out, rng_states, random_nums,seed):
    ddivs = g_ddivs
    pdivs = 60
    rand_0 = cuda.local.array(1, dtype=nb.float32)
    rand_1 = cuda.local.array(1, dtype=nb.float32)
    pf = cuda.local.array((ddivs,  ddivs), dtype=nb.float32)
    pf_theta = cuda.local.array(ddivs, dtype=nb.float32)
    pf_phi = cuda.local.array( ddivs, dtype=nb.float32)
    p_theta_cs = cuda.local.array(ddivs, dtype=nb.float32)
    p_phi_cs = cuda.local.array( ddivs, dtype=nb.float32)
    inverse_p_theta = cuda.local.array(pdivs, dtype=nb.float32)
    inverse_p_phi = cuda.local.array( pdivs, dtype=nb.float32)

    for i in range(ddivs):
        for j in range( ddivs):
            # theta = i * cmath.pi / ddivs
            # phi = j * cmath.pi * 2 / ddivs
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

    # get_rand(rand_0, rng_states, seed + 5)
    # get_rand(rand_1, rng_states, seed + 6)

    # rand_0[0]=random_nums[seed+1]
    # rand_1[0] = random_nums[seed + 2]
    # print('rand0',int(rand_0[0] * (pdivs - 1)),'shape',len(inverse_p_theta))
    # print('rand1', int(rand_1[0] * (pdivs - 1)), 'shape', len(inverse_p_phi))
    theta_out[0] = inverse_p_theta[int(random_nums[seed+1] * (pdivs - 1))] * cmath.pi / ddivs
    phi_out[0] = inverse_p_phi[int(random_nums[seed + 2] * (pdivs - 1))] * 2 * cmath.pi / ddivs
    # print('theta',theta_out[0])
    # print('phi',phi_out[0])


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
    out[0] = 2 * (Epar.real * Eper.imag - Epar.real * Eper.imag)


@cuda.jit(device=True, inline=True)
def rotate_phi(jones, phi, thread_idx):
    cosphi = cmath.cos(phi).real
    sinphi = cmath.sin(phi).real
    jones[thread_idx,0] = jones[thread_idx, 0] * cosphi + jones[thread_idx, 2] * sinphi
    jones[thread_idx,1] = jones[thread_idx, 1] * cosphi + jones[thread_idx, 3] * sinphi
    jones[thread_idx,2] = -jones[thread_idx, 0] * sinphi + jones[thread_idx, 2] * cosphi
    jones[thread_idx,3] = -jones[thread_idx, 1] * sinphi + jones[thread_idx, 3] * cosphi


@cuda.jit(device=True, inline=True)
def get_rand(rand, rng_states, seed):
    rand[0] = xoroshiro128p_uniform_float32(rng_states, seed)
    i=0
    while rand[0] == 0:
        print('gen random',rand[0])
        rand[0] = xoroshiro128p_uniform_float32(rng_states, seed + i)
        i+=1



@cuda.jit(device=True, inline=True)
def update_u(U, theta, phi, U_new, thread_idx):
    # costheta = cmath.cos(theta).real
    # sintheta = cmath.sin(theta).real
    # cosphi = cmath.cos(phi).real
    # sinphi = cmath.sin(phi).real
    #
    # if 1 - abs(U[thread_idx, 2]) < 1e-12:
    #     ux = sintheta * cosphi
    #     uy = sintheta * sinphi
    #     uz = costheta
    #     # print('abs')
    # else:
    #     if U[thread_idx, 0] != 0:
    #         phi0 = cmath.atan(U[thread_idx, 1] / U[thread_idx, 0]).real
    #     else:
    #         phi0 = cmath.pi / 2
    #
    #     theta0 = cmath.acos(U[thread_idx, 2]).real
    #
    #     # phi_fin = phi0 + phi
    #     # theta_fin = theta0 + theta
    #     phi_fin =  phi
    #     theta_fin =  theta
    #
    #     ux = cmath.sin(theta_fin).real * cmath.cos(phi_fin).real
    #     uy = cmath.sin(theta_fin).real * cmath.sin(phi_fin).real
    #     uz = cmath.cos(theta_fin).real
    #
    #     temp=cmath.sqrt(pow(ux,2)+pow(uy,2)+pow(uz,2)).real
    #     ux/=temp
    #     uy /= temp
    #     uz /= temp
        # temp = cmath.sqrt(1 - U[thread_idx, 2] * U[thread_idx, 2]).real
        # ux = sintheta * (U[thread_idx, 0] * U[thread_idx, 2] * cosphi - U[thread_idx, 1] * sinphi) / temp + U[
        #     thread_idx, 0] * costheta
        # uy = sintheta * (U[thread_idx, 1] * U[thread_idx, 2] * cosphi + U[thread_idx, 1] * sinphi) / temp + U[
        #     thread_idx, 1] * costheta
        # uz = -sintheta * cosphi * temp + U[thread_idx, 2] * costheta
    # U_new[0] = ux
    # U_new[1] = uy
    # U_new[2] = uz
    ux = U[thread_idx,0]
    uy = U[thread_idx,1]
    uz = U[thread_idx,2]

    costheta = cmath.cos(theta).real
    sintheta = cmath.sqrt(1.0 - costheta * costheta).real
    cosphi = cmath.cos(phi).real
    if (phi < cmath.pi):
        sinphi = cmath.sqrt(1.0 - cosphi * cosphi).real
    else:
        sinphi = -cmath.sqrt(1.0 - cosphi * cosphi).real
    sign=1 if uz>= 0 else -1
    if (1 - abs(uz) <= 1.0E-12):
        uxx = sintheta * cosphi
        uyy = sintheta * sinphi
        uzz = costheta * sign
    else:
        temp = cmath.sqrt(1.0 - uz * uz).real
        uxx = sintheta * (ux * uz * cosphi - uy * sinphi) / temp + ux * costheta
        uyy = sintheta * (uy * uz * cosphi + ux * sinphi) / temp + uy * costheta
        uzz = -sintheta * cosphi * temp + uz * costheta
    temp=cmath.sqrt(pow(uxx,2)+pow(uyy,2)+pow(uzz,2)).real
    uxx=uxx/temp
    uyy=uyy/temp
    uzz=uzz/temp
    U_new[0] = uxx
    U_new[1] = uyy
    U_new[2] = uzz
    # temp = cmath.sqrt(pow(U_new[0], 2) + pow(U_new[1], 2) + pow(U_new[2] , 2)).real
    # temp2=U_new[0]*U_new[0]+U_new[1]*U_new[1]+U_new[2]*U_new[2]
    # print('U_new after', U_new[ 0], U_new[ 1], U_new[ 2],temp)

#consumes 2 seeds
# todo check this function for bugs
@cuda.jit(device=True, inline=True)
def birefringence_rot(s, chi, lambda_0, n_o, n_e, jones, rng_states, thread_idx, B, U, U_epar,rand_nums,seed):
    if n_o != n_e or chi != 0:
        if B[0] == 2.0 and B[1] == 2.0 and B[2] == 2.0:
            # rand_num = xoroshiro128p_uniform_float32(rng_states, thread_idx)
            rand_num = rand_nums[seed]
            theta = cmath.pi * rand_num
            # rand_num = xoroshiro128p_uniform_float32(rng_states, thread_idx + 1)
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
            m[0] = (n_mat[0] - n_mat[1]) * cmath.sinh(Q_n * s) / (2 * Q_n) + cmath.cosh(Q_n * s)
            m[1] = (n_mat[0] - n_mat[1]) * cmath.sinh(Q_n * s) / (2 * Q_n) + cmath.cosh(Q_n * s)
            m[2] = (n_mat[2]) * cmath.sinh(Q_n * s) / (Q_n)
            m[3] = -m[2]

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


@cuda.jit(device=True, inline=True)
def region_info(position_x, position_y, region_number, mu_s, mu_a, g, n, thread_idx):
    # from [12]kirilin2010
    # upper stratum corneum
    # position_y = position_ys[thread_idx]
    y_0_p = y_0
    mu_s_0 = 35000.0
    mu_a_0 = 20.0
    g_0 = 0.9
    n_0 = 1.54
    # lower stratum corneum
    y_1_p = y_0_p + y_1
    mu_s_1 = 5000.0
    mu_a_1 = 15.0
    g_1 = 0.95
    n_1 = 1.34
    # epidermis
    y_2_p = y_1_p + y_2
    mu_s_2 = 12000.0
    mu_a_2 = 20.0
    g_2 = 0.85
    n_2 = 1.4
    # upper dermis with plexus
    y_3_p = y_2_p + y_3
    mu_s_3 = 12000.0
    mu_a_3 = 100.0
    g_3 = 0.9
    n_3 = 1.39
    # dermis
    y_4_p = y_3_p + y_4
    mu_s_4 = 7000.0
    mu_a_4 = 70.0
    g_4 = 0.87
    n_4 = 1.4
    # lower dermis with plexus
    y_5_p = y_4_p + y_5
    mu_s_5 = 12000.0
    mu_a_5 = 200.0
    g_5 = 0.95
    n_5 = 1.39

    if position_y < y_0_p:  # and position_y>0:
        region_number[0, 0] = 0
        mu_s[0, 0] = mu_s_0
        mu_a[0, 0] = mu_a_0
        g[0, 0] = g_0
        n[0, 0] = n_0
    elif position_y >= y_0_p and position_y < y_1_p:
        region_number[0, 0] = 1
        mu_s[0, 0] = mu_s_1
        mu_a[0, 0] = mu_a_1
        g[0, 0] = g_1
        n[0, 0] = n_1
    elif position_y >= y_1_p and position_y < y_2_p:
        region_number[0, 0] = 2
        mu_s[0, 0] = mu_s_2
        mu_a[0, 0] = mu_a_2
        g[0, 0] = g_2
        n[0, 0] = n_2
    elif position_y >= y_2_p and position_y < y_3_p:
        region_number[0, 0] = 3
        mu_s[0, 0] = mu_s_3
        mu_a[0, 0] = mu_a_3
        g[0, 0] = g_3
        n[0, 0] = n_3
    elif position_y >= y_3_p and position_y < y_4_p:
        region_number[0, 0] = 4
        mu_s[0, 0] = mu_s_4
        mu_a[0, 0] = mu_a_4
        g[0, 0] = g_4
        n[0, 0] = n_4
    elif position_y >= y_4_p and position_y < y_5_p:
        region_number[0, 0] = 5
        mu_s[0, 0] = mu_s_5
        mu_a[0, 0] = mu_a_5
        g[0, 0] = g_5
        n[0, 0] = n_5
    else:
        region_number[0, 0] = 6


@cuda.jit(device=True, inline=True)
def fresnel_reflection(incident_theta, normal_theta, n_1, n_2, transfer_theta):
    i_theta = normal_theta - incident_theta
    transfer_theta[0, 0] = cmath.asin(n_1 / n_2 * cmath.sin(i_theta)).real
    transfer_theta[0, 0] = normal_theta - transfer_theta[0, 0]
    # print('inc: ',i_theta, transfer_theta[0,0])


@cuda.jit(device=True, inline=True)
def fresnel_prob(incident_theta, transfer_theta, thread_idx, n_1, n_2, modes, prob):
    critical_angle = cmath.asin(n_2 / n_1).real
    if n_2 < n_1 and incident_theta > critical_angle:
        prob[0] = 1.0
        return

    if modes[thread_idx] == 0.0:  # non-polarized
        r = (1 / 2 * ((cmath.sin(incident_theta - transfer_theta) * cmath.sin(incident_theta - transfer_theta)) / (
                cmath.sin(incident_theta + transfer_theta) * cmath.sin(incident_theta + transfer_theta)) + (
                              cmath.tan(incident_theta - transfer_theta) * cmath.tan(
                          incident_theta - transfer_theta)) / (cmath.tan(incident_theta + transfer_theta) * cmath.tan(
            incident_theta + transfer_theta)))).real
        if r * r > 1.0:
            r = 1.0
        prob[0] = r * r
    elif modes[thread_idx] == 1.0:  # p-polarized
        brewster_angle = cmath.atan(n_2 / n_1)
        r = ((n_1 * cmath.cos(transfer_theta) - n_2 * cmath.cos(incident_theta)) / (
                n_1 * cmath.cos(transfer_theta) + n_2 * cmath.cos(incident_theta))).real
        if r * r > 1.0:
            r = 1.0

        prob[0] = r * r
    elif modes[thread_idx] == 2.0:  # s-polarized
        r = ((n_1 * cmath.cos(incident_theta) - n_2 * cmath.cos(transfer_theta)) / (
                n_1 * cmath.cos(incident_theta) + n_2 * cmath.cos(transfer_theta))).real
        if r * r > 1.0:
            r = 1.0

        prob[0] = r * r

#1 seed
@cuda.jit(device=True, inline=True)
def henyey_greenstein(g, func_val, rng_states, random_nums,seed,thread_idx):
    rand_num=random_nums[seed]
    if g[0, 0] != 0.0:
        # rand_num = xoroshiro128p_uniform_float32(rng_states, thread_idx)
        # rand_num = xoroshiro128p_uniform_float32(rng_states, int(rand_num) * 100 + thread_idx)
        func_val[0, 0] = math.acos(1.0 / 2.0 / g[0, 0] * (1.0 + g[0, 0] * g[0, 0] - (1.0 - g[0, 0] * g[0, 0]) / (
            pow(1.0 - g[0, 0] + 2 * g[0, 0] * rand_num, 2.0))))
    else:
        # rand_num = xoroshiro128p_uniform_float32(rng_states, thread_idx)
        func_val[0, 0] = math.acos(2.0 * rand_num - 1.0)


@cuda.jit(device=True, inline=True)
def collision_handling(amplitudes, old_position_xs, old_position_ys, new_position_x, new_position_y, length,
                       is_collision, old_directions, new_directions, thread_idx, adjusted_dist, collected, modes,random_nums, seed,
                       rng_states):
    y_0 = 2e-5
    y_1 = 3e-4
    y_2 = 15e-5
    y_3 = 2e-4
    y_4 = 2e-4
    y_5 = 3e-4

    y_0_t = y_0
    y_1_t = y_0_t + y_1
    y_2_t = y_1_t + y_2
    y_3_t = y_2_t + y_3
    y_4_t = y_3_t + y_4
    y_5_t = y_4_t + y_5

    y_array = cuda.local.array(6, dtype=float32)
    y_array[0] = y_0_t
    y_array[1] = y_1_t
    y_array[2] = y_2_t
    y_array[3] = y_3_t
    y_array[4] = y_4_t
    y_array[5] = y_5_t

    region_number_old = cuda.local.array((1, 1), dtype=float32)
    mu_s_old = cuda.local.array((1, 1), dtype=float32)
    mu_a_old = cuda.local.array((1, 1), dtype=float32)
    g_old = cuda.local.array((1, 1), dtype=float32)
    n_old = cuda.local.array((1, 1), dtype=float32)
    region_info(old_position_xs[thread_idx], old_position_ys[thread_idx], region_number_old, mu_s_old, mu_a_old, g_old,
                n_old, thread_idx)

    region_number_new = cuda.local.array((1, 1), dtype=float32)
    mu_s_new = cuda.local.array((1, 1), dtype=float32)
    mu_a_new = cuda.local.array((1, 1), dtype=float32)
    g_new = cuda.local.array((1, 1), dtype=float32)
    n_new = cuda.local.array((1, 1), dtype=float32)
    region_info(new_position_x[0, 0], new_position_y[0, 0], region_number_new, mu_s_new, mu_a_new, g_new, n_new,
                thread_idx)
    # print(region_number_old[0,0],' ',region_number_new[0,0])
    mu_t = mu_a_old[0, 0] + mu_s_old[0, 0]
    # amplitudes[thread_idx] -= 0.5
    if region_number_old[0, 0] != region_number_new[0, 0]:
        is_collision[0, 0] = 1.0
        y_lower = y_array[int(region_number_new[0, 0])]
        y_upper = y_array[int(region_number_new[0, 0] - 1)]
        if old_directions[thread_idx] > 0 and old_directions[thread_idx] < cmath.pi:
            delta_y = y_upper - old_position_ys[thread_idx]
        elif old_directions[thread_idx] < 0 and old_directions[thread_idx] > -cmath.pi:
            delta_y = y_lower - old_position_ys[thread_idx]
        # print(delta_y)
        if (cmath.cos(old_directions[thread_idx])) == 0:
            delta_x = 0
            distance = delta_y
            # print('yes')
        else:
            delta_x = delta_y * math.tan(old_directions[thread_idx])
            distance = delta_y / math.sin(old_directions[thread_idx])
            # print('no')
        # todo
        transfer_theta = cuda.local.array((1, 1), dtype=float32)
        fresnel_reflection(old_directions[thread_idx], math.pi / 2, n_old[0, 0], n_new[0, 0], transfer_theta)
        reflection_prob = cuda.local.array((1, 1), dtype=float32)
        fresnel_prob(cmath.pi / 2 - old_directions[thread_idx], cmath.pi / 2 - transfer_theta[0, 0], thread_idx,
                     n_old[0, 0], n_new[0, 0], modes, reflection_prob)

        amplitudes[thread_idx] *= math.exp(-mu_t * distance)
        adjusted_dist[thread_idx] += distance * math.sqrt(n_old[0, 0])

        length[0, 0] -= distance

        # print(length[0, 0])
        if length[0, 0] < 0:
            # print(length[0,0])
            length[0, 0] = 0
        adjusted_dist[thread_idx] += length[0, 0] * math.sqrt(n_new[0, 0])
        rand_num=random_nums[seed]
        # rand_num = xoroshiro128p_uniform_float32(rng_states, thread_idx)
        # if reflection_prob[0, 0]>0:
        #     print('fishy', reflection_prob[0, 0],old_directions[thread_idx])
        # print(reflection_prob[0, 0], rand_num)
        if reflection_prob[0, 0] > rand_num:
            # print('yes')
            mu_t = mu_a_new[0, 0] + mu_s_new[0, 0]
            # print(mu_t)
            # print(old_directions[thread_idx], transfer_theta[0, 0])
            old_directions[thread_idx] = transfer_theta[0, 0]

            amplitudes[thread_idx] *= math.exp(-mu_t * length[0, 0])
        else:
            # print('no')
            old_directions[thread_idx] = -old_directions[thread_idx]
            amplitudes[thread_idx] *= math.exp(-mu_t * length[0, 0])

        old_position_xs[thread_idx] = old_position_xs[thread_idx] + delta_x + length[0, 0] * math.cos(
            old_directions[thread_idx])
        old_position_ys[thread_idx] = old_position_ys[thread_idx] + delta_y + length[0, 0] * math.sin(
            old_directions[thread_idx])

        amplitudes[thread_idx] *= math.exp(-mu_t * length[0, 0])
    else:
        is_collision[0, 0] = 0.0
        old_position_xs[thread_idx] = old_position_xs[thread_idx] + math.cos(old_directions[thread_idx]) * length[0, 0]
        old_position_ys[thread_idx] = old_position_ys[thread_idx] + math.sin(old_directions[thread_idx]) * length[0, 0]
        adjusted_dist[thread_idx] += length[0, 0] * math.sqrt(n_old[0, 0])
        amplitudes[thread_idx] *= math.exp(-mu_t * length[0, 0])

#3 seeds
@cuda.jit(device=True, inline=True)
def scattering_direction(old_directions, g, rng_states, random_nums,seed,thread_idx):
    # rand_num = xoroshiro128p_uniform_float32(rng_states, thread_idx)
    rand_num=random_nums[seed]
    sign = 1.0 if rand_num > 0.5 else -1.0
    change = cuda.local.array((1, 1), dtype=float32)
    henyey_greenstein(g, change, rng_states, random_nums,seed+1,thread_idx)
    change[0, 0] = sign * change[0, 0]
    old_directions[thread_idx] = old_directions[thread_idx] + change[0, 0]


@cuda.jit(device=True, inline=True)
def step_length(mu_t, length, rng_states, random_nums,seed,thread_idx):
    # rand_num = xoroshiro128p_uniform_float32(rng_states, thread_idx)
    rand_num=random_nums[seed]
    r = cuda.local.array((1, 1), dtype=float32)
    r[0, 0] = - math.log(rand_num)
    length[0, 0] = operator.truediv(r[0, 0], mu_t[0, 0])


@cuda.jit(device=True, inline=True)
def take_step(amplitudes, old_position_xs, old_position_ys, directions, length, new_position_x, new_position_y,
              thread_idx, adjusted_dist, collected, modes, random_nums, seed,rng_states):
    new_position_x[0, 0] = old_position_xs[thread_idx] + math.cos(directions[thread_idx]) * length[0, 0]
    new_position_y[0, 0] = old_position_ys[thread_idx] + math.sin(directions[thread_idx]) * length[0, 0]
    is_collision = cuda.local.array((1, 1), dtype=float32)
    new_direction = cuda.local.array((1, 1), dtype=float32)
    collision_handling(amplitudes, old_position_xs, old_position_ys, new_position_x, new_position_y, length,
                       is_collision, directions, new_direction, thread_idx, adjusted_dist, collected, modes, random_nums, seed,rng_states)
    if is_collision[0, 0] > 0.0:
        direction = new_direction[0, 0]

    if old_position_ys[thread_idx] < 0:
        collected[thread_idx] = 1.0


@cuda.jit(device=True, inline=True)
def single_step(amplitudes, position_xs, position_ys, directions, polarizations, thread_idx, rng_states, adjusted_dist,
                collected, modes, random_nums, seed):
    region_number = cuda.local.array((1, 1), dtype=float32)
    mu_s = cuda.local.array((1, 1), dtype=float32)
    mu_a = cuda.local.array((1, 1), dtype=float32)
    g = cuda.local.array((1, 1), dtype=float32)
    n = cuda.local.array((1, 1), dtype=float32)
    mu_t = cuda.local.array((1, 1), dtype=float32)

    region_info(position_xs[thread_idx], position_ys[thread_idx], region_number, mu_s, mu_a, g, n, thread_idx)
    mu_t[0, 0] = mu_s[0, 0] + mu_a[0, 0]
    length = cuda.local.array((1, 1), dtype=float32)
    #1 seed
    step_length(mu_t, length, rng_states, random_nums, seed,thread_idx)
    #3 seeds
    scattering_direction(directions, g, rng_states, random_nums, seed+1,thread_idx)
    new_position_x = cuda.local.array((1, 1), dtype=float32)
    new_position_y = cuda.local.array((1, 1), dtype=float32)
    take_step(amplitudes, position_xs, position_ys, directions, length, new_position_x, new_position_y, thread_idx,
              adjusted_dist, collected, modes, random_nums, seed+4,rng_states)


@cuda.jit(device=True, inline=True)
def change_properties(amplitude, direction, polarization):
    pass


@cuda.jit(inline=False)
def process_steps(seed, amplitudes, direction_thetas, position_xs, position_ys, polarizations, steps, lengths, maxZs,
                  rng_states, adjusted_dist, collected, modes, random_nums,U, W, jones, mu_as, mu_ss, scat_events, jones_partial,
                  co_xy, cross_xy,
                  incoh_cross_xy, co_rz, cross_rz, incoh_cross_rz, co_xy_trad, cross_xy_trad, incoh_cross_xy_trad,
                  co_rz_trad, cross_rz_trad, incoh_cross_rz_trad, positions, s1, s2, m11, m12,
                  i_stokes_rz,q_stokes_rz,u_stokes_rz,v_stokes_rz):
    threadIdx = cuda.threadIdx
    blockIdx = cuda.blockIdx
    blockDim = cuda.blockDim
    thread_idx = blockDim.x * blockIdx.x + threadIdx.x

    if modes[thread_idx] == 3:
        polarized_photon(seed, random_nums[thread_idx],U, W, jones, mu_as, mu_ss, positions, s1, s2, m11, m12, scat_events, jones_partial,
                         co_xy,
                         cross_xy, incoh_cross_xy, co_rz, cross_rz, incoh_cross_rz, co_xy_trad, cross_xy_trad,
                         incoh_cross_xy_trad, co_rz_trad, cross_rz_trad, incoh_cross_rz_trad, i_stokes_rz,q_stokes_rz,u_stokes_rz,v_stokes_rz,collected, rng_states,
                         thread_idx)
    else:
        for i in range(20):
            pass
            while direction_thetas[thread_idx] > cmath.pi:
                direction_thetas[thread_idx] -= cmath.pi * 2
            while direction_thetas[thread_idx] < -cmath.pi:
                direction_thetas[thread_idx] += cmath.pi * 2
            if collected[thread_idx] == 1.0:
                break
            single_step(amplitudes, position_xs, position_ys, direction_thetas, polarizations, thread_idx, rng_states,
                        adjusted_dist, collected, modes,random_nums[thread_idx],i)
