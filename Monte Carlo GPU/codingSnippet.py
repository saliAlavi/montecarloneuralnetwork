from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import numba
import cmath
import math
from photon import *


@cuda.jit
def compute_pi( out):
    """Find the maximum value in values and store in result[0]"""
    fresnel_prob(np.pi/4, np.pi/3, out)
    thread_id=cuda.grid
    #out[thread_id] = math.acos(0.5)

@cuda.jit(device=True, inline=True)
def fresnel_prob(incident_theta, transfer_theta, prob):
    prob[0] = float32((1 / 2 * ((cmath.sin(incident_theta - transfer_theta) * cmath.sin(incident_theta - transfer_theta)) / (
            cmath.sin(incident_theta + transfer_theta) * cmath.sin(incident_theta + transfer_theta)) + (
                            cmath.tan(incident_theta - transfer_theta) * cmath.tan(
                        incident_theta - transfer_theta)) / (cmath.tan(incident_theta + transfer_theta) * cmath.tan(
        incident_theta + transfer_theta)))).real)


threads_per_block = 64
blocks = 24
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
out = np.zeros(threads_per_block * blocks, dtype=np.float32)

compute_pi[blocks, threads_per_block](out)
incident_theta=np.pi/4
transfer_theta=np.pi/3
a=(1 / 2 * ((cmath.sin(incident_theta - transfer_theta) * cmath.sin(incident_theta - transfer_theta)) / (
            cmath.sin(incident_theta + transfer_theta) * cmath.sin(incident_theta + transfer_theta)) + (
                            cmath.tan(incident_theta - transfer_theta) * cmath.tan(
                        incident_theta - transfer_theta)) / (cmath.tan(incident_theta + transfer_theta) * cmath.tan(
        incident_theta + transfer_theta)))).real
print('pi:', a)
print(out)