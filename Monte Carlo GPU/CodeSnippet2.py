from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import numba
import cmath
import math
from photon import *
import matplotlib.pyplot as plt


def fresnel_prob(incident_theta, transfer_theta):
    prob = 1 / 2 * ((np.sin(incident_theta - transfer_theta) * np.sin(incident_theta - transfer_theta)) / (
            cmath.sin(incident_theta + transfer_theta) * cmath.sin(incident_theta + transfer_theta)) + (
                            np.tan(incident_theta - transfer_theta) * np.tan(
                        incident_theta - transfer_theta)) / (np.tan(incident_theta + transfer_theta) * np.tan(
        incident_theta + transfer_theta)))
    return prob


def s_prob(incident_theta, transfer_theta,n_1, n_2):
    r = ((n_1 * cmath.cos(incident_theta) - n_2 * cmath.cos(transfer_theta)) / (
            n_1 * cmath.cos(incident_theta) + n_2 * cmath.cos(transfer_theta))).real
    x = (n_1 * cmath.cos(incident_theta) - n_2 * cmath.cos(transfer_theta)).real
    #x = (n_1 * cmath.cos(incident_theta) - n_2 * cmath.cos(transfer_theta)).real
    #print(incident_theta, transfer_theta, n_1, n_2, x, r)
    return r*r
prob=[]
for i in range(-0,1000):
    inc=np.pi/1000
    prob.append(s_prob(i*inc, 0.635119,1.39, 1.4))
plt.plot(prob)
plt.show()
print( (1.34 * cmath.cos(2.108569) - 1.4 * cmath.cos(2.083198)).real)
a=[1,3,4,5,3]
a=np.asarray(a)
print(a==3)