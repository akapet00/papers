"""Implementation of Figure 1 in the study by Izhikevich, E. M.: Which
model to use for cortical spiking neurons?, IEEE Transactions on Neural
Networks, doi: 10.1109/TNN.2004.832719.

Figure 1 of the outlined paper "summarizes neuro-computational features
of biological neurons" and the original code is available on author's
personal webpage:
https://www.izhikevich.org/publications/figure1.m
and the figure itself is available here:
https://www.izhikevich.org/publications/figure1.pdf

Author: Ante Lojic Kapetanovic
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def I(I0, exc_start, t):
    return I0*(t>exc_start)


def izhikevich(t, v, *args):
    dv, du = [0, 0]
    a, b, c, d, I0, exc_start = args
    if v[0] < 30:
        dv = 0.04*v[0]**2 + 5*v[0] + 140 - v[1] + I(I0, exc_start, t)
        du = a*(b*v[0]-v[1])
    else:
        v[0] = c   
        v[1] = v[1] + d
    return [dv, du]


# resolution and simulation time
tau = 0.25
t = np.arange(0, 100+tau, step=tau)
# Izhikevich model parameters
a = 0.02
b = 0.2
c = -65.
d = 6.
I0 = 14
exc_start = t.max()/10
args = [a, b, c, d, I0, exc_start]
# initial conditions for ODE system solver
v0 = -70
u0 = b*v0
y0 = [v0, u0]
# solution via Runge-Kutta order 4 method
sol = solve_ivp(
    fun=izhikevich,
    t_span=(t.min(), t.max()),
    y0=y0,
    args=args,
    t_eval=t,
    vectorized=True,
    )
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True)
ax[0].plot(sol.t, sol.y[0, :])
ax[0].set_ylabel('$V(t)$ [mV]')
ax[1].plot(sol.t, I(I0, exc_start, sol.t))
ax[1].set_ylabel('$I_{exc}(t)$ [$\\mu$A]')
plt.xlabel('$t$ [ms]')
plt.show()