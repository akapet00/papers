"""Implementation of Figure 1 in the study by Izhikevich, E. M.: Which
model to use for cortical spiking neurons?, IEEE Transactions on Neural
Networks, doi: 10.1109/TNN.2004.832719.

Figure 1 of the paper "summarizes neuro-computational features of
biological neurons" and the original code is available on author's
personal webpage at: https://www.izhikevich.org/publications/figure1.m
and the figure itself is available at:
https://www.izhikevich.org/publications/figure1.pdf

Author: Ante Lojic Kapetanovic
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def I(t, i_fun):
    """Wrapper for synaptic or injected dc-current function.
    
    Args
    ----
    t : numpy.ndarray
        a sequence of time points representing the solution domain
    i_fun : callable
        synaptic or injected dc-current function

    Returns
    -------
    numpy.ndarray
        time series for stimulating current
    """
    return i_fun(t)


def izhikevich(t, v, a, b, c, d, i_fun):
    """ODE system describing Izhikevih neuron model.

    More details:
    Izhikevich, E. M.: Simple model of spiking neurons, IEEE
    transactions on Neural Networks, doi: 10.1109/TNN.2003.820440

    Args
    ----
    t : numpy.ndarray
        a sequence of time points representing the solution domain
    v : list or tuple
        initial values for membrane potential, `v`,  and membrane
        recovery variable, `u`
    a : float
        time scale of the membrane recovery variable `u`
    b : float
        sensitivity of the recovery variable `u` to the subthreshold
        fluctuation of the membrane potential `v`
    c : float
        after-spike reset value of the membrane potential `v`
    d : float
        after-spike reset of the recovery variable `u`
    i_fun : callable
        synaptic or injected dc-current function

    Returns
    -------
    list
        time derivative of `v` and `u`
    """
    dv, du = [0, 0]
    if v[0] < 30:
        dv = 0.04*v[0]**2 + 5*v[0] + 140 - v[1] + I(t, i_fun)
        du = a*(b*v[0]-v[1])
    else:
        v[0] = c   
        v[1] = v[1] + d
    return [dv, du]


def visualize(t, v, i):
    """Utility function for simulation visualization.

    Args
    ----
    t : numpy.ndarray
        a sequence of time points representing the solution domain
    v : numpy.ndarray
        time series for membrane potential
    i : numpy.ndarray
        time series for synaptic or injected dc-current

    Returns
    -------
    numpy.ndarray
        two column axes of membrane potential in time and stimulating
        current in time, respectively
    """
    _, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True,
        gridspec_kw={'height_ratios': [4, 1]})
    ax[0].plot(t, v)
    ax[0].set_ylabel('$V(t)$ [mV]')
    ax[1].plot(t, i)
    ax[1].set_ylabel('$I_{exc}(t)$ [$\\mu$A]')
    plt.xlabel('$t$ [ms]')
    return ax


## example of inhibition-induced spiking
# Izhikevich model parameters
a = -0.02
b = -1
c = -60
d = 8
# initial conditions for ODE system solver
v0 = -63.8
u0 = b*v0
# resolution and simulation time
tau = 0.5
end = 350
t = np.arange(0, end+tau, step=tau)
# injected current
i_fun = np.vectorize(lambda t: 80 if (t<50) | (t>250) else 75)
# simulate and visualize
sol = solve_ivp(
    fun=izhikevich,
    t_span=(t.min(), t.max()),
    y0=(v0, u0),
    args=(a, b, c, d, i_fun),
    t_eval=t,
    vectorized=True,
    )
ax = visualize(
    t=sol.t,
    v=sol.y[0, :],
    i=i_fun(sol.t)
)
plt.show()