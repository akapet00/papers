"""Implementation of Figure 3 in the study by Izhikevich, E. M.: Simple
model of spiking neurons, IEEE Transactions on Neural Networks, doi:
10.1109/TNN.2003.820440.

Figure 3 of the paper depicts the simulation of a network of 1000
randomly coupled spiking neurons where the electrical behaviour of each
neuron is modelled by using Izhikevich novel neuron model.
Original code is available on author's homepage:
https://www.izhikevich.org/publications/net.m

Author: Ante Lojic Kapetanovic
"""

import matplotlib.pyplot as plt
import numpy as np

def model(n_neur, exc_to_inh_ratio):
    n_inh = int(n_neur/np.sum(exc_to_inh_ratio))
    n_exc = int(n_neur - n_inh)
    r_exc = np.random.random_sample(n_exc)
    r_inh = np.random.random_sample(n_inh)
    a = np.hstack((0.02*np.ones((n_exc, )), 0.02 + 0.08*r_inh))
    b = np.hstack((0.2*np.ones((n_exc,)), 0.25 - 0.05*r_inh))
    c = np.hstack((-65 + 15*r_exc**2, -65*np.ones((n_inh, ))))
    d = np.hstack((8 - 6*r_exc**2, 2*np.ones((n_inh, ))))
    S = np.hstack(
        (0.5*np.random.random_sample((n_exc + n_inh, n_exc)),
        -np.random.random_sample((n_exc + n_inh, n_inh))))
    return (a, b, c, d, S)

def simulation(tau, sim_dur, model):
    a, b, c, d, S = model
    # initial values
    v = -65*np.ones((n_exc + n_inh, ))
    u = b*v
    # simulation
    firing_time = []
    firing_neur = []
    for t in range(sim_dur):
        I = np.hstack(
            (5*np.random.standard_normal(n_exc, ),
            2*np.random.standard_normal(n_inh, )))
        fired = np.where(v>=30)[0]
        firing_time.extend(t*np.ones_like(fired))
        firing_neur.extend(fired)
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        I = I + np.sum(S[:, fired], 1)
        v = v + tau*(0.04*v**2 + 5*v + 140 - u + I)
        v = v + tau*(0.04*v**2 + 5*v + 140 - u + I)
        u = u + a*(b*v - u)
    return (firing_time, firing_neur, v)

def main():
    # network architecture
    n_neur = 1000
    exc_to_inh_ratio = (4, 1)
    neur_net = model(n_neur, exc_to_inh_ratio)

    # simulation time
    tau = 0.5  # for numerical stability
    sim_dur = 1000  # simulation time
    firing_time, firing_neur, v = simulation(tau, sim_dur, model)

    # visualization
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,
        gridspec_kw={'height_ratios': [4, 1]})
    ax[0].plot(firing_time, firing_neur, 'o', markersize=1, alpha=.7)
    ax[0].set_ylabel('neuron #')
    ax[1].plot(v, linewidth=.5)
    ax[1].set_xlabel('time [ms]')
    ax[1].set_ylabel('v [mV]')
    ax[1].set_ylim(np.min(v), 30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()