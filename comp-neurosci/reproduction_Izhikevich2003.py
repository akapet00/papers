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

import logging
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random_sample as rand
from numpy.random import standard_normal as randn
import seaborn
import tqdm


seaborn.set(style='whitegrid', context='paper', palette='colorblind', font='serif', font_scale=2)


class NeuralNet():
    def __init__(self):
        """Constructor."""
        pass

    def __str__(self):
        """String representation of the object."""
        short_desc = 'Spiking neural network based on Izhikevich neuron model'
        div = '-'*len(short_desc)
        if self.n_inh is None:
            return short_desc
        return (
            short_desc\
            + f'\n{div}\n'
            + f' Size of the network: {self.n_neur}\n'\
            + f' Excitatory neurons:  {self.n_exc}\n'\
            + f' Inhibitory neurons:  {self.n_inh}\n'\
            + f'{div}\n')
    
    def __repr__(self):
        """Object representation."""
        return self.__str__()

    def build(self, n_neur, exc_to_inh_ratio):
        """Build neural network.

        Args
        ----
        n_neur : int
            number of neurons in neural network
        exc_to_inh_ratio : tuple
            two element tuple representing the ratio between the number
            of excitatory neurons vs inhibitory neurons
        
        Returns
        -------
        None
        """
        self.n_neur = n_neur
        self.n_inh = int(self.n_neur/np.sum(exc_to_inh_ratio))
        self.n_exc = int(self.n_neur - self.n_inh)
        self.r_exc = rand(self.n_exc)
        self.r_inh = rand(self.n_inh)
        self.a = np.hstack(
            (0.02*np.ones((self.n_exc, )), 0.02 + 0.08*self.r_inh))
        self.b = np.hstack(
            (0.2*np.ones((self.n_exc,)), 0.25 - 0.05*self.r_inh))
        self.c = np.hstack(
            (-65 + 15*self.r_exc**2, -65*np.ones((self.n_inh, ))))
        self.d = np.hstack(
            (8 - 6*self.r_exc**2, 2*np.ones((self.n_inh, ))))
        self.S = np.hstack(
            (0.5*rand((self.n_exc + self.n_inh, self.n_exc)),
            -rand((self.n_exc + self.n_inh, self.n_inh))))

    def simulate(self, v_0, sim_dur, tau, visualize=False):
        """Run simulation.

        Args
        ----
        v_0 : float
            initial value of membrane potential
        sim_dur : int
            simulation duration, unit=[ms]
        tau : float
            simulation resolution, unit=[ms]
        visualize : bool, optional
            if True, return visualization of activated neurons over
            time and the final membrane potential dynamics
        
        Returns
        -------
        None
        """
        v = v_0*np.ones((self.n_exc + self.n_inh, ))
        u = self.b*v

        firing_time = []
        firing_neur = []
        start_time = time()
        for t in tqdm.trange(sim_dur):
            I = np.hstack(
                (5*randn(self.n_exc, ),
                2*randn(self.n_inh, )))
            fired = np.where(v>=30)[0]
            firing_time.extend(t*np.ones_like(fired))
            firing_neur.extend(fired)
            v[fired] = self.c[fired]
            u[fired] = u[fired] + self.d[fired]
            I = I + np.sum(self.S[:, fired], 1)
            for _ in range(int(1/tau)):
                v = v + tau*(0.04*v**2 + 5*v + 140 - u + I)
            u = u + self.a*(self.b*v - u)
        end_time = time() - start_time
        logging.info('\n')
        logging.info(f'Simulation finished. Elapsed: {round(end_time, 5)}s')
        if visualize:
            fig, ax = plt.subplots()
            ax.plot(firing_time, firing_neur, 'o', markersize=1, alpha=.7)
            ax.axhline(self.n_exc - 1, xmin=0, xmax=sim_dur-1, color='k', lw=2)
            ax.text(0.5, 0.5*self.n_exc/self.n_neur, 'excitatory',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='wheat', alpha=0.9))
            ax.text(0.5, self.n_exc/self.n_neur + 0.4*self.n_inh/self.n_neur,
                'inhibitory', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(facecolor='wheat', alpha=0.9))
            ax.set_xlabel('$t$ [ms]')
            ax.set_ylabel('neuron ID')
            return (fig, ax)


def main():
    n_neur = 1000              # total number of neurons in network
    exc_to_inh_ratio = (4, 1)  # excitatory vs inhibitory neurons ratio
    tau = 0.5                  # ms, for numerical stability
    sim_dur = 1000             # ms, simulation time
    v_0 = -65                  # mV, initial membrane potential value
    
    snn = NeuralNet()
    snn.build(n_neur, exc_to_inh_ratio)
    logging.info(snn)
    _ = snn.simulate(v_0, sim_dur, tau, visualize=True)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()