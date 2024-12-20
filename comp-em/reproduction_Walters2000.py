"""Implementation of Fig. 4 in the study by Walters, T. J. et al.
Heating and pain sensation produced in human skin by millimeter waves:
Comparison to a simple thermal model, Health Physics, 78(3):259-267,
2000.

Fig. 4 depicts the mean increase in skin temperature (markers) versus
fitted functions (curves) for a range of power densitites. The authors
used simple 1-D analytical approach, where blood perfusion as
the basic thermoregulatory-effect is not assessed.
This code compares authors' approach with a numerical FFT-based
approach that accounts for the blood perfusion and extrapolates from
a given data.

Author: Ante Lojic Kapetanovic
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0 as eps_0, mu_0, pi
from scipy.special import erfc
import seaborn

from utils import load_tissue_diel_properties
from utils import solve_bhte as solve_num


# set context to `seaborn`
seaborn.set(style='whitegrid', context='paper', palette='colorblind', font='serif', font_scale=2)

# bhte constants for dry skin from https://itis.swiss/virtual-population/tissue-properties/database/
k = 0.37
rho = 1109.
C = 3391.
m_b = 2.32e-4


def solve_anal(t, pen_depth, I_0, T_tr):
    """Return closed-form solution of the 1-D BHTE with no blood
    perfusion considered over given simulation period, t.

    Parameters
    ----------
    sim_time : numpy.ndarray
        Simulation time
    pen_depth : float
        energy penetration depth
    I_0 : float
        incident power density of the tissue surface
    T_tr : float
        transmission coefficient into the tisse
    
        
    Returns
    -------
    numpy.ndarray
        rise in temperature over exposure time, t
    """
    C_1 = 2 * I_0 * T_tr / np.sqrt(pi * k * rho * C)
    C_2 = I_0 * T_tr * pen_depth / k
    tau = 4 / pi * (C_2 / C_1) ** 2
    return (C_1 * np.sqrt(t) 
            - C_2 * (1 - np.exp(t / tau) * erfc(np.sqrt(t / tau))))


def main():
    # frequency
    f = 94 * 1e9

    # conductivity, relative permitivity and penetration depth from Gabriel1996
    sigma, eps_r, _, pen_depth = load_tissue_diel_properties('skin_dry', f)

    # incident power densities
    I_0 = np.linspace(0.95, 1.75, 9) * 1e4

    # air (vacuum) resistance
    Z_air = np.sqrt(mu_0 / eps_0)
    # dry skin resistance
    Z_skin_dry = np.sqrt(mu_0 / (eps_r * eps_0))
    # energy (Fresnel) transmission coefficient into the tissue
    T_tr = 2 * Z_skin_dry / (Z_air + Z_skin_dry)

    # simulation
    sim_time = 3
    N = 11
    t = np.linspace(0, sim_time, 21)
    delta_T_sur = np.empty((t.size, I_0.size))
    delta_T_sur_mb = np.empty((t.size, I_0.size))
    for col_idx, i_0 in enumerate(I_0):
        delta_T_sur[:, col_idx] = solve_anal(t, pen_depth, i_0, T_tr)
        delta_T_sur_mb[:, col_idx] = solve_num(t, N, pen_depth, k, rho, C, m_b, i_0, T_tr)[:, 0]

    # visualize
    fig = plt.figure()
    ax = fig.add_subplot()
    marker = ['ro', 'g^', 'bx', 'r^', 'gx', 'bo', 'rx', 'go', 'b^']
    line = ['r', 'g', 'b'] * 3
    for col_idx, i_0 in enumerate(I_0):
        ax.plot(t, delta_T_sur[:, col_idx], marker[col_idx], markersize=6, label=f'${i_0 / 1e4}$')
        ax.plot(t, delta_T_sur_mb[:, col_idx], line[col_idx], lw=2.5)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('$\\Delta T_{sur}$ [°C]')
    ax.legend(title='$I_0 \\times 10^4$ [W/m$^2$]', ncol=3, fontsize=11, title_fontsize=11)
    plt.show()
    

if __name__=='__main__':
    main()