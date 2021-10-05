"""Comparison between I-V curves obtained by using the smooth nonlinear
memristance approximation function introduced in the study by Bao, B.
C.; Liu, Z. and Xu, J. P.: Steady periodic memristor oscillator with
transient chaotic behaviours, Electronic Letters, doi:
10.1049/el.2010.3114, and by using the formulation that outlines the
actual physical memristive device, introduced in the study by Joglekar,
Y. N. and Wolf S.: The elusive memristor: properties of basic
electrical circuits, European Journal of Physics 30, doi:
10.1088/0143-0807/30/4/001. Parameters in the memristance formulation
by Joglekar et al. are fitted using the ideal memristance defined in
the formulation by Bao et al. 

Author: Ante Lojic Kapetanovic
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


#######################################################################
## plotting configuration
#######################################################################
def plotting_config(nrows=1, ncols=1):
    """Setup configuration for paper quality figures.

    Params
    ------
    nrows : int
        Number of subplots row-wise.
    ncols : int
        Number of subplots column-wise.

    Returns
    -------
    None
    """
    plt.style.use('seaborn-paper')
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'grid.linewidth': 0.7,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': [4.774*nrows, 2.950*ncols],
    })

#######################################################################
## util functions
#######################################################################
def mse(true, fit):
    """Return mean square error between true and fit data.

    Params
    ------
    true : float or numpy.ndarray
        measured data
    fit : float or numpy.ndarray
        fitted data
    
    Returns
    -------
    float
        mean square error between true and fit data
    """
    return np.mean((true - fit)**2)


def mae(true, fit):
    """Return mean absolute error between true and fit data.

    Params
    ------
    true : float or numpy.ndarray
        measured data
    fit : float or numpy.ndarray
        fitted data
    
    Returns
    -------
    float
        mean absolute error between true and fit data
    """
    return np.mean(np.abs(true - fit))
#######################################################################


def W(phi, a, b):
    """Return memductance, W(phi), obtained as dq(phi)/dphi.

    This function is introduced as smooth continuous squared
    monotone-increasing nonlinear approximation function for the actual
    memductance of a physical memristive device or system.

    Params
    -----
    phi : float
        flux-linkage
    a : float
        function constant
    b : float
        function constant

    Returns
    -------
    float
        the slope of flux-linkage
    """
    return a + 3*b*phi**2


def generate_memristance(phi, D, u_d, R_on):
    """Return memristance of a physical memrsitive device or system as
    a function of time assuming the device is phi-controlled.

    Args
    ----
    phi : numpy.ndarray
        flux time series, unit=[Wb]
    D : float
        memristor's length, unit=[m]
    u_d : float
        the mobility of dopants, unit=[m^2/Vs]
    R_on : float
        resistance of the fully doped memristor, unit=[Ohm]
    R_off : float
        resistance of the memristor if it is undoped, unit=[Ohm]
    
    
    Returns
    -------
    numpy.ndarray
        effective resistance of the memristor over time, unit=[Ohm]
    """
    w_0 = D/5  # length of the doped region at t=0, unit=[m]
    R_off = 20*R_on  # resistance of the fully undoped memristor, unit=[Ohm]
    R_d = R_off - R_on  # resistance difference between regions, unit=[Ohm]
    R_0 = R_on*(w_0/D) + R_off*(1-w_0/D)  # effective resistance at t=0
    Q_0 = D**2 / (u_d*R_on)  # initial charge of dopant
    return R_0 * np.sqrt(1 + 2*R_d*phi/(Q_0*R_0**2))  # effective resistance


## loss function
def loss(params, memristance, phi):
    """Return loss as l-2 norm between the theoretical memristance
    obtained via nonlinear approximation function and the memristance
    of an observed physical device or system.

    Args
    ----
    params : tuple
        free parameters of the `generate_memristance` function that are
        fitted
    m : numpy.ndarray
        theoretical memristance values
    phi : numpy.ndarray
        flux time series

    Returns
    -------
    float
        loss function value for a given iteration
    """
    return mse(memristance, generate_memristance(phi, *params))


def main():
    # configure simulation time
    tau = 0.1
    t = np.arange(0, 100, step=tau)

    # generate sinusodial voltage and flux linkage
    omega = 1
    V_0 = 1
    v = V_0*np.sin(omega*t)
    phi = V_0/omega*(1 - np.cos(omega*t))

    # theoretical memristance approximation
    a = 0.4
    b = 0.02
    memristance = 1/W(phi, a, b) 

    # unbounded minimization procedure
    res = minimize(
        fun=loss,
        x0=(10e-8, 1e-14, 10),
        args=(memristance, phi),
        method='Nelder-Mead',
        options={'xtol':1e-8},
    )
    opt_params = res.x
    print(
        f'\nFlux controled memristive device configuration:',
        f'\n D    = {opt_params[0]:3e} Ohm',
        f'\n u_d  = {opt_params[1]:3e} m^2/Vs',
        f'\n R_on = {opt_params[2]:3e} Ohm')
    memristance_fit = generate_memristance(phi, *opt_params)

    # visualization of I-V curves
    i = v/memristance
    i_fit = v/memristance_fit
    plotting_config(nrows=1, ncols=1)
    fig, ax = plt.subplots()
    ax.plot(v, i, 'ko', markersize=4, markevery=5, label='Bao et al.')
    ax.plot(v, i_fit, 'k-', label='Joglekar et al.')
    ax.set_xlabel('$v$ [V]')
    ax.set_ylabel('$i$ [A]')
    ax.legend(loc='best')
    ax.grid()
    # plt.show()
    fig.savefig(fname='compatibility.eps', format='eps', bbox_inches='tight')


if __name__ == "__main__":
    main()