"""I-V curves of the ideal memristor introduced and defined in the
study by Chua, L. O.: Memristor -- The missing circuit element, IEEE
transactions on circuit theory, doi: 10.1109/TCT.1971.1083337, using
the formulation outlined in the study by Joglekar, Y. N. and Wolf S.:
The elusive memristor: properties of basic electrical circuits,
European Journal of Physics 30, doi: 10.1088/0143-0807/30/4/001

Author: Ante Lojic Kapetanovic
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz


def generate_voltage(t, omega, V_0):
    """Return sinusoidal voltage characteristic.

    Args
    ----
    t : numpy.ndarray
        equally sequenced time points, unit=[s]
    omega : float
        angular frequency, unit=[1/s]
    V_0 : float
        source amplitude, unit=[V]

    Returns
    -------
    numpy.ndarray
        voltage time series, unit=[V]
    """
    return V_0*np.sin(omega*t)


def generate_flux(t, omega, V_0):
    """Return flux as the cumulative time integral of the sinusoidal
    voltage source, $\phi(t)=\int_{t} v(t) dt$.

    Args
    ----
    t : numpy.ndarray
        equally sequenced time points, unit=[s]
    omega : float
        angular frequency, unit=[1/s]
    V_0 : float
        source amplitude, unit=[V]

    Returns
    -------
    numpy.ndarray
        flux time series, unit=[Wb]
    """
    return V_0/omega*(1 - np.cos(omega*t))


def generate_memristance(R_on, R_off, D, w_0, u_d, nu, phi):
    """Return values of memristor's resistance over time.

    Args
    ----
    R_on : float
        resistance of the fully doped memristor, unit=[Ohm]
    R_off : float
        resistance of the memristor if it is undoped, unit=[Ohm]
    D : float
        memristor's length, unit=[m]
    w_0 : float
        effective length of the doped region at t=0, unit=[m]
    u_d : float
        the mobility of dopants, unit=[m^2/Vs]
    nu : float
        polarity of the memristor, either +1, which corresponds to the
        expansion, or -1, which corresponds to the contraction of the
        doped region
    phi : numpy.ndarray
        flux time series, unit=[Wb]
    
    Returns
    -------
    numpy.ndarray
        effective resistance of the memristor over time, unit=[Ohm]
    """
    # difference in resistance between doped and undoped region
    R_d = R_off - R_on
    # effective resistance of the memristor at t=0
    R_0 = R_on*(w_0/D) + R_off*(1-w_0/D)
    # the ammount of charge that is required to pass through the
    # memristor for the dopant boundary to move through distance D
    Q_0 = D**2 / (u_d*R_on)
    # memristance as a function of flux-linkage
    return R_0 * np.sqrt(1 - 2*nu*R_d*phi/(Q_0*R_0**2))
    

## a single ideal memristor electrical circuit
# generate voltage and calculate flux
t = np.linspace(0, 100, 1001)
V_0 = 1
omega = 0.1/1.5
v = generate_voltage(t, omega, V_0)
# phi_approx = cumtrapz(v, t, initial=0)
phi = generate_flux(t, omega, V_0)

# calculate memristance
R_on = 1
R_off = R_on * 20
D = 1e-8
w_0 = D/10
u_d = 1e-14
nu = -1
m = generate_memristance(R_on, R_off, D, w_0, u_d, nu, phi)

# q-phi and i-v characteristics
i = v/m
q = phi/m
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
axs[0].plot(phi, q)
axs[0].set_xlabel('$\\phi$')
axs[0].set_ylabel('$q$')
axs[1].plot(v, i)
axs[1].set_xlabel('$v$')
axs[1].set_ylabel('$i$')
plt.tight_layout()
plt.show()