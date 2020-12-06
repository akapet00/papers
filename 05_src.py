"""Implementation of the ideas behind the study by Bao, B. C.; Liu, Z.
and Xu, J. P.: Steady periodic memristor oscillator with transient
chaotic behaviours, Electronic Letters, doi: 10.1049/el.2010.3114

Author: Ante Lojic Kapetanovic
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def q(phi, a, b):
    """Charge as a flux-controlled memrisotr characteristic defined
    through a smooth continuous cubic monotone-increasing nonlinearity.

    Params
    ------
    phi : float
        flux-linkage
    a : float
        function constant, a>0
    b : float
        function constant, b>0

    Returns
    -------
    float
        flux-linkage value
    """
    assert (a>=0 and b>=0), '`a` and `b` must be positive values'
    return a*phi + b*phi**3

def W(phi, a, b):
    """Memductance W(phi) is obtained as dq(phi)/dphi and it represents
    the memrisotr constitutive relation.

    Params
    -----
    phi : float
        flux-linkage
    a : float
        function constant, a>0
    b : float
        function constant, b>0

    Returns
    -------
    float
        the slope of flux-linkage
    """
    assert (a>=0 and b>=0), '`a` and `b` must be positive values'
    return a + 3*b*phi**2


def cco(k, ic, *args):
    """Bao's implementation of the canonical Chua's oscillator with
    flux-controlled memristor;
    https://static-01.hindawi.com/articles/mpe/volume-2014/203123/figures/203123.fig.001.jpg
    where: x = v_1, y = i_3, z = v_2, w = phi, alpha = 1/C_1,
    beta = 1/C_2, gamma = G/C_2, L = 1

    Params
    ------
    ic : list or tuple
        initial conditions for x, y, z and w, respectively
    k : numpy.ndarray
        discrete sequence of time points for which to solve for cco
    args : list or tuple
        constants a, b, alpha, beta, gamma and L, respectively

    Returns
    -------
    tuple
        canonical Chua's oscillator ODE system
    """
    x, y, z, w = ic
    a, b, alpha, beta, gamma, L = args
    x_prime = k*alpha*(y - W(w, a, b)*x)
    y_prime = k*(z - x)
    z_prime = k/L*(-beta*y + gamma*z)
    w_prime = k*x 
    return (x_prime, y_prime, z_prime, w_prime)


def cco_jacobian(k, x, y, z, w, *args):
    """Jacobian matrix of Bao's implementation of the canonical Chua's
    oscillator with flux-controlled memristor ODE system.
    
    The equilibrium state of the ODE system is given by set:
    A = {(x, y, z, w)|x = y = z = 0, w = c},
    which corresponds to the w-axis, where c is a real constant.

    Params
    ------
    k : float
        time point
    x : float
        value of x in k-th time point
    y : float
        value of y in k-th time point
    z : float
        value of z in k-th time point
    w : float
        value of w in k-th time point
    args : list or tuple
        constants a, b, alpha, beta, gamma and L, respectively

    Returns
    -------
    numpy.ndarray
        Jacobian matrix at the equilibirium
    """
    a, b, alpha, beta, gamma, L = args
    return np.array([
        [-k*alpha*W(w, a, b), k*alpha,  0,        -6*k*alpha*b*x*w],
        [-k,                  0,        k,         0              ],
        [0,                  -k/L*beta, k/L*gamma, 0              ],
        [k,                   0,        0,         0              ]
    ])


def solve_cco(k, ic, args):
    """Solution of Bao's implementation of the canonical Chua's
    oscillator with flux-controlled memristor via Runge-Kutta method.

    Params
    ------
    ic : list or tuple
        initial conditions for x, y, z and w, respectively
    k : numpy.ndarray
        time scale factor as a solution domain
    args : list or tuple
        constants a, b, alpha, beta, gamma and L, respectively

    Returns
    -------
    tuple
        canonical Chua's oscillator ODE system solution
    """
    sol = solve_ivp(
        fun=cco,
        t_span=(k.min(), k.max()),
        y0=ic,
        method='RK45',
        t_eval=np.linspace(k.min(), k.max(), k.size),
        vectorized=True,
        args=args)
    x, y, z, w = sol.y
    return (x, y, z, w)


# configure parameters as in the paper
a = 0.2
b = 1
alpha = 1
beta = 0.65
gamma = 0.65
L = 1
args = [a, b, alpha, beta, gamma, L]
ic = [0, 10e-10, 0, 0]
t = np.linspace(0, 200, 100000)

x, y, z, w = solve_cco(t, ic, args)

# time series, v(t)
fig, ax = plt.subplots()
ax.plot(t, x)
ax.set_xlabel('t [s]')
ax.set_ylabel('v [V]')
plt.show()

# transient chaotic attractor
i = W(w, a, b)*x
fig, ax = plt.subplots()
ax.plot(x, i)
ax.set_xlabel('v [V]')
ax.set_ylabel('i [A]')
plt.show()

# Lyapunov exponents
dt = t[1] - t[0]
u = np.eye(4)
lyap = np.empty((t.size, 4))
for i, k  in enumerate(t):
    u_n = (np.eye(4) + cco_jacobian(k, x[i], y[i], z[i], w[i], *args)*dt) @ u
    q, r = np.linalg.qr(u_n)
    lyap[i, :] = np.log(np.abs(r.diagonal()))
    u = q
lyap_lambda = [sum([lyap[i, j] for i in range(t.size)]) / (dt*t.size)
    for j in range(4)]
print(f'Lyapunov exponents: {lyap_lambda}')