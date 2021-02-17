import csv
import os

import numpy as np
from scipy.integrate import odeint


SUPPORTED_TISSUES = [
    'air', 'blood', 'blood_vessel', 'body_fluid', 'bone_cancellous',
    'bone_cortical', 'bone_marrow', 'brain_grey_matter', 'brain_white_matter',
    'cerebellum', 'cerebro_spinal_fluid', 'dura', 'fat', 'muscle', 'skin_dry',
    'skin_wet',
    ]


def load_tissue_diel_properties(tissue, frequency):
    """Return conductivity, relative permitivity, loss tangent and
    penetration depth of a given tissue based on a given frequency.

    Parameters
    ----------
    tissue : str
        type of human tissue
    frequency : float
        radiation frequency
        
    Returns
    -------
    tuple
        tuple of 4 float values which represent conductivity, relative
        permitivity, loss tangent and penetration depth, respectively
    """
    if tissue not in SUPPORTED_TISSUES:
        raise ValueError(f'Unsupported tissue. Choose {SUPPORTED_TISSUES}.')
    if 1e9 > frequency > 100e9:
        raise ValueError('Invalid frequency. Choose in range [1, 100] GHz')
    tissue_diel_properties_path = os.path.join(
        'tissue_diel_properties', 'tissue_diel_properties.csv')
    with open(tissue_diel_properties_path) as f: 
        reader = csv.reader(f) 
        for row in reader:
            if str(row[0])==tissue and float(row[1])==frequency: 
                conductivity = float(row[2]) 
                relative_permitivity = float(row[3]) 
                loss_tangent = float(row[4]) 
                penetration_depth = float(row[5])
        return (conductivity, relative_permitivity, loss_tangent, penetration_depth)

    
def solve_bhte1d(t, N, pen_depth, k, rho, C, m_b, I_0, T_tr):
    """Numerical solution to 1-D Pennes' bioheat transfer equation by
    using Fast Fourier Transform on spatial coordinate.
    
    Parameters
    ----------
    t : numpy.ndarray
        simulation time; exposure time in seconds
    N : int
        number of collocation points
    pen_depth : float
        energy penetration depth
    k : float
        thermal conductivity of the tissue
    rho : float
        tissue density
    C : float
        heat capacity of the tissue
    m_b : float
        blood perfusion
    I_0 : float
        incident power density of the tissue surface
    T_tr : float
        transmission coefficient into the tisse
        
    Returns
    -------
    numpy.ndarray
        temperature distribution in time for each collocation point
    """
    dx = pen_depth / N
    x = np.arange(0, pen_depth, dx)
    kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    SAR = I_0 * T_tr / (rho * pen_depth) * np.exp(-x / pen_depth)
    SAR_fft = np.fft.fft(SAR)

    # initial conditions -- prior to radiofrequency exposure
    T0 = np.zeros_like(x)
    T0_fft = np.fft.fft(T0)

    # recasting complex numbers to an array for easier handling in SciPy
    T0_fft_ri = np.concatenate((T0_fft.real, T0_fft.imag))
    
    def rhs(T_fft_ri, t, kappa, k, rho, C, m_b, SAR_fft):
        T_fft = T_fft_ri[:N] + (1j) * T_fft_ri[N:]
        d_T_fft = (
            - np.power(kappa, 2) * k * T_fft / (rho * C)
            - rho * m_b * T_fft
            + SAR_fft / C)
        return np.concatenate((d_T_fft.real, d_T_fft.imag)).astype(np.float64)

    T_fft_ri = odeint(rhs, T0_fft_ri, t, args=(kappa, k, rho, C, m_b, SAR_fft))
    T_fft = T_fft_ri[:, :N] + (1j) * T_fft_ri[:, N:]

    T = np.empty_like(T_fft)
    for i in range(t.size):
        T[i, :] = np.fft.ifft(T_fft[i, :])
    return T.real