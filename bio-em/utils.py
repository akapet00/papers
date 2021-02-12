import csv
import os
import matplotlib
from matplotlib.transforms import Transform
import numpy as np 


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

    
class SqrtTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def transform_non_affine(self, a): 
        return np.array(a)**0.5

    def inverted(self):
        return InvertedSqrtTransform()


class InvertedSqrtTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def transform(self, a):
        return np.array(a)**2

    def inverted(self):
        return SqrtTransform()


class SqrtScale(matplotlib.scale.ScaleBase):
    name = 'sqrt'

    def __init__(self, axis, **kwargs):
        super().__init__(axis, **kwargs)

    def get_transform(self):
        return SqrtTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(matplotlib.ticker.AutoLocator())
        axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    
    
def register_sqrt_scale():
    """Register square-root axes scaler in the `matlab.scale`.

    Parameters
    ----------
    None
        
    Returns
    -------
    None
    """
    matplotlib.scale.register_scale(SqrtScale)