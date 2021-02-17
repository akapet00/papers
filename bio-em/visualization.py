import matplotlib
from matplotlib.transforms import Transform
import numpy as np 


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