import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

from CADRE.orbital_equations import CentralBody, set_body_units

_UNITS = {'p': 'DU', 'f': 'unitless', 'g': 'unitless', 'h': 'unitless', 'k': 'unitless',
          'L': 'rad', 'sma': 'DU', 'ecc': 'unitless', 'inc': 'rad', 'aop': 'rad', 'raan': 'rad',
          'ta': 'rad'}


class MEEToKep(JaxExplicitComp):
    """
    Convert modified equinoctial elements to classical Keplerian elements.

    The original declared the eccentricity in DU and the angles in DU/TU; they are unitless and
    radians here.
    """
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    inputs: dict[str, InputVar] = {
        name: InputVar(shape=('num_nodes',), val=1.0) for name in ('p', 'f', 'g', 'h', 'k', 'L')
    }

    outputs: dict[str, OutputVar] = {
        name: OutputVar(shape=('num_nodes',))
        for name in ('sma', 'ecc', 'inc', 'aop', 'raan', 'ta')
    }

    def model_post_init(self, context, /):
        super().model_post_init(context)
        set_body_units(self, _UNITS)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']

        outputs['sma'] = p / (1 - f**2 - g**2)
        outputs['ecc'] = jnp.sqrt(f**2 + g**2)
        outputs['inc'] = jnp.arctan2(2*jnp.sqrt(h**2+k**2), 1-h**2-k**2)
        outputs['raan'] = jnp.arctan2(k, h)
        outputs['aop'] = jnp.arctan2(g*h-f*k, f*h+g*k)
        outputs['ta'] = L - jnp.arctan2(g, f)
