import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

from CADRE.orbital_equations import CentralBody, set_body_units

_UNITS = {'sma': 'DU', 'ecc': 'unitless', 'inc': 'rad', 'aop': 'rad', 'raan': 'rad', 'ta': 'rad',
          'p': 'DU', 'f': 'unitless', 'g': 'unitless', 'h': 'unitless', 'k': 'unitless',
          'L': 'rad'}


class KepToMEE(JaxExplicitComp):
    """
    Convert classical Keplerian elements to modified equinoctial elements.

    This was a copy of CartesianToMEE's class name with angle units of DU/TU and an input
    referenced as 'TA' but declared 'ta'. Its hand-written partials also differentiated h and k
    with respect to ecc, which they do not depend on. It is renamed and differentiated by JAX.
    """
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    inputs: dict[str, InputVar] = {
        name: InputVar(shape=('num_nodes',), val=1.0)
        for name in ('sma', 'ecc', 'inc', 'aop', 'raan', 'ta')
    }

    outputs: dict[str, OutputVar] = {
        name: OutputVar(shape=('num_nodes',)) for name in ('p', 'f', 'g', 'h', 'k', 'L')
    }

    def model_post_init(self, context, /):
        super().model_post_init(context)
        set_body_units(self, _UNITS)
        self.outputs['L'].desc = 'true longitude'

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        a = inputs['sma']
        e = inputs['ecc']
        i = inputs['inc']
        w = inputs['aop']
        Om = inputs['raan']
        f = inputs['ta']

        outputs['p'] = a*(1 - e**2)
        outputs['f'] = e*jnp.cos(w + Om)
        outputs['g'] = e*jnp.sin(w + Om)
        outputs['h'] = jnp.tan(i/2)*jnp.cos(Om)
        outputs['k'] = jnp.tan(i/2)*jnp.sin(Om)
        outputs['L'] = Om + w + f
