import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

from CADRE.orbital_equations import CentralBody, set_body_units

#: Earth's J2 coefficient, as the legacy CADRE orbit model used it. The canonical units make the
#: Earth radius 1 DU (to 1e-7). The dymos 1.x version had 202.7e-6, the Moon's J2 (ADL-004).
J2 = 1.08264e-3


class SphericalHarmonicsComp(JaxExplicitComp):
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    inputs: dict[str, InputVar] = {
        'h': InputVar(shape=('num_nodes',), val=1.0),
        'k': InputVar(shape=('num_nodes',), val=1.0),
        'L': InputVar(shape=('num_nodes',), val=1.0, desc='true longitude'),
        'r': InputVar(shape=('num_nodes',), val=1.0, desc='geocentric radius'),
    }

    outputs: dict[str, OutputVar] = {
        'J2_r': OutputVar(shape=('num_nodes',),
                          desc='zonal gravity perturbation in the radial direction'),
        'J2_s': OutputVar(shape=('num_nodes',),
                          desc='zonal gravity perturbation in the tangential direction'),
        'J2_w': OutputVar(shape=('num_nodes',),
                          desc='zonal gravity perturbation in the normal direction'),
    }

    def model_post_init(self, context, /):
        super().model_post_init(context)
        set_body_units(self, {'h': 'unitless', 'k': 'unitless', 'L': 'rad', 'r': 'DU',
                              'J2_r': 'DU/TU**2', 'J2_s': 'DU/TU**2', 'J2_w': 'DU/TU**2'})

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        r = inputs['r']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']
        s_sq = 1 + h**2 + k**2

        outputs['J2_r'] = -3 * J2 * (1 - 12 * ((h * jnp.sin(L) - k * jnp.cos(L)) / s_sq) ** 2) / (2 * r**4)
        outputs['J2_s'] = -12 * J2 * ((h * jnp.sin(L) - k * jnp.cos(L)) * (
                h * jnp.cos(L) + k * jnp.sin(L)) / (s_sq ** 2)) / (r**4)
        outputs['J2_w'] = -6 * J2 * ((1 - h**2 - k**2) * (
                h * jnp.sin(L) - k * jnp.cos(L)) / (s_sq ** 2)) / (r**4)
