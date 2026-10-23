import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


class StateMuxComp(JaxExplicitComp):
    """Stack the Cartesian components into component-major (3, n) position and velocity."""
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        **{name: InputVar(shape=('num_nodes',), units='km') for name in ('x', 'y', 'z')},
        **{name: InputVar(shape=('num_nodes',), units='km/s') for name in ('vx', 'vy', 'vz')},
    }

    outputs: dict[str, OutputVar] = {
        'r_e2b_I': OutputVar(shape=(3, 'num_nodes'), units='km'),
        'v_e2b_I': OutputVar(shape=(3, 'num_nodes'), units='km/s'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['r_e2b_I'] = jnp.stack((inputs['x'], inputs['y'], inputs['z']))
        outputs['v_e2b_I'] = jnp.stack((inputs['vx'], inputs['vy'], inputs['vz']))
