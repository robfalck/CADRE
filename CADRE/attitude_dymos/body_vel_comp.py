import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


class BodyVelComp(JaxExplicitComp):
    """
    Determine velocity in the body frame.

    This  component was Attitude_Sideslip in the original version, which applied O_BI rather
    than its transpose. O_BI takes body vectors into the inertial frame, so ``v_B = O_BI^T v_I``.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'r_e2b_I': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Earth-to-satellite position vector in the inertial frame.'),
        'v_e2b_I': InputVar(shape=(3, 'num_nodes'), units='km/s',
                            desc='Earth-to-satellite velocity vector in the inertial frame.'),
        'w_B': InputVar(shape=(3, 'num_nodes'), units='rad/s',
                        desc='Angular velocity of the body frame wrt the inertial frame.'),
        'O_BI': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from body-fixed frame to '
                              'Earth-centered inertial frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'v_e2b_B': OutputVar(shape=(3, 'num_nodes'), units='km/s',
                             desc='Earth-to-satellite velocity vector in the body-fixed frame'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        # TODO: This formulation is missing the omega x r term from the basic kinematic equation
        # Need to add jnp.cross(w_B, r_e2b_I, axis=0)
        outputs['v_e2b_B'] = jnp.einsum('jin,jn->in', inputs['O_BI'], inputs['v_e2b_I'])
