import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


def _compute_angular_velocity(O_BI, Odot_BI):
    """
    Return the angular velocity of the body frame wrt the inertial frame.

    Parameters
    ----------
    O_BI : array-like (3, 3, n)
        The transformation matrix from the body to the inertial frame at n points in time.
    Odot_BI : array-like (3, 3, n)
        The time rate of change of O_BI.

    Returns
    -------
    w_B : array-like (3, n)
        Angular velocity of the body frame, in the body frame.

    Notes
    -----
    O_BI takes body vectors into the inertial frame, so ``Odot_BI = O_BI [w_B x]`` and
    ``[w_B x] = O_BI^T Odot_BI``. The original CADRE took the rows of ``Odot_BI O_BI^T``
    instead, which is the negative of w_B when O_BI is read as inertial-to-body, as the original
    did everywhere else.
    """
    W = jnp.einsum('kin,kjn->ijn', O_BI, Odot_BI)
    return jnp.stack((W[2, 1], W[0, 2], W[1, 0]))


class AngularVelocityComp(JaxExplicitComp):
    """
    Compute the angular velocity of the spacraft body frame relative to the inertial frame.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'O_BI': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from body-fixed frame to Earth-centered '
                              'inertial frame over time'),
        'Odot_BI': InputVar(shape=(3, 3, 'num_nodes'), units='1/s',
                            desc='Time derivative of O_BI.'),
    }

    outputs: dict[str, OutputVar] = {
        'w_B': OutputVar(shape=(3, 'num_nodes'), units='rad/s',
                         desc='Angular velocity of the body frame wrt the inertial frame.'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['w_B'] = _compute_angular_velocity(inputs['O_BI'], inputs['Odot_BI'])
