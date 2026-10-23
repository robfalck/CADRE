import jax.numpy as jnp
import numpy as np
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


#: Moment of inertia of the spacecraft, kg*m**2.
J = np.array([[0.018, 0.000, 0.000],
              [0.000, 0.018, 0.000],
              [0.000, 0.000, 0.006]])


def _compute_torque_req(w_B, wdot_B, J):
    """
    Return the torque required given the angular velocity and angular acceleration.

    Parameters
    ----------
    w_B : array-like (3, n)
        Angular velocity of the body frame.
    wdot_B : array-like (3, n)
        Angular acceleration of the body frame.
    J : array-like (3, 3)
        Moment of inertia of the spacecraft.

    Returns
    -------
    T_req : array-like (3, n)
        Required torque, J wdot + w x (J w).
    """
    return J @ wdot_B + jnp.cross(w_B, J @ w_B, axis=0)


class AttitudeTorqueComp(JaxExplicitComp):
    """
    Calculate the torque required given the angular velocity and its rate.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'w_B': InputVar(shape=(3, 'num_nodes'), units='rad/s',
                        desc='Angular velocity of the body frame wrt the inertial frame.'),
        'wdot_B': InputVar(shape=(3, 'num_nodes'), units='rad/s**2',
                           desc='Angular acceleration of the body frame wrt the inertial frame.'),
    }

    outputs: dict[str, OutputVar] = {
        'T_req': OutputVar(shape=(3, 'num_nodes'), units='N*m',
                           desc='Required torque vector in the body frame.'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['T_req'] = _compute_torque_req(inputs['w_B'], inputs['wdot_B'], J)
