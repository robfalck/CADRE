import jax
import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


def _dcm_from_quaternion(q):
    """
    Return the rotation matrix which takes body vectors into the inertial frame.

    Parameters
    ----------
    q : array-like (4, n)
        Scalar-first attitude quaternion of the body frame wrt the inertial frame.

    Returns
    -------
    O_BI : array-like (3, 3, n)
        Component-major rotation matrices, node axis last.
    """
    q0, q1, q2, q3 = q
    return jnp.stack([
        jnp.stack([q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)]),
        jnp.stack([2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)]),
        jnp.stack([2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2]),
    ])


def _dcm_and_rates(q, q_rate, q_rate2):
    """
    Return O_BI and its first and second time derivatives.

    The derivatives follow from the chain rule along the quaternion's rates:
    ``Odot = C'(q)[q_rate]`` and ``Oddot = C''(q)[q_rate, q_rate] + C'(q)[q_rate2]``.
    """
    def first(q, dq):
        return jax.jvp(_dcm_from_quaternion, (q,), (dq,))

    (O_BI, Odot_BI), (_, Oddot_BI) = jax.jvp(first, (q, q_rate), (q_rate, q_rate2))
    return O_BI, Odot_BI, Oddot_BI


class QuaternionAttitudeComp(JaxExplicitComp):
    """
    Compute the body orientation matrix and its time derivatives from an attitude quaternion.

    The quaternion is meant to be a dymos4 control, so that its first and second time
    derivatives are supplied by the transcription's control interpolation. Nothing here
    differentiates along the node axis; every node is independent.

    The quaternion is not normalized here. The unit-norm requirement is a constraint on
    the problem (see PointingComp).
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'q': InputVar(shape=(4, 'num_nodes'), val=0.5, units='unitless',
                      desc='Scalar-first attitude quaternion of the body frame wrt the '
                           'inertial frame.',
                      tags={'dymos.control_target:q'}),
        'q_rate': InputVar(shape=(4, 'num_nodes'), val=0.0, units='1/s',
                           desc='First time derivative of q.',
                           tags={'dymos.control_rate_target:q'}),
        'q_rate2': InputVar(shape=(4, 'num_nodes'), val=0.0, units='1/s**2',
                            desc='Second time derivative of q.',
                            tags={'dymos.control_rate2_target:q'}),
    }

    outputs: dict[str, OutputVar] = {
        'O_BI': OutputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                          desc='Rotation matrix from body-fixed frame to Earth-centered '
                               'inertial frame over time'),
        'Odot_BI': OutputVar(shape=(3, 3, 'num_nodes'), units='1/s',
                             desc='First time derivative of O_BI over time'),
        'Oddot_BI': OutputVar(shape=(3, 3, 'num_nodes'), units='1/s**2',
                              desc='Second time derivative of O_BI over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        O_BI, Odot_BI, Oddot_BI = _dcm_and_rates(inputs['q'], inputs['q_rate'], inputs['q_rate2'])
        outputs['O_BI'] = O_BI
        outputs['Odot_BI'] = Odot_BI
        outputs['Oddot_BI'] = Oddot_BI
