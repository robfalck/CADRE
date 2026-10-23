import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


class PointingComp(JaxExplicitComp):
    """
    Compute the pointing and quaternion-normalization residuals of the attitude.

    CADRE holds the body z axis along the third axis of the rolled frame (anti-velocity, see
    ORIComp) and leaves the roll about it free. The body z axis is aligned with it exactly when
    it has no component along the other two rolled-frame axes, so ``pointing_err`` has two
    entries per node; constrain both to zero. Since the body z axis is a unit vector only when
    the quaternion is, also constrain ``q_norm_err`` to zero.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'q': InputVar(shape=(4, 'num_nodes'), val=0.5, units='unitless',
                      desc='Scalar-first attitude quaternion of the body frame wrt the '
                           'inertial frame.',
                      tags={'dymos.control_target:q'}),
        'O_BI': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from body-fixed frame to Earth-centered '
                              'inertial frame over time'),
        'O_RI': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from rolled body-fixed frame to '
                              'Earth-centered inertial frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'pointing_err': OutputVar(shape=(2, 'num_nodes'), units='unitless',
                                  desc='Components of the body z axis along the rolled x and '
                                       'y axes.'),
        'q_norm_err': OutputVar(shape=('num_nodes',), units='unitless',
                                desc='Squared norm of the quaternion, minus one.'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        # The columns of O_BI and O_RI are the body and rolled axes in the inertial frame.
        z_B = inputs['O_BI'][:, 2]
        O_RI = inputs['O_RI']
        outputs['pointing_err'] = jnp.stack((jnp.sum(z_B * O_RI[:, 0], axis=0),
                                             jnp.sum(z_B * O_RI[:, 1], axis=0)))
        outputs['q_norm_err'] = jnp.sum(inputs['q']**2, axis=0) - 1.0
