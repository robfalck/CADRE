from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

from CADRE.attitude_dymos.angular_velocity_comp import _compute_angular_velocity


def _compute_angular_acceleration(O_BI, Odot_BI, Oddot_BI):
    """
    Return the time derivative of the body angular velocity.

    The angular velocity is bilinear in (O_BI, Odot_BI), so its rate is
    ``w(O_BI, Oddot_BI) + w(Odot_BI, Odot_BI)``.

    Parameters
    ----------
    O_BI : array-like (3, 3, n)
        The transformation matrix from the body to the inertial frame at n points in time.
    Odot_BI : array-like (3, 3, n)
        The first time derivative of O_BI.
    Oddot_BI : array-like (3, 3, n)
        The second time derivative of O_BI.

    Returns
    -------
    wdot_B : array-like (3, n)
        Angular acceleration of the body frame wrt the inertial frame.
    """
    return (_compute_angular_velocity(O_BI, Oddot_BI) +
            _compute_angular_velocity(Odot_BI, Odot_BI))


class AngularAccelerationComp(JaxExplicitComp):
    """
    Calculates time derivative of the body frame angular velocity wrt the inertial frame.

    This is computed pointwise from O_BI and its first and second time derivatives, which
    come from the attitude control's rates (see QuaternionAttitudeComp).
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'O_BI': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from body-fixed frame to Earth-centered '
                              'inertial frame over time'),
        'Odot_BI': InputVar(shape=(3, 3, 'num_nodes'), units='1/s',
                            desc='First time derivative of O_BI.'),
        'Oddot_BI': InputVar(shape=(3, 3, 'num_nodes'), units='1/s**2',
                             desc='Second time derivative of O_BI.'),
    }

    outputs: dict[str, OutputVar] = {
        'wdot_B': OutputVar(shape=(3, 'num_nodes'), units='rad/s**2',
                            desc='First time derivative of w_B over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['wdot_B'] = _compute_angular_acceleration(inputs['O_BI'], inputs['Odot_BI'],
                                                          inputs['Oddot_BI'])
