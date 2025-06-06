import openmdao.api as om
import jax
import jax.numpy as jnp


def _single_angular_velocity(O_BI, Odot_BI):
    """
    Return a the transformation matrix from the rolled frame to the inertial frame.

    Parameters
    ----------
    O_BI : array-like (3, 3)
        The transformation matrix from the body to the inertial frame at n points in time.
    Odot_BI : array-like (3, 3)
        The time rate of change of O_BI.
    """
    return jnp.vstack((jnp.dot(Odot_BI[2, :], O_BI[1, :]),
                       jnp.dot(Odot_BI[0, :], O_BI[2, :]),
                       jnp.dot(Odot_BI[1, :], O_BI[0, :])))


_compute_angular_velocity = jax.vmap(_single_angular_velocity, in_axes=(0, 0))
""" Vectorized form of the calculation of O_RI. """


class AngularVelocityComp(om.JaxExplicitComponent):
    """
    Compute the angular velocity of the spacraft body frame relative to the inertial frame.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('O_BI', shape=(nn, 3, 3), units='unitless',
                       desc='Rotation matrix from body-fixed frame to Earth-centered '
                            'inertial frame over time')

        self.add_input('Odot_BI', shape=(nn, 3, 3), units='1/s',
                       desc='Time derivative of O_BI.')

        # Outputs
        self.add_output('w_B', shape=(nn, 3), units='rad/s',
                        desc='Angular velocity of the body frame wrt the inertial frame.')

    def compute_primal(self, O_BI, Odot_BI):
        return _compute_angular_velocity(O_BI, Odot_BI)
