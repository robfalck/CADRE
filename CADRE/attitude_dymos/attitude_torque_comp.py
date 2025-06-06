import openmdao.api as om
import jax
import jax.numpy as jnp


def _single_torque(w_B, wdot_B, J):
    """
    Return the torque required given the angular velocity and angular acceleration.
    """
    wx = jnp.array([[ 0.0000,-w_B[2], w_B[1]],
                    [ w_B[2], 0.0000, -w_B[0]],
                    [-w_B[1], w_B[0], 0.0000]])
    return jnp.dot(J, wdot_B) + jnp.dot(wx, jnp.dot(J, w_B))


_compute_torque_req = jax.vmap(_single_torque, in_axes=(0, 0, None))
""" Vectorized form of the O_BR calculation. """


class AttitudeTorqueComp(om.JaxExplicitComponent):
    """
    Calculate the torque required given the angular velocity and its rate.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self._J = jnp.array([[0.018, 0.000, 0.000],
                             [0.000, 0.018, 0.000],
                             [0.000, 0.000, 0.006]])


    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('w_B', shape=(nn, 3), units='rad/s',
                       desc='Angular velocity of the body frame wrt the inertial frame.')

        self.add_input('wdot_B', shape=(nn, 3), units='rad/s**2',
                       desc='Angular acceleration of the body frame wrt the inertial frame.')

        # Outputs
        self.add_output('T_req', shape=(nn, 3), units='unitless',
                        desc='Required torque vectory in the body frame.')

    def get_self_statics(self):
        # return value must be hashable
        return self._J,

    def compute_primal(self, w_B, wdot_B):
        return _compute_torque_req(w_B, wdot_B, self._J)
