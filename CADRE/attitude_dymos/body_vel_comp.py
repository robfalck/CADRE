import openmdao.api as om
import jax.numpy as jnp


class BodyVelComp(om.JaxExplicitComponent):
    """
    Determine velocity in the body frame.

    This  component was Attitude_Sideslip in the original version.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r_e2b_I', shape=(nn, 3), units='km',
                       desc='Earth-to-satellite position vector in the inertial frame.')

        self.add_input('v_e2b_I', shape=(nn, 3), units='km/s',
                       desc='Earth-to-satellite velocity vector in the inertial frame.')

        self.add_input('w_B', shape=(nn, 3), units='rad/s',
                       desc='Angular velocity of the body frame wrt the inertial frame.')

        self.add_input('O_BI',  shape=(nn, 3, 3), units='unitless',
                       desc='Rotation matrix from body-fixed frame to '
                      'Earth-centered inertial frame over time')

        self.add_output('v_e2b_B', shape=(nn, 3), units='km/s',
                        desc='Earth-to-satellite velocity vector in the body-fixed frame')

    def compute_primal(self, r_e2b_I, v_e2b_I, w_B, O_BI):
        # TODO: This formulation is missing the omega x r term from the basic kinematic equation
        # Need to add jnp.cross(omega_dot, r, axis=-1)
        return jnp.einsum('kij,kj->ki', O_BI, v_e2b_I) # + jnp.cross(w_B, r_e2b_I, axis=-1)
