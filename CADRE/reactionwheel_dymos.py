import openmdao.api as om
import jax.numpy as jnp
import jax.debug

class ReactionWheel(om.JaxExplicitComponent):
    """
    Compute reaction wheel power.
    """
    # constants
    V = 4.0
    a = 4.9e-4
    b = 4.5e2
    I0 = 0.017

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        # Inputs
        self.add_input('w_B', shape=(n, 3), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        self.add_input('w_RW', shape=(n, 3), units='1/s',
                       desc='Angular velocity vector of reaction wheel over time')

        self.add_input('T_RW', shape=(n, 3), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        # Outputs
        self.add_output('P_RW', shape=(n, 3), units='W',
                        desc='Reaction wheel power over time')

        self.add_output('alpha_RW', shape=(n, 3), units='1/s**2',
                        desc='Angular acceleration vector of reaction wheel over time')

        # unit conversion of some kind
        self.J_RW = 2.8e-5

    def compute_primal(self, w_B, w_RW, T_RW):
        """
        Calculate outputs.
        """
        P_RW0 = (self.V * (self.a * w_RW[:, 0] +
                          self.b * T_RW[:, 0])**2 +
                          self.V * self.I0)
        P_RW1 = (self.V * (self.a * w_RW[:, 1] +
                          self.b * T_RW[:, 1])**2 +
                          self.V * self.I0)
        P_RW2 = (self.V * (self.a * w_RW[:, 2] +
                          self.b * T_RW[:, 2])**2 +
                          self.V * self.I0)

        P_RW = jnp.vstack((P_RW0, P_RW1, P_RW2))

        a2 = jnp.vstack((-w_B[:,2]*w_RW[:,1] + w_B[:,1]*w_RW[:,2],
                          w_B[:,2]*w_RW[:,0] - w_B[:,0]*w_RW[:,2],
                         -w_B[:,1]*w_RW[:,0] + w_B[:,0]*w_RW[:,1]))

        alpha_RW = -T_RW / self.J_RW - a2.T

        return P_RW, alpha_RW