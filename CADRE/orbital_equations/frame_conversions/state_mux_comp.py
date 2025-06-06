import openmdao.api as om
import jax.numpy as jnp


class StateMuxComp(om.JaxExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='km')
        self.add_input('y', shape=(nn,), units='km')
        self.add_input('z', shape=(nn,), units='km')
        self.add_input('vx', shape=(nn,), units='km/s')
        self.add_input('vy', shape=(nn,), units='km/s')
        self.add_input('vz', shape=(nn,), units='km/s')

        self.add_output('r_e2b_I', shape=(nn, 3), units='km')
        self.add_output('v_e2b_I', shape=(nn, 3), units='km/s')

    def compute_primal(self, x, y, z, vx, vy, vz):
        r_e2b_I = jnp.hstack((x[:, jnp.newaxis], y[:, jnp.newaxis], z[:, jnp.newaxis]))
        v_e2b_I = jnp.hstack((vx[:, jnp.newaxis], vy[:, jnp.newaxis], vz[:, jnp.newaxis]))
        return r_e2b_I, v_e2b_I

