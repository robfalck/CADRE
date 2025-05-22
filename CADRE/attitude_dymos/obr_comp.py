import openmdao.api as om
import jax
import jax.numpy as jnp


def _compute_O_BR(gam):
    """
    Return a the transformation matrix from body-fixed to rolled frame.
    """
    def _single_O_BR(gam):
        cgam = jnp.cos(gam)
        sgam = jnp.sin(gam)
        return jnp.array([
            [ cgam, sgam, 0.0],
            [-sgam, cgam, 0.0],
            [  0.0,  0.0, 1.0]
        ])
    return jax.vmap(_single_O_BR)(gam)


class OBRComp(om.JaxExplicitComponent):
    """
    Calculates the body-fixed orientation matrix.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('gamma', shape=(nn,), units='rad',
                       desc='Satellite roll angle over time')

        # Outputs
        self.add_output('O_BR', shape=(nn, 3, 3), units=None,
                        desc='Rotation matrix from body-fixed frame to rolled '
                        'body-fixed frame over time')

    def compute_primal(self, gamma):
        return _compute_O_BR(gamma)
