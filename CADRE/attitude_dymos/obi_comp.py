import openmdao.api as om
import jax.numpy as jnp


class OBIComp(om.JaxExplicitComponent):
    """
    Calculate the transformation matrix from the body frame to the inertial frame.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('O_BR', shape=(nn, 3, 3), units='unitless',
                       desc='Rotation matrix from body-fixed frame to rolled '
                       'body-fixed frame over time')

        self.add_input('O_RI',  shape=(nn, 3, 3), units='unitless',
                       desc='Rotation matrix from rolled body-fixed '
                       'frame to Earth-centered inertial frame over time')

        # Outputs
        self.add_output('O_BI',  shape=(nn, 3, 3), units='unitless',
                        desc='Rotation matrix from body-fixed frame to '
                        'Earth-centered inertial frame over time')

    def compute_primal(self, O_BR, O_RI):
        return jnp.einsum('nij,njk->nik', O_BR, O_RI)
