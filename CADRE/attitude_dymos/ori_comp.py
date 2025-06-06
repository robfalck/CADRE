import openmdao.api as om
import jax.numpy as jnp
import jax


def _single_O_RI(r, v):
    """
    Return a the transformation matrix from the rolled frame to the inertial frame.

    Parameters
    ----------
    r : array-like
        Inertial planet-to-spacecraft position vector.
    v : array_like
        Inertial planet-to-spacecraft velocity vector, expressed in the inertial frame.

    Returns
    -------
    O_RI : array-like
        Transformation matrix to transform from the rolled frame to the inertial frame.
    """
    rmag = jnp.sqrt(jnp.dot(r, r))
    vmag = jnp.sqrt(jnp.dot(v, v))

    r_norm = r / rmag
    v_norm = v / vmag

    vx = jnp.array([[       0.0, -v_norm[2],  v_norm[1]],
                    [ v_norm[2],         0., -v_norm[0]],
                    [-v_norm[1],  v_norm[0],         0.]])

    iB = jnp.dot(vx, r_norm)
    jB = -jnp.dot(vx, iB)

    return jnp.vstack((iB, jB, -v_norm))


_compute_O_RI = jax.vmap(_single_O_RI, in_axes=(0, 0))
""" Vectorized form of the calculation of O_RI. """


class ORIComp(om.JaxExplicitComponent):
    """
    Coordinate transformation from the interial plane to the rolled
    (forward facing) plane.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('r_e2b_I', shape=(nn, 3), units='km',
                       desc='Position vector from earth to satellite in '
                            'Earth-centered inertial frame over time')
        self.add_input('v_e2b_I', shape=(nn, 3), units='km/s',
                       desc='Velocity vector from earth to satellite in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('O_RI', shape=(nn, 3, 3), units='unitless',
                        desc='Rotation matrix from rolled body-fixed frame to '
                             'Earth-centered inertial frame over time')

    def compute_primal(self, r_e2b_I, v_e2b_I):
        return _compute_O_RI(r_e2b_I, v_e2b_I)
