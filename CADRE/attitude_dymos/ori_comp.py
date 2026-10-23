import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


def _compute_O_RI(r, v):
    """
    Return the transformation matrix from the rolled frame to the inertial frame.

    Parameters
    ----------
    r : array-like (3, n)
        Inertial planet-to-spacecraft position vector at n points in time.
    v : array_like (3, n)
        Inertial planet-to-spacecraft velocity vector, expressed in the inertial frame.

    Returns
    -------
    O_RI : array-like (3, 3, n)
        Transformation matrix to transform from the rolled frame to the inertial frame,
        ``x_I = O_RI x_R``. Its columns are the rolled axes expressed in the inertial frame.
    """
    # sqrt(sum(x**2)) rather than norm, which takes abs() and so is not complex-step safe.
    r_norm = r / jnp.sqrt(jnp.sum(r**2, axis=0))
    v_norm = v / jnp.sqrt(jnp.sum(v**2, axis=0))

    # iB = v_norm x r_norm, jB = -v_norm x iB
    iB = jnp.cross(v_norm, r_norm, axis=0)
    jB = -jnp.cross(v_norm, iB, axis=0)

    return jnp.stack((iB, jB, -v_norm), axis=1)


class ORIComp(JaxExplicitComp):
    """
    Coordinate transformation from the rolled (forward facing) frame to the inertial frame.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'r_e2b_I': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from earth to satellite in '
                                 'Earth-centered inertial frame over time'),
        'v_e2b_I': InputVar(shape=(3, 'num_nodes'), units='km/s',
                            desc='Velocity vector from earth to satellite in '
                                 'Earth-centered inertial frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'O_RI': OutputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                          desc='Rotation matrix from rolled body-fixed frame to '
                               'Earth-centered inertial frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['O_RI'] = _compute_O_RI(inputs['r_e2b_I'], inputs['v_e2b_I'])
