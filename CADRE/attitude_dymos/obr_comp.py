import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


def _compute_O_BR(gamma):
    """
    Return the transformation matrix from body-fixed to rolled frame at each node.

    ``x_R = O_BR x_B``, so the columns of O_BR are the body axes expressed in the rolled frame.

    Parameters
    ----------
    gamma : array-like (n,)
        Roll angle at n points in time.

    Returns
    -------
    O_BR : array-like (3, 3, n)
        Component-major rotation matrices, node axis last.
    """
    cgam = jnp.cos(gamma)
    sgam = jnp.sin(gamma)
    z = jnp.zeros_like(gamma)
    o = jnp.ones_like(gamma)
    return jnp.stack([jnp.stack([cgam, -sgam, z]),
                      jnp.stack([sgam, cgam, z]),
                      jnp.stack([z, z, o])])


class OBRComp(JaxExplicitComp):
    """
    Calculates the body-fixed orientation matrix.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'gamma': InputVar(shape=('num_nodes',), units='rad',
                          desc='Satellite roll angle over time'),
    }

    outputs: dict[str, OutputVar] = {
        'O_BR': OutputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                          desc='Rotation matrix from body-fixed frame to rolled '
                               'body-fixed frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['O_BR'] = _compute_O_BR(inputs['gamma'])
