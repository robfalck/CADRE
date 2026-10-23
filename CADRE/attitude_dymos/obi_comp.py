import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


class OBIComp(JaxExplicitComp):
    """
    Calculate the transformation matrix from the body frame to the inertial frame.

    ``x_I = O_RI x_R = O_RI O_BR x_B``, so ``O_BI = O_RI O_BR``.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'O_BR': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from body-fixed frame to rolled '
                              'body-fixed frame over time'),
        'O_RI': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from rolled body-fixed '
                              'frame to Earth-centered inertial frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'O_BI': OutputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                          desc='Rotation matrix from body-fixed frame to '
                               'Earth-centered inertial frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['O_BI'] = jnp.einsum('ijn,jkn->ikn', inputs['O_RI'], inputs['O_BR'])
