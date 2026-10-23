import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


class ReactionWheel(JaxExplicitComp):
    """
    Compute reaction wheel power.
    """
    # constants
    V: float = 4.0
    a: float = 4.9e-4
    b: float = 4.5e2
    I0: float = 0.017
    J_RW: float = 2.8e-5

    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'w_B': InputVar(shape=(3, 'num_nodes'), val=0.0, units='rad/s',
                        desc='Angular velocity vector in body-fixed frame over time'),
        'w_RW': InputVar(shape=(3, 'num_nodes'), units='rad/s',
                         desc='Angular velocity vector of reaction wheel over time',
                         tags={'dymos.state_target:w_RW'}),
        'T_RW': InputVar(shape=(3, 'num_nodes'), val=0.0, units='N*m',
                         desc='Torque vector of reaction wheel over time'),
    }

    outputs: dict[str, OutputVar] = {
        'P_RW': OutputVar(shape=(3, 'num_nodes'), units='W',
                          desc='Reaction wheel power over time'),
        'alpha_RW': OutputVar(shape=(3, 'num_nodes'), units='rad/s**2',
                              desc='Angular acceleration vector of reaction wheel over time',
                              tags={'dymos.state_rate_source:w_RW', 'dymos.state_units:rad/s'}),
    }

    def get_self_statics(self):
        return {'V': self.V, 'a': self.a, 'b': self.b, 'I0': self.I0, 'J_RW': self.J_RW}

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        """
        Calculate outputs.
        """
        w_B = inputs['w_B']
        w_RW = inputs['w_RW']
        T_RW = inputs['T_RW']

        V = statics['V']
        outputs['P_RW'] = V * (statics['a'] * w_RW + statics['b'] * T_RW)**2 + V * statics['I0']
        outputs['alpha_RW'] = -T_RW / statics['J_RW'] - jnp.cross(w_B, w_RW, axis=0)
