"""
Thermal discipline for CADRE, on om4.

The legacy model integrated the panel and body temperatures with its own RK4
(``ThermalTemperature``). Here ``ThermalTemperatureComp`` supplies the rate of a dymos4 state
``temperature`` instead.
"""
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

# constants
m_f = 0.4
m_b = 2.0
cp_f = 0.6e3
cp_b = 2.0e3
A_T = 2.66e-3

alpha_c = 0.9
alpha_r = 0.2
eps_c = 0.87
eps_r = 0.88

q_sol = 1360.0
K = 5.67051e-8


def _panel_map():
    """
    Return the (5, 12) matrix that sums each panel into its body (index 4) or fin (0-3).

    Panels 0-3 are on the body; panel p >= 4 is on fin (p + 1) % 4. Each entry is divided by the
    thermal mass, m * cp, of the body or fin.
    """
    M = np.zeros((5, 12))
    for p in range(12):
        if p < 4:
            M[4, p] = 1.0 / (m_b * cp_b)
        else:
            M[(p + 1) % 4, p] = 1.0 / (m_f * cp_f)
    return M


#: Each panel's contribution to the temperature rate of its body or fin.
PANEL_MAP = _panel_map()


class ThermalTemperatureComp(JaxExplicitComp):
    """
    Calculates the rate of change of the temperatures of the 4 fins and the body.

    The legacy model's analytic state Jacobian summed the emissivity over every cell of every
    panel for each panel, where its rate sums each panel's own cells; JAX differentiates the rate
    itself here.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'temperature': InputVar(shape=(5, 'num_nodes'), val=273.0, units='degK',
                                desc='Temperature for the 4 fins and body over time.',
                                tags={'dymos.state_target:temperature'}),
        'exposedArea': InputVar(shape=(7, 12, 'num_nodes'), val=0.0, units='m**2',
                                desc='Exposed area to the sun for each solar cell over time'),
        'cellInstd': InputVar(shape=(7, 12), val=1.0, units='unitless',
                              desc='Cell/Radiator indication', tags={'dymos.static_target'}),
        'LOS': InputVar(shape=('num_nodes',), val=0.0, units='unitless',
                        desc='Satellite to Sun line of sight over time'),
        'P_comm': InputVar(shape=('num_nodes',), val=1.0, units='W',
                           desc='Communication power over time'),
    }

    outputs: dict[str, OutputVar] = {
        'temperature_dot': OutputVar(shape=(5, 'num_nodes'), units='degK/s',
                                     desc='Rate of change of the fin and body temperatures.',
                                     tags={'dymos.state_rate_source:temperature',
                                           'dymos.state_units:degK'}),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        T = inputs['temperature']
        cellInstd = inputs['cellInstd']

        fact1 = q_sol*inputs['LOS']
        fact2 = K*A_T*T**4

        alpha = alpha_c*cellInstd + alpha_r - alpha_r*cellInstd
        s_eps = jnp.sum(eps_c*cellInstd + eps_r - eps_r*cellInstd, axis=0)
        s_al_ea = jnp.einsum('cp,cpn->pn', alpha, inputs['exposedArea'])*fact1

        f = PANEL_MAP @ s_al_ea - (PANEL_MAP @ s_eps)[:, jnp.newaxis]*fact2
        outputs['temperature_dot'] = f.at[4].add(4.0 * inputs['P_comm'] / m_b / cp_b)
