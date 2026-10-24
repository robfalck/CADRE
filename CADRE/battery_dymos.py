"""
Battery discipline for CADRE, on om4.

The legacy model integrated the state of charge with its own RK4 (``BatterySOC``). Here
``BatterySOCComp`` supplies the rate of a dymos4 state ``SOC`` instead.

The legacy ``BatteryConstraints`` aggregated the current and state-of-charge limits over the
whole time history with KS functions, so as not to need one constraint per time step. In
dymos4 those limits are path constraints, enforced at every node; ``battery_path_constraints``
returns them.
"""
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar
from dymos4.options import PathConstraint

# Constants
sigma = 1e-10
eta = 0.99
Cp = 2900.0*0.001*3600.0
IR = 0.9
T0 = 293.0
alpha = np.log(1/1.1**5)

#: Current and state-of-charge limits, from the legacy BatteryConstraints.
I_MIN = -10.0
I_MAX = 5.0
SOC_MIN = 0.2
SOC_MAX = 1.0


def _battery_voltage(SOC, T):
    """Return the battery voltage for a state of charge and battery (body) temperature."""
    voc = 3 + jnp.expm1(SOC) / (np.e-1)
    return IR * voc * (2.0 - jnp.exp(alpha*(T-T0)/T0))


_INPUTS = {
    'SOC': InputVar(shape=('num_nodes',), val=0.5, units='unitless',
                    desc='Battery state of charge over time',
                    tags={'dymos.state_target:SOC'}),
    'P_bat': InputVar(shape=('num_nodes',), val=0.0, units='W',
                      desc='Battery power over time'),
    'temperature': InputVar(shape=(5, 'num_nodes'), val=T0, units='degK',
                            desc='Temperature for the 4 fins and body over time; the battery '
                                 'is at the body temperature, the last entry.',
                            tags={'dymos.state_target:temperature'}),
}


class BatterySOCComp(JaxExplicitComp):
    """
    Computes the rate of change of the battery state of charge.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = _INPUTS

    outputs: dict[str, OutputVar] = {
        'SOC_dot': OutputVar(shape=('num_nodes',), units='1/s',
                             desc='Rate of change of the battery state of charge',
                             tags={'dymos.state_rate_source:SOC', 'dymos.state_units:unitless'}),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        SOC = inputs['SOC']
        I = inputs['P_bat'] / _battery_voltage(SOC, inputs['temperature'][4])  # noqa: E741
        outputs['SOC_dot'] = -sigma/24*SOC + eta/Cp*I


class BatteryPowerComp(JaxExplicitComp):
    """
    Current supplied by the battery.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = _INPUTS

    outputs: dict[str, OutputVar] = {
        'I_bat': OutputVar(shape=('num_nodes',), units='A',
                           desc='Battery Current over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['I_bat'] = inputs['P_bat'] / _battery_voltage(inputs['SOC'],
                                                              inputs['temperature'][4])


def battery_path_constraints(I_bat='ode.I_bat', SOC='states_SOC'):
    """
    Return the battery current and state-of-charge limits as dymos4 path constraints.

    Parameters
    ----------
    I_bat : str
        Phase-level name of the battery current.
    SOC : str
        Phase-level name of the state of charge.

    Returns
    -------
    list of PathConstraint
        ``I_MIN <= I_bat <= I_MAX`` and ``SOC_MIN <= SOC <= SOC_MAX``.
    """
    return [PathConstraint(name=I_bat, lower=I_MIN, upper=I_MAX, units='A'),
            PathConstraint(name=SOC, lower=SOC_MIN, upper=SOC_MAX)]
