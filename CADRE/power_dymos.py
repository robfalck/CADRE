"""
Power discipline for CADRE, on om4.

Not ported yet: ``Power_CellVoltage``, which interpolates the solar cell voltage table and is
blocked by ai/OM4_NEEDS.md #3. ``SolarPowerComp`` takes the panel voltages as an input.
"""
import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar


class SolarPowerComp(JaxExplicitComp):
    """
    Compute the output power of the solar panels.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'Isetpt': InputVar(shape=(12, 'num_nodes'), val=0.0, units='A',
                           desc='Currents of the solar panels'),
        'V_sol': InputVar(shape=(12, 'num_nodes'), val=0.0, units='V',
                          desc='Output voltage of solar panel over time'),
    }

    outputs: dict[str, OutputVar] = {
        'P_sol': OutputVar(shape=('num_nodes',), units='W', desc='Solar panels power over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['P_sol'] = jnp.sum(inputs['V_sol'] * inputs['Isetpt'], axis=0)


class PowerTotalComp(JaxExplicitComp):
    """
    Compute the battery power which is the sum of the loads.

    This includes a 2-Watt constant power usage that accounts for the scientific instruments on
    the satellite and small actuator inputs in response to disturbance torques.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'P_sol': InputVar(shape=('num_nodes',), val=0.0, units='W',
                          desc='Solar panels power over time'),
        'P_comm': InputVar(shape=('num_nodes',), val=0.0, units='W',
                           desc='Communication power over time'),
        'P_RW': InputVar(shape=(3, 'num_nodes'), val=0.0, units='W',
                         desc='Power used by reaction wheel over time'),
    }

    outputs: dict[str, OutputVar] = {
        'P_bat': OutputVar(shape=('num_nodes',), units='W', desc='Battery power over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['P_bat'] = (inputs['P_sol'] - 5.0*inputs['P_comm'] -
                            jnp.sum(inputs['P_RW'], axis=0) - 2.0)
