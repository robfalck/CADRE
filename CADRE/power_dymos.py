"""
Power discipline for CADRE, on om4.

The cell voltage is interpolated from ``cell_voltage_table()``, the legacy MBI fit of the cell
I-V curves (ADL-007), with om4's ``StructuredMetaModelComp`` (OpenMDAO/om4 PR #404,
ai/OM4_NEEDS.md #3).
"""
import jax.numpy as jnp
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

from om4.api import Group, InputVar, JaxExplicitComp, OutputVar, StructuredMetaModelComp
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from CADRE.tables import cell_voltage_table

#: The thermal node of each panel: the body (index 4) for panels 0 to 3, then the fins in turn.
PANEL_THERMAL_NODE = [4, 4, 4, 4, 0, 1, 2, 3, 0, 1, 2, 3]


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


class CellVoltagePointsComp(JaxExplicitComp):
    """
    The cell voltage table's inputs for every cell of every panel at every node.

    Each cell takes its panel's temperature and current, and its illuminated area
    ``LOS * exposedArea``, as ``Power_CellVoltage.setx`` built them. The points are ordered as
    ``(7, 12, num_nodes)``, flattened.
    """
    num_nodes: int = Field(default=1)

    @property
    def num_cells(self):
        return 7 * 12 * self.num_nodes

    inputs: dict[str, InputVar] = {
        'LOS': InputVar(shape=('num_nodes',), val=0.0, units='unitless',
                        desc='Line of Sight over Time'),
        'temperature': InputVar(shape=(5, 'num_nodes'), val=0.0, units='degK',
                                desc='Temperature of solar cells over time'),
        'exposedArea': InputVar(shape=(7, 12, 'num_nodes'), val=0.0, units='m**2',
                                desc='Exposed area to sun for each solar cell over time'),
        'Isetpt': InputVar(shape=(12, 'num_nodes'), val=0.0, units='A',
                           desc='Currents of the solar panels'),
    }

    # Defaults inside the table: a solver may evaluate the table before this has run.
    outputs: dict[str, OutputVar] = {
        'T_cell': OutputVar(shape=('num_cells',), val=273.0, units='degK',
                            desc='Temperature of each cell'),
        'A_cell': OutputVar(shape=('num_cells',), val=0.0, units='m**2',
                            desc='Illuminated area of each cell'),
        'I_cell': OutputVar(shape=('num_cells',), val=0.0, units='A', desc='Current of each cell'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        shape = (7, 12, self.num_nodes)
        T_panel = inputs['temperature'][jnp.array(PANEL_THERMAL_NODE)]
        outputs['T_cell'] = jnp.broadcast_to(T_panel, shape).ravel()
        outputs['A_cell'] = (inputs['LOS'] * inputs['exposedArea']).ravel()
        outputs['I_cell'] = jnp.broadcast_to(inputs['Isetpt'], shape).ravel()


class CellVoltageSumComp(JaxExplicitComp):
    """
    Sum the voltages of the 7 cells of each panel.
    """
    num_nodes: int = Field(default=1)

    @property
    def num_cells(self):
        return 7 * 12 * self.num_nodes

    inputs: dict[str, InputVar] = {
        'V_cell': InputVar(shape=('num_cells',), val=0.0, units='V',
                           desc='Voltage of each cell, ordered as (7, 12, num_nodes)'),
    }

    outputs: dict[str, OutputVar] = {
        'V_sol': OutputVar(shape=(12, 'num_nodes'), units='V',
                           desc='Output voltage of solar panel over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['V_sol'] = jnp.sum(inputs['V_cell'].reshape(7, 12, self.num_nodes), axis=0)


def _cell_voltage_subsystems(num_nodes, method, extrapolate):
    nn = num_nodes
    grid, V = cell_voltage_table()
    n_cells = 7 * 12 * nn
    return {
        'points': CellVoltagePointsComp(num_nodes=nn,
                                        promotes_inputs=['LOS', 'temperature', 'exposedArea', 'Isetpt'],
                                        promotes_outputs=['T_cell', 'A_cell', 'I_cell']),
        'table': StructuredMetaModelComp(
            grid={'T_cell': grid['T'], 'A_cell': grid['A'], 'I_cell': grid['I']},
            training_outputs={'V_cell': V}, method=method, vec_size=n_cells, extrapolate=extrapolate,
            inputs={'T_cell': InputVar(shape=(n_cells,), units='degK'),
                    'A_cell': InputVar(shape=(n_cells,), units='m**2'),
                    'I_cell': InputVar(shape=(n_cells,), units='A')},
            outputs={'V_cell': OutputVar(shape=(n_cells,), units='V')},
            promotes_inputs=['T_cell', 'A_cell', 'I_cell'], promotes_outputs=['V_cell']),
        'sum': CellVoltageSumComp(num_nodes=nn, promotes_inputs=['V_cell'], promotes_outputs=['V_sol']),
    }


class CellVoltageGroup(Group):
    """
    Output voltage of each solar panel (``Power_CellVoltage``).

    Each panel's voltage is the sum of its 7 cells', each interpolated in the cell's
    temperature, illuminated area and current from ``cell_voltage_table()``, the legacy MBI fit
    of the table, to about 6e-4 of the range.

    ``method`` is the ``StructuredMetaModelComp`` method. The default, ``'lagrange3'``, is
    complex-step safe, as CADRE's ODE checks need; ``'cubic'`` is more accurate (4e-4) and
    smoother, but not complex-step safe.

    With ``extrapolate=True`` a point outside the table is extrapolated (with a warning) rather
    than raising ``AnalysisError``.
    """
    num_nodes: int = Field(default=1)
    method: str = Field(default='lagrange3', description='Interpolation method of the table.')
    extrapolate: bool = Field(default=False, description='Extrapolate outside the table.')

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode='before')
    @classmethod
    def _build_from_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data['subsystems'] = _cell_voltage_subsystems(data.get('num_nodes', 1),
                                                      data.get('method', 'lagrange3'),
                                                      data.get('extrapolate', False))
        return data
