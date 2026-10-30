"""
The full CADRE ODE: orbit, attitude, reaction wheels, sun, solar, thermal, power, battery and comm.

This replaces the legacy ``CADRE`` group (``CADRE/CADRE_group.py``), which held the whole time
history of one design point and integrated it with its own RK4 components. Here every discipline
is pointwise and dymos4 integrates the states (ADL-006):

- the orbit, as modified equinoctial elements ``p, f, g, h, k, L`` (``TwoBodyEOM``, J2 only),
- the reaction wheel speeds ``w_RW``, the five ``temperature`` nodes, the battery ``SOC`` and
  the data downloaded ``Data``.

The attitude is the quaternion control ``q`` (ADL-002), constrained to point the body z axis
anti-velocity by the path constraints ``pointing_err = 0`` and ``q_norm_err = 0``; the legacy
roll angle ``gamma`` is subsumed by it. The panel currents ``Isetpt`` and the comm power
``P_comm`` are controls, and ``LD``, ``lat``, ``lon``, ``alt``, ``finAngle``, ``antAngle`` and
``cellInstd`` are static parameters, all matched by name.
"""
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

from om4.api import Connection, Group, InputDefault
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from CADRE.attitude_dymos import (AngularAccelerationComp, AngularVelocityComp,
                                  AttitudeTorqueComp, ORIComp, PointingComp,
                                  QuaternionAttitudeComp)
from CADRE.battery_dymos import BatteryPowerComp, BatterySOCComp
from CADRE.comm_dymos import (CommAntRotationComp, CommAntRotationMtxComp, CommBitRateComp,
                              CommDistanceComp, CommEarthsSpinComp, CommEarthsSpinMtxComp,
                              CommGSPosECIComp, CommGSPosEarthComp, CommLOSComp,
                              CommVectorAntComp, CommVectorBodyComp, CommVectorECIComp,
                              CommVectorSphericalComp, GainPatternGroup)
from CADRE.orbital_equations.EOMs import TwoBodyEOM
from CADRE.orbital_equations.frame_conversions import StateMuxComp
from CADRE.power_dymos import CellVoltageGroup, PowerTotalComp, SolarPowerComp
from CADRE.reactionwheel_dymos import ReactionWheel
from CADRE.solar_dymos import ExposedAreaGroup
from CADRE.sun_dymos import (SunLOSComp, SunPositionBodyComp, SunPositionECIComp,
                             SunPositionSphericalComp)
from CADRE.thermal_dymos import ThermalTemperatureComp


def _cadre_ode_subsystems(num_nodes, table_method):
    nn = num_nodes
    prom = {'promotes_inputs': ['*'], 'promotes_outputs': ['*']}
    tables = {'method': table_method, **prom}
    return {
        # Orbit
        'orbit': TwoBodyEOM(num_nodes=nn, **prom),
        'state_mux': StateMuxComp(num_nodes=nn, **prom),
        # Attitude
        'o_ri': ORIComp(num_nodes=nn, **prom),
        'attitude': QuaternionAttitudeComp(num_nodes=nn, **prom),
        'pointing': PointingComp(num_nodes=nn, **prom),
        'w_b': AngularVelocityComp(num_nodes=nn, **prom),
        'wdot_b': AngularAccelerationComp(num_nodes=nn, **prom),
        'torque': AttitudeTorqueComp(num_nodes=nn, **prom),
        # Reaction wheels
        'reaction_wheel': ReactionWheel(num_nodes=nn, **prom),
        # Sun
        'sun_eci': SunPositionECIComp(num_nodes=nn, **prom),
        'sun_body': SunPositionBodyComp(num_nodes=nn, **prom),
        'sun_spherical': SunPositionSphericalComp(num_nodes=nn, **prom),
        'sun_los': SunLOSComp(num_nodes=nn, **prom),
        # Solar
        'exposed_area': ExposedAreaGroup(num_nodes=nn, **tables),
        # Thermal
        'thermal': ThermalTemperatureComp(num_nodes=nn, **prom),
        # Power
        # Radau extrapolates each segment's controls to its end node, which can take Isetpt a
        # little below 0, the edge of the table; the legacy MBI also extended its tables.
        'cell_voltage': CellVoltageGroup(num_nodes=nn, extrapolate=True, **tables),
        'solar_power': SolarPowerComp(num_nodes=nn, **prom),
        'power_total': PowerTotalComp(num_nodes=nn, **prom),
        # Battery
        'battery_soc': BatterySOCComp(num_nodes=nn, **prom),
        'battery_power': BatteryPowerComp(num_nodes=nn, **prom),
        # Comm
        'gs_earth': CommGSPosEarthComp(num_nodes=nn, **prom),
        'earth_spin': CommEarthsSpinComp(num_nodes=nn, **prom),
        'earth_spin_mtx': CommEarthsSpinMtxComp(num_nodes=nn, **prom),
        'gs_eci': CommGSPosECIComp(num_nodes=nn, **prom),
        'vector_eci': CommVectorECIComp(num_nodes=nn, **prom),
        'vector_body': CommVectorBodyComp(num_nodes=nn, **prom),
        'ant_rotation': CommAntRotationComp(num_nodes=nn, **prom),
        'ant_rotation_mtx': CommAntRotationMtxComp(num_nodes=nn, **prom),
        'vector_ant': CommVectorAntComp(num_nodes=nn, **prom),
        'vector_spherical': CommVectorSphericalComp(num_nodes=nn, **prom),
        'comm_los': CommLOSComp(num_nodes=nn, **prom),
        'gain': GainPatternGroup(num_nodes=nn, **tables),
        'distance': CommDistanceComp(num_nodes=nn, **prom),
        'bit_rate': CommBitRateComp(num_nodes=nn, **prom),
    }


class CADREODE(Group):
    """
    The CADRE ODE for one design point.

    ``table_method`` is the interpolation method of the exposed area, cell voltage and gain
    tables (ADL-007): ``'lagrange3'`` (the default) is complex-step safe, ``'cubic'`` is smoother.
    """
    num_nodes: int = Field(default=1)
    table_method: str = Field(default='lagrange3', description='Interpolation method of the tables.')

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    connections: list[Connection] = Field(default=PydanticUndefined, init=False)
    input_defaults: dict[str, InputDefault] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode='before')
    @classmethod
    def _build_from_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data['subsystems'] = _cadre_ode_subsystems(data.get('num_nodes', 1),
                                                   data.get('table_method', 'lagrange3'))
        # The wheels supply the torque the prescribed attitude requires.
        data['connections'] = [Connection(src='T_req', tgt='T_RW')]
        # Inputs promoted together whose components disagree on the default.
        data['input_defaults'] = {
            'P_comm': InputDefault(val=0.0, units='W'),
            'temperature': InputDefault(val=273.0, units='degK'),
        }
        return data
