from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, VectorMagnitudeComp

from dymos import declare_state, declare_time, declare_parameter

from CADRE.battery_dymos import BatterySOCComp
from CADRE.orbit_eom import OrbitEOMComp
from CADRE.rw_dymos.rw_group import ReactionWheelGroup
from CADRE.solar_dymos import SolarExposedAreaComp
from CADRE.sun_dymos.sun_group import SunGroup
from CADRE.thermal_dymos import ThermalTemperatureComp
from CADRE.attitude_dymos.attitude_group import AttitudeGroup


@declare_time(units='s', targets=['time'])
@declare_parameter('r_e2b_I', targets=['r_e2b_I'], units='km', shape=(3,))
@declare_parameter('O_BI', targets=['O_BI'], shape=(3, 3))
@declare_parameter('w_B', targets=['w_B'], shape=(3,), units='1/s')
@declare_parameter('wdot_B', targets=['wdot_B'], shape=(3,), units='1/s**2')
@declare_state('w_RW', rate_source='rw_group.dXdt:w_RW', shape=(3,), targets=['w_RW'], units='rad/s')
# @declare_state('temperature', rate_source='thermal_temp_comp.dXdt:temperature', targets=['temperature'])
@declare_state('SOC', rate_source='battery_soc_comp.dXdt:SOC', targets=['SOC'])
# @declare_parameter('T_bat', targets=['T_bat'], units='degK')
@declare_parameter('LD', targets=['LD'], units='d', dynamic=False)  # Launch date, MJD
@declare_parameter('fin_angle', targets=['fin_angle'], units='deg', dynamic=False)  # Panel fin sweep angle
@declare_parameter('P_bat', targets=['P_bat'], units='W')
class CadreSystemsODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('sun_group', SunGroup(num_nodes=nn),
                           promotes_inputs=['r_e2b_I', 'O_BI', ('t', 'time'), 'LD'],
                           promotes_outputs=['azimuth', 'elevation'])

        self.add_subsystem('solar_comp', SolarExposedAreaComp(num_nodes=nn),
                           promotes_inputs=['fin_angle', 'azimuth', 'elevation'],
                           promotes_outputs=['exposedArea'])

        self.add_subsystem('rw_group', ReactionWheelGroup(num_nodes=nn),
                           promotes_inputs=['w_RW', 'w_B', 'wdot_B'],
                           promotes_outputs=['P_RW', 'T_RW', 'T_m'])

        # self.add_subsystem('thermal_temp_comp', ThermalTemperatureComp(num_nodes=nn),
        #                    promotes_inputs=['exposedArea', 'cellInstd', 'LOS', 'P_comm'])
        #
        self.add_subsystem('battery_soc_comp', BatterySOCComp(num_nodes=nn),
                           promotes_inputs=['SOC', 'P_bat'])
        #
        # # Only body tempearture is needed by battery.
        # body_idx = 5*np.arange(nn) + 4
        # self.connect('thermal_temp_comp.temperature', 'battery_soc_comp.T_bat',
        #              flat_src_indices=body_idx)
