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


@declare_time(units='s')
@declare_state('r_e2b_I', rate_source='orbit_eom_comp.dXdt:r_e2b_I', targets=['r_e2b_I'],
               units='km', shape=(3,))
@declare_state('v_e2b_I', rate_source='orbit_eom_comp.dXdt:v_e2b_I', targets=['v_e2b_I'],
               units='km/s', shape=(3,))
@declare_state('temperature', rate_source='thermal_temp_comp.dXdt:temperature', targets=['temperature'])
@declare_state('SOC', rate_source='battery_soc_comp.dXdt:SOC', targets=['SOC'])
@declare_state('w_RW', rate_source='rw_group.dXdt:w_RW', targets=['w_RW'])
@declare_parameter('T_bat', targets=['T_bat'], units='degK')
@declare_parameter('P_bat', targets=['P_bat'], units='W')
class CadreODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('rmag_comp',
                           VectorMagnitudeComp(vec_size=nn, length=3, in_name='r_e2b_I',
                                               mag_name='rmag_e2b', units='km'),
                           promotes_inputs=['r_e2b_I'], promotes_outputs=['rmag_e2b'])

        self.add_subsystem('orbit_eom_comp', OrbitEOMComp(num_nodes=nn),
                           promotes_inputs=['rmag_e2b', 'r_e2b_I', 'v_e2b_I'])

        self.add_subsystem('sun_group', SunGroup(num_nodes=nn),
                           promotes_inputs=['r_e2b_I', 'O_BI'],
                           promotes_outputs=['azimuth', 'elevation'])

        self.add_subsystem('solar_comp', SolarExposedAreaComp(num_nodes=nn),
                           promotes_inputs=['fin_angle', 'azimuth', 'elevation'],
                           promotes_outputs=['exposedArea'])

        self.add_subsystem('rw_group', ReactionWheelGroup(num_nodes=nn),
                           promotes_inputs=['w_RW', 'w_B', 'T_RW'],
                           promotes_outputs=['P_RW'])

        self.add_subsystem('thermal_temp_comp', ThermalTemperatureComp(num_nodes=nn),
                           promotes_inputs=['exposedArea', 'cellInstd', 'LOS', 'P_comm'])

        self.add_subsystem('battery_soc_comp', BatterySOCComp(num_nodes=nn),
                           promotes_inputs=['SOC', 'P_bat'])

        # Only body tempearture is needed by battery.
        body_idx = 5*np.arange(nn) + 4
        self.connect('thermal_temp_comp.temperature', 'battery_soc_comp.T_bat',
                     flat_src_indices=body_idx)
