from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, VectorMagnitudeComp, CrossProductComp

from dymos import declare_state, declare_time, declare_parameter
from dymos.phases.grid_data import GridData

from .orbit_eom import OrbitEOMComp
from .battery_dymos import BatterySOCComp
from .thermal_dymos import ThermalTemperatureComp
from .attitude_dymos.attitude_group import  AttitudeGroup


@declare_time(units='s')
@declare_state('r_e2b_I', rate_source='orbit_eom_comp.dXdt:r_e2b_I', targets=['r_e2b_I'],
               units='km', shape=(3,))
@declare_state('v_e2b_I', rate_source='orbit_eom_comp.dXdt:v_e2b_I', targets=['v_e2b_I'],
               units='km/s', shape=(3,))
# @declare_state('temperature', rate_source='thermal_temp_comp.dXdt:temperature', targets=['temperature'])
@declare_state('SOC', rate_source='battery_soc_comp.dXdt:SOC', targets=['SOC'])
@declare_parameter('T_bat', targets=['T_bat'], units='degK')
@declare_parameter('P_bat', targets=['P_bat'], units='W')
class CadreODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))
        self.options.declare('grid_data', types=(GridData,))

    def setup(self):
        nn = self.options['num_nodes']
        gd = self.options['grid_data']

        self.add_subsystem('rmag_comp',
                           VectorMagnitudeComp(vec_size=nn, length=3, in_name='r_e2b_I',
                                               mag_name='rmag_e2b', units='km'),
                           promotes_inputs=['r_e2b_I'], promotes_outputs=['rmag_e2b'])

        self.add_subsystem('vmag_comp',
                           VectorMagnitudeComp(vec_size=nn, length=3, in_name='v_e2b_I',
                                               mag_name='vmag_e2b_I', units='km/s'),
                           promotes_inputs=['v_e2b_I'], promotes_outputs=['vmag_e2b_I'])

        self.add_subsystem('specific_angular_momentum_comp',
                           CrossProductComp(a_name='r_e2b_I', a_units='km', b_name='v_e2b_I',
                                            b_units='km/s', c_name='h_e2b_I', c_units='km**2/s',
                                            vec_size=nn),
                           promotes_inputs=['r_e2b_I', 'v_e2b_I'], promotes_outputs=['h_e2b_I'])

        self.add_subsystem('hmag_comp',
                           VectorMagnitudeComp(vec_size=nn, length=3, in_name='h_e2b_I',
                                               mag_name='hmag_e2b_I', units='km**2/s'),
                           promotes_inputs=['h_e2b_I'], promotes_outputs=['hmag_e2b_I'])

        self.add_subsystem('orbit_eom_comp', OrbitEOMComp(num_nodes=nn),
                           promotes_inputs=['rmag_e2b', 'r_e2b_I', 'v_e2b_I'])

        self.add_subsystem('attitude_group', AttitudeGroup(num_nodes=nn, grid_data=gd),
                           promotes_inputs=['r_e2b_I', 'rmag_e2b_I', 'v_e2b_I', 'vmag_e2b_I',
                                            'h_e2b_I', 'hmag_e2b_I'])

        # self.add_subsystem('thermal_temp_comp', ThermalTemperatureComp(num_nodes=nn),
        #                    promotes_inputs=['temperature', 'exposedArea', 'cellInstd', 'LOS', 'P_comm'])

        self.add_subsystem('battery_soc_comp', BatterySOCComp(num_nodes=nn),
                           promotes_inputs=['SOC', 'P_bat'])

        # # Only body tempearture is needed by battery.
        # body_idx = 5*np.arange(nn) + 4
        # self.connect('thermal_temp_comp.temperature', 'battery_soc_comp.T_bat',
        #              src_indices=body_idx, flat_src_indices=True)
