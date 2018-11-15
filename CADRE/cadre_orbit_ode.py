from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, VectorMagnitudeComp, CrossProductComp, MuxComp

from dymos import declare_state, declare_time, declare_parameter

from .orbit_eom import OrbitEOMComp
from .battery_dymos import BatterySOCComp
from .thermal_dymos import ThermalTemperatureComp
from .attitude_dymos.attitude_group import  AttitudeGroup
from .rowwise_divide_comp import RowwiseDivideComp
from .ori_comp import ORIComp
from .obr_comp import OBRComp
from .obi_comp import OBIComp


@declare_time(units='s')
@declare_state('r_e2b_I', rate_source='orbit_eom_comp.dXdt:r_e2b_I', targets=['r_e2b_I'],
               units='km', shape=(3,))
@declare_state('v_e2b_I', rate_source='orbit_eom_comp.dXdt:v_e2b_I', targets=['v_e2b_I'],
               units='km/s', shape=(3,))
# @declare_state('temperature', rate_source='thermal_temp_comp.dXdt:temperature', targets=['temperature'])
# @declare_state('SOC', rate_source='battery_soc_comp.dXdt:SOC', targets=['SOC'])
# @declare_parameter('T_bat', targets=['T_bat'], units='degK')
# @declare_parameter('P_bat', targets=['P_bat'], units='W')
@declare_parameter('Gamma', targets=['Gamma'], units='rad')
class CadreODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('rmag_comp',
                           VectorMagnitudeComp(vec_size=nn, length=3, in_name='r_e2b_I',
                                               mag_name='rmag_e2b_I', units='km'),
                           promotes_inputs=['r_e2b_I'], promotes_outputs=['rmag_e2b_I'])

        # self.add_subsystem('vmag_comp',
        #                    VectorMagnitudeComp(vec_size=nn, length=3, in_name='v_e2b_I',
        #                                        mag_name='vmag_e2b_I', units='km/s'),
        #                    promotes_inputs=['v_e2b_I'], promotes_outputs=['vmag_e2b_I'])
        #
        # self.add_subsystem('specific_angular_momentum_comp',
        #                    CrossProductComp(a_name='r_e2b_I', a_units='km', b_name='v_e2b_I',
        #                                     b_units='km/s', c_name='h_e2b_I', c_units='km**2/s',
        #                                     vec_size=nn),
        #                    promotes_inputs=['r_e2b_I', 'v_e2b_I'], promotes_outputs=['h_e2b_I'])
        #
        # self.add_subsystem('hmag_comp',
        #                    VectorMagnitudeComp(vec_size=nn, length=3, in_name='h_e2b_I',
        #                                        mag_name='hmag_e2b_I', units='km**2/s'),
        #                    promotes_inputs=['h_e2b_I'], promotes_outputs=['hmag_e2b_I'])
        #
        # self.add_subsystem('r_unit_comp',
        #                    RowwiseDivideComp(a_name='r_e2b_I', b_name='rmag_e2b_I',
        #                                      c_name='runit_e2b_I',
        #                                      vec_size=nn, length=3, a_units='km',
        #                                      b_units='km', c_units=None),
        #                    promotes_inputs=['r_e2b_I', 'rmag_e2b_I'],
        #                    promotes_outputs=['runit_e2b_I'])
        #
        # self.add_subsystem('v_unit_comp',
        #                    RowwiseDivideComp(a_name='v_e2b_I', b_name='vmag_e2b_I',
        #                                      c_name='vunit_e2b_I',
        #                                      vec_size=nn, length=3, a_units='km/s',
        #                                      b_units='km/s', c_units=None),
        #                    promotes_inputs=['v_e2b_I', 'vmag_e2b_I'],
        #                    promotes_outputs=['vunit_e2b_I'])
        #
        # self.add_subsystem('h_unit_comp',
        #                    RowwiseDivideComp(a_name='h_e2b_I', b_name='hmag_e2b_I',
        #                                      c_name='hunit_e2b_I',
        #                                      vec_size=nn, length=3, a_units='km**2/s',
        #                                      b_units='km**2/s', c_units=None),
        #                    promotes_inputs=['h_e2b_I', 'hmag_e2b_I'],
        #                    promotes_outputs=['hunit_e2b_I'])

        self.add_subsystem('ori_comp',
                           ORIComp(num_nodes=nn),
                           promotes_inputs=['r_e2b_I', 'v_e2b_I'],
                           promotes_outputs=['O_RI'])

        self.add_subsystem('obr_comp',
                           OBRComp(num_nodes=nn),
                           promotes_inputs=['Gamma'],
                           promotes_outputs=['O_BR'])

        self.add_subsystem('obi_comp',
                           OBIComp(num_nodes=nn),
                           promotes_inputs=['O_BR', 'O_RI'],
                           promotes_outputs=['O_BI'])

        self.add_subsystem('orbit_eom_comp', OrbitEOMComp(num_nodes=nn),
                           promotes_inputs=['rmag_e2b_I', 'r_e2b_I', 'v_e2b_I'])

        # self.add_subsystem('attitude_group', AttitudeGroup(num_nodes=nn, grid_data=gd),
        #                    promotes_inputs=['r_e2b_I', 'rmag_e2b_I', 'v_e2b_I', 'vmag_e2b_I',
        #                                     'h_e2b_I', 'hmag_e2b_I'])

        # self.add_subsystem('thermal_temp_comp', ThermalTemperatureComp(num_nodes=nn),
        #                    promotes_inputs=['temperature', 'exposedArea', 'cellInstd', 'LOS', 'P_comm'])

        # self.add_subsystem('battery_soc_comp', BatterySOCComp(num_nodes=nn),
        #                    promotes_inputs=['SOC', 'P_bat'])

        # # Only body tempearture is needed by battery.
        # body_idx = 5*np.arange(nn) + 4
        # self.connect('thermal_temp_comp.temperature', 'battery_soc_comp.T_bat',
        #              src_indices=body_idx, flat_src_indices=True)
