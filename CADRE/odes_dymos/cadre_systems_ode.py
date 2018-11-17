from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from dymos import declare_state, declare_time, declare_parameter

from CADRE.battery_dymos import BatterySOCComp
from CADRE.rw_dymos.rw_group import ReactionWheelGroup
from CADRE.solar_dymos import SolarExposedAreaComp
from CADRE.sun_dymos.sun_group import SunGroup
from CADRE.thermal_dymos import ThermalTemperatureComp
from CADRE.comm_dymos import CommGroup
from CADRE.power_dymos import PowerGroup


@declare_time(units='s', targets=['time'])
@declare_parameter('r_e2b_I', targets=['r_e2b_I'], units='km', shape=(3,))
@declare_parameter('O_BI', targets=['O_BI'], shape=(3, 3))
@declare_parameter('w_B', targets=['w_B'], shape=(3,), units='1/s')
@declare_parameter('wdot_B', targets=['wdot_B'], shape=(3,), units='1/s**2')
@declare_state('w_RW', rate_source='rw_group.dXdt:w_RW', shape=(3,), targets=['w_RW'], units='rad/s')
@declare_state('temperature', rate_source='thermal_temp_comp.dXdt:temperature', targets=['temperature'], units='degK', shape=(5,))
@declare_state('SOC', rate_source='battery_soc_comp.dXdt:SOC', targets=['SOC'])
@declare_state('data', rate_source='comm_group.dXdt:data', units='Gibyte')
@declare_parameter('LD', targets=['LD'], units='d', dynamic=False)  # Launch date, MJD
@declare_parameter('fin_angle', targets=['fin_angle'], units='rad', dynamic=False)  # Panel fin sweep angle
@declare_parameter('P_comm', targets=['P_comm'], units='W')
@declare_parameter('antAngle', targets=['antAngle'], units='rad')
@declare_parameter('cellInstd', targets=['cellInstd'], units=None, shape=(7, 12), dynamic=False)
@declare_parameter('Isetpt', targets=['Isetpt'], units='A', shape=(12,), dynamic=True)
class CadreSystemsODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('sun_group', SunGroup(num_nodes=nn),
                           promotes_inputs=['r_e2b_I', 'O_BI', ('t', 'time'), 'LD'],
                           promotes_outputs=['azimuth', 'elevation', 'LOS'])

        self.add_subsystem('solar_comp', SolarExposedAreaComp(num_nodes=nn),
                           promotes_inputs=['fin_angle', 'azimuth', 'elevation'],
                           promotes_outputs=['exposed_area'])

        self.add_subsystem('comm_group', CommGroup(num_nodes=nn),
                           promotes_inputs=[('t', 'time'), 'r_e2b_I', 'antAngle', 'P_comm', 'O_BI'])

        self.add_subsystem('rw_group', ReactionWheelGroup(num_nodes=nn),
                           promotes_inputs=['w_RW', 'w_B', 'wdot_B'],
                           promotes_outputs=['P_RW', 'T_RW', 'T_m'])

        self.add_subsystem('thermal_temp_comp', ThermalTemperatureComp(num_nodes=nn),
                           promotes_inputs=['temperature', 'exposed_area', 'cellInstd',
                                            'LOS', 'P_comm'])

        self.add_subsystem('power_group', PowerGroup(num_nodes=nn),
                           promotes_inputs=['LOS', 'temperature', 'exposed_area', 'Isetpt',
                                            'P_comm', 'P_RW'],
                           promotes_outputs=['P_bat', 'P_sol'])

        self.add_subsystem('battery_soc_comp', BatterySOCComp(num_nodes=nn),
                           promotes_inputs=['SOC', 'P_bat', 'temperature'])
