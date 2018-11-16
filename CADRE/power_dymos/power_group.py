"""
Reaction Wheel discipline for CADRE.
"""
from __future__ import print_function, division, absolute_import

import os

from openmdao.api import Group

from CADRE.power_dymos.power_cell_voltage import PowerCellVoltage
from CADRE.power_dymos.power_solar_power import PowerSolarPower
from CADRE.power_dymos.power_total import PowerTotal


class PowerGroup(Group):
    """
    The Sun subsystem for CADRE.

    Externally sourced inputs:
    w_RW     - Reaction wheel angular velocity state.
    w_B      - Body fixed angular velocity vector of satellite.
    T_RW     - Reaction wheel torque.
    """

    def initialize(self):
        fpath = os.path.dirname(os.path.realpath(__file__))
        self.options.declare('num_nodes', types=(int,),
                             desc="Number of time points.")
        self.options.declare('filename', fpath + '/../data/Power/curve.dat',
                             desc="File containing surrogate model for voltage.")

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('cell_voltage_comp',
                           PowerCellVoltage(num_nodes=nn, filename=self.options['filename']),
                           promotes_inputs=['LOS', 'temperature', 'exposed_area', 'Isetpt'],
                           promotes_outputs=['V_sol'])

        self.add_subsystem('solar_power_comp',
                           PowerSolarPower(num_nodes=nn),
                           promotes_inputs=['Isetpt', 'V_sol'], promotes_outputs=['P_sol'])

        self.add_subsystem('total_power_comp',
                           PowerTotal(num_nodes=nn),
                           promotes_inputs=['P_sol', 'P_comm', 'P_RW'],
                           promotes_outputs=['P_bat'])
