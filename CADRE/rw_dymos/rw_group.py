"""
Reaction Wheel discipline for CADRE.
"""
from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from CADRE.rw_dymos.rw_dynamics import ReactionWheelDynamics
from CADRE.rw_dymos.rw_motor import ReactionWheelMotorComp
from CADRE.rw_dymos.rw_power import ReactionWheelPowerComp


class ReactionWheelGroup(Group):
    """
    The Sun subsystem for CADRE.

    Externally sourced inputs:
    w_RW     - Reaction wheel angular velocity state.
    w_B      - Body fixed angular velocity vector of satellite.
    T_RW     - Reaction wheel torque.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int,),
                             desc="Number of time points.")
        self.options.declare('J_RW', 2.8e-5,
                             desc="Mass moment of inertia of the reaction wheel.")

    def setup(self):
        nn = self.options['num_nodes']
        J_RW = self.options['J_RW']

        self.add_subsystem('rw_dynamics',
                           ReactionWheelDynamics(num_nodes=nn, J_RW=J_RW),
                           promotes_inputs=['w_B', 'w_RW'], promotes_outputs=['dXdt:w_RW'])

        self.add_subsystem('rw_motor',
                           ReactionWheelMotorComp(num_nodes=nn, J_RW=J_RW),
                           promotes_inputs=['T_RW', 'w_B', 'w_RW'])

        self.add_subsystem('rw_power',
                           ReactionWheelPowerComp(num_nodes=nn),
                           promotes_inputs=['w_RW'], promotes_outputs=['P_RW'])

        self.connect('rw_motor.T_m', ('rw_power.T_RW', 'rw_dynamics.T_RW'))
