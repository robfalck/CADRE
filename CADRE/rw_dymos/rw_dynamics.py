"""
Reaction Wheel discipline for CADRE: Reaction Wheel Dynamics component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


class ReactionWheel_Dynamics(ExplicitComponent):
    """
    Compute the angular velocity vector of reaction wheel.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")
        self.options.declare('J_RW', 2.8e-5,
                             desc="Mass moment of inertia of the reaction wheel.")

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('w_B', np.zeros((nn, 3)), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        self.add_input('T_RW', np.zeros((nn, 3)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        self.add_input('w_RW', np.zeros((nn, 3)), units='1/s',
                       desc='Angular velocity vector of reaction wheel over time')

        # Outputs
        self.add_output('dXdt:w_RW', np.zeros((nn, 3)), units='1/s',
                        desc='Rate of change of reaction wheel angular velocity vector over time')

        #self.options['state_var'] = 'w_RW'
        #self.options['init_state_var'] = 'w_RW0'
        #self.options['external_vars'] = ['w_B', 'T_RW']

        self.jy = np.zeros((3, 3))

        self.djy_dx = np.zeros((3, 3, 3))
        self.djy_dx[:, :, 0] = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
        self.djy_dx[:, :, 1] = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
        self.djy_dx[:, :, 2] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
