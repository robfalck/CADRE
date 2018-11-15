"""
Reaction Wheel discipline for CADRE: Reaction Wheel Dynamics component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


class ReactionWheelDynamics(ExplicitComponent):
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
        J_RW = self.options['J_RW']

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

        ar = np.arange(3*nn)

        self.declare_partials(of='dXdt:w_RW', wrt='T_RW', rows=ar, cols=ar, val=-1.0/J_RW)

        # Sparsity pattern across a cross product.
        row = np.array([2, 0, 1])
        col = np.array([1, 2, 0])
        row1 = np.tile(row, nn) + np.repeat(3*np.arange(nn), 3)
        col1 = np.tile(col, nn) + np.repeat(3*np.arange(nn), 3)

        row = np.array([1, 2, 0])
        col = np.array([2, 0, 1])
        row2 = np.tile(row, nn) + np.repeat(3*np.arange(nn), 3)
        col2 = np.tile(col, nn) + np.repeat(3*np.arange(nn), 3)

        rows = np.concatenate([row1, row2])
        cols = np.concatenate([col1, col2])

        self.declare_partials(of='dXdt:w_RW', wrt='w_B', rows=rows, cols=cols)
        self.declare_partials(of='dXdt:w_RW', wrt='w_RW', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        J_RW = self.options['J_RW']

        w_B = inputs['w_B']
        T_RW = inputs['T_RW']
        state = inputs['w_RW']

        jy = np.zeros((nn, 3, 3))
        jy[:, 0, 1] = -w_B[:, 2]
        jy[:, 0, 2] = w_B[:, 1]
        jy[:, 1, 0] = w_B[:, 2]
        jy[:, 1, 2] = -w_B[:, 0]
        jy[:, 2, 0] = -w_B[:, 1]
        jy[:, 2, 1] = w_B[:, 0]

        outputs['dXdt:w_RW'] = -T_RW / J_RW - np.einsum("ijk,ik->ij", jy, state)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        J_RW = self.options['J_RW']

        w_B = inputs['w_B']
        state = inputs['w_RW']

        d_wRW = w_B.flatten()
        partials['dXdt:w_RW', 'w_RW'][:3*nn] = -d_wRW
        partials['dXdt:w_RW', 'w_RW'][3*nn:] = d_wRW

        d_wB = state.flatten()
        partials['dXdt:w_RW', 'w_B'][:3*nn] = d_wB
        partials['dXdt:w_RW', 'w_B'][3*nn:] = -d_wB
