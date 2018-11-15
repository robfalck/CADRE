"""
Reaction Wheel discipline for CADRE: Reaction Wheel Motor component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


class ReactionWheelMotorComp(ExplicitComponent):
    """
    Compute reaction wheel motor torque.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")
        self.options.declare('J_RW', 2.8e-5,
                             desc="Mass moment of inertia of the reaction wheel.")

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('T_RW', np.zeros((nn, 3)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        self.add_input('w_B', np.zeros((nn, 3)), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        self.add_input('w_RW', np.zeros((nn, 3)), units='1/s',
                       desc='Angular velocity vector of reaction wheel over time')

        # Outputs
        self.add_output('T_m', np.ones((nn, 3)), units='N*m',
                        desc='Torque vector of motor over time')

        row_col = np.arange(3*nn)

        self.declare_partials('T_m', 'T_RW', rows=row_col, cols=row_col, val=-1.0)

        # (0., -w_B[i, 2], w_B[i, 1])
        # (w_B[i, 2], 0., -w_B[i, 0])
        # (-w_B[i, 1], w_B[i, 0], 0.)
        base = np.repeat(3*np.arange(nn), 3)
        row1 = np.tile(np.array([1, 2, 0]), nn) + base
        col1 = np.tile(np.array([2, 0, 1]), nn) + base
        row2 = np.tile(np.array([2, 0, 1]), nn) + base
        col2 = np.tile(np.array([1, 2, 0]), nn) + base
        rows = np.concatenate([row1, row2])
        cols = np.concatenate([col1, col2])

        self.declare_partials('T_m', 'w_B', rows=rows, cols=cols)
        self.declare_partials('T_m', 'w_RW', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        nn = self.options['num_nodes']
        J_RW = self.options['J_RW']

        T_RW = inputs['T_RW']
        w_B = inputs['w_B']
        w_RW = inputs['w_RW']
        T_m = outputs['T_m']

        w_Bx = np.zeros((3, 3))
        h_RW = J_RW * w_RW[:]
        for i in range(0, nn):
            w_Bx[0, :] = (0., -w_B[i, 2], w_B[i, 1])
            w_Bx[1, :] = (w_B[i, 2], 0., -w_B[i, 0])
            w_Bx[2, :] = (-w_B[i, 1], w_B[i, 0], 0.)

            T_m[i, :] = -T_RW[i, :] - np.dot(w_Bx, h_RW[i, :])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        J_RW = self.options['J_RW']

        w_B = inputs['w_B']
        w_RW = inputs['w_RW']

        d_TwB = w_RW.flatten() * J_RW
        partials['T_m', 'w_B'][:nn*3] = -d_TwB
        partials['T_m', 'w_B'][nn*3:] = d_TwB

        d_TRW = w_B.flatten() * J_RW
        partials['T_m', 'w_RW'][:nn*3] = d_TRW
        partials['T_m', 'w_RW'][nn*3:] = -d_TRW