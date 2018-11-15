"""
Reaction Wheel discipline for CADRE: Reaction Wheel Motor component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


class ReactionWheelTorqueComp(ExplicitComponent):
    """
    Compute the required reaction wheel tourque.
    """
    J = np.zeros((3, 3))
    J[0, :] = (0.018, 0., 0.)
    J[1, :] = (0., 0.018, 0.)
    J[2, :] = (0., 0., 0.006)

    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")

        self.dwx_dw = np.zeros((3, 3, 3))

        self.dwx_dw[0, :, 0] = (0., 0., 0.)
        self.dwx_dw[1, :, 0] = (0., 0., -1.)
        self.dwx_dw[2, :, 0] = (0., 1., 0.)

        self.dwx_dw[0, :, 1] = (0., 0., 1.)
        self.dwx_dw[1, :, 1] = (0., 0., 0.)
        self.dwx_dw[2, :, 1] = (-1., 0, 0.)

        self.dwx_dw[0, :, 2] = (0., -1., 0)
        self.dwx_dw[1, :, 2] = (1., 0., 0.)
        self.dwx_dw[2, :, 2] = (0., 0., 0.)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('w_B', np.zeros((nn, 3)), units='1/s',
                       desc='Angular velocity in body-fixed frame over time')

        self.add_input('wdot_B', np.zeros((nn, 3)), units='1/s**2',
                       desc='Time derivative of w_B over time')

        # Outputs
        self.add_output('T_RW', np.zeros((nn, 3)), units='N*m',
                        desc='Total reaction wheel torque over time')

        rows = np.tile(np.array([0, 0, 0]), 3*nn) + np.repeat(np.arange(3*nn), 3)
        col = np.tile(np.array([0, 1, 2]), 3)
        cols = np.tile(col, nn) + np.repeat(3*np.arange(nn), 9)

        self.declare_partials('T_RW', 'w_B', rows=rows, cols=cols)
        self.declare_partials('T_RW', 'wdot_B', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        w_B = inputs['w_B']
        wdot_B = inputs['wdot_B']
        T_RW = outputs['T_RW']

        wx = np.zeros((3, 3))
        for i in range(0, self.options['num_nodes']):
            wx[0, :] = (0., -w_B[i, 2], w_B[i, 1])
            wx[1, :] = (w_B[i, 2], 0., -w_B[i, 0])
            wx[2, :] = (-w_B[i, 1], w_B[i, 0], 0.)
            T_RW[i, :] = np.dot(self.J, wdot_B[i, :]) + \
                np.dot(wx, np.dot(self.J, w_B[i, :]))

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        w_B = inputs['w_B']

        dT_dw = np.zeros((nn, 3, 3))
        dT_dwdot = np.zeros((nn, 3, 3))
        wx = np.zeros((3, 3))

        for i in range(0, nn):
            wx[0, :] = (0., -w_B[i, 2], w_B[i, 1])
            wx[1, :] = (w_B[i, 2], 0., -w_B[i, 0])
            wx[2, :] = (-w_B[i, 1], w_B[i, 0], 0.)

            dT_dwdot[i, :, :] = self.J
            dT_dw[i, :, :] = np.dot(wx, self.J)

            for k in range(0, 3):
                dT_dw[i, :, k] += np.dot(self.dwx_dw[:, :, k], np.dot(self.J, w_B[i, :]))

        partials['T_RW', 'w_B'] = dT_dw.flatten()
        partials['T_RW', 'wdot_B'] = dT_dwdot.flatten()
