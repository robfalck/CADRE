from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class AngularVelocityComp(ExplicitComponent):
    """
    Calculates angular velocity vector from the satellite's orientation
    matrix and its derivative.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('O_BI', np.zeros((nn, 3, 3)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered '
                       'inertial frame over time')

        self.add_input('Odot_BI', np.zeros((nn, 3, 3)), units=None,
                       desc='First derivative of O_BI over time')

        # Outputs
        self.add_output('w_B', np.zeros((nn, 3)), units='1/s',
                        desc='Angular velocity vector in body-fixed frame over time')

        self.dw_dOdot = np.zeros((nn, 3, 3, 3))
        self.dw_dO = np.zeros((nn, 3, 3, 3))

        row = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0])
        col = np.array([6, 7, 8, 0, 1, 2, 3, 4, 5])
        rows = np.tile(row, nn) + np.repeat(3*np.arange(nn), 9)
        cols = np.tile(col, nn) + np.repeat(9*np.arange(nn), 9)

        self.declare_partials('w_B', 'O_BI', rows=rows, cols=cols)

        self.dw_dOdot = np.zeros((nn, 3, 3, 3))
        self.dw_dO = np.zeros((nn, 3, 3, 3))

        row = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
        col = np.array([3, 4, 5, 6, 7, 8, 0, 1, 2])
        rows = np.tile(row, nn) + np.repeat(3*np.arange(nn), 9)
        cols = np.tile(col, nn) + np.repeat(9*np.arange(nn), 9)

        self.declare_partials('w_B', 'Odot_BI', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_BI = inputs['O_BI']
        Odot_BI = inputs['Odot_BI']
        w_B = outputs['w_B']

        w_B[:, 0] = np.einsum("ij,ij->i", Odot_BI[:, 2, :], O_BI[:, 1, :])
        w_B[:, 1] = np.einsum("ij,ij->i", Odot_BI[:, 0, :], O_BI[:, 2, :])
        w_B[:, 2] = np.einsum("ij,ij->i", Odot_BI[:, 1, :], O_BI[:, 0, :])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        O_BI = inputs['O_BI']
        Odot_BI = inputs['Odot_BI']

        partials['w_B', 'O_BI'] = Odot_BI.flatten()
        partials['w_B', 'Odot_BI'] = O_BI.flatten()