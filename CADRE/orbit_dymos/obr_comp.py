from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class OBRComp(ExplicitComponent):
    """
    Calculates the rotation matrix from body fixed to the roll frame.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('Gamma', np.zeros(nn), units='rad',
                       desc='Satellite roll angle over time')

        # Outputs
        self.add_output('O_BR', np.zeros((nn, 3, 3)), units=None,
                        desc='Rotation matrix from body-fixed frame to rolled body-fixed '
                             'frame over time')

        rows = np.tile(9*np.arange(nn), 4) + np.repeat(np.array([0, 1, 3, 4]), nn)
        cols = np.tile(np.arange(nn), 4)

        self.declare_partials('O_BR', 'Gamma', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        Gamma = inputs['Gamma']
        O_BR = outputs['O_BR']

        O_BR[:, 0, 0] = np.cos(Gamma)
        O_BR[:, 0, 1] = np.sin(Gamma)
        O_BR[:, 1, 0] = -O_BR[:, 0, 1]
        O_BR[:, 1, 1] = O_BR[:, 0, 0]
        O_BR[:, 2, 2] = 1.0

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        Gamma = inputs['Gamma']

        sin_gam = np.sin(Gamma)
        cos_gam = np.cos(Gamma)
        partials['O_BR', 'Gamma'][:nn] = -sin_gam
        partials['O_BR', 'Gamma'][nn:2*nn] = cos_gam
        partials['O_BR', 'Gamma'][2*nn:3*nn] = -cos_gam
        partials['O_BR', 'Gamma'][3*nn:4*nn] = -sin_gam