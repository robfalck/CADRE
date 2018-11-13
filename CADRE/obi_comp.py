from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class OBIComp(ExplicitComponent):
    """
    Calculates the rotation matrix from body fixed to the roll frame.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('O_BR', np.zeros((nn, 3, 3)), units=None,
                        desc='Rotation matrix from body-fixed frame to rolled body-fixed '
                             'frame over time')
        self.add_input('O_RI', np.zeros((nn, 3, 3)), units=None,
                        desc='Rotation matrix from rolled body-fixed to inertial '
                             'frame over time')
        self.add_output('O_BI', np.zeros((nn, 3, 3)), units=None,
                        desc='Rotation matrix from rolled body-fixed to inertial '
                             'frame over time')

        rows = np.tile(9*np.arange(nn), 4) + np.repeat(np.array([0, 1, 3, 4]), nn)
        cols = np.tile(np.arange(nn), 4)

        self.declare_partials('O_BI', 'O_BR', val=1.0)
        self.declare_partials('O_BI', 'O_RI', val=1.0)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_RI = inputs['O_RI']
        O_BR = inputs['O_BR']
        outputs['O_BI'] = np.matmul(O_RI, O_BR)

    # def compute_partials(self, inputs, partials):
    #     """
    #     Calculate and save derivatives. (i.e., Jacobian)
    #     """
    #     nn = self.options['num_nodes']
    #     Gamma = inputs['Gamma']
    #
    #     sin_gam = np.sin(Gamma)
    #     cos_gam = np.cos(Gamma)
    #     partials['O_BR', 'Gamma'][:nn] = -sin_gam
    #     partials['O_BR', 'Gamma'][nn:2*nn] = cos_gam
    #     partials['O_BR', 'Gamma'][2*nn:3*nn] = -cos_gam
    #     partials['O_BR', 'Gamma'][3*nn:4*nn] = -sin_gam