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

        template = np.kron(np.eye(nn, dtype=int), np.kron(np.ones((3, 3), dtype=int), np.eye(3, dtype=int)))
        rs, cs = template.nonzero()
        self.declare_partials('O_BI', 'O_BR', rows=rs, cols=cs, val=1.0)

        template = np.kron(np.eye(nn * 3, dtype=int), np.ones((3, 3), dtype=int))
        cs, rs = template.nonzero()  # Note we're transposing the matrix at each node here
        self.declare_partials('O_BI', 'O_RI', rows=rs, cols=cs)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_RI = inputs['O_RI']
        O_BR = inputs['O_BR']
        outputs['O_BI'] = np.matmul(O_RI, O_BR)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        partials['O_BI', 'O_BR'] = np.tile(inputs['O_RI'], 3).ravel()
        partials['O_BI', 'O_RI'] = np.tile(np.reshape(inputs['O_BR'], (nn, 9)), 3).ravel()
