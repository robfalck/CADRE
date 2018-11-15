"""
Reaction Wheel discipline for CADRE: Reaction Wheel Power component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


class ReactionWheelPowerComp(ExplicitComponent):
    """
    Compute reaction wheel power.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")

        # Curve fit from manufacturer-provided data.
        # I = I_0 + (a*w + b*T)**2
        self.options.declare('I0', 0.017,
                             desc="Base Current for zero torque and angular velocity.")
        self.options.declare('a', 4.9e-4,
                             desc="Current coefficient for angular velocity.")
        self.options.declare('b', 4.5e2,
                             desc="Current coefficient for torque.")
        self.options.declare('V', 4.0,
                             desc='Reaction wheel voltage.')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('w_RW', np.zeros((nn, 3)), units='1/s',
                       desc='Angular velocity vector of reaction wheel over time')

        self.add_input('T_RW', np.zeros((nn, 3)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        # Outputs
        self.add_output('P_RW', np.ones((nn, 3)), units='W',
                        desc='Reaction wheel power over time')

        row_col = np.arange(3*nn)

        self.declare_partials('P_RW', 'w_RW', rows=row_col, cols=row_col)
        self.declare_partials('P_RW', 'T_RW', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        a = self.options['a']
        b = self.options['b']
        V = self.options['V']
        I0 = self.options['I0']

        w_RW = inputs['w_RW']
        T_RW = inputs['T_RW']

        outputs['P_RW'] = V * ((a * w_RW + b * T_RW)**2 + I0)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        a = self.options['a']
        b = self.options['b']
        V = self.options['V']
        I0 = self.options['I0']

        w_RW = inputs['w_RW']
        T_RW = inputs['T_RW']

        prod = 2.0 * V * (a * w_RW + b * T_RW)
        dP_dw = a * prod
        dP_dT = b * prod

        partials['P_RW', 'w_RW'] = dP_dw.flatten()
        partials['P_RW', 'T_RW'] = dP_dT.flatten()