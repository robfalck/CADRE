"""
Power discipline for CADRE: Power Solar Power component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


class PowerSolarPower(ExplicitComponent):
    """
    Compute the output power of the solar panels.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('Isetpt', np.zeros((nn, 12)), units='A',
                       desc='Currents of the solar panels')

        self.add_input('V_sol', np.zeros((nn, 12)), units='V',
                       desc='Output voltage of solar panel over time')

        self.add_output('P_sol', np.zeros((nn, )), units='W',
                        desc='Solar panels power over time')

        rows = np.tile(np.repeat(0, 12), nn) + np.repeat(np.arange(nn), 12)
        cols = np.arange(12*nn)

        self.declare_partials('P_sol', 'Isetpt', rows=rows, cols=cols)
        self.declare_partials('P_sol', 'V_sol', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        V_sol = inputs['V_sol']
        Isetpt = inputs['Isetpt']

        outputs['P_sol'] = np.einsum('ij,ij->i', V_sol, Isetpt)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        V_sol = inputs['V_sol']
        Isetpt = inputs['Isetpt']

        partials['P_sol', 'Isetpt'] = V_sol.flatten()
        partials['P_sol', 'V_sol'] = Isetpt.flatten()