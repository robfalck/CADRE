"""
Power discipline for CADRE: Power Total component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


class PowerTotal(ExplicitComponent):
    """
    Compute the battery power which is the sum of the loads. This
    includes a 2-Watt constant power usage that accounts for the scientific
    instruments on the satellite and small actuator inputs in response to
    disturbance torques.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('P_sol', np.zeros((nn, ), order='F'), units='W',
                       desc='Solar panels power over time')

        self.add_input('P_comm', np.zeros((nn, ), order='F'), units='W',
                       desc='Communication power over time')

        self.add_input('P_RW', np.zeros((nn, 3, ), order='F'), units='W',
                       desc='Power used by reaction wheel over time')

        # Outputs
        self.add_output('P_bat', np.zeros((nn, ), order='F'), units='W',
                        desc='Battery power over time')

        row_col = np.arange(nn)

        self.declare_partials('P_bat', 'P_sol', rows=row_col, cols=row_col, val=1.0)
        self.declare_partials('P_bat', 'P_comm', rows=row_col, cols=row_col, val=-5.0)

        rows = np.tile(np.repeat(0, 3), nn) + np.repeat(np.arange(nn), 3)
        cols = np.arange(3*nn)

        self.declare_partials('P_bat', 'P_RW', rows=rows, cols=cols, val=-1.0)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['P_bat'] = inputs['P_sol'] - 5.0*inputs['P_comm'] - np.sum(inputs['P_RW'], 1) - 2.0
