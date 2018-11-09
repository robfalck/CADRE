from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent

from ..kinematics import computepositionspherical, computepositionsphericaljacobian

class CommVectorSphericalComp(ExplicitComponent):
    """
    Convert satellite-ground vector into Az-El.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('r_b2g_A', np.zeros((nn, 3)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')

        # Outputs
        self.add_output('azimuthGS', np.zeros(nn), units='rad',
                        desc='Azimuth angle from satellite to ground station in '
                             'Earth-fixed frame over time')

        self.add_output('elevationGS', np.zeros(nn), units='rad',
                        desc='Elevation angle from satellite to ground station '
                             'in Earth-fixed frame over time')

        rows = np.tile(np.array([0, 0, 0]), nn) + np.repeat(np.arange(nn), 3)
        cols = np.arange(nn*3)

        self.declare_partials('elevationGS', 'r_b2g_A', rows=rows, cols=cols)

        self.declare_partials('azimuthGS', 'r_b2g_A', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        azimuthGS, elevationGS = computepositionspherical(self.options['num_nodes'], inputs['r_b2g_A'])
        outputs['azimuthGS'] = azimuthGS
        outputs['elevationGS'] = elevationGS

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']

        Ja1, Ja2 = computepositionsphericaljacobian(nn, 3*nn, inputs['r_b2g_A'])

        partials['azimuthGS', 'r_b2g_A'] = Ja1
        partials['elevationGS', 'r_b2g_A'] = Ja2
