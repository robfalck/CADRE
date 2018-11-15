"""
Sun discipline for CADRE: Sun Position Spherical component.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent

from CADRE.kinematics import computepositionspherical, computepositionsphericaljacobian


class SunPositionSphericalComp(ExplicitComponent):
    """
    Compute the elevation angle of the Sun in the body-fixed frame.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('r_e2s_B', np.zeros((nn, 3)), units='km',
                       desc='Position vector from Earth to Sun in body-fixed '
                            'frame over time.')

        # Outputs
        self.add_output('azimuth', np.zeros((nn, )), units='rad',
                        desc='Ezimuth angle of the Sun in the body-fixed frame '
                             'over time.')

        self.add_output('elevation', np.zeros((nn, )), units='rad',
                        desc='Elevation angle of the Sun in the body-fixed frame '
                             'over time.')

        rows = np.tile(np.array([0, 0, 0]), nn) + np.repeat(np.arange(nn), 3)
        cols = np.arange(nn*3)

        self.declare_partials('elevation', 'r_e2s_B', rows=rows, cols=cols)

        self.declare_partials('azimuth', 'r_e2s_B', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        nn = self.options['num_nodes']

        azimuth, elevation = computepositionspherical(nn, inputs['r_e2s_B'])

        outputs['azimuth'] = azimuth
        outputs['elevation'] = elevation

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']

        Ja1, Ja2 = computepositionsphericaljacobian(nn, 3*nn, inputs['r_e2s_B'])

        partials['azimuth', 'r_e2s_B'] = Ja1
        partials['elevation', 'r_e2s_B'] = Ja2