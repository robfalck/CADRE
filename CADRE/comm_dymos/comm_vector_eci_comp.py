from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class CommVectorECIComp(ExplicitComponent):
    """
    Determine vector between satellite and ground station in ECI frame (I).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('r_e2g_I', np.zeros((nn, 3)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        self.add_input('r_e2b_I', np.zeros((nn, 3)), units='km',
                       desc='Position vector from earth to satellite '
                            'in Earth-centered inertial frame over time')

        # Outputs
        self.add_output('r_b2g_I', np.zeros((nn, 3)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in Earth-centered inertial frame over time')

        ar = np.arange(3*nn)

        self.declare_partials('r_b2g_I', 'r_e2g_I', rows=ar, cols=ar, val=1.0)
        self.declare_partials('r_b2g_I', 'r_e2b_I', rows=ar, cols=ar, val=-1.0)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_b2g_I'] = inputs['r_e2g_I'] - inputs['r_e2b_I']
