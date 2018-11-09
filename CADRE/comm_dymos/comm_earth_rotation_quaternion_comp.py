from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class CommEarthRotationQuaternionComp(ExplicitComponent):
    """
    Returns the ECI to ECEF quaternion as a function of time.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))
        self.options.declare('omega', types=(float,), desc='Earth rotation rate (rad/s)',
                             default=2*np.pi/3600.0/24.0)

        self.options.declare('gha0', default=0.0, desc='Greenwich hour angle at epoch (rad)')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('t', np.zeros(nn), units='s',
                       desc='Time since epoch')

        # Outputs
        self.add_output('q_IE', np.zeros((nn, 4)), units=None,
                        desc='Quarternion to convert ECI (I) to ECEF (E)')

        row = 4*np.arange(nn)
        col = np.arange(nn)
        rows = np.concatenate([row, row+3])
        cols = np.concatenate([col, col])

        self.declare_partials('q_IE', 't', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        t = inputs['t']
        q_E = outputs['q_IE']

        omega = self.options['omega']
        gha = self.options['gha0'] + omega * t

        q_E[:, 0] = np.cos(gha)
        q_E[:, 3] = -np.sin(gha)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        t = inputs['t']

        nn = self.options['num_nodes']

        omega = self.options['omega']
        gha = self.options['gha0'] + omega * t

        partials['q_IE', 't'][:nn] = -np.sin(gha) * omega
        partials['q_IE', 't'][nn:2*nn] = -np.cos(gha) * omega
