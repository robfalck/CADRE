from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class CommAntQuaternionComp(ExplicitComponent):
    """
    Fixed antenna angle to time history of the quaternion.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('antAngle',  np.zeros(nn), units='rad')

        # Outputs
        self.add_output('q_AB', np.zeros((nn, 4)), units=None,
                        desc='Quarternion matrix in antenna angle frame over time')

        rs=np.arange(nn*4)
        cs=np.repeat(np.arange(nn), 4)

        # TODO sparsity
        self.declare_partials('q_AB', 'antAngle', rows=rs, cols=cs)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """

        antAngle = inputs['antAngle']
        q_AB = outputs['q_AB']

        rt2 = np.sqrt(2)
        q_AB[:, 0] = np.cos(antAngle/2.)
        q_AB[:, 1] = np.sin(antAngle/2.) / rt2
        q_AB[:, 2] = - np.sin(antAngle/2.) / rt2
        q_AB[:, 3] = 0.0

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        antAngle = inputs['antAngle']
        dq_dt = np.reshape(partials['q_AB', 'antAngle'], (nn, 4))

        rt2 = np.sqrt(2)
        dq_dt[:, 0] = - np.sin(antAngle / 2.) / 2.
        dq_dt[:, 1] = np.cos(antAngle / 2.) / rt2 / 2.
        dq_dt[:, 2] = - np.cos(antAngle / 2.) / rt2 / 2.
        dq_dt[:, 3] = 0.0

