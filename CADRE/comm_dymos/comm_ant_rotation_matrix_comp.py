from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class CommAntRotationMatrixComp(ExplicitComponent):
    """
    Translate antenna angle into the body frame.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('q_AB', np.zeros((nn, 4)),
                       desc='Quarternion matrix in antenna angle frame over time')

        # Outputs
        self.add_output('O_AB', np.zeros((nn, 3, 3)), units=None,
                        desc='Rotation matrix from antenna angle to body-fixed '
                             'frame over time')

        row = np.tile(np.array([0, 0, 0, 0]), 9) + np.repeat(np.arange(9), 4)
        col = np.tile(np.arange(4), 9)
        rows = np.tile(row, nn) + np.repeat(9*np.arange(nn), 36)
        cols = np.tile(col, nn) + np.repeat(4*np.arange(nn), 36)

        self.declare_partials('O_AB', 'q_AB', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        q_A = inputs['q_AB']
        O_AB = outputs['O_AB']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        for i in range(0, self.options['num_nodes']):
            A[0, :] = ( q_A[i, 0], -q_A[i, 3],  q_A[i, 2])  # noqa: E201
            A[1, :] = ( q_A[i, 3],  q_A[i, 0], -q_A[i, 1])  # noqa: E201
            A[2, :] = (-q_A[i, 2],  q_A[i, 1],  q_A[i, 0])  # noqa: E201
            A[3, :] = ( q_A[i, 1],  q_A[i, 2],  q_A[i, 3])  # noqa: E201

            B[0, :] = ( q_A[i, 0],  q_A[i, 3], -q_A[i, 2])  # noqa: E201
            B[1, :] = (-q_A[i, 3],  q_A[i, 0],  q_A[i, 1])  # noqa: E201
            B[2, :] = ( q_A[i, 2], -q_A[i, 1],  q_A[i, 0])  # noqa: E201
            B[3, :] = ( q_A[i, 1],  q_A[i, 2],  q_A[i, 3])  # noqa: E201

            O_AB[i, :, :] = np.dot(A.T, B)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        J = np.empty((nn, 3, 3, 4))
        q_A = inputs['q_AB']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))
        dA_dq = np.zeros((4, 3, 4))
        dB_dq = np.zeros((4, 3, 4))

        # dA_dq
        dA_dq[0, :, 0] = (1, 0, 0)
        dA_dq[1, :, 0] = (0, 1, 0)
        dA_dq[2, :, 0] = (0, 0, 1)
        dA_dq[3, :, 0] = (0, 0, 0)

        dA_dq[0, :, 1] = (0, 0, 0)
        dA_dq[1, :, 1] = (0, 0, -1)
        dA_dq[2, :, 1] = (0, 1, 0)
        dA_dq[3, :, 1] = (1, 0, 0)

        dA_dq[0, :, 2] = (0, 0, 1)
        dA_dq[1, :, 2] = (0, 0, 0)
        dA_dq[2, :, 2] = (-1, 0, 0)
        dA_dq[3, :, 2] = (0, 1, 0)

        dA_dq[0, :, 3] = (0, -1, 0)
        dA_dq[1, :, 3] = (1, 0, 0)
        dA_dq[2, :, 3] = (0, 0, 0)
        dA_dq[3, :, 3] = (0, 0, 1)

        # dB_dq
        dB_dq[0, :, 0] = (1, 0, 0)
        dB_dq[1, :, 0] = (0, 1, 0)
        dB_dq[2, :, 0] = (0, 0, 1)
        dB_dq[3, :, 0] = (0, 0, 0)

        dB_dq[0, :, 1] = (0, 0, 0)
        dB_dq[1, :, 1] = (0, 0, 1)
        dB_dq[2, :, 1] = (0, -1, 0)
        dB_dq[3, :, 1] = (1, 0, 0)

        dB_dq[0, :, 2] = (0, 0, -1)
        dB_dq[1, :, 2] = (0, 0, 0)
        dB_dq[2, :, 2] = (1, 0, 0)
        dB_dq[3, :, 2] = (0, 1, 0)

        dB_dq[0, :, 3] = (0, 1, 0)
        dB_dq[1, :, 3] = (-1, 0, 0)
        dB_dq[2, :, 3] = (0, 0, 0)
        dB_dq[3, :, 3] = (0, 0, 1)

        for i in range(0, nn):
            A[0, :] = ( q_A[i, 0], -q_A[i, 3],  q_A[i, 2])  # noqa: E201
            A[1, :] = ( q_A[i, 3],  q_A[i, 0], -q_A[i, 1])  # noqa: E201
            A[2, :] = (-q_A[i, 2],  q_A[i, 1],  q_A[i, 0])  # noqa: E201
            A[3, :] = ( q_A[i, 1],  q_A[i, 2],  q_A[i, 3])  # noqa: E201

            B[0, :] = ( q_A[i, 0],  q_A[i, 3], -q_A[i, 2])  # noqa: E201
            B[1, :] = (-q_A[i, 3],  q_A[i, 0],  q_A[i, 1])  # noqa: E201
            B[2, :] = ( q_A[i, 2], -q_A[i, 1],  q_A[i, 0])  # noqa: E201
            B[3, :] = ( q_A[i, 1],  q_A[i, 2],  q_A[i, 3])  # noqa: E201

            for k in range(0, 4):
                J[i, :, :, k] = np.dot(dA_dq[:, :, k].T, B) + np.dot(A.T, dB_dq[:, :, k])

        partials['O_AB', 'q_AB'] = J.flatten()
