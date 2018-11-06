"""
Bspline module for CADRE
"""

from six.moves import range
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent

from MBI import MBI


class BsplineParameters(ExplicitComponent):
    """
    Creates a Bspline interpolant for several CADRE variables
    so that their time histories can be shaped with m control points
    instead of n time points.
    """

    def __init__(self, n, m):
        super(BsplineParameters, self).__init__()

        self.n = n
        self.m = m

        self.deriv_cached = False

    def setup(self):
        m = self.m
        n = self.n

        # Inputs
        self.add_input('t1', 0., units='s', desc='Start time')

        self.add_input('t2', 43200., units='s', desc='End time')

        self.add_input('CP_P_comm', np.zeros((m, )), units='W',
                       desc='Communication power at the control points')

        self.add_input('CP_gamma', np.zeros((m, )), units='rad',
                       desc='Satellite roll angle at control points')

        self.add_input('CP_Isetpt', np.zeros((m, 12)), units='A',
                       desc='Currents of the solar panels at the control points')

        # Outputs
        self.add_output('P_comm', np.ones((n, )), units='W',
                        desc='Communication power over time')

        self.add_output('Gamma', 0.1*np.ones((n, )), units='rad',
                        desc='Satellite roll angle over time')

        self.add_output('Isetpt', 0.2*np.ones((n, 12)), units='A',
                        desc='Currents of the solar panels over time')

        self.declare_partials('P_comm', 'CP_P_comm')
        self.declare_partials('Gamma', 'CP_gamma')

        rowm = np.repeat(0, m)
        colm = 12*np.arange(m)
        rown = np.tile(rowm, n) + np.repeat(12*np.arange(n), m)
        coln = np.tile(colm, n)
        rows = np.tile(rown, 12) + np.repeat(np.arange(12), n*m)
        cols = np.tile(coln, 12) + np.repeat(np.arange(12), n*m)

        self.declare_partials('Isetpt', 'CP_Isetpt', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        # Only need to do this once.
        if self.deriv_cached is False:
            t1 = inputs['t1']
            t2 = inputs['t2']
            n = self.n

            self.B = MBI(np.zeros(n), [np.linspace(t1, t2, n)], [self.m], [4]).getJacobian(0, 0)
            self.Bdot = MBI(np.zeros(n), [np.linspace(t1, t2, n)], [self.m], [4]).getJacobian(1, 0)

            self.BT = self.B.transpose()
            self.BdotT = self.Bdot.transpose()

            self.deriv_cached = True

        outputs['P_comm'] = self.B.dot(inputs['CP_P_comm'])
        outputs['Gamma'] = self.B.dot(inputs['CP_gamma'])
        for k in range(12):
            outputs['Isetpt'][:, k] = self.B.dot(inputs['CP_Isetpt'][:, k])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        B = self.B
        partials['P_comm', 'CP_P_comm'] = B
        partials['Gamma', 'CP_gamma'] = B

        # TODO : need to fix the time issue so that we can declare very sparse derivatives.
        partials['Isetpt', 'CP_Isetpt'] = np.tile(B.todense().flat, 12)


