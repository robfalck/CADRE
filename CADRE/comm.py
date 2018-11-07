"""
Communications Discpline for CADRE
"""

import os
from six.moves import range

import numpy as np
import scipy.sparse
from MBI import MBI

from openmdao.core.explicitcomponent import ExplicitComponent

from CADRE.kinematics import fixangles, computepositionspherical, \
    computepositionsphericaljacobian, computepositionrotd,\
    computepositionrotdjacobian

from CADRE import rk4


class Comm_DataDownloaded(rk4.RK4):
    """
    Integrate the incoming data rate to compute the time history of data
    downloaded from the satelite.
    """

    def __init__(self, n_times, h):
        super(Comm_DataDownloaded, self).__init__(n_times, h)

        self.n_times = n_times

    def setup(self):
        n_times = self.n_times

        # Inputs
        self.add_input('Dr', np.zeros((n_times, )), units='Gibyte/s',
                       desc='Download rate over time')

        # Initial State
        self.add_input('Data0', np.zeros((1, )), units='Gibyte',
                       desc='Initial downloaded data state')

        # States
        self.add_output('Data', np.zeros((n_times, 1)), units='Gibyte',
                        desc='Downloaded data state over time')

        self.options['state_var'] = 'Data'
        self.options['init_state_var'] = 'Data0'
        self.options['external_vars'] = ['Dr']

        self.dfdy = np.array([[0.]])
        self.dfdx = np.array([[1.]])

    def f_dot(self, external, state):
        return external[0]

    def df_dy(self, external, state):
        return self.dfdy

    def df_dx(self, external, state):
        return self.dfdx


class Comm_AntRotation(ExplicitComponent):
    """
    Fixed antenna angle to time history of the quaternion.
    """

    def __init__(self, n):
        super(Comm_AntRotation, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('antAngle', 0.0, units='rad')

        # Outputs
        self.add_output('q_A', np.zeros((n, 4)), units=None,
                        desc='Quarternion matrix in antenna angle frame over time')

        self.declare_partials('q_A', 'antAngle')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """

        antAngle = inputs['antAngle']
        q_A = outputs['q_A']

        rt2 = np.sqrt(2)
        q_A[:, 0] = np.cos(antAngle/2.)
        q_A[:, 1] = np.sin(antAngle/2.) / rt2
        q_A[:, 2] = - np.sin(antAngle/2.) / rt2
        q_A[:, 3] = 0.0

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        antAngle = inputs['antAngle']
        dq_dt = np.zeros(4)

        rt2 = np.sqrt(2)
        dq_dt[0] = - np.sin(antAngle / 2.) / 2.
        dq_dt[1] = np.cos(antAngle / 2.) / rt2 / 2.
        dq_dt[2] = - np.cos(antAngle / 2.) / rt2 / 2.
        dq_dt[3] = 0.0

        partials['q_A', 'antAngle'] = np.tile(dq_dt, n)


class Comm_AntRotationMtx(ExplicitComponent):
    """
    Translate antenna angle into the body frame.
    """

    def __init__(self, n):
        super(Comm_AntRotationMtx, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('q_A', np.zeros((n, 4)),
                       desc='Quarternion matrix in antenna angle frame over time')

        # Outputs
        self.add_output('O_AB', np.zeros((n, 3, 3)), units=None,
                        desc='Rotation matrix from antenna angle to body-fixed '
                             'frame over time')

        row = np.tile(np.array([0, 0, 0, 0]), 9) + np.repeat(np.arange(9), 4)
        col = np.tile(np.arange(4), 9)
        rows = np.tile(row, n) + np.repeat(9*np.arange(n), 36)
        cols = np.tile(col, n) + np.repeat(4*np.arange(n), 36)

        self.declare_partials('O_AB', 'q_A', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        q_A = inputs['q_A']
        O_AB = outputs['O_AB']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        for i in range(0, self.n):
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
        n = self.n
        J = np.empty((n, 3, 3, 4))
        q_A = inputs['q_A']

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

        for i in range(0, self.n):
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

        partials['O_AB', 'q_A'] = J.flatten()


class Comm_BitRate(ExplicitComponent):
    """
    Compute the data rate the satellite receives.
    """

    # constants
    pi = 2 * np.arccos(0.)
    c = 299792458
    Gr = 10 ** (12.9 / 10.)
    Ll = 10 ** (-2.0 / 10.)
    f = 437e6
    k = 1.3806503e-23
    SNR = 10 ** (5.0 / 10.)
    T = 500.
    alpha = c ** 2 * Gr * Ll / 16.0 / pi ** 2 / f ** 2 / k / SNR / T / 1e6

    def __init__(self, n):
        super(Comm_BitRate, self).__init__()
        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('P_comm', np.zeros(n), units='W',
                       desc='Communication power over time')

        self.add_input('gain', np.zeros(n), units=None,
                       desc='Transmitter gain over time')

        self.add_input('GSdist', np.zeros(n), units='km',
                       desc='Distance from ground station to satellite over time')

        self.add_input('CommLOS', np.zeros(n), units=None,
                       desc='Satellite to ground station line of sight over time')

        # Outputs
        self.add_output('Dr', np.zeros(n), units='Gibyte/s',
                        desc='Download rate over time')

        row_col = np.arange(n)

        self.declare_partials('Dr', 'P_comm', rows=row_col, cols=row_col)
        self.declare_partials('Dr', 'gain', rows=row_col, cols=row_col)
        self.declare_partials('Dr', 'GSdist', rows=row_col, cols=row_col)
        self.declare_partials('Dr', 'CommLOS', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        """
         Calculate outputs.
        """
        P_comm = inputs['P_comm']
        gain = inputs['gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']
        Dr = outputs['Dr']

        for i in range(0, self.n):
            if np.abs(GSdist[i]) > 1e-10:
                S2 = GSdist[i] * 1e3
            else:
                S2 = 1e-10
            Dr[i] = self.alpha * P_comm[i] * gain[i] * CommLOS[i] / S2 ** 2

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        P_comm = inputs['P_comm']
        gain = inputs['gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']

        S2 = np.zeros(n)

        for i in range(n):
            if np.abs(GSdist[i]) > 1e-10:
                S2[i] = GSdist[i] * 1e3
            else:
                S2[i] = 1e-10

        rec_S2_sq = 1.0 / S2 ** 2
        partials['Dr', 'P_comm'] = self.alpha * gain * CommLOS * rec_S2_sq
        partials['Dr', 'gain'] = self.alpha * P_comm * CommLOS * rec_S2_sq
        partials['Dr', 'GSdist'] = -2.0 * 1e3 * self.alpha * P_comm * gain * CommLOS * rec_S2_sq / S2
        partials['Dr', 'CommLOS'] = self.alpha * P_comm * gain * rec_S2_sq


class Comm_Distance(ExplicitComponent):
    """
    Calculates distance from ground station to satellite.
    """

    def __init__(self, n):
        super(Comm_Distance, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_A', np.zeros((n, 3)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')

        # Outputs
        self.add_output('GSdist', np.zeros(n), units='km',
                        desc='Distance from ground station to satellite over time')

        rows = np.tile(np.array([0, 0, 0]), n) + np.repeat(np.arange(n), 3)
        cols = np.tile(np.arange(3), n) + np.repeat(3*np.arange(n), 3)

        self.declare_partials('GSdist', 'r_b2g_A', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_b2g_A = inputs['r_b2g_A']
        GSdist = outputs['GSdist']

        for i in range(0, self.n):
            GSdist[i] = np.dot(r_b2g_A[i, :], r_b2g_A[i, :])**0.5

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_b2g_A = inputs['r_b2g_A']

        J = np.zeros((self.n, 3))
        for i in range(0, self.n):
            norm = np.dot(r_b2g_A[i, :], r_b2g_A[i, :])**0.5
            if norm > 1e-10:
                J[i, :] = r_b2g_A[i, :] / norm

        partials['GSdist', 'r_b2g_A'] = J.flatten()


class Comm_EarthsSpin(ExplicitComponent):
    """
    Returns the Earth quaternion as a function of time.
    """

    def __init__(self, n):
        super(Comm_EarthsSpin, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('t', np.zeros(n), units='s',
                       desc='Time')

        # Outputs
        self.add_output('q_E', np.zeros((n, 4)), units=None,
                        desc='Quarternion matrix in Earth-fixed frame over time')

        row = 4*np.arange(n)
        col = np.arange(n)
        rows = np.concatenate([row, row+3])
        cols = np.concatenate([col, col])

        self.declare_partials('q_E', 't', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        t = inputs['t']
        q_E = outputs['q_E']

        fact = np.pi / 3600.0 / 24.0
        theta = fact * t

        q_E[:, 0] = np.cos(theta)
        q_E[:, 3] = -np.sin(theta)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        t = inputs['t']

        n = self.n

        fact = np.pi / 3600.0 / 24.0
        theta = fact * t

        partials['q_E', 't'][:n] = -np.sin(theta) * fact
        partials['q_E', 't'][n:2*n] = -np.cos(theta) * fact


class Comm_EarthsSpinMtx(ExplicitComponent):
    """
    Quaternion to rotation matrix for the earth spin.
    """

    def __init__(self, n):
        super(Comm_EarthsSpinMtx, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('q_E', np.zeros((n, 4)), units=None,
                       desc='Quarternion matrix in Earth-fixed frame over time')

        # Outputs
        self.add_output('O_IE', np.zeros((n, 3, 3)), units=None,
                        desc='Rotation matrix from Earth-centered inertial frame to '
                             'Earth-fixed frame over time')

        row = np.tile(np.array([0, 0, 0, 0]), 9) + np.repeat(np.arange(9), 4)
        col = np.tile(np.arange(4), 9)
        rows = np.tile(row, n) + np.repeat(9*np.arange(n), 36)
        cols = np.tile(col, n) + np.repeat(4*np.arange(n), 36)

        self.declare_partials('O_IE', 'q_E', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        q_E = inputs['q_E']
        O_IE = outputs['O_IE']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        for i in range(0, self.n):
            A[0, :] = ( q_E[i, 0], -q_E[i, 3],  q_E[i, 2])  # noqa: E201
            A[1, :] = ( q_E[i, 3],  q_E[i, 0], -q_E[i, 1])  # noqa: E201
            A[2, :] = (-q_E[i, 2],  q_E[i, 1],  q_E[i, 0])  # noqa: E201
            A[3, :] = ( q_E[i, 1],  q_E[i, 2],  q_E[i, 3])  # noqa: E201

            B[0, :] = ( q_E[i, 0],  q_E[i, 3], -q_E[i, 2])  # noqa: E201
            B[1, :] = (-q_E[i, 3],  q_E[i, 0],  q_E[i, 1])  # noqa: E201
            B[2, :] = ( q_E[i, 2], -q_E[i, 1],  q_E[i, 0])  # noqa: E201
            B[3, :] = ( q_E[i, 1],  q_E[i, 2],  q_E[i, 3])  # noqa: E201

            O_IE[i, :, :] = np.dot(A.T, B)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        q_E = inputs['q_E']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        dA_dq = np.zeros((4, 3, 4))
        dB_dq = np.zeros((4, 3, 4))

        J = np.zeros((self.n, 3, 3, 4))

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

        for i in range(0, self.n):
            A[0, :] = ( q_E[i, 0], -q_E[i, 3],  q_E[i, 2])  # noqa: E201
            A[1, :] = ( q_E[i, 3],  q_E[i, 0], -q_E[i, 1])  # noqa: E201
            A[2, :] = (-q_E[i, 2],  q_E[i, 1],  q_E[i, 0])  # noqa: E201
            A[3, :] = ( q_E[i, 1],  q_E[i, 2],  q_E[i, 3])  # noqa: E201

            B[0, :] = ( q_E[i, 0],  q_E[i, 3], -q_E[i, 2])  # noqa: E201
            B[1, :] = (-q_E[i, 3],  q_E[i, 0],  q_E[i, 1])  # noqa: E201
            B[2, :] = ( q_E[i, 2], -q_E[i, 1],  q_E[i, 0])  # noqa: E201
            B[3, :] = ( q_E[i, 1],  q_E[i, 2],  q_E[i, 3])  # noqa: E201

            for k in range(0, 4):
                J[i, :, :, k] = np.dot(dA_dq[:, :, k].T, B) + np.dot(A.T, dB_dq[:, :, k])

        partials['O_IE', 'q_E'] = J.flatten()


class Comm_GainPattern(ExplicitComponent):
    """
    Determines transmitter gain based on an external az-el map.
    """

    def __init__(self, n, rawG_file=None):
        super(Comm_GainPattern, self).__init__()

        self.n = n

        if not rawG_file:
            fpath = os.path.dirname(os.path.realpath(__file__))
            rawG_file = fpath + '/data/Comm/Gain.txt'

        rawGdata = np.genfromtxt(rawG_file)
        rawG = (10 ** (rawGdata / 10.0)).reshape((361, 361), order='F')

        pi = np.pi
        az = np.linspace(0, 2 * pi, 361)
        el = np.linspace(0, 2 * pi, 361)

        self.MBI = MBI(rawG, [az, el], [15, 15], [4, 4])
        self.x = np.zeros((self.n, 2), order='F')

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('azimuthGS', np.zeros(n), units='rad',
                       desc='Azimuth angle from satellite to ground station in '
                            'Earth-fixed frame over time')

        self.add_input('elevationGS', np.zeros(n), units='rad',
                       desc='Elevation angle from satellite to ground station '
                            'in Earth-fixed frame over time')

        # Outputs
        self.add_output('gain', np.zeros(n), units=None,
                        desc='Transmitter gain over time')

        row_col = np.arange(n)

        self.declare_partials('gain', 'elevationGS', rows=row_col, cols=row_col)
        self.declare_partials('gain', 'azimuthGS', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        result = fixangles(self.n, inputs['azimuthGS'], inputs['elevationGS'])
        self.x[:, 0] = result[0]
        self.x[:, 1] = result[1]
        outputs['gain'] = self.MBI.evaluate(self.x)[:, 0]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        partials['gain', 'azimuthGS'] = self.MBI.evaluate(self.x, 1)[:, 0]
        partials['gain', 'elevationGS'] = self.MBI.evaluate(self.x, 2)[:, 0]


class Comm_GSposEarth(ExplicitComponent):
    """
    Returns position of the ground station in Earth frame.
    """

    # Constants
    Re = 6378.137
    d2r = np.pi / 180.

    def __init__(self, n):
        super(Comm_GSposEarth, self).__init__()

        self.n = n

    def setup(self):
        # Inputs
        self.add_input('lon', 0.0, units='rad',
                       desc='Longitude of ground station in Earth-fixed frame')
        self.add_input('lat', 0.0, units='rad',
                       desc='Latitude of ground station in Earth-fixed frame')
        self.add_input('alt', 0.0, units='km',
                       desc='Altitude of ground station in Earth-fixed frame')

        # Outputs
        self.add_output('r_e2g_E', np.zeros((self.n, 3)), units='km',
                        desc='Position vector from earth to ground station in '
                             'Earth-fixed frame over time')

        self.declare_partials('r_e2g_E', 'lon')
        self.declare_partials('r_e2g_E', 'lat')
        self.declare_partials('r_e2g_E', 'alt')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        lat = inputs['lat']
        lon = inputs['lon']
        alt = inputs['alt']
        r_e2g_E = outputs['r_e2g_E']

        cos_lat = np.cos(self.d2r * lat)
        r_GS = (self.Re + alt)

        r_e2g_E[:, 0] = r_GS * cos_lat * np.cos(self.d2r*lon)
        r_e2g_E[:, 1] = r_GS * cos_lat * np.sin(self.d2r*lon)
        r_e2g_E[:, 2] = r_GS * np.sin(self.d2r*lat)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        lat = inputs['lat']
        lon = inputs['lon']
        alt = inputs['alt']

        dr_dlon = np.zeros(3)
        dr_dlat = np.zeros(3)
        dr_dalt = np.zeros(3)

        cos_lat = np.cos(self.d2r * lat)
        sin_lat = np.sin(self.d2r * lat)
        cos_lon = np.cos(self.d2r * lon)
        sin_lon = np.sin(self.d2r * lon)

        r_GS = (self.Re + alt)

        dr_dlon[0] = -self.d2r * r_GS * cos_lat * sin_lon
        dr_dlat[0] = -self.d2r * r_GS * sin_lat * cos_lon
        dr_dalt[0] = cos_lat * cos_lon

        dr_dlon[1] = self.d2r * r_GS * cos_lat * cos_lon
        dr_dlat[1] = -self.d2r * r_GS * sin_lat * sin_lon
        dr_dalt[1] = cos_lat * sin_lon

        dr_dlon[2] = 0.
        dr_dlat[2] = self.d2r * r_GS * cos_lat
        dr_dalt[2] = sin_lat

        partials['r_e2g_E', 'lon'] = np.tile(dr_dlon, n)
        partials['r_e2g_E', 'lat'] = np.tile(dr_dlat, n)
        partials['r_e2g_E', 'alt'] = np.tile(dr_dalt, n)


class Comm_GSposECI(ExplicitComponent):
    """
    Convert time history of ground station position from earth frame
    to inertial frame.
    """
    def __init__(self, n):
        super(Comm_GSposECI, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_IE', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from Earth-centered inertial '
                            'frame to Earth-fixed frame over time')

        self.add_input('r_e2g_E', np.zeros((n, 3)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-fixed frame over time')

        # Outputs
        self.add_output('r_e2g_I', np.zeros((n, 3)), units='km',
                        desc='Position vector from earth to ground station in '
                             'Earth-centered inertial frame over time')

        rows = np.tile(np.repeat(0, 3), 3*n) + np.repeat(np.arange(3*n), 3)
        cols = np.arange(9*n)

        self.declare_partials('r_e2g_I', 'O_IE', rows=rows, cols=cols)

        row = np.tile(np.repeat(0, 3), 3) + np.repeat(np.arange(3), 3)
        col = np.tile(np.arange(3), 3)
        rows = np.tile(row, n) + np.repeat(3*np.arange(n), 9)
        cols = np.tile(col, n) + np.repeat(3*np.arange(n), 9)

        self.declare_partials('r_e2g_I', 'r_e2g_E', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_IE = inputs['O_IE']
        r_e2g_E = inputs['r_e2g_E']
        r_e2g_I = outputs['r_e2g_I']

        for i in range(0, self.n):
            r_e2g_I[i, :] = np.dot(O_IE[i, :, :], r_e2g_E[i, :])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        O_IE = inputs['O_IE']
        r_e2g_E = inputs['r_e2g_E']

        J1 = np.zeros((self.n, 3, 3))

        for k in range(0, 3):
            for v in range(0, 3):
                J1[:, k, v] = r_e2g_E[:, v]

        partials['r_e2g_I', 'O_IE'] = J1.flatten()
        partials['r_e2g_I', 'r_e2g_E'] = O_IE.flatten()


class Comm_LOS(ExplicitComponent):
    """
    Determines if the Satellite has line of sight with the ground stations.
    """

    # constants
    Re = 6378.137

    def __init__(self, n):
        super(Comm_LOS, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_I', np.zeros((n, 3)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('r_e2g_I', np.zeros((n, 3)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('CommLOS', np.zeros(n), units=None,
                        desc='Satellite to ground station line of sight over time')

        rows = np.tile(np.array([0, 0, 0]), n) + np.repeat(np.arange(n), 3)
        cols = np.tile(np.arange(3), n) + np.repeat(3*np.arange(n), 3)

        self.declare_partials('CommLOS', 'r_b2g_I', rows=rows, cols=cols)
        self.declare_partials('CommLOS', 'r_e2g_I', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        n = self.n
        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']
        CommLOS = outputs['CommLOS']

        Rb = 100.0
        for i in range(n):
            proj = np.dot(r_b2g_I[i, :], r_e2g_I[i, :]) / self.Re

            if proj > 0:
                CommLOS[i] = 0.
            elif proj < -Rb:
                CommLOS[i] = 1.
            else:
                x = (proj - 0) / (-Rb - 0)
                CommLOS[i] = 3 * x ** 2 - 2 * x ** 3

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        Re = self.Re
        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']

        dLOS_drb = np.zeros((n, 3))
        dLOS_dre = np.zeros((n, 3))

        Rb = 100.0
        for i in range(n):

            proj = np.dot(r_b2g_I[i, :], r_e2g_I[i, :]) / Re

            if proj > 0:
                continue

            elif proj < -Rb:
                continue

            else:
                x = (proj - 0) / (-Rb - 0)
                dx_dproj = -1. / Rb
                dLOS_dx = 6 * x - 6 * x ** 2
                dproj_drb = r_e2g_I[i, :] / Re
                dproj_dre = r_b2g_I[i, :] / Re

                dLOS_drb[i, :] = dLOS_dx * dx_dproj * dproj_drb
                dLOS_dre[i, :] = dLOS_dx * dx_dproj * dproj_dre

        partials['CommLOS', 'r_b2g_I'] = dLOS_drb.flatten()
        partials['CommLOS', 'r_e2g_I'] = dLOS_dre.flatten()


class Comm_VectorAnt(ExplicitComponent):
    """
    Transform from antenna to body frame
    """
    def __init__(self, n):
        super(Comm_VectorAnt, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_B', np.zeros((n, 3)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in body-fixed frame over time')

        self.add_input('O_AB', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from antenna angle to body-fixed '
                            'frame over time')

        # Outputs
        self.add_output('r_b2g_A', np.zeros((n, 3)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in antenna angle frame over time')

        row = np.tile(np.repeat(0, 3), 3) + np.repeat(np.arange(3), 3)
        col = np.tile(np.arange(3), 3)
        rows = np.tile(row, n) + np.repeat(3*np.arange(n), 9)
        cols = np.tile(col, n) + np.repeat(3*np.arange(n), 9)

        self.declare_partials('r_b2g_A', 'r_b2g_B', rows=rows, cols=cols)

        row = np.tile(np.array([0, 0, 0]), n) + np.repeat(3*np.arange(n), 3)
        col = np.tile(np.arange(3), n) + np.repeat(9*np.arange(n), 3)

        rows = np.concatenate([row, row+1, row+2])
        cols = np.concatenate([col, col+3, col+6])

        self.declare_partials('r_b2g_A', 'O_AB', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_b2g_A'] = np.einsum('ijk,ik->ij', inputs['O_AB'], inputs['r_b2g_B'])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        O_AB = inputs['O_AB']
        r_b2g_B = inputs['r_b2g_B']

        partials['r_b2g_A', 'r_b2g_B'] = O_AB.flatten()

        nn = 3*n
        dO_AB = r_b2g_B.flatten()
        partials['r_b2g_A', 'O_AB'][:nn] = dO_AB
        partials['r_b2g_A', 'O_AB'][nn:2*nn] = dO_AB
        partials['r_b2g_A', 'O_AB'][2*nn:3*nn] = dO_AB


class Comm_VectorBody(ExplicitComponent):
    """
    Transform from body to inertial frame.
    """
    def __init__(self, n):
        super(Comm_VectorBody, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_I', np.zeros((n, 3)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('O_BI', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered'
                            'inertial frame over time')

        # Outputs
        self.add_output('r_b2g_B', np.zeros((n, 3,)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in body-fixed frame over time')

        row = np.tile(np.repeat(0, 3), 3) + np.repeat(np.arange(3), 3)
        col = np.tile(np.arange(3), 3)
        rows = np.tile(row, n) + np.repeat(3*np.arange(n), 9)
        cols = np.tile(col, n) + np.repeat(3*np.arange(n), 9)

        self.declare_partials('r_b2g_B', 'r_b2g_I', rows=rows, cols=cols)

        row = np.tile(np.array([0, 0, 0]), n) + np.repeat(3*np.arange(n), 3)
        col = np.tile(np.arange(3), n) + np.repeat(9*np.arange(n), 3)

        rows = np.concatenate([row, row+1, row+2])
        cols = np.concatenate([col, col+3, col+6])

        self.declare_partials('r_b2g_B', 'O_BI', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_b2g_B'] = np.einsum('ijk,ik->ij', inputs['O_BI'], inputs['r_b2g_I'])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        O_BI = inputs['O_BI']
        r_b2g_I = inputs['r_b2g_I']

        partials['r_b2g_B', 'r_b2g_I'] = O_BI.flatten()

        nn = 3*n
        dO_BI = r_b2g_I.flatten()
        partials['r_b2g_B', 'O_BI'][:nn] = dO_BI
        partials['r_b2g_B', 'O_BI'][nn:2*nn] = dO_BI
        partials['r_b2g_B', 'O_BI'][2*nn:3*nn] = dO_BI


class Comm_VectorECI(ExplicitComponent):
    """
    Determine vector between satellite and ground station.
    """
    def __init__(self, n):
        super(Comm_VectorECI, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2g_I', np.zeros((n, 3)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        self.add_input('r_e2b_I', np.zeros((n, 6)), units=None,
                       desc='Position and velocity vector from earth to satellite '
                            'in Earth-centered inertial frame over time')

        # Outputs
        self.add_output('r_b2g_I', np.zeros((n, 3)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in Earth-centered inertial frame over time')

        row_col = np.arange(3*n)
        vals = np.ones(3*n)

        self.declare_partials('r_b2g_I', 'r_e2g_I', rows=row_col, cols=row_col, val=vals)

        cols = np.tile(np.array([0, 1, 2]), n) + np.repeat(6*np.arange(n), 3)
        vals = -vals

        self.declare_partials('r_b2g_I', 'r_e2b_I', rows=row_col, cols=cols, val=vals)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_b2g_I'] = inputs['r_e2g_I'] - inputs['r_e2b_I'][:, :3]


class Comm_VectorSpherical(ExplicitComponent):
    """
    Convert satellite-ground vector into Az-El.
    """
    def __init__(self, n):
        super(Comm_VectorSpherical, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_A', np.zeros((n, 3)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')

        # Outputs
        self.add_output('azimuthGS', np.zeros(n), units='rad',
                        desc='Azimuth angle from satellite to ground station in '
                             'Earth-fixed frame over time')

        self.add_output('elevationGS', np.zeros(n), units='rad',
                        desc='Elevation angle from satellite to ground station '
                             'in Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        azimuthGS, elevationGS = computepositionspherical(self.n, inputs['r_b2g_A'])
        outputs['azimuthGS'] = azimuthGS
        outputs['elevationGS'] = elevationGS

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.Ja1, self.Ji1, self.Jj1, self.Ja2, self.Ji2, self.Jj2 = \
            computepositionsphericaljacobian(self.n, 3 * self.n, inputs['r_b2g_A'])

        self.J1 = scipy.sparse.csc_matrix((self.Ja1, (self.Ji1, self.Jj1)),
                                          shape=(self.n, 3 * self.n))
        self.J2 = scipy.sparse.csc_matrix((self.Ja2, (self.Ji2, self.Jj2)),
                                          shape=(self.n, 3 * self.n))
        self.J1T = self.J1.transpose()
        self.J2T = self.J2.transpose()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 'r_b2g_A' in d_inputs:
                r_b2g_A = d_inputs['r_b2g_A'].reshape((3 * self.n))
                if 'azimuthGS' in d_outputs:
                    d_outputs['azimuthGS'] += self.J1.dot(r_b2g_A)
                if 'elevationGS' in d_outputs:
                    d_outputs['elevationGS'] += self.J2.dot(r_b2g_A)
        else:
            if 'r_b2g_A' in d_inputs:
                if 'azimuthGS' in d_outputs:
                    az_GS = d_outputs['azimuthGS']
                    d_inputs['r_b2g_A'] += (self.J1T.dot(az_GS)).reshape((self.n, 3))
                if 'elevationGS' in d_outputs:
                    el_GS = d_outputs['elevationGS']
                    d_inputs['r_b2g_A'] += (self.J2T.dot(el_GS)).reshape((self.n, 3))
