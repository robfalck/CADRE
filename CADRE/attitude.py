"""
Attitude discipline for CADRE.
"""
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent

from CADRE.kinematics import computepositionrotd, computepositionrotdjacobian


class Attitude_Angular(ExplicitComponent):
    """
    Calculates angular velocity vector from the satellite's orientation
    matrix and its derivative.
    """
    def __init__(self, n=2):
        super(Attitude_Angular, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_BI', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered '
                       'inertial frame over time')

        self.add_input('Odot_BI', np.zeros((n, 3, 3)), units=None,
                       desc='First derivative of O_BI over time')

        # Outputs
        self.add_output('w_B', np.zeros((n, 3)), units='1/s',
                        desc='Angular velocity vector in body-fixed frame over time')

        self.dw_dOdot = np.zeros((n, 3, 3, 3))
        self.dw_dO = np.zeros((n, 3, 3, 3))

        row = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0])
        col = np.array([6, 7, 8, 0, 1, 2, 3, 4, 5])
        rows = np.tile(row, n) + np.repeat(3*np.arange(n), 9)
        cols = np.tile(col, n) + np.repeat(9*np.arange(n), 9)

        self.declare_partials('w_B', 'O_BI', rows=rows, cols=cols)

        self.dw_dOdot = np.zeros((n, 3, 3, 3))
        self.dw_dO = np.zeros((n, 3, 3, 3))

        row = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
        col = np.array([3, 4, 5, 6, 7, 8, 0, 1, 2])
        rows = np.tile(row, n) + np.repeat(3*np.arange(n), 9)
        cols = np.tile(col, n) + np.repeat(9*np.arange(n), 9)

        self.declare_partials('w_B', 'Odot_BI', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_BI = inputs['O_BI']
        Odot_BI = inputs['Odot_BI']
        w_B = outputs['w_B']

        w_B[:, 0] = np.einsum("ij,ij->i", Odot_BI[:, 2, :], O_BI[:, 1, :])
        w_B[:, 1] = np.einsum("ij,ij->i", Odot_BI[:, 0, :], O_BI[:, 2, :])
        w_B[:, 2] = np.einsum("ij,ij->i", Odot_BI[:, 1, :], O_BI[:, 0, :])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        O_BI = inputs['O_BI']
        Odot_BI = inputs['Odot_BI']

        partials['w_B', 'O_BI'] = Odot_BI.flatten()
        partials['w_B', 'Odot_BI'] = O_BI.flatten()


class Attitude_AngularRates(ExplicitComponent):
    """
    Calculates time derivative of angular velocity vector.

    This time derivative is a central difference at interior time points, and forward/backward
    at the start and end time.
    """

    def __init__(self, n=2, h=28.8):
        super(Attitude_AngularRates, self).__init__()

        self.n = n
        self.h = h

    def setup(self):
        n = self.n
        h = self.h

        # Inputs
        self.add_input('w_B', np.zeros((n, 3)), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        # Outputs
        self.add_output('wdot_B', np.zeros((n, 3)), units='1/s**2',
                        desc='Time derivative of w_B over time')

        # Upper and Lower Diag
        row1 = np.arange(3*(n-1))
        col1 = row1 + 3
        val1a = 0.5 * np.ones(3*(n-1))
        val1a[:3] = 1.0
        val1b = 0.5 * np.ones(3*(n-1))
        val1b[-3:] = 1.0

        val1a *= (1.0 / h)
        val1b *= (1.0 / h)

        # Central Diag
        row_col2 = np.array((0, 1, 2, 3*n-3, 3*n-2, 3*n-1))
        val2 = np.array((-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)) * (1.0 / h)

        rows = np.concatenate((row1, col1, row_col2))
        cols = np.concatenate((col1, row1, row_col2))
        val = np.concatenate((val1a, -val1b, val2))

        self.declare_partials('wdot_B', 'w_B', rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        h = self.h

        w_B = inputs['w_B']
        wdot_B = outputs['wdot_B']

        wdot_B[0, :] = w_B[1, :] - w_B[0, :]
        wdot_B[1:-1, :] = (w_B[2:, :] - w_B[:-2, :]) * 0.5
        wdot_B[-1, :] = w_B[-1, :] - w_B[-2, :]
        wdot_B *= 1.0 / h


class Attitude_Attitude(ExplicitComponent):
    """
    Coordinate transformation from the inertial plane to the rolled
    (forward facing) plane.
    """

    dvx_dv = np.zeros((3, 3, 3))
    dvx_dv[0, :, 0] = (0., 0., 0.)
    dvx_dv[1, :, 0] = (0., 0., -1.)
    dvx_dv[2, :, 0] = (0., 1., 0.)

    dvx_dv[0, :, 1] = (0., 0., 1.)
    dvx_dv[1, :, 1] = (0., 0., 0.)
    dvx_dv[2, :, 1] = (-1., 0., 0.)

    dvx_dv[0, :, 2] = (0., -1., 0.)
    dvx_dv[1, :, 2] = (1., 0., 0.)
    dvx_dv[2, :, 2] = (0., 0., 0.)

    def __init__(self, n=2):
        super(Attitude_Attitude, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2b_I', np.zeros((n, 6)), units=None,
                       desc='Position and velocity vector from earth to satellite in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('O_RI', np.zeros((n, 3, 3)), units=None,
                        desc='Rotation matrix from rolled body-fixed frame to '
                             'Earth-centered inertial frame over time')

        row1 = np.repeat(np.arange(3), 3)
        col1 = np.tile(np.arange(3), 3)
        col2 = col1 + 3
        row3 = row1 + 3
        row5 = row1 + 6

        row = np.concatenate([row1, row1, row3, row3, row5])
        col = np.concatenate([col1, col2, col1, col2, col2])

        rows = np.tile(row, n) + np.repeat(9*np.arange(n), 45)
        cols = np.tile(col, n) + np.repeat(6*np.arange(n), 45)

        self.declare_partials('O_RI', 'r_e2b_I', rows=rows, cols=cols)


    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_e2b_I = inputs['r_e2b_I']
        O_RI = outputs['O_RI']

        O_RI[:] = np.zeros(O_RI.shape)
        for i in range(0, self.n):

            r = r_e2b_I[i, 0:3]
            v = r_e2b_I[i, 3:]

            normr = np.sqrt(np.dot(r, r))
            normv = np.sqrt(np.dot(v, v))

            # Prevent overflow
            if normr < 1e-10:
                normr = 1e-10
            if normv < 1e-10:
                normv = 1e-10

            r = r / normr
            v = v / normv

            vx = np.zeros((3, 3))
            vx[0, :] = (0., -v[2], v[1])
            vx[1, :] = (v[2], 0., -v[0])
            vx[2, :] = (-v[1], v[0], 0.)

            iB = np.dot(vx, r)
            jB = -np.dot(vx, iB)

            O_RI[i, 0, :] = iB
            O_RI[i, 1, :] = jB
            O_RI[i, 2, :] = -v

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_e2b_I = inputs['r_e2b_I']

        diB_dv = np.zeros((3, 3))
        djB_dv = np.zeros((3, 3))

        for i in range(0, self.n):

            r = r_e2b_I[i, 0:3]
            v = r_e2b_I[i, 3:]

            normr = np.sqrt(np.dot(r, r))
            normv = np.sqrt(np.dot(v, v))

            # Prevent overflow
            if normr < 1e-10:
                normr = 1e-10
            if normv < 1e-10:
                normv = 1e-10

            r = r / normr
            v = v / normv

            dr_dr = np.zeros((3, 3))
            dv_dv = np.zeros((3, 3))

            for k in range(0, 3):
                dr_dr[k, k] += 1.0 / normr
                dv_dv[k, k] += 1.0 / normv
                dr_dr[:, k] -= r * r_e2b_I[i, k] / normr ** 2
                dv_dv[:, k] -= v * r_e2b_I[i, 3 + k] / normv ** 2

            vx = np.zeros((3, 3))
            vx[0, :] = (0., -v[2], v[1])
            vx[1, :] = (v[2], 0., -v[0])
            vx[2, :] = (-v[1], v[0], 0.)

            iB = np.dot(vx, r)

            diB_dr = vx
            diB_dv[:, 0] = np.dot(self.dvx_dv[:, :, 0], r)
            diB_dv[:, 1] = np.dot(self.dvx_dv[:, :, 1], r)
            diB_dv[:, 2] = np.dot(self.dvx_dv[:, :, 2], r)

            djB_diB = -vx
            djB_dv[:, 0] = -np.dot(self.dvx_dv[:, :, 0], iB)
            djB_dv[:, 1] = -np.dot(self.dvx_dv[:, :, 1], iB)
            djB_dv[:, 2] = -np.dot(self.dvx_dv[:, :, 2], iB)

            n0 = i*45
            partials['O_RI', 'r_e2b_I'][n0:n0+9] = np.dot(diB_dr, dr_dr).flatten()
            partials['O_RI', 'r_e2b_I'][n0+9:n0+18] = np.dot(diB_dv, dv_dv).flatten()

            partials['O_RI', 'r_e2b_I'][n0+18:n0+27] = np.dot(np.dot(djB_diB, diB_dr), dr_dr).flatten()
            partials['O_RI', 'r_e2b_I'][n0+27:n0+36] = np.dot(np.dot(djB_diB, diB_dv) + djB_dv, dv_dv).flatten()

            partials['O_RI', 'r_e2b_I'][n0+36:n0+45] = -dv_dv.flatten()


class Attitude_Roll(ExplicitComponent):
    """
    Calculates the body-fixed orientation matrix.
    """
    def __init__(self, n=2):
        super(Attitude_Roll, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('Gamma', np.zeros(n), units='rad',
                       desc='Satellite roll angle over time')

        # Outputs
        self.add_output('O_BR', np.zeros((n, 3, 3)), units=None,\
                        desc='Rotation matrix from body-fixed frame to rolled ' 'body-fixed\
                        frame over time')

        rows = np.tile(9*np.arange(n), 4) + np.repeat(np.array([0, 1, 3, 4]), n)
        cols = np.tile(np.arange(n), 4)

        self.declare_partials('O_BR', 'Gamma', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        Gamma = inputs['Gamma']
        O_BR = outputs['O_BR']

        O_BR[:] = np.zeros((self.n, 3, 3))
        O_BR[:, 0, 0] = np.cos(Gamma)
        O_BR[:, 0, 1] = np.sin(Gamma)
        O_BR[:, 1, 0] = -O_BR[:, 0, 1]
        O_BR[:, 1, 1] = O_BR[:, 0, 0]
        O_BR[:, 2, 2] = np.ones(self.n)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        Gamma = inputs['Gamma']

        sin_gam = np.sin(Gamma)
        cos_gam = np.cos(Gamma)
        partials['O_BR', 'Gamma'][:n] = -sin_gam
        partials['O_BR', 'Gamma'][n:2*n] = cos_gam
        partials['O_BR', 'Gamma'][2*n:3*n] = -cos_gam
        partials['O_BR', 'Gamma'][3*n:4*n] = -sin_gam


class Attitude_RotationMtx(ExplicitComponent):
    """
    Multiplies transformations to produce the orientation matrix of the
    body frame with respect to inertial.
    """

    def __init__(self, n=2):
        super(Attitude_RotationMtx, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_BR', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from body-fixed frame to rolled '
                       'body-fixed frame over time')

        self.add_input('O_RI', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from rolled body-fixed '
                       'frame to Earth-centered inertial frame over time')

        # Outputs
        self.add_output('O_BI', np.zeros((n, 3, 3)), units=None,
                        desc='Rotation matrix from body-fixed frame to '
                        'Earth-centered inertial frame over time')

        row = np.repeat(np.arange(3), 3)
        row1 = np.tile(row, n) + np.repeat(9*np.arange(n), 9)
        col = np.tile(np.arange(3), 3)
        col1 = np.tile(col, n) + np.repeat(9*np.arange(n), 9)

        # Transpose here instead of in compute_partials
        rows = np.concatenate([col1, col1+3, col1+6])
        cols = np.concatenate([row1, row1+3, row1+6])

        self.declare_partials('O_BI', 'O_BR', rows=rows, cols=cols)

        row = np.repeat(3*np.arange(3), 3)
        row1 = np.tile(row, n) + np.repeat(9*np.arange(n), 9)

        col = np.tile(np.array([0, 3, 6]), 3)
        col1 = np.tile(col, n) + np.repeat(9*np.arange(n), 9)

        rows = np.concatenate([row1, row1+1, row1+2])
        cols = np.concatenate([col1, col1+1, col1+2])

        self.declare_partials('O_BI', 'O_RI', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_BR = inputs['O_BR']
        O_RI = inputs['O_RI']
        O_BI = outputs['O_BI']

        for i in range(0, self.n):
            O_BI[i, :, :] = np.dot(O_BR[i, :, :], O_RI[i, :, :])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        O_BR = inputs['O_BR']
        O_RI = inputs['O_RI']

        nn = 9*n
        dO_BR = O_RI.flatten()
        partials['O_BI', 'O_BR'][:nn] = dO_BR
        partials['O_BI', 'O_BR'][nn:2*nn] = dO_BR
        partials['O_BI', 'O_BR'][2*nn:3*nn] = dO_BR

        dO_RI = O_BR.flatten()
        partials['O_BI', 'O_RI'][:nn] = dO_RI
        partials['O_BI', 'O_RI'][nn:2*nn] = dO_RI
        partials['O_BI', 'O_RI'][2*nn:3*nn] = dO_RI


class Attitude_RotationMtxRates(ExplicitComponent):
    """
    Calculates time derivative of body frame orientation matrix.
    """

    def __init__(self, n=2, h=28.2):
        super(Attitude_RotationMtxRates, self).__init__()

        self.n = n
        self.h = h

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_BI', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered '
                            'inertial frame over time')

        # Outputs
        self.add_output('Odot_BI', np.zeros((n, 3, 3)), units=None,
                        desc='First derivative of O_BI over time')

        base1 = np.arange(9)
        base2 = np.arange(9*(n - 2))
        nn = 9*(n - 1)

        rows = np.concatenate([base1, base1, base2+9, base2+9, base1+nn, base1+nn])
        cols = np.concatenate([base1+9, base1, base2+18, base2, base1+nn, base1+nn-9])

        #self.declare_partials('Odot_BI', 'O_BI', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_BI = inputs['O_BI']
        h = self.h
        Odot_BI = outputs['Odot_BI']

        Odot_BI[0, :, :] = O_BI[1, :, :]
        Odot_BI[0, :, :] -= O_BI[0, :, :]
        Odot_BI[1:-1, :, :] = O_BI[2:, :, :] / 2.0
        Odot_BI[1:-1, :, :] -= O_BI[:-2, :, :] / 2.0
        Odot_BI[-1, :, :] = O_BI[-1, :, :]
        Odot_BI[-1, :, :] -= O_BI[-2, :, :]
        Odot_BI *= 1.0/h

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dOdot_BI = d_outputs['Odot_BI']

        h = self.h

        if mode == 'fwd':
            if 'O_BI' in d_inputs:
                dO_BI = d_inputs['O_BI']
                dOdot_BI[0, :, :] += dO_BI[1, :, :] / h
                dOdot_BI[0, :, :] -= dO_BI[0, :, :] / h
                dOdot_BI[1:-1, :, :] += dO_BI[2:, :, :] / (2.0*h)
                dOdot_BI[1:-1, :, :] -= dO_BI[:-2, :, :] / (2.0*h)
                dOdot_BI[-1, :, :] += dO_BI[-1, :, :] / h
                dOdot_BI[-1, :, :] -= dO_BI[-2, :, :] / h
        else:
            if 'O_BI' in d_inputs:
                dO_BI = np.zeros(d_inputs['O_BI'].shape)
                dO_BI[1, :, :] += dOdot_BI[0, :, :] / h
                dO_BI[0, :, :] -= dOdot_BI[0, :, :] / h
                dO_BI[2:, :, :] += dOdot_BI[1:-1, :, :] / (2.0*h)
                dO_BI[:-2, :, :] -= dOdot_BI[1:-1, :, :] / (2.0*h)
                dO_BI[-1, :, :] += dOdot_BI[-1, :, :] / h
                dO_BI[-2, :, :] -= dOdot_BI[-1, :, :] / h
                d_inputs['O_BI'] += dO_BI


class Attitude_Sideslip(ExplicitComponent):
    """
    Determine velocity in the body frame.
    """

    def __init__(self, n=2):
        super(Attitude_Sideslip, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2b_I', np.zeros((n, 6)), units=None,
                       desc='Position and velocity vector from earth to satellite '
                       'in Earth-centered inertial frame over time')

        self.add_input('O_BI', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from body-fixed frame to '
                       'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('v_e2b_B', np.zeros((n, 3)), units='m/s',
                        desc='Velocity vector from earth to satellite'
                        'in body-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_e2b_I = inputs['r_e2b_I']
        O_BI = inputs['O_BI']
        v_e2b_B = outputs['v_e2b_B']

        v_e2b_B[:] = computepositionrotd(self.n, r_e2b_I[:, 3:], O_BI)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_e2b_I = inputs['r_e2b_I']
        O_BI = inputs['O_BI']

        self.J1, self.J2 = computepositionrotdjacobian(self.n,
                                                       r_e2b_I[:, 3:],
                                                       O_BI)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dv_e2b_B = d_outputs['v_e2b_B']

        if mode == 'fwd':
            for k in range(3):
                if 'O_BI' in inputs:
                    for u in range(3):
                        for v in range(3):
                            dv_e2b_B[:, k] += self.J1[:, k, u, v] * \
                                d_inputs['O_BI'][:, u, v]
                if 'r_e2b_I' in inputs:
                    for j in range(3):
                        dv_e2b_B[:, k] += self.J2[:, k, j] * \
                            d_inputs['r_e2b_I'][:, 3+j]

        else:
            for k in range(3):
                if 'O_BI' in inputs:
                    for u in range(3):
                        for v in range(3):
                            d_inputs['O_BI'][:, u, v] += self.J1[:, k, u, v] * \
                                dv_e2b_B[:, k]
                if 'r_e2b_I' in inputs:
                    for j in range(3):
                        d_inputs['r_e2b_I'][:, 3+j] += self.J2[:, k, j] * \
                            dv_e2b_B[:, k]


class Attitude_Torque(ExplicitComponent):
    """
    Compute the required reaction wheel tourque.
    """

    J = np.zeros((3, 3))
    J[0, :] = (0.018, 0., 0.)
    J[1, :] = (0., 0.018, 0.)
    J[2, :] = (0., 0., 0.006)

    def __init__(self, n=2):
        super(Attitude_Torque, self).__init__()

        self.n = n

        self.dT_dwdot = np.zeros((n, 3, 3))
        self.dwx_dw = np.zeros((3, 3, 3))

        self.dwx_dw[0, :, 0] = (0., 0., 0.)
        self.dwx_dw[1, :, 0] = (0., 0., -1.)
        self.dwx_dw[2, :, 0] = (0., 1., 0.)

        self.dwx_dw[0, :, 1] = (0., 0., 1.)
        self.dwx_dw[1, :, 1] = (0., 0., 0.)
        self.dwx_dw[2, :, 1] = (-1., 0, 0.)

        self.dwx_dw[0, :, 2] = (0., -1., 0)
        self.dwx_dw[1, :, 2] = (1., 0., 0.)
        self.dwx_dw[2, :, 2] = (0., 0., 0.)

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('w_B', np.zeros((n, 3)), units='1/s',
                       desc='Angular velocity in body-fixed frame over time')

        self.add_input('wdot_B', np.zeros((n, 3)), units='1/s**2',
                       desc='Time derivative of w_B over time')

        # Outputs
        self.add_output('T_tot', np.zeros((n, 3)), units='N*m',
                        desc='Total reaction wheel torque over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        w_B = inputs['w_B']
        wdot_B = inputs['wdot_B']
        T_tot = outputs['T_tot']

        wx = np.zeros((3, 3))
        for i in range(0, self.n):
            wx[0, :] = (0., -w_B[i, 2], w_B[i, 1])
            wx[1, :] = (w_B[i, 2], 0., -w_B[i, 0])
            wx[2, :] = (-w_B[i, 1], w_B[i, 0], 0.)
            T_tot[i, :] = np.dot(self.J, wdot_B[i, :]) + \
                np.dot(wx, np.dot(self.J, w_B[i, :]))

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        w_B = inputs['w_B']

        self.dT_dw = np.zeros((self.n, 3, 3))
        wx = np.zeros((3, 3))

        for i in range(0, self.n):
            wx[0, :] = (0., -w_B[i, 2], w_B[i, 1])
            wx[1, :] = (w_B[i, 2], 0., -w_B[i, 0])
            wx[2, :] = (-w_B[i, 1], w_B[i, 0], 0.)

            self.dT_dwdot[i, :, :] = self.J
            self.dT_dw[i, :, :] = np.dot(wx, self.J)

            for k in range(0, 3):
                self.dT_dw[i, :, k] += np.dot(self.dwx_dw[:, :, k],
                                              np.dot(self.J, w_B[i, :]))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dT_tot = d_outputs['T_tot']

        if mode == 'fwd':
            for k in range(3):
                for j in range(3):
                    if 'w_B' in d_inputs:
                        dT_tot[:, k] += self.dT_dw[:, k, j] * \
                            d_inputs['w_B'][:, j]
                    if 'wdot_B' in d_inputs:
                        dT_tot[:, k] += self.dT_dwdot[:, k, j] * \
                            d_inputs['wdot_B'][:, j]
        else:
            for k in range(3):
                for j in range(3):
                    if 'w_B' in d_inputs:
                        d_inputs['w_B'][:, j] += self.dT_dw[:, k, j] * \
                            dT_tot[:, k]
                    if 'wdot_B' in d_inputs:
                        d_inputs['wdot_B'][:, j] += self.dT_dwdot[:, k, j] * \
                            dT_tot[:, k]
