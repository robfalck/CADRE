"""
Reaction wheel discipline for CADRE
"""

from six.moves import range
import numpy as np

from openmdao.api import ExplicitComponent

from CADRE import rk4


class ReactionWheel_Motor(ExplicitComponent):
    """
    Compute reaction wheel motor torque.
    """

    def __init__(self, n):
        super(ReactionWheel_Motor, self).__init__()
        self.n = n

        # Constant
        self.J_RW = 2.8e-5

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('T_RW', np.zeros((n, 3)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        self.add_input('w_B', np.zeros((n, 3)), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        self.add_input('w_RW', np.zeros((n, 3)), units='1/s',
                       desc='Angular velocity vector of reaction wheel over time')

        # Outputs
        self.add_output('T_m', np.ones((n, 3)), units='N*m',
                        desc='Torque vector of motor over time')

        row_col = np.arange(3*n)

        self.declare_partials('T_m', 'T_RW', rows=row_col, cols=row_col, val=-1.0)

        # (0., -w_B[i, 2], w_B[i, 1])
        # (w_B[i, 2], 0., -w_B[i, 0])
        # (-w_B[i, 1], w_B[i, 0], 0.)
        row1 = np.tile(np.array([1, 2, 0]), n) + np.repeat(3*np.arange(n), 3)
        col1 = np.tile(np.array([2, 0, 1]), n) + np.repeat(3*np.arange(n), 3)
        row2 = np.tile(np.array([2, 0, 1]), n) + np.repeat(3*np.arange(n), 3)
        col2 = np.tile(np.array([1, 2, 0]), n) + np.repeat(3*np.arange(n), 3)
        rows = np.concatenate([row1, row2])
        cols = np.concatenate([col1, col2])

        self.declare_partials('T_m', 'w_B', rows=rows, cols=cols)
        self.declare_partials('T_m', 'w_RW', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        T_RW = inputs['T_RW']
        w_B = inputs['w_B']
        w_RW = inputs['w_RW']
        T_m = outputs['T_m']

        w_Bx = np.zeros((3, 3))
        h_RW = self.J_RW * w_RW[:]
        for i in range(0, self.n):
            w_Bx[0, :] = (0., -w_B[i, 2], w_B[i, 1])
            w_Bx[1, :] = (w_B[i, 2], 0., -w_B[i, 0])
            w_Bx[2, :] = (-w_B[i, 1], w_B[i, 0], 0.)

            T_m[i, :] = -T_RW[i, :] - np.dot(w_Bx, h_RW[i, :])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        w_B = inputs['w_B']
        w_RW = inputs['w_RW']

        d_TwB = w_RW.flatten() * self.J_RW
        partials['T_m', 'w_B'][:n*3] = -d_TwB
        partials['T_m', 'w_B'][n*3:] = d_TwB

        d_TRW = w_B.flatten() * self.J_RW
        partials['T_m', 'w_RW'][:n*3] = d_TRW
        partials['T_m', 'w_RW'][n*3:] = -d_TRW


class ReactionWheel_Power(ExplicitComponent):
    """
    Compute reaction wheel power.
    """
    # constants
    V = 4.0
    a = 4.9e-4
    b = 4.5e2
    I0 = 0.017

    def __init__(self, n):
        super(ReactionWheel_Power, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('w_RW', np.zeros((n, 3)), units='1/s',
                       desc='Angular velocity vector of reaction wheel over time')

        self.add_input('T_RW', np.zeros((n, 3)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        # Outputs
        self.add_output('P_RW', np.ones((n, 3)), units='W',
                        desc='Reaction wheel power over time')

        row_col = np.arange(3*n)

        self.declare_partials('P_RW', 'w_RW', rows=row_col, cols=row_col)
        self.declare_partials('P_RW', 'T_RW', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        n = self.n
        w_RW = inputs['w_RW']
        T_RW = inputs['T_RW']
        P_RW = outputs['P_RW']

        outputs['P_RW'] = (self.V * (self.a * w_RW + self.b * T_RW)**2 + self.V * self.I0)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        w_RW = inputs['w_RW']
        T_RW = inputs['T_RW']

        prod = 2 * self.V * (self.a * w_RW + self.b * T_RW)
        dP_dw = self.a * prod
        dP_dT = self.b * prod

        partials['P_RW', 'w_RW'] = dP_dw.flatten()
        partials['P_RW', 'T_RW'] = dP_dT.flatten()


class ReactionWheel_Torque(ExplicitComponent):
    """
    Compute torque vector of reaction wheel.
    """

    def __init__(self, n):
        super(ReactionWheel_Torque, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('T_tot', np.zeros((n, 3)), units='N*m',
                       desc='Total reaction wheel torque over time')

        # Outputs
        self.add_output('T_RW', np.zeros((n, 3)), units='N*m',
                        desc='Torque vector of reaction wheel over time')

        row_col = np.arange(3*n)

        self.declare_partials('T_RW', 'T_tot', rows=row_col, cols=row_col, val=1.0)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['T_RW'][:] = inputs['T_tot'][:]


class ReactionWheel_Dynamics(rk4.RK4):
    """
    Compute the angular velocity vector of reaction wheel.
    """

    def __init__(self, n_times, h):
        super(ReactionWheel_Dynamics, self).__init__(n_times, h)

        self.n_times = n_times

    def setup(self):
        n_times = self.n_times

        # Inputs
        self.add_input('w_B', np.zeros((n_times, 3)), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        self.add_input('T_RW', np.zeros((n_times, 3)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        self.add_input('w_RW0', np.zeros((3, )), units='1/s',
                       desc='Initial angular velocity vector of reaction wheel')

        # Outputs
        self.add_output('w_RW', np.zeros((n_times, 3)), units='1/s',
                        desc='Angular velocity vector of reaction wheel over time')

        self.options['state_var'] = 'w_RW'
        self.options['init_state_var'] = 'w_RW0'
        self.options['external_vars'] = ['w_B', 'T_RW']

        self.jy = np.zeros((3, 3))

        self.djy_dx = np.zeros((3, 3, 3))
        self.djy_dx[:, :, 0] = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
        self.djy_dx[:, :, 1] = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
        self.djy_dx[:, :, 2] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]

        # unit conversion of some kind
        self.J_RW = 2.8e-5

    def f_dot(self, external, state):

        self.jy[0, :] = [0., -external[2], external[1]]
        self.jy[1, :] = [external[2], 0., -external[0]]
        self.jy[2, :] = [-external[1], external[0], 0.]

        # TODO: sort out unit conversion here with T_RW
        return (-external[3:]/2.8e-5 + self.jy.dot(state))

    def df_dy(self, external, state):

        self.jy[0, :] = [0., -external[2], external[1]]
        self.jy[1, :] = [external[2], 0., -external[0]]
        self.jy[2, :] = [-external[1], external[0], 0.]
        return -self.jy

    def df_dx(self, external, state):
        self.jx = np.zeros((3, 6))

        for i in range(3):
            self.jx[i, 0:3] = -self.djy_dx[:, :, i].dot(state)
            self.jx[i, i+3] = -1.0 / self.J_RW
        return self.jx
