"""
Sun discipline for CADRE
"""

from six.moves import range
import numpy as np
import scipy.sparse

from openmdao.core.explicitcomponent import ExplicitComponent

from CADRE.kinematics import computepositionrotd, computepositionrotdjacobian
from CADRE.kinematics import computepositionspherical, computepositionsphericaljacobian


class Sun_LOS(ExplicitComponent):
    """
    Compute the Satellite to sun line of sight.
    """

    def __init__(self, n=2):
        super(Sun_LOS, self).__init__()

        self.n = n

        # Earth's radius is 6378 km. 0.85 is the alpha in John Hwang's paper
        self.r1 = 6378.137 * 0.85
        self.r2 = 6378.137

    def setup(self):
        n = self.n

        self.add_input('r_e2b_I', np.zeros((n, 6), order='F'), units=None,
                       desc='Position and velocity vectors from '
                            'Earth to satellite in Earth-centered '
                            'inertial frame over time.')

        self.add_input('r_e2s_I', np.zeros((n, 3), order='F'), units='km',
                       desc='Position vector from Earth to sun in Earth-centered '
                            'inertial frame over time.')

        self.add_output('LOS', np.zeros((n, ), order='F'), units=None,
                        desc='Satellite to sun line of sight over time')

        rows = np.tile(np.repeat(0, 3), n) + np.repeat(np.arange(n), 3)
        cols = np.tile(np.arange(3), n) + np.repeat(6*np.arange(n), 3)

        self.declare_partials('LOS', 'r_e2b_I', rows=rows, cols=cols)

        rows = np.tile(np.repeat(0, 3), n) + np.repeat(np.arange(n), 3)
        cols = np.arange(n*3)

        self.declare_partials('LOS', 'r_e2s_I', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        n = self.n
        r_e2b_I = inputs['r_e2b_I']
        r_e2s_I = inputs['r_e2s_I']
        LOS = outputs['LOS']

        for i in range(n):
            r_b = r_e2b_I[i, :3]
            r_s = r_e2s_I[i, :3]
            dot = np.dot(r_b, r_s)
            cross = np.cross(r_b, r_s)
            dist = np.sqrt(cross.dot(cross))

            if dot >= 0.0:
                LOS[i] = 1.0
            elif dist <= self.r1:
                LOS[i] = 0.0
            elif dist >= self.r2:
                LOS[i] = 1.0
            else:
                x = (dist - self.r1) / (self.r2 - self.r1)
                LOS[i] = 3*x**2 - 2*x**3

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_e2b_I = inputs['r_e2b_I']
        r_e2s_I = inputs['r_e2s_I']

        nj = 3*self.n

        Jab = np.zeros(shape=(nj, ), dtype=np.float)
        Jas = np.zeros(shape=(nj, ), dtype=np.float)

        for i in range(self.n):
            r_b = r_e2b_I[i, :3]
            r_s = r_e2s_I[i, :3]
            dot = np.dot(r_b, r_s)
            cross = np.cross(r_b, r_s)
            dist = np.sqrt(np.dot(cross, cross))

            if dot >= 0.0:
                continue

            elif dist <= self.r1:
                continue

            elif dist >= self.r2:
                continue

            else:
                x = (dist-self.r1)/(self.r2-self.r1)
                # LOS = 3*x**2 - 2*x**3
                ddist_dcross = cross/dist
                dcross_drb = crossMatrix(-r_s)
                dcross_drs = crossMatrix(r_b)
                dx_ddist = 1.0/(self.r2 - self.r1)
                dLOS_dx = 6*x - 6*x**2
                dLOS_drb = dLOS_dx * dx_ddist * np.dot(ddist_dcross, dcross_drb)
                dLOS_drs = dLOS_dx * dx_ddist * np.dot(ddist_dcross, dcross_drs)

                Jab[i*3:i*3+3] = dLOS_drb
                Jas[i*3:i*3+3] = dLOS_drs

        partials['LOS', 'r_e2b_I'] = Jab
        partials['LOS', 'r_e2s_I'] = Jas


def crossMatrix(v):
    # so m[1,0] is v[2], for example
    m = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return m


class Sun_PositionBody(ExplicitComponent):
    """
    Position vector from earth to sun in body-fixed frame.
    """

    def __init__(self, n=2):
        super(Sun_PositionBody, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_BI', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from the Earth-centered inertial frame '
                            'to the satellite frame.')

        self.add_input('r_e2s_I', np.zeros((n, 3)), units='km',
                       desc='Position vector from Earth to Sun in Earth-centered '
                            'inertial frame over time.')

        # Outputs
        self.add_output('r_e2s_B', np.zeros((n, 3)), units='km',
                        desc='Position vector from Earth to Sun in body-fixed '
                             'frame over time.')


        row = np.tile(np.repeat(0, 3), 3) + np.repeat(np.arange(3), 3)
        col = np.tile(np.arange(3), 3)
        rows = np.tile(row, n) + np.repeat(3*np.arange(n), 9)
        cols = np.tile(col, n) + np.repeat(3*np.arange(n), 9)

        self.declare_partials('r_e2s_B', 'r_e2s_I', rows=rows, cols=cols)

        row = np.tile(np.array([0, 0, 0]), n) + np.repeat(3*np.arange(n), 3)
        col = np.tile(np.arange(3), n) + np.repeat(9*np.arange(n), 3)

        rows = np.concatenate([row, row+1, row+2])
        cols = np.concatenate([col, col+3, col+6])

        self.declare_partials('r_e2s_B', 'O_BI', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_e2s_B'] = np.einsum('ijk,ik->ij', inputs['O_BI'], inputs['r_e2s_I'])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        O_BI = inputs['O_BI']
        r_e2s_I = inputs['r_e2s_I']

        partials['r_e2s_B', 'r_e2s_I'] = O_BI.flatten()

        nn = 3*n
        dO_AB = r_e2s_I.flatten()
        partials['r_e2s_B', 'O_BI'][:nn] = dO_AB
        partials['r_e2s_B', 'O_BI'][nn:2*nn] = dO_AB
        partials['r_e2s_B', 'O_BI'][2*nn:3*nn] = dO_AB


class Sun_PositionECI(ExplicitComponent):
    """
    Compute the position vector from Earth to Sun in Earth-centered inertial frame.
    """

    # constants
    d2r = np.pi/180.

    def __init__(self, n=2):
        super(Sun_PositionECI, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('LD', 0.0, units=None)

        self.add_input('t', np.zeros((n, )), units='s', desc='Time')

        # Outputs
        self.add_output('r_e2s_I', np.zeros((n, 3)), units='km',
                        desc='Position vector from Earth to Sun in Earth-centered '
                             'inertial frame over time.')

        self.declare_partials('r_e2s_I', 'LD')

        rows = np.arange(n*3)
        cols = np.tile(np.repeat(0, 3), n) + np.repeat(np.arange(n), 3)

        self.declare_partials('r_e2s_I', 't', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        n = self.n
        r_e2s_I = outputs['r_e2s_I']

        T = inputs['LD'] + inputs['t'][:]/3600./24.
        for i in range(0, n):
            L = self.d2r*280.460 + self.d2r*0.9856474*T[i]
            g = self.d2r*357.528 + self.d2r*0.9856003*T[i]
            Lambda = L + self.d2r*1.914666*np.sin(g) + self.d2r*0.01999464*np.sin(2*g)
            eps = self.d2r*23.439 - self.d2r*3.56e-7*T[i]
            r_e2s_I[i, 0] = np.cos(Lambda)
            r_e2s_I[i, 1] = np.sin(Lambda)*np.cos(eps)
            r_e2s_I[i, 2] = np.sin(Lambda)*np.sin(eps)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n
        tconv = (1.0 / 3600. / 24.)

        T = inputs['LD'] + inputs['t'][:] * tconv
        dr_dt = np.empty(3)

        Ja = np.zeros(3*n)
        dL_dt = self.d2r * 0.9856474
        dg_dt = self.d2r * 0.9856003
        deps_dt = -self.d2r*3.56e-7

        for i in range(n):
            L = self.d2r*280.460 + self.d2r*0.9856474*T[i]
            g = self.d2r*357.528 + self.d2r*0.9856003*T[i]
            Lambda = L + self.d2r*1.914666*np.sin(g) + self.d2r*0.01999464*np.sin(2*g)
            eps = self.d2r*23.439 - self.d2r*3.56e-7*T[i]

            dlambda_dt = (dL_dt + self.d2r*1.914666*np.cos(g)*dg_dt +
                          self.d2r*0.01999464*np.cos(2*g)*2*dg_dt)

            dr_dt[0] = -np.sin(Lambda)*dlambda_dt
            dr_dt[1] = np.cos(Lambda)*np.cos(eps)*dlambda_dt - np.sin(Lambda)*np.sin(eps)*deps_dt
            dr_dt[2] = np.cos(Lambda)*np.sin(eps)*dlambda_dt + np.sin(Lambda)*np.cos(eps)*deps_dt

            Ja[i*3:i*3+3] = dr_dt

        dr_e2s = Ja.flatten()
        partials['r_e2s_I', 'LD'] = dr_e2s
        partials['r_e2s_I', 't'] = dr_e2s * tconv


class Sun_PositionSpherical(ExplicitComponent):
    """
    Compute the elevation angle of the Sun in the body-fixed frame.
    """

    def __init__(self, n=2):
        super(Sun_PositionSpherical, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2s_B', np.zeros((n, 3)), units='km',
                       desc='Position vector from Earth to Sun in body-fixed '
                            'frame over time.')

        # Outputs
        self.add_output('azimuth', np.zeros((n, )), units='rad',
                        desc='Ezimuth angle of the Sun in the body-fixed frame '
                             'over time.')

        self.add_output('elevation', np.zeros((n, )), units='rad',
                        desc='Elevation angle of the Sun in the body-fixed frame '
                             'over time.')

        rows = np.tile(np.array([0, 0, 0]), n) + np.repeat(np.arange(n), 3)
        cols = np.arange(n*3)

        self.declare_partials('elevation', 'r_e2s_B', rows=rows, cols=cols)

        self.declare_partials('azimuth', 'r_e2s_B', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        azimuth, elevation = computepositionspherical(self.n, inputs['r_e2s_B'])

        outputs['azimuth'] = azimuth
        outputs['elevation'] = elevation

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.n

        Ja1, Ja2 = computepositionsphericaljacobian(n, 3*n, inputs['r_e2s_B'])

        partials['azimuth', 'r_e2s_B'] = Ja1
        partials['elevation', 'r_e2s_B'] = Ja2
