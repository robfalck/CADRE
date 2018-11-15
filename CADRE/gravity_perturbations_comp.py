"""
Orbit discipline for CADRE
"""
from math import sqrt

from six.moves import range
import numpy as np

from openmdao.api import ExplicitComponent

from CADRE import rk4


# Constants
mu = 398600.44
Re = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

C1 = -mu
C2 = -1.5*mu*J2*Re**2
C3 = -2.5*mu*J3*Re**3
C4 = 1.875*mu*J4*Re**4


class GravityPerturbationsComp(ExplicitComponent):
    """
    Computes the Earth to body position vector in Earth-centered intertial frame.
    """

    def initialize(self):

        self.options.declare('num_nodes', types=(int,))
        self.options.declare('GM', types=(float,), default=mu)  # GM of earth (km**3/s**2)

    def setup(self):
        nn = self.options['num_nodes']

        self.z_hat = np.zeros((nn, 3))
        self.z_hat[:, 2] = 1

        self.add_input('r_e2b_I', 1000.0*np.ones((nn, 3)), units='km',
                       desc='Position vectors from earth to satellite '
                            'in Earth-centered inertial frame over time')

        self.add_input('rmag_e2b_I', 1000.0*np.ones((nn,)), units='km',
                       desc='Radius magnitude from earth to satellite '
                            'in Earth-centered inertial frame over time')

        self.add_output('a_pert:J2', 0.0001*np.ones((nn, 3)), units='km/s**2',
                        desc='Acceleration vectors from earth to satellite '
                             'in Earth-centered inertial frame over time')

        temp = np.eye(3, dtype=int)
        temp[:, -1] = [3, 3, 2]
        template = np.kron(np.eye(nn, dtype=int), temp)
        rs, cs = np.nonzero(template)
        nonzero_template = template[rs, cs].ravel()
        self.diag_idxs = np.where(nonzero_template <= 2)[0]
        self.col3_idxs = np.where(nonzero_template >= 2)[0]

        self.declare_partials(of='a_pert:J2', wrt='r_e2b_I', rows=rs, cols=cs, val=1.0)

        rs = np.arange(nn * 3, dtype=int)
        cs = np.repeat(np.arange(nn, dtype=int), 3)

        self.declare_partials(of='a_pert:J2', wrt='rmag_e2b_I', rows=rs, cols=cs, val=1.0)

    def compute(self, inputs, outputs):
        r = inputs['r_e2b_I']
        rmag = inputs['rmag_e2b_I']

        Reor = Re / rmag
        zhat = self.z_hat
        rz = r[:, 2]
        rz2or2 = (rz / rmag)**2

        a2 = -1.5 * mu * J2 * Reor**2 / rmag**3
        b2 = (1 - 5 * rz2or2[:, np.newaxis]) * r
        c2 = 2 * rz[:, np.newaxis] * zhat

        outputs['a_pert:J2'] = a2[:, np.newaxis] * (b2 + c2)

        print()
        print(outputs['a_pert:J2'])

    def compute_partials(self, inputs, partials):
        r = inputs['r_e2b_I']
        rmag = inputs['rmag_e2b_I']

        Reor = Re / rmag
        zhat = self.z_hat
        rz = r[:, 2]

        rzor = rz / rmag
        rz2or2 = rzor * rzor

        drzor_drmag = -rz / rmag**2
        drz2or2_drmag = 2 * rzor * drzor_drmag


        a2 = -1.5 * mu * J2 * Reor**2 / rmag**3
        b2 = (1 - 5 * rz2or2[:, np.newaxis]) * r
        c2 = 2 * rz[:, np.newaxis] * zhat

        da2_drmag = 5 * 1.5 * mu * J2 * Reor**2 * rmag**-4
        db2_drmag = -5 * (drz2or2_drmag[:, np.newaxis] * r)

        drzor_dr = zhat / rmag[:, np.newaxis]
        drz2or2_dr = 2 * rzor[:, np.newaxis] * drzor_dr
        db2_dr = - 5 * (drz2or2_dr * r + rz2or2[:, np.newaxis])
        dc2_dr = 2 * zhat

        a2[:, np.newaxis] * (b2 + c2)

        # X = da2_dr * b2 + a2 * db2_dr + \
        #     da2_dr * c2 + a2 * dc2_dr

        X = a2[:, np.newaxis] * (db2_dr + dc2_dr)
        print('X')
        print(X)
        print('done')


        # da2_dr = 0
        # db2_dr = -5 * (drz2or2_dr * r + rz2or2[:, np.newaxis])
        # dc2_dr = 2 * zhat

        partials['a_pert:J2', 'rmag_e2b_I'] = (da2_drmag[:, np.newaxis] * b2 +
                                               a2[:, np.newaxis] * db2_drmag +
                                               da2_drmag[:, np.newaxis] * c2).ravel()

        for i in range(self.options['num_nodes']):
            rvec = inputs['r_e2b_I'][i, :]
            x = rvec[0]
            y = rvec[1]
            z = rvec[2]

            z2 = z * z
            z3 = z2 * z
            z4 = z3 * z

            r = sqrt(x * x + y * y + z2)

            r2 = r * r
            r3 = r2 * r
            r4 = r3 * r
            r5 = r4 * r
            r6 = r5 * r
            r7 = r6 * r
            r8 = r7 * r

            dr = np.array([x, y, z]) / r

            T2 = 1 - 5 * z2 / r2
            T3 = 3 * z - 7 * z3 / r2
            T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
            T3z = 3 * z - 0.6 * r2 / z
            T4z = 4 - 28.0 / 3.0 * z2 / r2

            dT2 = (10 * z2) / (r3) * dr
            dT2[2] -= 10. * z / r2

            dT3 = 14 * z3 / r3 * dr
            dT3[2] -= 21. * z2 / r2 - 3

            dT4 = (28 * z2 / r3 - 84. * z4 / r5) * dr
            dT4[2] -= 28 * z / r2 - 84 * z3 / r4

            dT3z = -1.2 * r / z * dr
            dT3z[2] += 0.6 * r2 / z2 + 3

            dT4z = 56.0 / 3.0 * z2 / r3 * dr
            dT4z[2] -= 56.0 / 3.0 * z / r2

            eye = np.identity(3)

            dfdy = np.zeros((3, 3))

            dfdy[:, :] += eye * (C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 + C4 / r7 * T4)
            fact = (-3 * C1 / r4 - 5 * C2 / r6 * T2 - 7 * C3 / r8 * T3 - 7 * C4 / r8 * T4)
            dfdy[:, 0] += dr[0] * rvec * fact
            dfdy[:, 1] += dr[1] * rvec * fact
            dfdy[:, 2] += dr[2] * rvec * fact
            dfdy[:, 0] += rvec * (C2 / r5 * dT2[0] + C3 / r7 * dT3[0] + C4 / r7 * dT4[0])
            dfdy[:, 1] += rvec * (C2 / r5 * dT2[1] + C3 / r7 * dT3[1] + C4 / r7 * dT4[1])
            dfdy[:, 2] += rvec * (C2 / r5 * dT2[2] + C3 / r7 * dT3[2] + C4 / r7 * dT4[2])
            dfdy[2, :] += dr * z * (-5 * C2 / r6 * 2 - 7 * C3 / r8 * T3z - 7 * C4 / r8 * T4z)
            dfdy[2, :] += z * (C3 / r7 * dT3z + C4 / r7 * dT4z)
            dfdy[2, 2] += (C2 / r5 * 2 + C3 / r7 * T3z + C4 / r7 * T4z)

            print('dfdy')
            print(dfdy)

        # partials['a_pert:J2', 'r_e2b_I'] = a2[:, np.newaxis] * db2_dr
        # partials['a_pert:J2', 'r_e2b_I'][self.col3_idxs] += (da2_dr * c2 + a2[:, np.newaxis] * dc2_dr).ravel()
        # print()
        # print((da2_dr * c2 + a2[:, np.newaxis] * dc2_dr).ravel())


        # a2[:, np.newaxis] * (b2 + c2)

        # partials['a_pert:J2', 'r_e2b_I'] = ((1 - 5 * rz2or2[:, np.newaxis]) * np.eye(4)).ravel()



if __name__ == '__main__':
    print('foo')