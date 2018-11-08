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


class OrbitEOMComp(ExplicitComponent):
    """
    Computes the Earth to body position vector in Earth-centered intertial frame.
    """

    def initialize(self):

        self.options.declare('num_nodes', types=(int,))
        self.options.declare('GM', types=(float,), default=mu)  # GM of earth (km**3/s**2)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r_e2b_I', 1000.0*np.ones((nn, 3)), units='km',
                       desc='Position vectors from earth to satellite '
                            'in Earth-centered inertial frame over time')

        self.add_input('v_e2b_I', 1000.0*np.ones((nn, 3)), units='km/s',
                       desc='Velocity vectors from earth to satellite '
                            'in Earth-centered inertial frame over time')

        self.add_input('rmag_e2b', 1000.0*np.ones((nn,)), units='km',
                       desc='Position and velocity vectors from earth to satellite '
                            'in Earth-centered inertial frame over time')

        self.add_input('a_pert_I', np.zeros((nn, 3)), units='km/s**2',
                       desc='Perturbing accelerations in the Earth-centered inertial '
                            'frame over time')

        self.add_output('dXdt:r_e2b_I', 1000.0*np.ones((nn, 3)), units='km/s',
                        desc='Velocity vectors from earth to satellite '
                             'in Earth-centered inertial frame over time')

        self.add_output('dXdt:v_e2b_I', 1000.0*np.ones((nn, 3)), units='km/s**2',
                        desc='Acceleration vectors from earth to satellite '
                             'in Earth-centered inertial frame over time')

        ar = np.arange(3 * nn, dtype=int)

        self.declare_partials(of='dXdt:r_e2b_I', wrt='v_e2b_I', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='dXdt:v_e2b_I', wrt='r_e2b_I', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='dXdt:v_e2b_I', wrt='a_pert_I', rows=ar, cols=ar, val=1.0)

        rs = np.arange(nn * 3, dtype=int)
        cs = np.repeat(np.arange(nn, dtype=int), 3)

        self.declare_partials(of='dXdt:v_e2b_I', wrt='rmag_e2b', rows=rs, cols=cs, val=1.0)

    def compute(self, inputs, outputs):
        r = inputs['r_e2b_I']
        rmag = inputs['rmag_e2b']
        v = inputs['v_e2b_I']
        GM = self.options['GM']
        a_pert_I = inputs['a_pert_I']

        outputs['dXdt:r_e2b_I'] = v
        outputs['dXdt:v_e2b_I'] = (-GM * r) / rmag[:, np.newaxis]**3 + a_pert_I

    def compute_partials(self, inputs, partials):
        r = inputs['r_e2b_I']
        rmag = inputs['rmag_e2b']
        GM = self.options['GM']

        partials['dXdt:v_e2b_I', 'rmag_e2b'] = (3 * GM * r / rmag[:, np.newaxis]**4).ravel()
        partials['dXdt:v_e2b_I', 'r_e2b_I'] = -GM / np.repeat(rmag, 3)**3



if __name__ == '__main__':
    print('foo')