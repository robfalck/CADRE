"""
Sun discipline for CADRE: Sun LOS component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


def cross_deriv(v):
    """
    Compute derivative across a cross product of two 3 dimensional vectors.

    Given c = a x b, dc_da = cross_deriv(b), and dc_db = -cross_deriv(a)
    """
    m = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return m


class SunLOSComp(ExplicitComponent):
    """
    Compute the Satellite to sun line of sight.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")
        self.options.declare('Re', types=(float, ), default=6378.137,
                             desc="Radius of the Earth (km).")
        self.options.declare('alpha', types=(float, ), default=0.85,
                             desc="LOS smoothing factor.")

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r_e2b_I', np.zeros((nn, 3)), units='km',
                       desc='Position vector from '
                            'Earth to satellite in Earth-centered '
                            'inertial frame over time.')

        self.add_input('r_e2s_I', np.zeros((nn, 3)), units='km',
                       desc='Position vector from Earth to sun in Earth-centered '
                            'inertial frame over time.')

        self.add_output('LOS', np.zeros((nn, )), units=None,
                        desc='Satellite to sun line of sight over time')

        rows = np.tile(np.repeat(0, 3), nn) + np.repeat(np.arange(nn), 3)
        cols = np.arange(nn*3)

        self.declare_partials('LOS', 'r_e2b_I', rows=rows, cols=cols)
        self.declare_partials('LOS', 'r_e2s_I', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        nn = self.options['num_nodes']
        r2 = self.options['Re']
        r1 = r2 * self.options['alpha']

        r_e2b_I = inputs['r_e2b_I']
        r_e2s_I = inputs['r_e2s_I']
        LOS = outputs['LOS']

        for i in range(nn):
            r_b = r_e2b_I[i, :]
            r_s = r_e2s_I[i, :]
            dot = np.dot(r_b, r_s)
            cross = np.cross(r_b, r_s)
            dist = np.sqrt(cross.dot(cross))

            if dot >= 0.0:
                LOS[i] = 1.0
            elif dist <= r1:
                LOS[i] = 0.0
            elif dist >= r2:
                LOS[i] = 1.0
            else:
                x = (dist - r1) / (r2 - r1)
                LOS[i] = 3*x**2 - 2*x**3

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        nj = 3 * nn
        r2 = self.options['Re']
        r1 = r2 * self.options['alpha']

        r_e2b_I = inputs['r_e2b_I']
        r_e2s_I = inputs['r_e2s_I']

        Jab = np.zeros(shape=(nj, ), dtype=r_e2b_I.dtype)
        Jas = np.zeros(shape=(nj, ), dtype=r_e2b_I.dtype)

        for i in range(nn):
            r_b = r_e2b_I[i, :]
            r_s = r_e2s_I[i, :]
            dot = np.dot(r_b, r_s)

            if dot >= 0.0:
                continue

            cross = np.cross(r_b, r_s)
            dist = np.sqrt(np.dot(cross, cross))

            if dist <= r1 or dist >= r2:
                continue

            else:
                x = (dist - r1)/(r2 - r1)
                # LOS = 3*x**2 - 2*x**3
                ddist_dcross = cross / dist
                dcross_drb = cross_deriv(-r_s)
                dcross_drs = cross_deriv(r_b)
                dx_ddist = 1.0/(r2 - r1)
                dLOS_dx = 6*x - 6*x**2
                dLOS_drb = dLOS_dx * dx_ddist * np.dot(ddist_dcross, dcross_drb)
                dLOS_drs = dLOS_dx * dx_ddist * np.dot(ddist_dcross, dcross_drs)

                Jab[i*3:i*3+3] = dLOS_drb
                Jas[i*3:i*3+3] = dLOS_drs

        partials['LOS', 'r_e2b_I'] = Jab
        partials['LOS', 'r_e2s_I'] = Jas

