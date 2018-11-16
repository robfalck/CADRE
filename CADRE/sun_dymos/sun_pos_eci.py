"""
Sun discipline for CADRE: Sun Position ECI component.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class SunPositionECIComp(ExplicitComponent):
    """
    Compute the position vector from Earth to Sun in Earth-centered inertial frame.
    """

    # constants
    d2r = np.pi/180.

    def initialize(self):
        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('LD', 5233.5, units='d',
                       desc='Launch day.')

        self.add_input('t', np.zeros((nn, )), units='s',
                       desc='Time vector from simulation.')

        # Outputs
        self.add_output('r_e2s_I', np.zeros((nn, 3)), units='km',
                        desc='Position vector from Earth to Sun in Earth-centered '
                             'inertial frame over time.')

        self.declare_partials('r_e2s_I', 'LD')

        rows = np.arange(nn*3)
        cols = np.tile(np.repeat(0, 3), nn) + np.repeat(np.arange(nn), 3)

        self.declare_partials('r_e2s_I', 't', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        d2r = self.d2r
        sec_to_day = 1.0 / 3600. / 24.

        r_e2s_I = outputs['r_e2s_I']

        T = inputs['LD'] + inputs['t'][:] * sec_to_day
        L = d2r*280.460 + d2r*0.9856474*T
        g = d2r*357.528 + d2r*0.9856003*T
        Lambda = L + d2r*1.914666 * np.sin(g) + d2r * 0.01999464 * np.sin(2*g)
        eps = d2r * 23.439 - d2r * 3.56e-7 * T

        r_e2s_I[:, 0] = np.cos(Lambda)
        r_e2s_I[:, 1] = np.sin(Lambda) * np.cos(eps)
        r_e2s_I[:, 2] = np.sin(Lambda) * np.sin(eps)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        d2r = self.d2r
        sec_to_day = 1.0 / 3600. / 24.

        T = inputs['LD'] + inputs['t'][:] * sec_to_day
        dr_dt = np.empty((nn, 3))

        dL_dt = d2r * 0.9856474
        dg_dt = d2r * 0.9856003
        deps_dt = -d2r*3.56e-7

        L = d2r*280.460 + d2r*0.9856474*T
        g = d2r*357.528 + d2r*0.9856003*T
        Lambda = L + d2r*1.914666 * np.sin(g) + d2r * 0.01999464 * np.sin(2*g)
        eps = d2r * (23.439 - 3.56e-7 * T)

        dlambda_dt = (dL_dt + d2r * 1.914666 * np.cos(g) * dg_dt +
                      d2r * 0.01999464 * np.cos(2*g) * 2 * dg_dt)

        dr_dt[:, 0] = -np.sin(Lambda)*dlambda_dt
        dr_dt[:, 1] = np.cos(Lambda)*np.cos(eps)*dlambda_dt - np.sin(Lambda)*np.sin(eps)*deps_dt
        dr_dt[:, 2] = np.cos(Lambda)*np.sin(eps)*dlambda_dt + np.sin(Lambda)*np.cos(eps)*deps_dt

        dr_e2s = dr_dt.flatten()
        partials['r_e2s_I', 'LD'] = dr_e2s
        partials['r_e2s_I', 't'] = dr_e2s * sec_to_day
