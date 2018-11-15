from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class CommLOSComp(ExplicitComponent):
    """
    Determines if the Satellite has line of sight with the ground stations.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))
        self.options.declare('Re', types=(float,), default=6378.137,
                             desc="Radius of the Earth (km).")

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('r_b2g_I', np.zeros((nn, 3)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('r_e2g_I', np.zeros((nn, 3)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('CommLOS', np.zeros(nn), units=None,
                        desc='Satellite to ground station line of sight over time')

        rows = np.tile(np.array([0, 0, 0]), nn) + np.repeat(np.arange(nn), 3)
        cols = np.tile(np.arange(3), nn) + np.repeat(3*np.arange(nn), 3)

        self.declare_partials('CommLOS', 'r_b2g_I', rows=rows, cols=cols)
        self.declare_partials('CommLOS', 'r_e2g_I', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        nn = self.options['num_nodes']
        Re = self.options['Re']
        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']
        CommLOS = outputs['CommLOS']

        Rb = 100.0
        for i in range(nn):
            proj = np.dot(r_b2g_I[i, :], r_e2g_I[i, :]) / Re

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
        nn = self.options['num_nodes']
        Re = self.options['Re']
        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']

        dLOS_drb = np.zeros((nn, 3))
        dLOS_dre = np.zeros((nn, 3))

        Rb = 100.0
        for i in range(nn):

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