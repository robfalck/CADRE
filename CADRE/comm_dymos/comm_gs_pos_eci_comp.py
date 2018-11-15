from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class CommGSPosECIComp(ExplicitComponent):
    """
    Convert time history of ground station position from earth frame
    to inertial frame.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))
        self.options.declare('lat', types=(float,), default=42.2708,
                             desc='ground station latitude (degrees)')
        self.options.declare('lon', types=(float,), default=-83.7264,
                             desc='ground station longitude (degrees)')
        self.options.declare('alt', types=(float,), default=0.256,
                             desc='ground station altitude (km)')
        self.options.declare('Re', types=(float,), default=6378.137,
                             desc='Earth equatorial radius (km)')

    def setup(self):
        n = self.options['num_nodes']
        lat = self.options['lat']
        lon = self.options['lon']
        alt = self.options['alt']
        Re = self.options['Re']

        cos_lat = np.cos(np.radians(lat))
        sin_lat = np.sin(np.radians(lat))
        cos_lon = np.cos(np.radians(lon))
        sin_lon = np.sin(np.radians(lon))
        r_GS = (Re + alt)
        self.r_e2g_ECEF = np.array([r_GS * cos_lat * cos_lon,
                                    r_GS * cos_lat * sin_lon,
                                    r_GS * sin_lat])

        # Inputs
        self.add_input('O_IE', np.zeros((n, 3, 3)), units=None,
                       desc='Rotation matrix from ECI (I) to ECEF (E)')

        # Outputs
        self.add_output('r_e2g_I', np.zeros((n, 3)), units='km',
                        desc='Position vector from earth to ground station in '
                             'Earth-centered inertial frame over time')

        rows = np.tile(np.repeat(0, 3), 3*n) + np.repeat(np.arange(3*n), 3)
        cols = np.arange(9*n)

        self.declare_partials('r_e2g_I', 'O_IE', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_IE = inputs['O_IE']
        r_e2g_ECEF = self.r_e2g_ECEF
        r_e2g_I = outputs['r_e2g_I']

        for i in range(0, self.options['num_nodes']):
            r_e2g_I[i, :] = np.dot(O_IE[i, :, :], r_e2g_ECEF)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_e2g_ECEF = self.r_e2g_ECEF

        J1 = np.zeros((self.options['num_nodes'], 3, 3))

        for k in range(0, 3):
            for v in range(0, 3):
                J1[:, k, v] = r_e2g_ECEF[v]

        partials['r_e2g_I', 'O_IE'] = J1.flatten()