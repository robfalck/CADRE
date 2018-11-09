from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class CommDataRateComp(ExplicitComponent):
    """
    Compute the data bit rate based on the satellite position, communications power, and
    transmitter gain.
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

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        self.S2 = np.zeros(nn)

        # Inputs
        self.add_input('P_comm', np.zeros(nn), units='W',
                       desc='Communication power over time')

        self.add_input('gain', np.zeros(nn), units=None,
                       desc='Transmitter gain over time')

        self.add_input('GSdist', np.zeros(nn), units='km',
                       desc='Distance from ground station to satellite over time')

        self.add_input('CommLOS', np.zeros(nn), units=None,
                       desc='Satellite to ground station line of sight over time')

        # Outputs
        self.add_output('dXdt:data', np.zeros(nn), units='Gibyte/s',
                        desc='Download rate over time')

        row_col = np.arange(nn)

        self.declare_partials('dXdt:data', 'P_comm', rows=row_col, cols=row_col)
        self.declare_partials('dXdt:data', 'gain', rows=row_col, cols=row_col)
        self.declare_partials('dXdt:data', 'GSdist', rows=row_col, cols=row_col)
        self.declare_partials('dXdt:data', 'CommLOS', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        """
         Calculate outputs.
        """
        P_comm = inputs['P_comm']
        gain = inputs['gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']
        Dr = outputs['dXdt:data']
        S2 = self.S2

        # Where distance to ground station nonzero, or close to zero
        nz_idxs = np.where(np.abs(GSdist) > 1.0E-10)
        zero_idxs = np.where(np.abs(GSdist) <= 1.0E-10)

        S2[nz_idxs] = GSdist[nz_idxs] * 1e3
        S2[zero_idxs] = 1.0E-10

        Dr[nz_idxs] = self.alpha * P_comm[nz_idxs] * gain[nz_idxs] * CommLOS[nz_idxs] / S2[nz_idxs] ** 2
        Dr[zero_idxs] = self.alpha * P_comm[zero_idxs] * gain[zero_idxs] * CommLOS[zero_idxs] / S2[zero_idxs]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        P_comm = inputs['P_comm']
        gain = inputs['gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']
        S2 = self.S2

        # Where distance to ground station nonzero, or close to zero
        nz_idxs = np.where(np.abs(GSdist) > 1.0E-10)
        zero_idxs = np.where(np.abs(GSdist) <= 1.0E-10)

        S2[nz_idxs] = GSdist[nz_idxs] * 1e3
        S2[zero_idxs] = 1.0E-10

        rec_S2_sq = 1.0 / S2 ** 2
        partials['dXdt:data', 'P_comm'] = self.alpha * gain * CommLOS * rec_S2_sq
        partials['dXdt:data', 'gain'] = self.alpha * P_comm * CommLOS * rec_S2_sq
        partials['dXdt:data', 'GSdist'] = -2.0 * 1e3 * self.alpha * P_comm * gain * CommLOS * rec_S2_sq / S2
        partials['dXdt:data', 'CommLOS'] = self.alpha * P_comm * gain * rec_S2_sq