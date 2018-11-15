"""
Thermal discipline for CADRE
"""
from __future__ import print_function, division, absolute_import

from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent

from dymos.utils.indexing import get_src_indices_by_row

# constants
m_f = 0.4
m_b = 2.0
cp_f = 0.6e3
cp_b = 2.0e3
A_T = 2.66e-3

alpha_c = 0.9
alpha_r = 0.2
eps_c = 0.87
eps_r = 0.88

q_sol = 1360.0
K = 5.67051e-8


class ThermalTemperatureComp(ExplicitComponent):
    """
    Calculates the temperature distribution on the solar panels.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int, ))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('temperature', np.zeros((nn, 5)), units='degK',
                        desc='Temperature for the 4 fins and body over time.')
        # TODO upper and lower bounds become constraints on problem?
                        #lower=50, upper=400)

        self.add_input('exposed_area', np.zeros((nn, 7, 12)), units='m**2',
                       desc='Exposed area to the sun for each solar cell over time')

        self.add_input('cellInstd', np.ones((7, 12)), units=None,
                       desc='Cell/Radiator indication')

        self.add_input('LOS', np.zeros((nn, )), units=None,
                       desc='Satellite to Sun line of sight over time')

        self.add_input('P_comm', np.ones((nn, )), units='W',
                       desc='Communication power over time')

        # Outputs
        self.add_output('dXdt:temperature', np.zeros((nn, 5)), units='degK/s',
                        desc='Rate of change of temperature over time')

        #self.options['state_var'] = 'temperature'
        #self.options['init_state_var'] = 'T0'
        #self.options['external_vars'] = ['exposed_area', 'LOS', 'P_comm']
        #self.options['fixed_external_vars'] = ['cellInstd']

        # Precompute the Panel data so that we can use the sparsity patterns.

        self.panel_f_i = np.empty(12, dtype=np.int)
        self.panel_m = np.empty(12)
        self.panel_cp = np.empty(12)

        for p in range(12):

            # Body
            if p < 4:
                f_i = 4
                m = m_b
                cp = cp_b

            # Fin
            else:
                f_i = (p+1) % 4
                m = m_f
                cp = cp_f

            self.panel_f_i[p] = f_i
            self.panel_m[p] = m
            self.panel_cp[p] = cp

        # Derivatives

        row_col = np.arange(nn*5)

        self.declare_partials(of='dXdt:temperature', wrt='temperature', rows=row_col, cols=row_col)

        row = np.tile(np.array(self.panel_f_i), 7)
        rows = np.tile(row, nn) + np.repeat(5*np.arange(nn), 84)
        cols = np.tile(np.arange(84), nn) + np.repeat(84*np.arange(nn), 84)

        self.declare_partials(of='dXdt:temperature', wrt='exposed_area', rows=rows, cols=cols)

        cols = np.tile(np.arange(84), nn)

        self.declare_partials(of='dXdt:temperature', wrt='cellInstd', rows=rows, cols=cols)

        rows = np.arange(5*nn)
        cols = np.tile(np.repeat(0, 5), nn) + np.repeat(np.arange(nn), 5)

        self.declare_partials(of='dXdt:temperature', wrt='LOS', rows=rows, cols=cols)

        rows = np.tile(4, nn) + 5*np.arange(nn)
        cols = np.arange(nn)

        self.declare_partials(of='dXdt:temperature', wrt='P_comm', rows=rows, cols=cols, val=4.0/m_b/cp_b)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        temperature = inputs['temperature']
        exposed_area = inputs['exposed_area']
        cellInstd = inputs['cellInstd']
        LOS = inputs['LOS']
        P_comm = inputs['P_comm']

        # revised implementation from ThermalTemperature.f90
        alpha = alpha_c * cellInstd + alpha_r - alpha_r * cellInstd
        s_eps = np.sum(eps_c*cellInstd + eps_r - eps_r*cellInstd, 0)

        fact1 = q_sol * LOS
        fact2 = K * A_T * temperature**4

        s_al_ea = np.sum(alpha*exposed_area, 1) * fact1[:, np.newaxis]

        # Panels
        outputs['dXdt:temperature'][:] = 0.0
        for p in range(12):
            f_i = self.panel_f_i[p]
            cp = self.panel_cp[p]
            m = self.panel_m[p]

            # Cells
            outputs['dXdt:temperature'][:, f_i] += (s_al_ea[:, p] - s_eps[p]*fact2[:, f_i]) * (1.0/(m*cp))

        outputs['dXdt:temperature'][:, 4] += 4.0 * P_comm / m_b / cp_b

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        temperature = inputs['temperature']
        exposed_area = inputs['exposed_area']
        cellInstd = inputs['cellInstd']
        LOS = inputs['LOS']
        P_comm = inputs['P_comm']

        # revised implementation from ThermalTemperature.f90
        d_temperature = np.zeros((nn, 5), dtype=temperature.dtype)
        d_exposed_area = np.zeros((nn, 7, 12), dtype=temperature.dtype)
        d_cellInstd = np.zeros((nn, 7, 12), dtype=temperature.dtype)
        d_LOS = np.zeros((nn, 5), dtype=temperature.dtype)

        alpha = alpha_c*cellInstd + alpha_r - alpha_r*cellInstd
        sum_eps = 4.0 * K * A_T * np.sum(eps_c*cellInstd + eps_r - eps_r*cellInstd, 0)

        dalpha_dw = alpha_c - alpha_r
        deps_dw = eps_c - eps_r

        alpha_A_sum = np.sum(alpha * exposed_area, 1)

        # Panels
        for p in range(0, 12):
            f_i = self.panel_f_i[p]
            cp = self.panel_cp[p]
            m = self.panel_m[p]

            # Cells
            fact = temperature[:, f_i]**3 * (1.0 / (m*cp))
            fact3 = q_sol/(m*cp)
            fact1 = fact3*LOS
            fact2 = K * A_T * temperature[:, f_i] * fact

            d_temperature[:, f_i] -= sum_eps[p] * fact
            d_LOS[:, f_i] += alpha_A_sum[:, p] * fact3
            d_cellInstd[:, :, p] = dalpha_dw * exposed_area[:, :, p] * fact1[:, np.newaxis] - \
                (deps_dw * fact2)[:, np.newaxis]
            d_exposed_area[:, :, p] = np.outer(fact1, alpha[:, p])

        partials['dXdt:temperature', 'temperature'] = d_temperature.flatten()
        partials['dXdt:temperature', 'exposed_area'] = d_exposed_area.flatten()
        partials['dXdt:temperature', 'cellInstd'] = d_cellInstd.flatten()
        partials['dXdt:temperature', 'LOS'] = d_LOS.flatten()
