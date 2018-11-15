"""
Battery discipline for CADRE
"""

import numpy as np

from openmdao.api import ExplicitComponent
from openmdao.components.ks_comp import KSfunction

# Constants
sigma = 1e-10
eta = 0.99
Cp = 2900.0*0.001*3600.0
IR = 0.9
T0 = 293.0
alpha = np.log(1/1.1**5)


class BatterySOCComp(ExplicitComponent):
    """
    Computes the rate of change of the battery state of charge and the battery current draw.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('SOC', np.ones((nn,)), units=None,
                       desc='battery state of charge over time')

        self.add_input('P_bat', np.zeros((nn,)), units='W',
                       desc='Battery power over time')

        self.add_input('temperature', 273.0 * np.ones((nn, 5)), units='degK',
                       desc='Battery temperature over time')

        # Outputs
        self.add_output('dXdt:SOC', np.zeros((nn,)), units='1/s',
                        desc='Rate of change of state of charge over time')

        self.add_output('I_bat', np.zeros((nn,)), units='A',
                        desc='Battery current draw over time')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='dXdt:SOC', wrt='SOC', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:SOC', wrt='P_bat', rows=ar, cols=ar)

        self.declare_partials(of='I_bat', wrt='SOC', rows=ar, cols=ar)
        self.declare_partials(of='I_bat', wrt='P_bat', rows=ar, cols=ar)

        cs = np.arange(4, nn*5, 5, dtype=int)

        self.declare_partials(of='I_bat', wrt='temperature', rows=ar, cols=cs)
        self.declare_partials(of='dXdt:SOC', wrt='temperature', rows=ar, cols=cs)

        # self.options['state_var'] = 'SOC'
        # self.options['init_state_var'] = 'iSOC'
        # self.options['external_vars'] = ['P_bat', 'temperature']

    def compute(self, inputs, outputs):
        SOC = inputs['SOC']
        P = inputs['P_bat']
        T = inputs['temperature'][:, 4]

        voc = 3 + np.expm1(SOC) / (np.e - 1)
        V = IR * voc * (2.0 - np.exp(alpha*(T-T0)/T0))
        I = P/V  # noqa: E741

        outputs['dXdt:SOC'] = sigma/24*SOC - eta/Cp*I
        outputs['I_bat'] = I

    def compute_partials(self, inputs, partials):
        SOC = inputs['SOC']
        P = inputs['P_bat']
        T = inputs['temperature'][:, 4]

        voc = 3 + np.expm1(SOC) / (np.e-1)
        dVoc_dSOC = np.exp(SOC) / (np.e-1)

        tmp = 2 - np.exp(alpha*(T-T0)/T0)
        V = IR * voc * tmp
        dV_dvoc = IR * tmp

        # I = P/V

        dV_dSOC = IR * dVoc_dSOC * tmp
        dI_dSOC = -P/V**2 * dV_dSOC

        dV_dT = - IR * voc * np.exp(alpha*(T-T0)/T0) * alpha/T0
        dI_dT = - P/V**2 * dV_dT
        dI_dP = 1.0/V

        partials['dXdt:SOC', 'SOC'] = sigma/24 - eta/Cp*dI_dSOC
        partials['dXdt:SOC', 'P_bat'] = -eta/Cp*dI_dP
        partials['dXdt:SOC', 'temperature'] = -eta/Cp*dI_dT

        tmp2 = -P/(V**2)
        partials['I_bat', 'temperature'] = tmp2 * dV_dT
        partials['I_bat', 'SOC'] = tmp2 * dV_dvoc * dVoc_dSOC
        partials['I_bat', 'P_bat'] = 1.0 / V
