"""
Rate collector to assemble all temperature rates
"""
from __future__ import print_function, division, absolute_import
import numpy as np

from openmdao.api import ExplicitComponent


class TemperatureRateCollectComp(ExplicitComponent):
    """
    Computes the Earth to body position vector in Earth-centered intertial frame.
    """

    def initialize(self):

        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('dXdt:T_bat', 0.0*np.ones((nn,)), units='degK/s',
                       desc='Rate of change of the battery temperature')

        self.add_input('dXdt:T_fins', 0.0*np.ones((nn, 4)), units='degK/s',
                       desc='Rate of change of the temperature of each fin')

        self.add_output('dXdt:temperature', 0.0*np.ones((nn, 5)), units='degK/s',
                        desc='Rate of change of the temperature vector')

        rs = np.arange(4, nn*5, 5, dtype=int)
        cs = np.arange(nn, dtype=int)

        self.declare_partials(of='dXdt:temperature', wrt='dXdt:T_bat', rows=rs, cols=cs, val=1.0)

        temp=np.zeros((5, 4), dtype=int)
        temp[:4, :4] = np.eye(4, dtype=int)
        template = np.kron(np.eye(nn, dtype=int), temp)
        rs, cs = np.nonzero(template)

        self.declare_partials(of='dXdt:temperature', wrt='dXdt:T_fins', rows=rs, cols=cs, val=1.0)

    def compute(self, inputs, outputs):

        outputs['dXdt:temperature'][:, :4] = inputs['dXdt:T_fins']
        outputs['dXdt:temperature'][:, 4] = inputs['dXdt:T_bat']
