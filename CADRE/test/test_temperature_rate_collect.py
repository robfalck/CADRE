from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from CADRE.temperature_rate_collect_comp import TemperatureRateCollectComp

class TestTemperatureRateCollect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = cls.nn = 6

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('dXdt:T_bat', val=np.ones(nn,))
        ivc.add_output('dXdt:T_fins', val=np.ones((nn, 4)))

        cls.p.model.add_subsystem('temperature_rate_collect',
                                  TemperatureRateCollectComp(num_nodes=nn),
                                  promotes_inputs=['*'], promotes_outputs=['*'])

        cls.p.setup(check=True, force_alloc_complex=True)

        cls.p['dXdt:T_bat'] = np.random.rand(nn)
        cls.p['dXdt:T_fins'] = np.random.rand(nn, 4)

        cls.p.run_model()

    def test_results(self):
        expected = np.zeros((self.nn, 5))
        expected[:, :4] = self.p['dXdt:T_fins']
        expected[:, 4] = self.p['dXdt:T_bat']

        assert_rel_error(self, self.p['dXdt:temperature'], expected)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(method='cs')
        assert_check_partials(cpd)
