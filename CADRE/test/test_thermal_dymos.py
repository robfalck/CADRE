from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from CADRE.thermal_dymos import ThermalTemperatureComp


class TestBatteryDymos(unittest.TestCase):

    def test_derivatives(self):
        nn = 5

        prob = Problem(model=Group())

        ivc = prob.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('temperature', val=np.ones((nn, 5)))
        ivc.add_output('exposedArea', val=np.ones((nn, 7, 12)))
        ivc.add_output('cellInstd', val=np.ones((7, 12)))
        ivc.add_output('LOS', val=np.ones((nn, )))
        ivc.add_output('P_comm', val=np.ones((nn, )))

        prob.model.add_subsystem('comp', ThermalTemperatureComp(num_nodes=nn),
                                  promotes_inputs=['*'], promotes_outputs=['*'])

        prob.setup(check=True, force_alloc_complex=True)

        prob['temperature'] = 273 + np.random.random((nn, 5)) * 100
        prob['exposedArea'] = np.random.random((nn, 7, 12))
        prob['cellInstd'] = np.random.random((7, 12))
        prob['LOS'] = np.random.random((nn, ))
        prob['P_comm'] = np.random.random((nn, ))

        prob.run_model()

        J = self.prob.check_partials(method='cs')
        assert_check_partials(J)


if __name__ == '__main__':
    unittest.main()
