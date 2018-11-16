from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group, VectorMagnitudeComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from CADRE.ori_comp import ORIComp

class TestOrbitEOM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = 4

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('r_e2b_I', val=np.ones((nn, 3)))
        ivc.add_output('v_e2b_I', val=np.ones((nn, 3)))
        # ivc.add_output('hunit_e2b_I', val=np.ones((nn, 3)))

        cls.p.model.add_subsystem('ori_comp', ORIComp(num_nodes=nn),
                                  promotes_inputs=['*'], promotes_outputs=['*'])

        cls.p.setup(check=True, force_alloc_complex=True)

        cls.p['r_e2b_I'] = np.random.rand(nn, 3)*10000
        cls.p['v_e2b_I'] = np.random.rand(nn, 3)*10
        # cls.p['hunit_e2b_I'] = np.random.rand(nn, 3)

        cls.p.run_model()


    def test_results(self):
        pass

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(method='fd')
        assert_check_partials(cpd)
