from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.orbit_dymos.gravity_perturbations_comp import GravityPerturbationsComp

class TestOrbitEOM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = 4

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('r_e2b_I', val=np.ones((nn, 3)))
        ivc.add_output('rmag_e2b_I', val=np.ones((nn)))
        # ivc.add_output('hunit_e2b_I', val=np.ones((nn, 3)))

        cls.p.model.add_subsystem('gp_comp', GravityPerturbationsComp(num_nodes=nn),
                                  promotes_inputs=['*'], promotes_outputs=['*'])

        cls.p.setup(check=True, force_alloc_complex=True)

        cls.p['r_e2b_I'] = np.random.rand(nn, 3)*6000

        for i in range(nn):
            r_i = cls.p['r_e2b_I'][i, :]
            cls.p['rmag_e2b_I'][i] = np.sqrt(np.dot(r_i, r_i))

        cls.p.run_model()


    def test_results(self):
        pass

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(method='fd')
        assert_check_partials(cpd)
