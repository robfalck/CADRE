from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.orbit_dymos.obr_comp import OBRComp

class TestOBRComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = 3

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('Gamma', val=np.ones((nn,)))

        cls.p.model.add_subsystem('obr_comp', OBRComp(num_nodes=nn),
                                  promotes_inputs=['*'], promotes_outputs=['*'])

        cls.p.setup(check=True, force_alloc_complex=True)

        cls.p['Gamma'] = np.random.rand(nn)*np.pi

        cls.p.run_model()


    def test_results(self):
        pass

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(method='cs')
        assert_check_partials(cpd)
