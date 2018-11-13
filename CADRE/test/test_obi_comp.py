from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.obi_comp import OBIComp

class TestOBRComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = 5

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('O_BR', val=np.ones((nn, 3, 3)))
        ivc.add_output('O_RI', val=np.ones((nn, 3, 3)))

        cls.p.model.add_subsystem('obr_comp', OBIComp(num_nodes=nn),
                                  promotes_inputs=['*'], promotes_outputs=['*'])

        cls.p.setup(check=True, force_alloc_complex=True)

        cls.p['O_BR'] = np.random.rand(nn, 3, 3)
        cls.p['O_RI'] = np.random.rand(nn, 3, 3)

        cls.p.run_model()


    def test_results(self):
        pass

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(method='cs')
        assert_check_partials(cpd)
