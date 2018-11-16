from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group, VectorMagnitudeComp
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.orbit_dymos.orbit_eom import OrbitEOMComp

class TestOrbitEOM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = 10

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('r_e2b_I', val=np.ones((nn, 3)))
        ivc.add_output('v_e2b_I', val=np.ones((nn, 3)))
        ivc.add_output('a_pert_I', val=np.ones((nn, 3)))

        cls.p.model.add_subsystem('rmag_comp', VectorMagnitudeComp(vec_size=nn, length=3,
                                                                   in_name='r_e2b_I',
                                                                   mag_name='rmag_e2b',
                                                                   units='km'),
                                  promotes_inputs=['*'], promotes_outputs=['*'])

        cls.p.model.add_subsystem('orbit_eom_comp', OrbitEOMComp(num_nodes=nn),
                                  promotes_inputs=['*'], promotes_outputs=['*'])

        cls.p.setup(check=True, force_alloc_complex=True)

        cls.p['r_e2b_I'] = np.random.rand(nn, 3) * 1000
        cls.p['v_e2b_I'] = np.random.rand(nn, 3) * 10
        cls.p['a_pert_I'] = np.random.rand(nn, 3)

        cls.p.run_model()


    def test_results(self):
        pass

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(method='cs')
        assert_check_partials(cpd)
