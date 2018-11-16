from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group, DirectSolver, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from CADRE.attitude_dymos.VectorUnitizeComp import VectorUnitizeComp

GM = 398600.44
rmag = 7000.0
period = 2 * np.pi * np.sqrt(rmag ** 3 / GM)
vcirc = np.sqrt(GM / rmag)
duration = period / 1

class TestCadreODE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nn = 1

        p = cls.p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('a', val=np.zeros((cls.nn, 3)), units='km')

        p.model.add_subsystem('vec_norm_comp',
                              VectorUnitizeComp(in_name='a', out_name='a_norm', vec_size=cls.nn,
                                                length=3, units='km'),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.setup(check=True, force_alloc_complex=True)

        # p['a'] = np.random.rand(cls.nn, 3)
        p['a'] = np.array([[4.2, 2, 3]]) / np.sqrt(1**2 + 2**2 + 3**2)
        print(p['a'])

        p.run_model()

    def test_results(self):
        a = self.p['a']
        for i in range(self.nn):
            a_norm = a[i, :] / np.sqrt(np.dot(a[i, :], a[i, :]))
            assert_rel_error(self, self.p['a_norm'][i, :], a_norm)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False)
        assert_check_partials(cpd, atol=1.0E-5, rtol=1.0)
