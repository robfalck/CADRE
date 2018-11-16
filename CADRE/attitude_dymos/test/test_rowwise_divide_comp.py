from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from CADRE.rowwise_divide_comp import RowwiseDivideComp

GM = 398600.44
rmag = 7000.0
period = 2 * np.pi * np.sqrt(rmag ** 3 / GM)
vcirc = np.sqrt(GM / rmag)
duration = period / 1

class TestCadreODE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nn = 5

        p = cls.p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('a', val=np.zeros((cls.nn, 3)), units='km')
        ivc.add_output('a_mag', val=np.zeros((cls.nn,)), units='km')

        p.model.add_subsystem('vec_norm_comp',
                              RowwiseDivideComp(a_name='a', b_name='a_mag', c_name='a_unit',
                                                vec_size=cls.nn, length=3, a_units='km',
                                                b_units='km'),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.setup(check=True, force_alloc_complex=True)

        p['a'] = np.random.rand(cls.nn, 3)
        for i in range(cls.nn):
            p['a_mag'][i] = np.sqrt(np.dot(p['a'][i, :], p['a'][i, :]))

        p.run_model()

    def test_results(self):
        a = self.p['a']
        for i in range(self.nn):
            a_unit = a[i, :] / np.sqrt(np.dot(a[i, :], a[i, :]))
            assert_rel_error(self, self.p['a_unit'][i, :], a_unit)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False)
        assert_check_partials(cpd, atol=1.0E-5, rtol=1.0)
