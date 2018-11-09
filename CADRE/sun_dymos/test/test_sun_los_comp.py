"""
Unit test vs saved data from John's CMF implementation.
"""
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem

from CADRE.sun_dymos.sun_los_comp import SunLOSComp
from CADRE.test.util import load_validation_data

#
# load saved data from John's CMF implementation.
#
n, m, h, setd = load_validation_data(idx='5')


class TestSunLOSDymos(unittest.TestCase):

    def test_component(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', SunLOSComp(num_nodes=1500),
                                        promotes=['*'])

        prob.setup()

        prob['r_e2b_I'] = setd['r_e2b_I'][:3, :].T
        prob['r_e2s_I'] = setd['r_e2s_I'].T

        prob.run_model()

        for var in ['LOS']:

            tval = setd[var].T

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))


if __name__ == "__main__":
    unittest.main()