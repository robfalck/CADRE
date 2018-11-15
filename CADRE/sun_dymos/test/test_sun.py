"""
Unit test vs saved data from John's CMF implementation.
"""
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.sun_dymos.sun_group import SunGroup
from CADRE.sun_dymos.sun_los_comp import SunLOSComp
from CADRE.sun_dymos.sun_pos_eci import SunPositionECIComp
from CADRE.sun_dymos.sun_pos_spherical import SunPositionSphericalComp
from CADRE.test.util import load_validation_data

#
# load saved data from John's CMF implementation.
#
n, m, h, setd = load_validation_data(idx='5')


class TestSun(unittest.TestCase):

    def test_SunLOSComp(self):
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

    def test_SunPositionECI(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', SunPositionECIComp(num_nodes=1500),
                                        promotes=['*'])

        prob.setup()

        prob['LD'] = setd['LD']
        prob['t'] = setd['t'].T

        prob.run_model()

        for var in ['r_e2s_I']:

            tval = setd[var].T

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))

    def test_SunPositionECI(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', SunPositionSphericalComp(num_nodes=1500),
                                        promotes=['*'])

        prob.setup()

        prob['r_e2s_B'] = setd['r_e2s_B'].T

        prob.run_model()

        for var in ['azimuth', 'elevation']:

            tval = setd[var].T

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))

    def test_partials(self):

        # this subrange has a few points in the smoothing region.
        idx = np.arange(108, 113)

        prob = Problem(model = SunGroup(num_nodes=len(idx)))

        prob.setup()

        prob['t'] = setd['t'][idx].T
        prob['LD'] = 0.0
        prob['r_e2b_I'] = setd['r_e2b_I'][:3, idx].T

        prob.run_model()

        np.set_printoptions(linewidth=100000, edgeitems=10000)
        J = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(J, atol=3.0E-5, rtol=1.0E-2)

if __name__ == "__main__":
    unittest.main()