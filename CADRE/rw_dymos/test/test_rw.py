"""
Unit test vs saved data from John's CMF implementation.
"""
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.rw_dymos.rw_group import ReactionWheelGroup
from CADRE.rw_dymos.rw_motor import ReactionWheelMotorComp
from CADRE.rw_dymos.rw_power import ReactionWheelPowerComp
from CADRE.test.util import load_validation_data

#
# load saved data from John's CMF implementation.
#
n, m, h, setd = load_validation_data(idx='5')


class TestReactionWheel(unittest.TestCase):

    def test_ReactionWheelMotorComp(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', ReactionWheelMotorComp(num_nodes=1500),
                                        promotes=['*'])

        prob.setup()

        prob['T_RW'] = setd['T_RW'].T
        prob['w_B'] = setd['w_B'].T
        prob['w_RW'] = setd['w_RW'].T

        prob.run_model()

        for var in ['T_m']:

            tval = setd[var].T

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))

    def test_ReactionWheelPowerComp(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', ReactionWheelPowerComp(num_nodes=1500),
                                        promotes=['*'])

        prob.setup()

        prob['w_RW'] = setd['w_RW'].T
        prob['T_RW'] = setd['T_RW'].T

        prob.run_model()

        for var in ['P_RW']:

            tval = setd[var].T

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))

    def test_partials(self):

        prob = Problem(model = ReactionWheelGroup(num_nodes=5))

        prob.setup()

        prob['rw_motor.w_RW'] = setd['w_RW'][:, :5].T
        prob['rw_power.w_RW'] = setd['w_RW'][:, :5].T
        prob['rw_dynamics.w_RW'] = setd['w_RW'][:, :5].T
        prob['rw_motor.w_B'] = setd['w_B'][:, :5].T
        prob['rw_dynamics.w_B'] = setd['w_B'][:, :5].T
        prob['T_RW'] = setd['T_RW'][:, :5].T

        prob.run_model()

        np.set_printoptions(linewidth=100000, edgeitems=10000)
        J = prob.check_partials(method='cs', compact_print=True, step_calc='rel')
        assert_check_partials(J, atol=3.0E-4, rtol=1.0E-3)

if __name__ == "__main__":
    unittest.main()