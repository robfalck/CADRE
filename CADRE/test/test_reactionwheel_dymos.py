import unittest

import numpy as np
from numpy.testing import assert_allclose

from om4.api import Problem
from dymos4.grid_data import GaussLobattoGrid
from dymos4.options import Control, State, Time
from dymos4.phase.phase import Phase
from dymos4.transcriptions.picard import PicardShooting

from CADRE.reactionwheel_dymos import ReactionWheel
from CADRE.testing_utils import assert_partials_ok, wrap

N = 5


class TestReactionWheel(unittest.TestCase):

    def test_values_and_partials(self):
        rng = np.random.default_rng(0)
        w_B = rng.normal(size=(3, N))
        w_RW = 100 * rng.normal(size=(3, N))
        T_RW = 1e-5 * rng.normal(size=(3, N))
        p = wrap(ReactionWheel(num_nodes=N))
        p.set_val('w_B', w_B)
        p.set_val('w_RW', w_RW)
        p.set_val('T_RW', T_RW)
        p.run_model()

        rw = ReactionWheel()
        assert_allclose(p.get_val('P_RW'), rw.V * (rw.a * w_RW + rw.b * T_RW)**2 + rw.V * rw.I0)
        assert_allclose(p.get_val('alpha_RW'), -T_RW / rw.J_RW - np.cross(w_B, w_RW, axis=0))
        # P_RW is strongly curved in T_RW (2 V b**2 ~ 1.6e6), too much for a forward FD reference.
        assert_partials_ok(p.check_partials(differentiator='cs'))


class TestReactionWheelDymos(unittest.TestCase):

    def test_simple_prop(self):
        """With the body not rotating, a constant torque spins the wheel up linearly."""
        T = 1.0e-5
        duration = 10.0
        phase = Phase(
            ode=ReactionWheel(),
            transcription=PicardShooting(grid_data=GaussLobattoGrid(num_segments=1,
                                                                    nodes_per_seg=20)),
            time=Time(units='s', opt_initial=False, opt_duration=False, duration_val=duration),
            states={'w_RW': State(opt_initial=False)},
            controls={'T_RW': Control(opt=False, units='N*m', shape=(3,))},
        )
        prob = Problem(model=phase)
        phase.set_time_val(t_initial=0.0, t_duration=duration)
        phase.set_state_val('w_RW', [[0, 0, 0], [0, 0, 0]])
        phase.set_control_val('T_RW', [[T, 0, 0], [T, 0, 0]])
        prob.run_model()

        t = prob.get_val('t')
        w_RW = prob.get_val('states_w_RW')
        assert_allclose(w_RW[0], -T * t / ReactionWheel().J_RW, rtol=1e-9)
        assert_allclose(w_RW[1:], np.zeros((2, t.size)), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
