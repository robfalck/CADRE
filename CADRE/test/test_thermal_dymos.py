import unittest

import numpy as np
from numpy.testing import assert_allclose

from om4.api import Problem
from dymos4.grid_data import GaussLobattoGrid
from dymos4.options import State, Time
from dymos4.phase.phase import Phase
from dymos4.transcriptions.picard import PicardShooting

from CADRE import thermal_dymos as th
from CADRE.thermal_dymos import ThermalTemperatureComp
from CADRE.testing_utils import assert_partials_ok, wrap

N = 4


def _legacy_f_dot(exposedArea, LOS, P_comm, cellInstd, state):
    """The legacy ThermalTemperature.f_dot, verbatim, at one time."""
    fact1 = th.q_sol*LOS
    fact2 = th.K*th.A_T*state**4

    alpha = th.alpha_c*cellInstd + th.alpha_r - th.alpha_r*cellInstd
    s_eps = np.sum(th.eps_c*cellInstd + th.eps_r - th.eps_r*cellInstd, 0)
    s_al_ea = np.sum(alpha*exposedArea, 0)*fact1

    f = np.zeros((5, ))
    for p in range(0, 12):
        if p < 4:
            f_i = 4
            m = th.m_b
            cp = th.cp_b
        else:
            f_i = (p+1) % 4
            m = th.m_f
            cp = th.cp_f
        f[f_i] += (s_al_ea[p] - s_eps[p]*fact2[f_i])/(m*cp)

    f[4] += 4.0 * P_comm / th.m_b / th.cp_b
    return f


def _inputs(rng):
    return {
        'temperature': 250. + 100. * rng.random((5, N)),
        'exposedArea': 0.01 * rng.random((7, 12, N)),
        'cellInstd': (rng.random((7, 12)) > 0.3).astype(float),
        'LOS': np.array([0.0, 0.4, 1.0, 1.0]),
        'P_comm': np.array([0.0, 2.0, 5.0, 1.0]),
    }


class TestThermalTemperatureComp(unittest.TestCase):

    def test_matches_legacy(self):
        vals = _inputs(np.random.default_rng(0))
        p = wrap(ThermalTemperatureComp(num_nodes=N))
        for name, val in vals.items():
            p.set_val(name, val)
        p.run_model()
        for i in range(N):
            expected = _legacy_f_dot(vals['exposedArea'][..., i], vals['LOS'][i],
                                     vals['P_comm'][i], vals['cellInstd'],
                                     vals['temperature'][:, i])
            assert_allclose(p.get_val('temperature_dot')[:, i], expected)

    def test_partials(self):
        vals = _inputs(np.random.default_rng(1))
        p = wrap(ThermalTemperatureComp(num_nodes=N))
        for name, val in vals.items():
            p.set_val(name, val)
        p.run_model()
        assert_partials_ok(p.check_partials(differentiator='cs'))


class TestThermalDymos(unittest.TestCase):

    def test_radiative_cooling(self):
        """In the dark with no comm power, each fin and the body cool as dT/dt = -c T**4."""
        duration = 5000.0
        T_init = np.array([300., 310., 320., 330., 290.])

        phase = Phase(
            ode=ThermalTemperatureComp(),
            transcription=PicardShooting(grid_data=GaussLobattoGrid(num_segments=10,
                                                                    nodes_per_seg=8)),
            time=Time(units='s', opt_initial=False, opt_duration=False, duration_val=duration),
            states={'temperature': State(opt_initial=False)},
        )
        prob = Problem(model=phase)
        phase.set_time_val(t_initial=0.0, t_duration=duration)
        phase.set_state_val('temperature', [T_init, T_init])
        prob.set_val('ode.P_comm', 0.0)
        prob.set_val('ode.LOS', 0.0)
        prob.run_model()

        # With every panel a cell, each panel's emissivity sum is 7 eps_c.
        c = (th.PANEL_MAP @ np.full(12, 7 * th.eps_c)) * th.K * th.A_T
        t = prob.get_val('t')
        expected = (T_init[:, np.newaxis]**-3 + 3 * c[:, np.newaxis] * t)**(-1 / 3)
        assert_allclose(prob.get_val('states_temperature'), expected, rtol=1e-8)
        # The fins (small thermal mass) cool much more than the body.
        self.assertLess(prob.get_val('final_states_temperature')[0], 0.8 * T_init[0])


if __name__ == '__main__':
    unittest.main()
