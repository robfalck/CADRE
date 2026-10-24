import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from om4.api import Problem
from dymos4.grid_data import GaussLobattoGrid
from dymos4.options import State, Time
from dymos4.phase.phase import Phase
from dymos4.transcriptions.picard import PicardShooting

from CADRE import battery_dymos as bat
from CADRE.battery_dymos import BatteryPowerComp, BatterySOCComp, battery_path_constraints
from CADRE.testing_utils import assert_partials_ok, wrap

N = 4


def _voltage(SOC, T):
    voc = 3 + np.expm1(SOC) / (np.e - 1)
    return bat.IR * voc * (2.0 - np.exp(bat.alpha * (T - bat.T0) / bat.T0))


def _inputs():
    return {
        'SOC': np.array([0.2, 0.5, 0.8, 1.0]),
        'P_bat': np.array([-5.0, 0.0, 3.0, 10.0]),
        'temperature': np.vstack((np.full((4, N), 280.), [270., 293., 310., 330.])),
    }


def _run(comp, vals):
    p = wrap(comp)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


class TestBatteryComps(unittest.TestCase):

    def test_soc_rate(self):
        vals = _inputs()
        p = _run(BatterySOCComp(num_nodes=N), vals)
        I = vals['P_bat'] / _voltage(vals['SOC'], vals['temperature'][4])  # noqa: E741
        assert_allclose(p.get_val('SOC_dot'), -bat.sigma / 24 * vals['SOC'] + bat.eta / bat.Cp * I)
        assert_partials_ok(p.check_partials(differentiator='cs'))

    def test_current(self):
        vals = _inputs()
        p = _run(BatteryPowerComp(num_nodes=N), vals)
        assert_allclose(p.get_val('I_bat'),
                        vals['P_bat'] / _voltage(vals['SOC'], vals['temperature'][4]))
        assert_partials_ok(p.check_partials(differentiator='cs'))

    def test_path_constraints(self):
        I_con, SOC_con = battery_path_constraints()
        self.assertEqual((I_con.name, I_con.lower, I_con.upper), ('ode.I_bat', -10.0, 5.0))
        self.assertEqual((SOC_con.name, SOC_con.lower, SOC_con.upper), ('states_SOC', 0.2, 1.0))


class TestBatteryDymos(unittest.TestCase):

    def test_discharge(self):
        """A constant draw at the reference temperature, against scipy's integrator."""
        P = -20.0
        duration = 3600.0
        phase = Phase(
            ode=BatterySOCComp(),
            transcription=PicardShooting(grid_data=GaussLobattoGrid(num_segments=4,
                                                                    nodes_per_seg=8)),
            time=Time(units='s', opt_initial=False, opt_duration=False, duration_val=duration),
            states={'SOC': State(opt_initial=False)},
        )
        prob = Problem(model=phase)
        phase.set_time_val(t_initial=0.0, t_duration=duration)
        phase.set_state_val('SOC', 0.9)
        prob.set_val('ode.P_bat', P)
        prob.run_model()

        def rate(t, SOC):
            return -bat.sigma / 24 * SOC + bat.eta / bat.Cp * P / _voltage(SOC, bat.T0)

        t = prob.get_val('t')
        # Segment boundaries repeat in t, which t_eval rejects, so use the dense output.
        sol = solve_ivp(rate, (0, duration), [0.9], dense_output=True, rtol=1e-12, atol=1e-12)
        assert_allclose(prob.get_val('states_SOC'), sol.sol(t)[0], rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
