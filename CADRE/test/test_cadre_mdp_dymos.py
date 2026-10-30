"""
The full CADRE model on dymos4, checked against the legacy model's stored results.

``mdp_execute.pkl`` holds the legacy MDP's objective and constraints at its initial design (two
design points); ``data1346.pkl`` holds the legacy time histories of all six design points at an
optimized design. Each design point here is simulated (Radau, Newton on each segment) with the
same design and compared.
"""
import os
import pickle
import unittest

import numpy as np
from numpy.testing import assert_allclose

from om4.api import Problem
from om4.derivatives.finite_difference import FiniteDifference

from CADRE.cadre_mdp_dymos import (DURATION, LAT, LON, ALT, CADREMDP, launch_points, orbit_guess,
                                   quaternion_from_dcm, roll_quaternion, simulate)
from CADRE.cadre_ode import CADREODE
from CADRE.test.test_tables import LEGACY
from CADRE.testing_utils import assert_partials_ok, wrap

with open(os.path.join(os.path.dirname(__file__), 'mdp_execute.pkl'), 'rb') as f:
    MDP_EXECUTE = pickle.load(f)


class TestCADREODE(unittest.TestCase):

    def test_partials(self):
        n = 3
        pt = launch_points()[0]
        frac = np.linspace(0.0, 0.05, n)
        mee, r, v = orbit_guess(pt['r0'], pt['v0'], frac)
        p = wrap(CADREODE(num_nodes=n))
        for name, val in mee.items():
            p.set_val(name, val)
        rng = np.random.default_rng(0)
        vals = {'q': roll_quaternion(r, v, np.pi / 4), 'q_rate': 1e-4 * rng.random((4, n)),
                'q_rate2': 1e-7 * rng.random((4, n)), 't': frac * DURATION,
                'temperature': 273 + 10 * rng.random((5, n)), 'SOC': 0.5 + 0.1 * rng.random(n),
                'w_RW': rng.random((3, n)), 'Isetpt': 0.1 + 0.2 * rng.random((12, n)),
                'P_comm': 1 + rng.random(n), 'LD': pt['LD'], 'lat': LAT, 'lon': LON, 'alt': ALT,
                'finAngle': np.pi / 4, 'antAngle': 0.1, 'cellInstd': np.ones((7, 12))}
        for name, val in vals.items():
            p.set_val(name, val)
        p.run_model()
        # Not complex step: the azimuth and elevation (sun_dymos.azimuth_elevation) use
        # jnp.arctan2 and jnp.mod, which do not take complex inputs. Central differences, as the
        # reaction wheel power is strongly quadratic in the torque. No one step suits every
        # component (km positions, 1e-3 m**2 cell areas), so this catches wrong partials in the
        # assembled ODE; each component's own test checks them tightly.
        assert_partials_ok(p.check_partials(differentiator=FiniteDifference.central(step=1e-7)), tol=1e-3)


class TestLegacyInitialDesign(unittest.TestCase):
    """The legacy MDP's initial design: roll pi/4, 0.2 A per panel, 0.1 W comm, SOC 0.5."""

    def test_two_points(self):
        total = 0.0
        for i, pt in enumerate(launch_points()[:2]):
            prob = simulate(pt['LD'], pt['r0'], pt['v0'], num_segments=100, nodes_per_seg=4)
            SOC = prob.get_val('states_SOC')
            # The battery drains far below its limits at this design; the legacy con5.
            assert_allclose(SOC[0] - SOC[-1], MDP_EXECUTE[f'pt{i}_con5.val'], rtol=1e-3)
            total += prob.get_val('states_Data')[-1]
        # The data downloaded is small and sensitive to the attitude and the control grid.
        assert_allclose(-total, MDP_EXECUTE['obj.val'], rtol=0.1)


class TestLegacyOptimizedDesign(unittest.TestCase):
    """Design point 0 of the legacy optimized design (data1346.pkl)."""

    def test_point_0(self):
        P = '0:'
        frac = LEGACY['t'] / DURATION
        pt = launch_points()[0]
        mee, _, _ = orbit_guess(pt['r0'], pt['v0'], np.zeros(1))

        from CADRE.cadre_mdp_dymos import cadre_phase, set_initial_guess
        phase = cadre_phase(pt['LD'], {k: v[0] for k, v in mee.items()}, num_segments=300,
                            nodes_per_seg=4, solve_segments=True)
        prob = Problem(model=phase)
        set_initial_guess(phase, pt['r0'], pt['v0'], SOC0=float(LEGACY[P + 'iSOC'][0]))
        # The legacy O_BI is inertial to body (ADL-005).
        phase.set_control_val('q', quaternion_from_dcm(np.transpose(LEGACY[P + 'O_BI'], (1, 0, 2))).T,
                              time_vals=frac)
        phase.set_control_val('Isetpt', LEGACY[P + 'Isetpt'].T, time_vals=frac)
        phase.set_control_val('P_comm', np.ravel(LEGACY[P + 'P_comm']), time_vals=frac)
        for name in ('finAngle', 'antAngle', 'cellInstd'):
            phase.set_parameter_val(name, LEGACY[name])
        prob.run_model()

        assert_allclose(prob.get_val('states_Data')[-1], np.ravel(LEGACY[P + 'Data'])[-1], rtol=0.03)
        assert_allclose(prob.get_val('states_SOC')[-1], np.ravel(LEGACY[P + 'SOC'])[-1], atol=0.01)
        t = prob.get_val('t')
        T_legacy = np.array([np.interp(t, LEGACY['t'], row) for row in LEGACY[P + 'temperature']])
        assert_allclose(prob.get_val('states_temperature'), T_legacy, atol=1.5)


class TestCADREMDP(unittest.TestCase):

    def test_wiring(self):
        mdp = CADREMDP(npts=2, num_segments=6, nodes_per_seg=3)
        self.assertEqual(set(mdp.design_vars), {'cellInstd', 'finAngle', 'antAngle'})
        prob = Problem(model=mdp)
        mdp.set_initial_guess(simulate_states=False)
        prob.set_val('pt0.final_states_Data', 3.0)
        prob.set_val('pt1.final_states_Data', 4.0)
        prob.set_val('pt1.final_states_SOC', 0.4)
        prob.set_val('finAngle', 0.3)
        prob.run_model()
        assert_allclose(prob.get_val('mdp.obj'), [-7.0])
        assert_allclose(prob.get_val('mdp.SOC_periodic_1'), [0.5 - 0.4])
        # One shared fin angle reaches both design points.
        for i in range(2):
            assert_allclose(prob.get_val(f'pt{i}.ode.exposed_area.table.finAngle'), 0.3)


if __name__ == '__main__':
    unittest.main()
