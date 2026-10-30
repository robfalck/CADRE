import unittest

import numpy as np
from numpy.testing import assert_allclose

from CADRE.orbital_equations.EOMs import TwoBodyDynamicsComp, TwoBodyEOM
from CADRE.orbital_equations.EOMs.perturbations.perturbations import Perturbations
from CADRE.orbital_equations.EOMs.perturbations.spherical_harmonics_comp import J2, SphericalHarmonicsComp
from CADRE.orbital_equations.frame_conversions import RComp
from CADRE.orbital_equations.frame_conversions.mee_to_cart import mee_to_cart
from CADRE.orbital_equations.EOMs.perturbations.total_perturbation import TotalPerturbation
from CADRE.testing_utils import assert_partials_ok, wrap

N = 5
MEE = {'p': np.full(N, 1.3), 'f': np.full(N, 0.1), 'g': np.full(N, -0.05),
       'h': np.full(N, 0.2), 'k': np.full(N, 0.1), 'L': np.linspace(0.3, 5.0, N)}


def _run(comp, vals):
    p = wrap(comp)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


class TestTwoBodyDynamicsComp(unittest.TestCase):

    def test_unperturbed_circular(self):
        L = np.linspace(0, 6, N)
        p = _run(TwoBodyDynamicsComp(num_nodes=N),
                 {'p': np.full(N, 4.0), 'f': 0 * L, 'g': 0 * L, 'h': 0 * L, 'k': 0 * L, 'L': L})
        # Only the true longitude moves, at the mean motion of a radius-4 circular orbit.
        assert_allclose(p.get_val('L_dot'), np.full(N, 4.0**-1.5))
        for name in ('p_dot', 'f_dot', 'g_dot', 'h_dot', 'k_dot'):
            assert_allclose(p.get_val(name), np.zeros(N), atol=1e-15)

    def test_tags_and_units(self):
        comp = TwoBodyDynamicsComp(num_nodes=N)
        self.assertIn('dymos.state_rate_source:p', comp.outputs['p_dot'].tags)
        self.assertIn('dymos.state_units:DU_earth', comp.outputs['p_dot'].tags)
        self.assertIn('dymos.state_target:L', comp.inputs['L'].tags)
        self.assertEqual(comp.outputs['p_dot'].units, 'DU_earth/TU_earth')

    def test_partials_with_perturbations(self):
        rng = np.random.default_rng(0)
        vals = dict(MEE)
        for name in ('a_r', 'a_s', 'a_w'):
            vals[name] = 1.0e-2 * rng.normal(size=N)
        p = _run(TwoBodyDynamicsComp(num_nodes=N), vals)
        assert_partials_ok(p.check_partials(), tol=1e-5)


class TestPerturbationComps(unittest.TestCase):

    def test_spherical_harmonics_matches_cartesian_j2(self):
        """The radial, transverse and normal components of the Cartesian J2 acceleration."""
        p = _run(RComp(num_nodes=N), {k: MEE[k] for k in ('p', 'f', 'g', 'L')})
        r = p.get_val('r')
        p = _run(SphericalHarmonicsComp(num_nodes=N), {'h': MEE['h'], 'k': MEE['k'], 'L': MEE['L'], 'r': r})
        a_rsw = np.vstack([p.get_val(f'J2_{c}') for c in 'rsw'])

        # The legacy model's Cartesian J2 acceleration, with mu = Re = 1.
        cart = np.asarray(mee_to_cart(*(MEE[k] for k in ('p', 'f', 'g', 'h', 'k', 'L'))))
        pos, vel = cart[:3], cart[3:]
        rr = np.sqrt(np.sum(pos**2, axis=0))
        z2 = (pos[2] / rr)**2
        a = -1.5 * J2 * pos / rr**5 * (1 - 5 * z2)
        a[2] -= 3.0 * J2 * pos[2] / rr**5
        r_hat = pos / rr
        w_hat = np.cross(pos, vel, axis=0)
        w_hat /= np.sqrt(np.sum(w_hat**2, axis=0))
        s_hat = np.cross(w_hat, r_hat, axis=0)
        expected = np.vstack([np.sum(a * e, axis=0) for e in (r_hat, s_hat, w_hat)])
        assert_allclose(a_rsw, expected, rtol=1e-10, atol=1e-14)

    def test_spherical_harmonics_partials(self):
        p = _run(SphericalHarmonicsComp(num_nodes=N),
                 {'h': MEE['h'], 'k': MEE['k'], 'L': MEE['L'], 'r': np.linspace(1.1, 1.5, N)})
        assert_partials_ok(p.check_partials())

    def test_total_perturbation(self):
        rng = np.random.default_rng(1)
        comp = TotalPerturbation(num_nodes=N)
        vals = {name: rng.normal(size=N) for name in comp.inputs}
        p = _run(comp, vals)
        assert_allclose(p.get_val('a_r'), vals['T_r'] + vals['J2_r'] + vals['D_r'] + vals['sb_r'])
        assert_allclose(p.get_val('a_s'), vals['T_s'] + vals['J2_s'] + vals['D_s'] + vals['sb_s'])
        assert_allclose(p.get_val('a_w'), vals['T_w'] + vals['J2_w'] + vals['sb_w'])
        assert_partials_ok(p.check_partials())

    def test_perturbations_group(self):
        p = _run(Perturbations(num_nodes=N), MEE)
        # With no thrust, drag or third body, the total perturbation is J2 alone.
        assert_allclose(p.get_val('a_r'), p.get_val('J2_r'))
        assert_partials_ok(p.check_partials())


class TestTwoBodyEOM(unittest.TestCase):

    def test_rebuilds_at_new_num_nodes(self):
        eom = TwoBodyEOM(num_nodes=3)
        rebuilt = type(eom).model_validate(eom.model_dump() | {'num_nodes': N})
        self.assertEqual(rebuilt.subsystems['spacecraft_dynamics'].inputs['p'].shape, (N,))

    def test_run_and_partials(self):
        p = _run(TwoBodyEOM(num_nodes=N), MEE)
        self.assertTrue(np.all(np.isfinite(p.get_val('L_dot'))))
        assert_partials_ok(p.check_partials(), tol=1e-5)


if __name__ == '__main__':
    unittest.main()
