import unittest

import numpy as np
from numpy.testing import assert_allclose

from CADRE.orbital_equations.frame_conversions import (CartesianToMEE, KepToMEE, MEEToCart,
                                                       MEEToKep, RComp, StateMuxComp)
from CADRE.orbital_equations.frame_conversions.mee_to_cart import mee_to_cart
from CADRE.testing_utils import assert_partials_ok, wrap

N = 5
MEE_NAMES = ('p', 'f', 'g', 'h', 'k', 'L')
CART_NAMES = ('x', 'y', 'z', 'vx', 'vy', 'vz')
KEP_NAMES = ('sma', 'ecc', 'inc', 'aop', 'raan', 'ta')

#: An eccentric, inclined orbit sampled at N true longitudes.
MEE = {'p': np.full(N, 1.3), 'f': np.full(N, 0.1), 'g': np.full(N, -0.05),
       'h': np.full(N, 0.2), 'k': np.full(N, 0.1), 'L': np.linspace(0.3, 5.0, N)}


def _run(comp, vals):
    p = wrap(comp)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


class TestMEEToCart(unittest.TestCase):

    def test_circular_equatorial(self):
        L = np.linspace(0, 2 * np.pi, N)
        x, y, z, vx, vy, vz = (np.asarray(a) for a in
                               mee_to_cart(np.ones(N), 0 * L, 0 * L, 0 * L, 0 * L, L))
        assert_allclose(x, np.cos(L), atol=1e-15)
        assert_allclose(y, np.sin(L), atol=1e-15)
        assert_allclose(z, 0 * L, atol=1e-15)
        assert_allclose(vx, -np.sin(L), atol=1e-15)
        assert_allclose(vy, np.cos(L), atol=1e-15)
        assert_allclose(vz, 0 * L, atol=1e-15)

    def test_component_and_partials(self):
        p = _run(MEEToCart(num_nodes=N), MEE)
        expected = mee_to_cart(*(MEE[n] for n in MEE_NAMES))
        for name, val in zip(CART_NAMES, expected):
            assert_allclose(p.get_val(name), val)
        assert_partials_ok(p.check_partials(), tol=1e-5)


class TestCartesianToMEE(unittest.TestCase):

    def test_inverts_mee_to_cart(self):
        cart = mee_to_cart(*(MEE[n] for n in MEE_NAMES))
        p = _run(CartesianToMEE(num_nodes=N), dict(zip(CART_NAMES, cart)))
        for name in MEE_NAMES[:-1]:
            assert_allclose(p.get_val(name), MEE[name], atol=1e-12)
        assert_allclose(np.mod(p.get_val('L'), 2 * np.pi), MEE['L'], atol=1e-12)
        assert_partials_ok(p.check_partials(), tol=1e-5)


class TestKeplerianConversions(unittest.TestCase):

    def test_round_trip_and_partials(self):
        kep = {'sma': np.full(N, 1.5), 'ecc': np.full(N, 0.2), 'inc': np.full(N, 0.4),
               'aop': np.full(N, 0.7), 'raan': np.full(N, 1.1), 'ta': np.linspace(0.1, 2.5, N)}
        p = _run(KepToMEE(num_nodes=N), kep)
        mee = {name: p.get_val(name) for name in MEE_NAMES}
        assert_partials_ok(p.check_partials(), tol=1e-5)

        p2 = _run(MEEToKep(num_nodes=N), mee)
        for name in KEP_NAMES:
            assert_allclose(p2.get_val(name), kep[name], atol=1e-12)
        assert_partials_ok(p2.check_partials(), tol=1e-5)

    def test_mee_to_cart_agrees_with_keplerian_radius(self):
        kep_p = 1.5 * (1 - 0.2**2)
        p = _run(KepToMEE(num_nodes=N), {'sma': np.full(N, 1.5), 'ecc': np.full(N, 0.2),
                                         'inc': np.full(N, 0.4), 'aop': np.full(N, 0.7),
                                         'raan': np.full(N, 1.1), 'ta': np.linspace(0.1, 2.5, N)})
        cart = mee_to_cart(*(p.get_val(n) for n in MEE_NAMES))
        r = np.sqrt(sum(np.asarray(c)**2 for c in cart[:3]))
        assert_allclose(r, kep_p / (1 + 0.2 * np.cos(np.linspace(0.1, 2.5, N))))


class TestRComp(unittest.TestCase):

    def test_values_and_partials(self):
        p = _run(RComp(num_nodes=N), {n: MEE[n] for n in ('p', 'f', 'g', 'L')})
        cart = mee_to_cart(*(MEE[n] for n in MEE_NAMES))
        assert_allclose(p.get_val('r'), np.sqrt(sum(np.asarray(c)**2 for c in cart[:3])))
        assert_partials_ok(p.check_partials(), tol=1e-5)


class TestStateMuxComp(unittest.TestCase):

    def test_values_and_partials(self):
        rng = np.random.default_rng(0)
        vals = {name: rng.normal(size=N) for name in CART_NAMES}
        p = _run(StateMuxComp(num_nodes=N), vals)
        assert_allclose(p.get_val('r_e2b_I'), np.stack([vals[n] for n in ('x', 'y', 'z')]))
        assert_allclose(p.get_val('v_e2b_I'), np.stack([vals[n] for n in ('vx', 'vy', 'vz')]))
        assert_partials_ok(p.check_partials(), tol=1e-5)

    def test_converts_units(self):
        from om4.api import Group, Problem
        from CADRE.orbital_equations import DU_earth_m, TU_earth_s
        model = Group(subsystems={
            'mee': MEEToCart(num_nodes=N, promotes_inputs=['*'], promotes_outputs=['*']),
            'mux': StateMuxComp(num_nodes=N, promotes_inputs=['*'], promotes_outputs=['*']),
        })
        p = Problem(model=model)
        for name, val in MEE.items():
            p.set_val(name, val)
        p.run_model()
        cart = mee_to_cart(*(MEE[n] for n in MEE_NAMES))
        assert_allclose(p.get_val('r_e2b_I')[0], np.asarray(cart[0]) * DU_earth_m / 1000.)
        assert_allclose(p.get_val('v_e2b_I')[2],
                        np.asarray(cart[5]) * DU_earth_m / TU_earth_s / 1000.)


if __name__ == '__main__':
    unittest.main()
