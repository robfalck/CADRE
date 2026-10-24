import unittest

import numpy as np
from numpy.testing import assert_allclose

from CADRE.orbital_equations import DU_earth_m, TU_earth_s
from CADRE.orbital_equations.frame_conversions.mee_to_cart import mee_to_cart
from CADRE.orbital_equations.orbit_initial_comp import OrbitInitialComp
from CADRE.testing_utils import assert_partials_ok, wrap

ELEMENTS = {'altPerigee': 450., 'altApogee': 650., 'RAAN': 66.279, 'Inc': 82.072,
            'argPerigee': 30.0, 'trueAnomaly': 337.987}


def _legacy_rv(altPerigee, altApogee, RAAN, Inc, argPerigee, trueAnomaly):
    """The legacy Orbit_Initial.compute_rv, condensed."""
    Re = 6378.137
    mu = 398600.44

    def rot(axis, angle):
        S = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return (np.eye(3) + S * np.sin(angle) +
                (1 - np.cos(angle)) * (np.outer(axis, axis) - np.eye(3)))

    d2r = np.pi / 180.0
    r_perigee = Re + altPerigee
    r_apogee = Re + altApogee
    e = (r_apogee - r_perigee) / (r_apogee + r_perigee)
    a = (r_perigee + r_apogee) / 2
    p = a * (1 - e**2)
    nu = d2r * trueAnomaly
    rmag0 = p / (1 + e * np.cos(nu))
    r0_P = rmag0 * np.array([np.cos(nu), np.sin(nu), 0])
    v0_P = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])
    O_IP = (rot(np.array([0, 0, 1]), RAAN * d2r) @ rot(np.array([1, 0, 0]), Inc * d2r) @
            rot(np.array([0, 0, 1]), argPerigee * d2r))
    return O_IP @ r0_P, O_IP @ v0_P


class TestOrbitInitialComp(unittest.TestCase):

    def test_matches_legacy_position_and_velocity(self):
        p = wrap(OrbitInitialComp())
        for name, val in ELEMENTS.items():
            p.set_val(name, val)
        p.run_model()

        mee = [p.get_val('p0', units='DU_earth')] + [p.get_val(f'{n}0') for n in 'fghkL']
        cart = np.asarray(mee_to_cart(*mee)).ravel()
        r = cart[:3] * DU_earth_m / 1000.
        v = cart[3:] * DU_earth_m / TU_earth_s / 1000.

        r_legacy, v_legacy = _legacy_rv(**ELEMENTS)
        # The legacy model's Re and mu differ from DU_earth and MU_earth in the 7th digit.
        assert_allclose(r, r_legacy, rtol=1e-6)
        assert_allclose(v, v_legacy, rtol=1e-6)

    def test_partials(self):
        p = wrap(OrbitInitialComp())
        for name, val in ELEMENTS.items():
            p.set_val(name, val)
        p.run_model()
        assert_partials_ok(p.check_partials(), tol=1e-5)


if __name__ == '__main__':
    unittest.main()
