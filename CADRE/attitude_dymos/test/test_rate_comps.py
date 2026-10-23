"""Tests of QuaternionAttitudeComp, AngularVelocityComp, AngularAccelerationComp and AttitudeTorqueComp."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from CADRE.attitude_dymos import (AngularAccelerationComp, AngularVelocityComp,
                                  AttitudeTorqueComp, QuaternionAttitudeComp)
from CADRE.attitude_dymos.angular_velocity_comp import _compute_angular_velocity
from CADRE.attitude_dymos.attitude_torque_comp import J
from CADRE.attitude_dymos.quaternion_attitude_comp import _dcm_from_quaternion
from CADRE.testing_utils import assert_partials_ok, wrap

N = 5
T = np.linspace(0.0, 2.0, N)
H = 1.0e-4


def _q(t):
    """
    A smooth, non-unit attitude quaternion history, (4, n).

    Not normalized on purpose: the components must not rely on it.
    """
    t = np.atleast_1d(t)
    return np.stack((np.cos(0.3 * t**2), 0.2 * np.sin(t), 0.1 + 0.05 * t, np.sin(0.4 * t)))


def _q_rate(t):
    return (_q(t + H) - _q(t - H)) / (2 * H)


def _q_rate2(t):
    return (_q(t + H) - 2 * _q(t) + _q(t - H)) / H**2


def _O(t):
    return np.asarray(_dcm_from_quaternion(_q(t)))


def _w(t):
    O = _O(t)
    Odot = (_O(t + H) - _O(t - H)) / (2 * H)
    return np.asarray(_compute_angular_velocity(O, Odot))


class TestQuaternionAttitudeComp(unittest.TestCase):

    def _prob(self):
        p = wrap(QuaternionAttitudeComp(num_nodes=N))
        p.set_val('q', _q(T))
        p.set_val('q_rate', _q_rate(T))
        p.set_val('q_rate2', _q_rate2(T))
        p.run_model()
        return p

    def test_unit_quaternion_is_a_rotation(self):
        q = _q(T) / np.linalg.norm(_q(T), axis=0)
        O = np.asarray(_dcm_from_quaternion(q))
        for i in range(N):
            assert_allclose(O[..., i] @ O[..., i].T, np.eye(3), atol=1e-14)
            assert_allclose(np.linalg.det(O[..., i]), 1.0)

    def test_values_match_time_differences(self):
        p = self._prob()
        assert_allclose(p.get_val('O_BI'), _O(T))
        assert_allclose(p.get_val('Odot_BI'), (_O(T + H) - _O(T - H)) / (2 * H),
                        rtol=1e-6, atol=1e-7)
        assert_allclose(p.get_val('Oddot_BI'), (_O(T + H) - 2 * _O(T) + _O(T - H)) / H**2,
                        rtol=1e-4, atol=1e-4)

    def test_partials(self):
        assert_partials_ok(self._prob().check_partials(), tol=1e-5)


class TestAngularVelocityComp(unittest.TestCase):

    def test_values_and_partials(self):
        O = _O(T)
        Odot = (_O(T + H) - _O(T - H)) / (2 * H)
        p = wrap(AngularVelocityComp(num_nodes=N))
        p.set_val('O_BI', O)
        p.set_val('Odot_BI', Odot)
        p.run_model()

        # O_BI takes body vectors to inertial, so Odot_BI = O_BI [w_B x].
        for i in range(N):
            W = O[..., i].T @ Odot[..., i]
            w = p.get_val('w_B')[:, i]
            assert_allclose(w, [W[2, 1], W[0, 2], W[1, 0]])
        assert_partials_ok(p.check_partials())

    def test_spin_about_body_z(self):
        """A body spinning at +omega about its z axis (shared with inertial z) has w_B = omega z."""
        omega = 0.3
        t = np.linspace(0, 2, N)
        c, s = np.cos(omega * t), np.sin(omega * t)
        z, o = np.zeros(N), np.ones(N)
        O = np.stack([np.stack([c, -s, z]), np.stack([s, c, z]), np.stack([z, z, o])])
        Odot = omega * np.stack([np.stack([-s, -c, z]), np.stack([c, -s, z]),
                                 np.stack([z, z, z])])
        p = wrap(AngularVelocityComp(num_nodes=N))
        p.set_val('O_BI', O)
        p.set_val('Odot_BI', Odot)
        p.run_model()
        assert_allclose(p.get_val('w_B'), np.stack([z, z, omega * o]), atol=1e-15)


class TestAngularAccelerationComp(unittest.TestCase):

    def test_values_and_partials(self):
        p = wrap(AngularAccelerationComp(num_nodes=N))
        p.set_val('O_BI', _O(T))
        p.set_val('Odot_BI', (_O(T + H) - _O(T - H)) / (2 * H))
        p.set_val('Oddot_BI', (_O(T + H) - 2 * _O(T) + _O(T - H)) / H**2)
        p.run_model()

        h = 1.0e-3
        assert_allclose(p.get_val('wdot_B'), (_w(T + h) - _w(T - h)) / (2 * h),
                        rtol=1e-4, atol=1e-4)
        assert_partials_ok(p.check_partials())


class TestAttitudeTorqueComp(unittest.TestCase):

    def test_values_and_partials(self):
        rng = np.random.default_rng(3)
        w = rng.normal(size=(3, N))
        wdot = rng.normal(size=(3, N))
        p = wrap(AttitudeTorqueComp(num_nodes=N))
        p.set_val('w_B', w)
        p.set_val('wdot_B', wdot)
        p.run_model()
        for i in range(N):
            expected = J @ wdot[:, i] + np.cross(w[:, i], J @ w[:, i])
            assert_allclose(p.get_val('T_req')[:, i], expected)
        assert_partials_ok(p.check_partials())


if __name__ == '__main__':
    unittest.main()
