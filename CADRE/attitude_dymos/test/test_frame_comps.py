"""Tests of ORIComp, OBIComp, BodyVelComp and PointingComp."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from CADRE.attitude_dymos import BodyVelComp, OBIComp, ORIComp, PointingComp
from CADRE.attitude_dymos.obr_comp import _compute_O_BR
from CADRE.attitude_dymos.ori_comp import _compute_O_RI
from CADRE.testing_utils import assert_partials_ok, wrap

N = 6


def _circular_orbit(n=N):
    """Position and velocity, component-major (3, n), on an inclined circular orbit."""
    nu = np.linspace(0, 1.5 * np.pi, n)
    inc = 0.3
    r = 7000.0 * np.stack((np.cos(nu), np.cos(inc) * np.sin(nu), np.sin(inc) * np.sin(nu)))
    v = 7.5 * np.stack((-np.sin(nu), np.cos(inc) * np.cos(nu), np.sin(inc) * np.cos(nu)))
    return r, v


def _quaternion_from_dcm(O):
    """
    Return the scalar-first quaternion whose body-to-inertial matrix is O, at each node.

    The body-to-inertial matrix is the active rotation scipy builds from the quaternion.
    """
    q_xyzw = Rotation.from_matrix(np.moveaxis(O, -1, 0)).as_quat()
    return np.vstack((q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]))


class TestORIComp(unittest.TestCase):

    def test_values(self):
        r, v = _circular_orbit()
        p = wrap(ORIComp(num_nodes=N))
        p.set_val('r_e2b_I', r)
        p.set_val('v_e2b_I', v)
        p.run_model()

        O_RI = p.get_val('O_RI')
        self.assertEqual(O_RI.shape, (3, 3, N))
        v_hat = v / np.linalg.norm(v, axis=0)
        assert_allclose(O_RI[:, 2], -v_hat)
        # On a circular orbit r is perpendicular to v, so the rolled frame is orthonormal.
        for i in range(N):
            assert_allclose(O_RI[..., i] @ O_RI[..., i].T, np.eye(3), atol=1e-12)

    def test_partials(self):
        r, v = _circular_orbit()
        p = wrap(ORIComp(num_nodes=N))
        p.set_val('r_e2b_I', r)
        p.set_val('v_e2b_I', v)
        p.run_model()
        assert_partials_ok(p.check_partials(), tol=1e-5)


class TestOBIComp(unittest.TestCase):

    def test_values_and_partials(self):
        rng = np.random.default_rng(0)
        O_BR = rng.normal(size=(3, 3, N))
        O_RI = rng.normal(size=(3, 3, N))
        p = wrap(OBIComp(num_nodes=N))
        p.set_val('O_BR', O_BR)
        p.set_val('O_RI', O_RI)
        p.run_model()
        for i in range(N):
            assert_allclose(p.get_val('O_BI')[..., i], O_RI[..., i] @ O_BR[..., i])
        assert_partials_ok(p.check_partials())


class TestBodyVelComp(unittest.TestCase):

    def test_values_and_partials(self):
        rng = np.random.default_rng(1)
        r, v = _circular_orbit()
        O_BI = rng.normal(size=(3, 3, N))
        p = wrap(BodyVelComp(num_nodes=N))
        p.set_val('r_e2b_I', r)
        p.set_val('v_e2b_I', v)
        p.set_val('w_B', rng.normal(size=(3, N)))
        p.set_val('O_BI', O_BI)
        p.run_model()
        for i in range(N):
            assert_allclose(p.get_val('v_e2b_B')[:, i], O_BI[..., i].T @ v[:, i])
        assert_partials_ok(p.check_partials())


class TestPointingComp(unittest.TestCase):

    def _run(self, q, O_RI):
        from CADRE.attitude_dymos.quaternion_attitude_comp import _dcm_from_quaternion
        p = wrap(PointingComp(num_nodes=N))
        p.set_val('q', q)
        p.set_val('O_BI', np.asarray(_dcm_from_quaternion(q)))
        p.set_val('O_RI', O_RI)
        p.run_model()
        return p

    def test_any_roll_about_the_pointing_axis_satisfies_it(self):
        r, v = _circular_orbit()
        O_RI = np.asarray(_compute_O_RI(r, v))
        gamma = np.linspace(-2.0, 2.0, N)
        O_BI = np.einsum('ijn,jkn->ikn', O_RI, np.asarray(_compute_O_BR(gamma)))
        q = _quaternion_from_dcm(O_BI)

        p = self._run(q, O_RI)
        assert_allclose(p.get_val('pointing_err'), np.zeros((2, N)), atol=1e-12)
        assert_allclose(p.get_val('q_norm_err'), np.zeros(N), atol=1e-12)

    def test_mispointing_is_detected(self):
        r, v = _circular_orbit()
        O_RI = np.asarray(_compute_O_RI(r, v))
        tilt = Rotation.from_euler('x', 0.1).as_matrix()
        O_BI = np.einsum('ijn,jk->ikn', O_RI, tilt)
        p = self._run(_quaternion_from_dcm(O_BI), O_RI)
        assert_allclose(np.abs(p.get_val('pointing_err')[1]), np.sin(0.1) * np.ones(N))

    def test_partials(self):
        rng = np.random.default_rng(2)
        p = wrap(PointingComp(num_nodes=N))
        p.set_val('q', rng.normal(size=(4, N)))
        p.set_val('O_BI', rng.normal(size=(3, 3, N)))
        p.set_val('O_RI', rng.normal(size=(3, 3, N)))
        p.run_model()
        assert_partials_ok(p.check_partials())


if __name__ == '__main__':
    unittest.main()
