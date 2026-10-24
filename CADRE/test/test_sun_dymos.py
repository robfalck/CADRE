import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from CADRE.sun_dymos import (SunLOSComp, SunPositionBodyComp, SunPositionECIComp,
                             SunPositionSphericalComp)
from CADRE.testing_utils import assert_partials_ok, wrap

R_E = 6378.137

#: Days from the ephemeris epoch (J2000) to the 2000 March equinox and June solstice.
EQUINOX = 79.3
SOLSTICE = 171.5


def _run(comp, vals):
    p = wrap(comp)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


class TestSunPositionECIComp(unittest.TestCase):

    def test_equinox_and_solstice(self):
        t = np.array([0.0, SOLSTICE - EQUINOX]) * 86400.
        p = _run(SunPositionECIComp(num_nodes=2), {'LD': EQUINOX, 't': t})
        r = p.get_val('r_e2s_I')
        assert_allclose(np.linalg.norm(r, axis=0), 1.0)
        # At the March equinox the Sun is along the vernal equinox direction, +x.
        assert_allclose(r[:, 0], [1.0, 0.0, 0.0], atol=0.01)
        # At the June solstice it is 90 degrees along the ecliptic, at the obliquity above +y.
        eps = np.radians(23.439)
        assert_allclose(r[:, 1], [0.0, np.cos(eps), np.sin(eps)], atol=0.01)

    def test_time_units(self):
        p = _run(SunPositionECIComp(num_nodes=3), {'LD': 0.0, 't': np.array([0, 1, 2]) * 86400.})
        p2 = _run(SunPositionECIComp(num_nodes=3), {'LD': 1.0, 't': np.array([-1, 0, 1]) * 86400.})
        assert_allclose(p.get_val('r_e2s_I'), p2.get_val('r_e2s_I'))

    def test_partials(self):
        p = _run(SunPositionECIComp(num_nodes=4),
                 {'LD': 5000.0, 't': np.linspace(0, 86400., 4)})
        assert_partials_ok(p.check_partials(), tol=1e-5)


class TestSunLOSComp(unittest.TestCase):

    def _los(self, r_b, r_s=(1.0, 0.0, 0.0)):
        r_b = np.asarray(r_b, dtype=float).T
        r_s = np.tile(np.asarray(r_s, dtype=float)[:, np.newaxis], (1, r_b.shape[1]))
        p = _run(SunLOSComp(num_nodes=r_b.shape[1]), {'r_e2b_I': r_b, 'r_e2s_I': r_s})
        return p

    def test_regions(self):
        r1, r2 = 0.85 * R_E, R_E
        d_mid = 0.5 * (r1 + r2)
        p = self._los([[7000., 0., 0.],        # sunlit side
                       [-7000., 0., 0.],       # directly behind the Earth: umbra
                       [-7000., 0., 0.5 * r1],  # behind, inside r1: umbra
                       [-7000., 1.1 * r2, 0.],  # behind, outside r2: sunlit
                       [-7000., d_mid, 0.]])    # behind, in the penumbra band
        assert_allclose(p.get_val('LOS'), [1.0, 0.0, 0.0, 1.0, 0.5])

    def test_partials(self):
        r1, r2 = 0.85 * R_E, R_E
        # Points in every region, away from the region boundaries.
        p = self._los([[7000., 100., 0.], [-7000., 0.2 * r1, 0.], [-7000., 1.2 * r2, 0.],
                       [-7000., 0.3 * r1 + 0.7 * r2, 200.]],
                      r_s=(0.9, 0.3, np.sqrt(1 - 0.9**2 - 0.3**2)))
        assert_partials_ok(p.check_partials(), tol=1e-5)
        # The derivative must stay finite on the Earth-Sun line, where the distance is zero.
        p = self._los([[-7000., 0., 0.], [7000., 0., 0.]])
        for pairs in p.check_partials().values():
            for data in pairs.values():
                self.assertTrue(np.all(np.isfinite(np.asarray(data['J']))))


class TestSunPositionBodyComp(unittest.TestCase):

    def test_values_and_partials(self):
        n = 4
        O_BI = np.stack([Rotation.from_euler('zyx', [0.3 * i, 0.2, -0.1 * i]).as_matrix()
                         for i in range(n)], axis=-1)
        rng = np.random.default_rng(0)
        r_s = rng.normal(size=(3, n))
        r_s /= np.linalg.norm(r_s, axis=0)
        p = _run(SunPositionBodyComp(num_nodes=n), {'O_BI': O_BI, 'r_e2s_I': r_s})
        for i in range(n):
            # O_BI takes body vectors into inertial, so its transpose takes them back.
            assert_allclose(O_BI[..., i] @ p.get_val('r_e2s_B')[:, i], r_s[:, i])
        assert_partials_ok(p.check_partials())


class TestSunPositionSphericalComp(unittest.TestCase):

    def test_values(self):
        s = 1 / np.sqrt(2)
        r = np.array([[1., 0., 0.], [0., 1., 0.], [-s, -s, 0.], [0., -s, s], [s, 0, -s]]).T
        p = _run(SunPositionSphericalComp(num_nodes=5), {'r_e2s_B': r})
        assert_allclose(p.get_val('azimuth'), [0., np.pi / 2, 5 * np.pi / 4, 3 * np.pi / 2, 0.],
                        atol=1e-15)
        assert_allclose(p.get_val('elevation'),
                        [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 4, 3 * np.pi / 4])

    def test_partials(self):
        r = np.array([[0.5, 0.2, 0.3], [-0.4, 0.6, -0.2], [0.1, -0.7, 0.5]]).T
        p = _run(SunPositionSphericalComp(num_nodes=3), {'r_e2s_B': r})
        assert_partials_ok(p.check_partials(), tol=1e-5)


if __name__ == '__main__':
    unittest.main()
