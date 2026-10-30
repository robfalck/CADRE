import unittest

import numpy as np

from CADRE.solar_dymos import ExposedAreaGroup, ExposedAreaTransposeComp
from CADRE.test.test_tables import CASE, LEGACY, assert_close_to_range
from CADRE.testing_utils import assert_partials_ok, wrap

#: Every 100th node of the legacy case, a spread of sun angles.
NODES = slice(0, None, 100)


def _run(system, vals):
    p = wrap(system)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


class TestExposedAreaHelpers(unittest.TestCase):

    def test_transpose(self):
        area = np.random.default_rng(0).random((3, 7, 12))
        p = _run(ExposedAreaTransposeComp(num_nodes=3), {'exposed_area_nodes': area})
        np.testing.assert_allclose(p.get_val('exposedArea'), np.moveaxis(area, 0, -1))
        assert_partials_ok(p.check_partials(differentiator='cs'))


class TestExposedAreaGroup(unittest.TestCase):

    def _run_legacy(self):
        azimuth = LEGACY[CASE + 'azimuth'][NODES]
        return _run(ExposedAreaGroup(num_nodes=azimuth.size),
                    {'finAngle': LEGACY['finAngle'], 'azimuth': azimuth,
                     'elevation': LEGACY[CASE + 'elevation'][NODES]})

    def test_matches_legacy(self):
        p = self._run_legacy()
        assert_close_to_range(p.get_val('exposedArea'), LEGACY[CASE + 'exposedArea'][..., NODES], 5e-4)

    def test_partials(self):
        assert_partials_ok(self._run_legacy().check_partials(differentiator='cs'))


if __name__ == '__main__':
    unittest.main()
