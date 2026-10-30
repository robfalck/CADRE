import os
import pickle
import unittest

import numpy as np
from scipy.interpolate import NdBSpline, make_interp_spline

from CADRE.kinematics import fixangles
from CADRE.tables import (antenna_gain_fit, antenna_gain_table, cell_voltage_fit, cell_voltage_table,
                          exposed_area_fit, exposed_area_table, raw_antenna_gain_table,
                          raw_cell_voltage_table, raw_exposed_area_table)

#: Inputs and outputs of the legacy model, which evaluated its tables with MBI.
with open(os.path.join(os.path.dirname(__file__), 'data1346.pkl'), 'rb') as f:
    LEGACY = pickle.load(f)

CASE = '0:'
N = LEGACY[CASE + 'azimuth'].size


def cubic(grid, values):
    """Tensor-product not-a-knot cubic interpolation, as om4's StructuredMetaModelComp 'cubic'."""
    axes = list(grid.values())
    coef = values.reshape(*[x.size for x in axes], -1)
    knots = []
    for axis, x in enumerate(axes):
        spline = make_interp_spline(x, np.moveaxis(coef, axis, 0), k=3, axis=0)
        coef = np.moveaxis(spline.c, 0, axis)
        knots.append(spline.t)
    spline = NdBSpline(tuple(knots), coef, 3)
    return lambda points: spline(points).reshape(len(points), *values.shape[len(axes):])


def assert_close_to_range(actual, desired, tol):
    """Assert the largest difference is within ``tol`` of the range of ``desired``."""
    err = np.max(np.abs(actual - desired)) / np.ptp(desired)
    assert err <= tol, f'max error {err:.3e} of the range exceeds {tol:.1e}'


def solar_points():
    return np.column_stack([np.full(N, LEGACY['finAngle'][0]), LEGACY[CASE + 'azimuth'],
                            LEGACY[CASE + 'elevation']])


def legacy_exposed_area(area):
    """``(N, 7, 12)`` to the legacy ``(7, 12, N)``."""
    return np.moveaxis(area, 0, -1)


def voltage_points():
    """One point per node, cell and panel, built as ``Power_CellVoltage.setx`` did."""
    T, LOS, area, I = (LEGACY[CASE + k] for k in ('temperature', 'LOS', 'exposedArea', 'Isetpt'))
    T_panel = np.array([T[4 if p < 4 else p % 4] for p in range(12)])
    return np.column_stack([np.broadcast_to(T_panel, (7, 12, N)).ravel(), (LOS * area).ravel(),
                            np.broadcast_to(I, (7, 12, N)).ravel()])


def legacy_panel_voltage(V):
    """Sum the cell voltages of each panel, ``(7 * 12 * N,)`` to ``(12, N)``."""
    return V.reshape(7, 12, N).sum(axis=0)


def gain_points():
    return np.column_stack(fixangles(N, LEGACY[CASE + 'azimuthGS'], LEGACY[CASE + 'elevationGS']))


class TestMBIFit(unittest.TestCase):
    """The SciPy fits reproduce the legacy MBI outputs, to MBI's conjugate-gradient tolerance."""

    def test_exposed_area(self):
        assert_close_to_range(legacy_exposed_area(exposed_area_fit()(solar_points())),
                              LEGACY[CASE + 'exposedArea'], 5e-4)

    def test_cell_voltage(self):
        assert_close_to_range(legacy_panel_voltage(cell_voltage_fit()(voltage_points())),
                              LEGACY[CASE + 'V_sol'], 2e-4)

    def test_antenna_gain(self):
        assert_close_to_range(antenna_gain_fit()(gain_points()), LEGACY[CASE + 'gain'], 5e-4)

    def test_raw_tables_differ_from_the_fits(self):
        # Why the tables are smoothed: interpolating the raw table misses the legacy values.
        grid, area = raw_exposed_area_table()
        err = np.max(np.abs(legacy_exposed_area(cubic(grid, area)(solar_points())) -
                            LEGACY[CASE + 'exposedArea']))
        self.assertGreater(err / np.ptp(LEGACY[CASE + 'exposedArea']), 0.1)


class TestTables(unittest.TestCase):
    """Cubic interpolation of the sampled tables reproduces the fits and the legacy outputs."""

    cases = {
        'exposed_area': (exposed_area_table, exposed_area_fit, raw_exposed_area_table, 2e-4),
        'cell_voltage': (cell_voltage_table, cell_voltage_fit, raw_cell_voltage_table, 6e-4),
        'antenna_gain': (antenna_gain_table, antenna_gain_fit, raw_antenna_gain_table, 2e-4),
    }

    def test_grids(self):
        for name, (table, _, raw, _) in self.cases.items():
            with self.subTest(name):
                grid, values = table()
                raw_grid, raw_values = raw()
                self.assertEqual(list(grid), list(raw_grid))
                self.assertEqual(values.shape, tuple(x.size for x in grid.values()) +
                                 raw_values.shape[len(raw_grid):])
                for key, x in grid.items():
                    self.assertTrue(np.all(np.diff(x) > 0.0))
                    np.testing.assert_allclose(x[[0, -1]], raw_grid[key][[0, -1]], rtol=1e-12, atol=1e-15)
                self.assertFalse(values.flags.writeable)
                self.assertIs(table()[1], values)

    def test_interpolation_matches_the_fit(self):
        rng = np.random.default_rng(0)
        for name, (table, fit, _, tol) in self.cases.items():
            with self.subTest(name):
                grid, values = table()
                points = np.column_stack([rng.uniform(x[0], x[-1], 5000) for x in grid.values()])
                assert_close_to_range(cubic(grid, values)(points), fit()(points), tol)

    def test_exposed_area_matches_legacy(self):
        assert_close_to_range(legacy_exposed_area(cubic(*exposed_area_table())(solar_points())),
                              LEGACY[CASE + 'exposedArea'], 5e-4)

    def test_cell_voltage_matches_legacy(self):
        assert_close_to_range(legacy_panel_voltage(cubic(*cell_voltage_table())(voltage_points())),
                              LEGACY[CASE + 'V_sol'], 5e-4)

    def test_antenna_gain_matches_legacy(self):
        assert_close_to_range(cubic(*antenna_gain_table())(gain_points()), LEGACY[CASE + 'gain'], 5e-4)


if __name__ == '__main__':
    unittest.main()
