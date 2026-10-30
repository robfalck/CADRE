import unittest

import numpy as np
from numpy.testing import assert_allclose

from CADRE.power_dymos import (CellVoltageGroup, CellVoltagePointsComp, CellVoltageSumComp,
                               PowerTotalComp, SolarPowerComp)
from CADRE.test.test_tables import CASE, LEGACY, assert_close_to_range
from CADRE.testing_utils import assert_partials_ok, wrap

N = 4


def _run(comp, vals):
    p = wrap(comp)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


class TestPowerComps(unittest.TestCase):

    def test_solar_power(self):
        rng = np.random.default_rng(0)
        vals = {'Isetpt': 0.4 * rng.random((12, N)), 'V_sol': 2.0 + rng.random((12, N))}
        p = _run(SolarPowerComp(num_nodes=N), vals)
        assert_allclose(p.get_val('P_sol'), np.sum(vals['Isetpt'] * vals['V_sol'], axis=0))
        assert_partials_ok(p.check_partials())

    def test_total(self):
        rng = np.random.default_rng(1)
        vals = {'P_sol': 10 * rng.random(N), 'P_comm': rng.random(N),
                'P_RW': 0.1 * rng.random((3, N))}
        p = _run(PowerTotalComp(num_nodes=N), vals)
        assert_allclose(p.get_val('P_bat'), vals['P_sol'] - 5.0 * vals['P_comm'] -
                        np.sum(vals['P_RW'], axis=0) - 2.0)
        assert_partials_ok(p.check_partials())



class TestCellVoltage(unittest.TestCase):

    def test_points(self):
        rng = np.random.default_rng(2)
        vals = {'LOS': rng.random(N), 'temperature': 250 + 50 * rng.random((5, N)),
                'exposedArea': 1e-3 * rng.random((7, 12, N)), 'Isetpt': 0.3 * rng.random((12, N))}
        p = _run(CellVoltagePointsComp(num_nodes=N), vals)
        T, A, I = (p.get_val(k).reshape(7, 12, N) for k in ('T_cell', 'A_cell', 'I_cell'))
        for c in range(7):
            for panel in range(12):
                node = 4 if panel < 4 else panel % 4
                assert_allclose(T[c, panel], vals['temperature'][node])
                assert_allclose(A[c, panel], vals['LOS'] * vals['exposedArea'][c, panel])
                assert_allclose(I[c, panel], vals['Isetpt'][panel])
        assert_partials_ok(p.check_partials(differentiator='cs'))

    def test_sum(self):
        V = np.random.default_rng(3).random(7 * 12 * N)
        p = _run(CellVoltageSumComp(num_nodes=N), {'V_cell': V})
        assert_allclose(p.get_val('V_sol'), V.reshape(7, 12, N).sum(axis=0))
        assert_partials_ok(p.check_partials(differentiator='cs'))

    def _run_legacy(self, step=100):
        nodes = slice(0, None, step)
        vals = {k: LEGACY[CASE + k][..., nodes] for k in ('LOS', 'temperature', 'exposedArea', 'Isetpt')}
        return _run(CellVoltageGroup(num_nodes=vals['LOS'].size), vals), nodes

    def test_matches_legacy(self):
        p, nodes = self._run_legacy()
        assert_close_to_range(p.get_val('V_sol'), LEGACY[CASE + 'V_sol'][:, nodes], 1e-3)

    def test_partials(self):
        # Fewer nodes: the dense reference has 84 columns per node.
        assert_partials_ok(self._run_legacy(step=300)[0].check_partials(differentiator='cs'))


if __name__ == '__main__':
    unittest.main()
