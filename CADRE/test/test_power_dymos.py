import unittest

import numpy as np
from numpy.testing import assert_allclose

from CADRE.power_dymos import PowerTotalComp, SolarPowerComp
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


if __name__ == '__main__':
    unittest.main()
