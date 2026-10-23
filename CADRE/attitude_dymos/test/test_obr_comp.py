import unittest

import numpy as np
from numpy.testing import assert_allclose

from CADRE.attitude_dymos.obr_comp import OBRComp
from CADRE.testing_utils import assert_partials_ok, wrap


class TestOBRComp(unittest.TestCase):

    def test_obr_comp(self):
        n = 1000
        p = wrap(OBRComp(num_nodes=n))
        p.set_val('gamma', np.pi/2 * np.sin(np.linspace(0, 100, n)))
        p.run_model()

        gam = p.get_val('gamma')
        sgam = np.sin(gam)
        cgam = np.cos(gam)
        z = np.zeros(n)
        o = np.ones(n)

        O_BR = p.get_val('O_BR')
        self.assertEqual(O_BR.shape, (3, 3, n))

        assert_allclose(O_BR[0, 0], cgam)
        assert_allclose(O_BR[0, 1], -sgam)
        assert_allclose(O_BR[0, 2], z)
        assert_allclose(O_BR[1, 0], sgam)
        assert_allclose(O_BR[1, 1], cgam)
        assert_allclose(O_BR[1, 2], z)
        assert_allclose(O_BR[2, 0], z)
        assert_allclose(O_BR[2, 1], z)
        assert_allclose(O_BR[2, 2], o)

    def test_partials(self):
        p = wrap(OBRComp(num_nodes=5))
        p.set_val('gamma', np.linspace(-1, 2, 5))
        p.run_model()
        assert_partials_ok(p.check_partials())


if __name__ == '__main__':
    unittest.main()
