import unittest

import openmdao.api as om
import numpy as np

from openmdao.utils.assert_utils import assert_near_equal

from CADRE.attitude_dymos.obr_comp import OBRComp


class TestOBRComp(unittest.TestCase):

    def test_obr_comp(self):
        p = om.Problem()
        p.model.add_subsystem('obr_comp', OBRComp(num_nodes=1000), promotes=['*'])

        p.setup()

        p.set_val('gamma', np.pi/2 * np.sin(np.linspace(0, 100, 1000)))

        p.run_model()

        gam = p.get_val('gamma')
        sgam = np.sin(gam)
        cgam = np.cos(gam)

        O_BR = p.get_val('O_BR')

        z = np.zeros(1000)
        o = np.ones(1000)

        assert_near_equal(O_BR[:, 0, 0], cgam)
        assert_near_equal(O_BR[:, 0, 1], sgam)
        assert_near_equal(O_BR[:, 0, 2], z)
        assert_near_equal(O_BR[:, 1, 0], -sgam)
        assert_near_equal(O_BR[:, 1, 1], cgam)
        assert_near_equal(O_BR[:, 1, 2], z)
        assert_near_equal(O_BR[:, 2, 0], z)
        assert_near_equal(O_BR[:, 2, 1], z)
        assert_near_equal(O_BR[:, 2, 2], o)





if __name__ == '__main__':
    unittest.main()