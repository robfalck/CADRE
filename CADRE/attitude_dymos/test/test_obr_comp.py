import unittest

import openmdao.api as om
import numpy as np

from CADRE.attitude_dymos.OBRComp import OBRComp


class TestOBRComp(unittest.TestCase):

    def test_obr_comp(self):
        p = om.Problem()
        p.model.add_subsystem('obr_comp', OBRComp(num_nodes=1000), promotes=['*'])

        p.setup()

        p.set_val('gamma', np.pi/2 * np.sin(np.linspace(0, 100, 1000)))

        p.run_model()

        print(p.get_val('gamma'))
        print(p.get_val('O_BR'))



if __name__ == '__main__':
    unittest.main()