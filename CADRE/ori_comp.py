from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class ORIComp(ExplicitComponent):
    """
    Coordinate transformation from the inertial plane to the rolled
    (forward facing) plane.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('runit_e2b_I', np.zeros((nn, 3)), units=None,
                       desc='Unitized osition vector from earth to satellite in '
                            'Earth-centered inertial frame over time')

        self.add_input('vunit_e2b_I', np.zeros((nn, 3)), units=None,
                       desc='Unitized velocity vector from earth to satellite in '
                            'Earth-centered inertial frame over time')

        self.add_input('hunit_e2b_I', np.zeros((nn, 3)), units=None,
                       desc='Unitized orbit normal vector from earth to satellite in '
                            'Earth-centered inertial frame over time')
        # Outputs
        self.add_output('O_RI', np.zeros((nn, 3, 3)), units=None,
                        desc='Rotation matrix from rolled body-fixed frame to '
                             'Earth-centered inertial frame over time')

        cs = np.arange(nn * 3, dtype=int)

        rs = np.arange(nn * 3, dtype=int) + np.repeat(0 + 6 * np.arange(nn, dtype=int), 3)
        self.declare_partials('O_RI', 'runit_e2b_I', rows=rs, cols=cs, val=1.0)

        rs = np.arange(nn * 3, dtype=int) + np.repeat(3 + 6 * np.arange(nn, dtype=int), 3)
        self.declare_partials('O_RI', 'vunit_e2b_I', rows=rs, cols=cs, val=1.0)

        rs = np.arange(nn * 3, dtype=int) + np.repeat(6 + 6 * np.arange(nn, dtype=int), 3)
        self.declare_partials('O_RI', 'hunit_e2b_I', rows=rs, cols=cs, val=1.0)


    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        rhat = inputs['runit_e2b_I']
        vhat = inputs['vunit_e2b_I']
        hhat = inputs['hunit_e2b_I']

        outputs['O_RI'][:, 0, :] = rhat
        outputs['O_RI'][:, 1, :] = vhat
        outputs['O_RI'][:, 2, :] = hhat
