from __future__ import print_function, division, absolute_import

import os

import numpy as np

from openmdao.api import ExplicitComponent

from MBI import MBI

from ..kinematics import fixangles

class CommGainPatternComp(ExplicitComponent):
    """
    Determines transmitter gain based on an external az-el map.
    """
    #
    # def __init__(self, num_nodes, rawG_file=None):
    #     super(CommGainPatternComp, self).__init__(num_nodes=num_nodes)
    #
    #     if not rawG_file:
    #         fpath = os.path.dirname(os.path.realpath(__file__))
    #         rawG_file = fpath + '/data/Comm/Gain.txt'
    #
    #     rawGdata = np.genfromtxt(rawG_file)
    #     rawG = (10 ** (rawGdata / 10.0)).reshape((361, 361), order='F')
    #
    #     pi = np.pi
    #     az = np.linspace(0, 2 * pi, 361)
    #     el = np.linspace(0, 2 * pi, 361)
    #
    #     self.MBI = MBI(rawG, [az, el], [15, 15], [4, 4])
    #     self.x = np.zeros((self.n, 2), order='F')

    def initialize(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

        rawG_file = os.path.join(parent_path, 'data/Comm/Gain.txt')

        self.options.declare('num_nodes', types=(int,))
        self.options.declare('rawG_file', types=(str,), default=rawG_file)

    def setup(self):

        nn = self.options['num_nodes']

        rawGdata = np.genfromtxt(self.options['rawG_file'])
        rawG = (10 ** (rawGdata / 10.0)).reshape((361, 361), order='F')

        pi = np.pi
        az = np.linspace(0, 2 * pi, 361)
        el = np.linspace(0, 2 * pi, 361)

        self.MBI = MBI(rawG, [az, el], [15, 15], [4, 4])
        self.MBI.seterr('raise')

        self.x = np.zeros((nn, 2), order='F')

        # Inputs
        self.add_input('azimuthGS', np.zeros(nn), units='rad',
                       desc='Azimuth angle from satellite to ground station in '
                            'Earth-fixed frame over time')

        self.add_input('elevationGS', np.zeros(nn), units='rad',
                       desc='Elevation angle from satellite to ground station '
                            'in Earth-fixed frame over time')

        # Outputs
        self.add_output('gain', np.zeros(nn), units=None,
                        desc='Transmitter gain over time')

        row_col = np.arange(nn)

        self.declare_partials('gain', 'elevationGS', rows=row_col, cols=row_col)
        self.declare_partials('gain', 'azimuthGS', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        result = fixangles(self.options['num_nodes'], inputs['azimuthGS'], inputs['elevationGS'])
        self.x[:, 0] = result[0]
        self.x[:, 1] = result[1]
        outputs['gain'] = self.MBI.evaluate(self.x)[:, 0]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        partials['gain', 'azimuthGS'] = self.MBI.evaluate(self.x, 1)[:, 0]
        partials['gain', 'elevationGS'] = self.MBI.evaluate(self.x, 2)[:, 0]
