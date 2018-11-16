"""
Solar discipline for CADRE
"""

import os
from six.moves import range
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent

from CADRE.kinematics import fixangles
from MBI import MBI


class Solar_ExposedArea(ExplicitComponent):
    """
    Exposed area calculation for a given solar cell

    p: panel ID [0,11]
    c: cell ID [0,6]
    a: fin angle [0,90]
    z: azimuth [0,360]
    e: elevation [0,180]
    LOS: line of sight with the sun [0,1]
    """

    def __init__(self, n, raw1_file=None, raw2_file=None):
        super(Solar_ExposedArea, self).__init__()

        self.n = n
        self.raw1_file = raw1_file
        self.raw2_file = raw2_file

    def setup(self):
        n = self.n
        raw1_file = self.raw1_file
        raw2_file = self.raw2_file

        fpath = os.path.dirname(os.path.realpath(__file__))
        if not raw1_file:
            raw1_file = fpath + '/data/Solar/Area10.txt'
        if not raw2_file:
            raw2_file = fpath + '/data/Solar/Area_all.txt'

        raw1 = np.genfromtxt(raw1_file)
        raw2 = np.loadtxt(raw2_file)

        nc = self.nc = 7
        self.np = 12

        self.na = 10
        self.nz = 73
        self.ne = 37
        angle = np.zeros(self.na)
        azimuth = np.zeros(self.nz)
        elevation = np.zeros(self.ne)

        index = 0
        for i in range(self.na):
            angle[i] = raw1[index]
            index += 1
        for i in range(self.nz):
            azimuth[i] = raw1[index]
            index += 1

        index -= 1
        azimuth[self.nz - 1] = 2.0 * np.pi
        for i in range(self.ne):
            elevation[i] = raw1[index]
            index += 1

        angle[0] = 0.0
        angle[-1] = np.pi / 2.0
        azimuth[0] = 0.0
        azimuth[-1] = 2 * np.pi
        elevation[0] = 0.0
        elevation[-1] = np.pi

        counter = 0
        data = np.zeros((self.na, self.nz, self.ne, self.np * self.nc))
        flat_size = self.na * self.nz * self.ne
        for p in range(self.np):
            for c in range(nc):
                data[:, :, :, counter] = \
                    raw2[nc * p + c][119:119 + flat_size].reshape((self.na,
                                                                   self.nz,
                                                                   self.ne))
                counter += 1

        self.MBI = MBI(data, [angle, azimuth, elevation],
                             [4, 10, 8],
                             [4, 4, 4])

        self.MBI.seterr('raise')

        self.x = np.zeros((self.n, 3))
        self.Jfin = None
        self.Jaz = None
        self.Jel = None

        # Inputs
        self.add_input('finAngle', 0.0, units='rad',
                       desc='Fin angle of solar panel')

        self.add_input('azimuth', np.zeros((n, )), units='rad',
                       desc='Azimuth angle of the sun in the body-fixed frame over time')

        self.add_input('elevation', np.zeros((n, )), units='rad',
                       desc='Elevation angle of the sun in the body-fixed frame over time')

        # Outputs
        self.add_output('exposedArea', np.zeros((n, self.nc, self.np)),
                        desc='Exposed area to sun for each solar cell over time',
                        units='m**2', lower=-5e-3, upper=1.834e-1)

        self.declare_partials('exposedArea', 'finAngle')

        nn = self.nc * self.np
        rows = np.tile(np.arange(nn), n) + np.repeat(nn*np.arange(n), nn)
        cols = np.tile(np.repeat(0, nn), n) + np.repeat(np.arange(n), nn)

        self.declare_partials('exposedArea', 'azimuth', rows=rows, cols=cols)
        self.declare_partials('exposedArea', 'elevation', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        self.setx(inputs)
        P = self.MBI.evaluate(self.x)
        outputs['exposedArea'] = P.reshape(self.n, self.nc, self.np, order='F')

    def setx(self, inputs):
        """
        Sets our state array
        """
        result = fixangles(self.n, inputs['azimuth'], inputs['elevation'])
        self.x[:, 0] = inputs['finAngle']
        self.x[:, 1] = result[0]
        self.x[:, 2] = result[1]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        Jfin = self.MBI.evaluate(self.x, 1).reshape(self.n, self.nc, self.np, order='F')
        Jaz = self.MBI.evaluate(self.x, 2).reshape(self.n, self.nc, self.np, order='F')
        Jel = self.MBI.evaluate(self.x, 3).reshape(self.n, self.nc, self.np, order='F')

        partials['exposedArea', 'finAngle'] = Jfin.flatten()
        partials['exposedArea', 'azimuth'] = Jaz.flatten()
        partials['exposedArea', 'elevation'] = Jel.flatten()

