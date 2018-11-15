"""
Solar discipline for CADRE
"""
from __future__ import print_function, division, absolute_import
from six.moves import range
import os

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent

from CADRE.kinematics import fixangles
from MBI import MBI


class SolarExposedAreaComp(ExplicitComponent):
    """
    Exposed area calculation for a given solar cell

    p: panel ID [0,11]
    c: cell ID [0,6]
    a: fin angle [0,90]
    z: azimuth [0,360]
    e: elevation [0,180]
    LOS: line of sight with the sun [0,1]
    """
    def initialize(self):
        fpath = os.path.dirname(os.path.realpath(__file__))

        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")
        self.options.declare('raw1_file', fpath + '/data/Solar/Area10.txt',
                             desc="angle, azimuth, elevation points for exposed area interpolation.")
        self.options.declare('raw2_file', fpath + '/data/Solar/Area_all.txt',
                             desc="exposed area at points in raw1_file for exposed area interpolation.")

    def setup(self):
        nn = self.options['num_nodes']
        raw1_file = self.options['raw1_file']
        raw2_file = self.options['raw2_file']

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

        self.x = np.zeros((nn, 3))

        # Inputs
        self.add_input('fin_angle', 0.0, units='rad',
                       desc='Fin angle of solar panel')

        self.add_input('azimuth', np.zeros((nn, )), units='rad',
                       desc='Azimuth angle of the sun in the body-fixed frame over time')

        self.add_input('elevation', np.zeros((nn, )), units='rad',
                       desc='Elevation angle of the sun in the body-fixed frame over time')

        # Outputs
        self.add_output('exposed_area', np.zeros((nn, self.nc, self.np)),
                        desc='Exposed area to sun for each solar cell over time',
                        units='m**2', lower=-5e-3, upper=1.834e-1)

        self.declare_partials('exposed_area', 'fin_angle')

        ncp = self.nc * self.np
        rows = np.tile(np.arange(ncp), nn) + np.repeat(ncp*np.arange(nn), ncp)
        cols = np.tile(np.repeat(0, ncp), nn) + np.repeat(np.arange(nn), ncp)

        self.declare_partials('exposed_area', 'azimuth', rows=rows, cols=cols)
        self.declare_partials('exposed_area', 'elevation', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        nn = self.options['num_nodes']

        self.setx(inputs)
        P = self.MBI.evaluate(self.x)
        outputs['exposed_area'] = P.reshape(nn, self.nc, self.np, order='F')

    def setx(self, inputs):
        """
        Sets our state array
        """
        nn = self.options['num_nodes']

        result = fixangles(nn, inputs['azimuth'], inputs['elevation'])
        self.x[:, 0] = inputs['fin_angle']
        self.x[:, 1] = result[0]
        self.x[:, 2] = result[1]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']

        Jfin = self.MBI.evaluate(self.x, 1).reshape(nn, self.nc, self.np, order='F')
        Jaz = self.MBI.evaluate(self.x, 2).reshape(nn, self.nc, self.np, order='F')
        Jel = self.MBI.evaluate(self.x, 3).reshape(nn, self.nc, self.np, order='F')

        partials['exposed_area', 'fin_angle'] = Jfin.flatten()
        partials['exposed_area', 'azimuth'] = Jaz.flatten()
        partials['exposed_area', 'elevation'] = Jel.flatten()

