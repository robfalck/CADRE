from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from dymos.phases.grid_data import GridData

from .angular_velocity_comp import AngularVelocityComp


class AttitudeGroup(Group):
    """
    The Comm subsystem for CADRE.

    Externally sourced inputs:
    O_BI      - parameter
    Odot_BI   - parameter
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

        # self.options.declare('lat_gs', types=(float,), default=42.2708,
        #                      desc='ground station latitude (degrees)')
        # self.options.declare('lon_gs', types=(float,), default=-83.7264,
        #                      desc='ground station longitude (degrees)')
        # self.options.declare('alt_gs', types=(float,), default=0.256,
        #                      desc='ground station altitude (km)')
        self.options.declare('Re', types=(float,), default=6378.137,
                             desc='Earth equatorial radius (km)')

    def setup(self):
        nn = self.options['num_nodes']
