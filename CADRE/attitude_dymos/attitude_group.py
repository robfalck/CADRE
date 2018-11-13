from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from dymos.phases.grid_data import GridData

from .ode_rate_comp import ODERateComp
from CADRE.rowwise_divide_comp import RowwiseDivideComp


class AttitudeGroup(Group):
    """
    The Comm subsystem for CADRE.

    Externally sourced inputs:
    t        - time
    r_e2b_I  - state
    antAngle - parameter
    P_comm   - parameter
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')
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
        gd = self.options['grid_data']



        obi_rate_comp = self.add_subsystem('obi_rate_comp',
                                           ODERateComp(grid_data=gd),
                                           promotes_inputs=['O_BI'],
                                           promotes_outputs=[('dYdt:O_BI', 'Odot_BI')])

        obi_rate_comp.add_rate('O_BI', shape=(3, 3), units=None)
        #
        # self.add_subsystem('angular_velocity_comp',
        #                    AngularVelocityComp(num_nodes=nn),
        #                    promotes_inputs=['O_BI', 'Odot_BI'],
        #                    promotes_outputs=['w_B'])
