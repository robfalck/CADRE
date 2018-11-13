"""
Sun discipline for CADRE.
"""
from __future__ import print_function, division, absolute_import

from openmdao.api import Group, MatrixVectorProductComp

from CADRE.sun_dymos.sun_los_comp import SunLOSComp
from CADRE.sun_dymos.sun_pos_eci import SunPositionECIComp
from CADRE.sun_dymos.sun_pos_spherical import SunPositionSphericalComp


class SunGroup(Group):
    """
    The Sun subsystem for CADRE.

    Externally sourced inputs:
    t        - time
    LD       - launch date
    r_e2b_I  - state: Position vector of satelite wrt Earth.
    O_BI     - rotation matrix from body fixed frame to earth centered frame
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int,),
                             desc="Number of time points.")

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('sun_position_eci',
                           SunPositionECIComp(num_nodes=nn),
                           promotes_inputs=['t', 'LD'], promotes_outputs=['r_e2s_I'])

        self.add_subsystem('sun_los',
                           SunLOSComp(num_nodes=nn),
                           promotes_inputs=['r_e2s_I', 'r_e2b_I'], promotes_outputs=['LOS'])

        self.add_subsystem('sun_position_body',
                           MatrixVectorProductComp(vec_size=nn, A_name='O_BI', x_name='r_e2s_I',
                                                   b_name='r_e2s_B', x_units='km', b_units='km'),
                           promotes_inputs=['O_BI', 'r_e2s_I'], promotes_outputs=['r_e2s_B'])

        self.add_subsystem('sun_position_spherical',
                           SunPositionSphericalComp(num_nodes=nn),
                           promotes_inputs=['r_e2s_B'], promotes_outputs=['azimuth', 'elevation'])

