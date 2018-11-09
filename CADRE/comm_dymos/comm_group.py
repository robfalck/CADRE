from __future__ import print_function, division, absolute_import

from openmdao.api import Group, VectorMagnitudeComp, MatrixVectorProductComp

from .comm_ant_quaternion_comp import CommAntQuaternionComp
from .comm_ant_rotation_matrix_comp import CommAntRotationMatrixComp
from .comm_data_rate_comp import CommDataRateComp
from .comm_earth_rotation_quaternion_comp import CommEarthRotationQuaternionComp
from .comm_earth_rotation_matrix_comp import CommEarthRotationMatrixComp
from .comm_gain_pattern_comp import CommGainPatternComp
from .comm_gs_pos_eci_comp import CommGSPosECIComp
from .comm_los_comp import CommLOSComp
from .comm_vector_eci_comp import CommVectorECIComp
from .comm_vector_spherical_comp import CommVectorSphericalComp


class CommGroup(Group):
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
        self.options.declare('lat_gs', types=(float,), default=42.2708,
                             desc='ground station latitude (degrees)')
        self.options.declare('lon_gs', types=(float,), default=-83.7264,
                             desc='ground station longitude (degrees)')
        self.options.declare('alt_gs', types=(float,), default=0.256,
                             desc='ground station altitude (km)')
        self.options.declare('Re', types=(float,), default=6378.137,
                             desc='Earth equatorial radius (km)')

    def setup(self):
        nn = self.options['num_nodes']
        lat_gs = self.options['lat_gs']
        lon_gs = self.options['lon_gs']
        alt_gs = self.options['alt_gs']

        self.add_subsystem('ant_quaternion_comp',
                           CommAntQuaternionComp(num_nodes=nn),
                           promotes_inputs=['antAngle'], promotes_outputs=['q_AB'])

        self.add_subsystem('ant_rotation_matrix_comp',
                           CommAntRotationMatrixComp(num_nodes=nn),
                           promotes_inputs=['q_AB'], promotes_outputs=['O_AB'])

        self.add_subsystem('comm_earth_rotation_quaternion_comp',
                           CommEarthRotationQuaternionComp(num_nodes=nn, gha0=0.0),
                           promotes_inputs=['t'], promotes_outputs=['q_IE'])

        self.add_subsystem('comm_earth_rotation_matrix_comp',
                           CommEarthRotationMatrixComp(num_nodes=nn),
                           promotes_inputs=['q_IE'], promotes_outputs=['O_IE'])

        self.add_subsystem('comm_gs_pos_eci_comp',
                           CommGSPosECIComp(num_nodes=nn, lat=lat_gs, lon=lon_gs, alt=alt_gs),
                           promotes_inputs=['O_IE'], promotes_outputs=['r_e2g_I'])

        self.add_subsystem('comm_vector_eci_comp', CommVectorECIComp(num_nodes=nn),
                           promotes_inputs=['r_e2g_I', 'r_e2b_I'], promotes_outputs=['r_b2g_I'])

        self.add_subsystem('comm_vector_body_comp',
                           MatrixVectorProductComp(vec_size=nn, A_name='O_BI', x_name='r_b2g_I',
                                                   b_name='r_b2g_B', x_units='km', b_units='km'),
                           promotes_inputs=['O_BI', 'r_b2g_I'], promotes_outputs=['r_b2g_B'])

        self.add_subsystem('comm_vector_ant_comp',
                           MatrixVectorProductComp(vec_size=nn, A_name='O_AB', x_name='r_b2g_B',
                                                   b_name='r_b2g_A', x_units='km', b_units='km'),
                           promotes_inputs=['O_AB', 'r_b2g_B'], promotes_outputs=['r_b2g_A'])

        self.add_subsystem('comm_vector_spherical_comp',
                           CommVectorSphericalComp(num_nodes=nn),
                           promotes_inputs=['r_b2g_A'],
                           promotes_outputs=['elevationGS', 'azimuthGS'])

        self.add_subsystem('comm_distance_comp',
                           VectorMagnitudeComp(vec_size=nn, in_name='r_b2g_A', mag_name='GSdist',
                                               units='km'),
                           promotes_inputs=['r_b2g_A'], promotes_outputs=['GSdist'])

        self.add_subsystem('comm_gain_pattern_comp',
                           CommGainPatternComp(num_nodes=nn),
                           promotes_inputs=['elevationGS', 'azimuthGS'],
                           promotes_outputs=['gain'])

        self.add_subsystem('comm_los_comp', CommLOSComp(num_nodes=nn),
                           promotes_inputs=['r_b2g_I', 'r_e2g_I'], promotes_outputs=['CommLOS'])

        self.add_subsystem('data_rate_comp',
                           CommDataRateComp(num_nodes=nn),
                           promotes_inputs=['P_comm', 'gain', 'CommLOS', 'GSdist'],
                           promotes_outputs=['dXdt:data'])
