from __future__ import print_function, division, absolute_import

from openmdao.api import Group, VectorMagnitudeComp

from dymos import declare_state, declare_time, declare_parameter

from CADRE.orbit_dymos.orbit_eom import OrbitEOMComp
from CADRE.orbit_dymos.gravity_perturbations_comp import GravityPerturbationsComp
from CADRE.orbit_dymos.ori_comp import ORIComp
from CADRE.orbit_dymos.obr_comp import OBRComp
from CADRE.orbit_dymos.obi_comp import OBIComp


@declare_time(units='s')
@declare_state('r_e2b_I', rate_source='orbit_eom_comp.dXdt:r_e2b_I', targets=['r_e2b_I'],
               units='km', shape=(3,))
@declare_state('v_e2b_I', rate_source='orbit_eom_comp.dXdt:v_e2b_I', targets=['v_e2b_I'],
               units='km/s', shape=(3,))
@declare_parameter('Gamma', targets=['Gamma'], units='rad')
class CadreOrbitODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('rmag_comp',
                           VectorMagnitudeComp(vec_size=nn, length=3, in_name='r_e2b_I',
                                               mag_name='rmag_e2b_I', units='km'),
                           promotes_inputs=['r_e2b_I'], promotes_outputs=['rmag_e2b_I'])

        self.add_subsystem('grav_pert_comp', GravityPerturbationsComp(num_nodes=nn),
                           promotes_inputs=['r_e2b_I', 'rmag_e2b_I'],
                           promotes_outputs=[('a_pert:J2', 'a_pert_I')])

        self.add_subsystem('ori_comp',
                           ORIComp(num_nodes=nn),
                           promotes_inputs=['r_e2b_I', 'v_e2b_I'],
                           promotes_outputs=['O_RI'])

        self.add_subsystem('obr_comp',
                           OBRComp(num_nodes=nn),
                           promotes_inputs=['Gamma'],
                           promotes_outputs=['O_BR'])

        self.add_subsystem('obi_comp',
                           OBIComp(num_nodes=nn),
                           promotes_inputs=['O_BR', 'O_RI'],
                           promotes_outputs=['O_BI'])

        self.add_subsystem('orbit_eom_comp', OrbitEOMComp(num_nodes=nn),
                           promotes_inputs=['rmag_e2b_I', 'r_e2b_I', 'v_e2b_I', 'a_pert_I'])
