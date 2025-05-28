import openmdao.api as om
from CADRE.orbital_equations.EOMs.perturbations.total_perturbation import TotalPerturbation
from CADRE.orbital_equations.EOMs.perturbations.spherical_harmonics_comp import SphericalHarmonicsComp
# from CADRE.EOMs.perturbations.secondary_body.secondary_body_comp import SecondaryBodyEffect
from CADRE.orbital_equations.frame_conversions.r_comp import RComp


class Perturbations(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('central_body', values=('earth', 'sun', 'moon'))
        self.options.declare('sc_params', types=dict)

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        self.add_subsystem('r_comp', RComp(num_nodes=nn, central_body=cb),
                           promotes_inputs=['p', 'f', 'g', 'L'], promotes_outputs=['r'])

        self.add_subsystem('spherical_harmonics', subsys=SphericalHarmonicsComp(num_nodes=nn, central_body=cb),
                           promotes_inputs=['h', 'k', 'L', 'r'], promotes_outputs=['J2_r', 'J2_s', 'J2_w'])

        # self.add_subsystem('atmospheric_drag', subsys=AerodynamicDrag(num_nodes=nn,
        #                                                               mu=mu, sc_params=sc),
        #                    promotes_inputs=['p', 'f', 'g', 'L', 'r', 'v'], promotes_outputs=['D_r', 'D_s'])

        # self.add_subsystem('secondary_body', subsys=SecondaryBodyEffect(num_nodes=nn),
        #                    promotes_inputs=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'],
        #                    promotes_outputs=['sb_r', 'sb_s', 'sb_w'])

        self.add_subsystem('total_perturbation', subsys=TotalPerturbation(num_nodes=nn, central_body=cb),
                           promotes_inputs=['J2_r', 'J2_s', 'J2_w', 'T_r', 'T_s', 'T_w',
                                            'D_r', 'D_s', 'sb_r', 'sb_s', 'sb_w'],
                           promotes_outputs=['a_r', 'a_s', 'a_w'])
