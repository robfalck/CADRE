import openmdao.api as om
from CADRE.orbital_equations.EOMs import TwoBodyDynamicsComp
from CADRE.orbital_equations.EOMs.perturbations.perturbations import Perturbations
from CADRE.orbital_equations.frame_conversions import MEEToCart


class TwoBodyEOM(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('central_body', values=('earth', 'sun', 'moon'))
        self.options.declare('sc_params', types=dict)

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']
        sc_params = self.options['sc_params']
        # kep_elements_constraints = self.options['kep_elements_constraints']

        self.add_subsystem('mee_to_cart', subsys=MEEToCart(num_nodes=nn, central_body=cb),
                           promotes_inputs=['p', 'f', 'g', 'h', 'k', 'L'],
                           promotes_outputs=['x', 'y', 'z', 'vx', 'vy', 'vz'])

        # self.add_subsystem('propulsion', subsys=PropulsionMEE(num_nodes=nn, guidance_direction=gd, sc_params=sc_params),
        #                    promotes_inputs=['m', 'throttle', 'u_r', 'u_s', 'u_w'],
        #                    promotes_outputs=['T_r', 'T_s', 'T_w', 'm_dot'])

        self.add_subsystem('perturbations', subsys=Perturbations(num_nodes=nn, sc_params=sc_params, central_body=cb),
                           promotes_inputs=['T_r', 'T_s', 'T_w'],
                           promotes_outputs=['a_r', 'a_s', 'a_w'])

        self.add_subsystem('spacecraft_dynamics', subsys=TwoBodyDynamicsComp(num_nodes=nn, central_body=cb),
                           promotes_inputs=['p', 'f', 'g', 'h', 'k', 'L', 'a_r', 'a_s', 'a_w'],
                           promotes_outputs=['p_dot', 'f_dot', 'g_dot', 'h_dot', 'k_dot', 'L_dot'])


