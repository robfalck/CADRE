import numpy as np
import openmdao.api as om
import CADRE.orbital_equations

from openmdao.utils.assert_utils import assert_check_partials


class TwoBodyDynamicsComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('central_body', values=('earth', 'sun', 'moon'))

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        # Modified equinoctial elements
        self.add_input('p', val=np.ones(nn), desc='semi-latus rectum', units=f'DU_{cb}')
        self.add_input('f', val=np.ones(nn), units='unitless')
        self.add_input('g', val=np.ones(nn), units='unitless')
        self.add_input('h', val=np.ones(nn), units='unitless')
        self.add_input('k', val=np.ones(nn), units='unitless')
        self.add_input('L', val=np.ones(nn), desc='true longitude', units='rad')

        # Perturbations
        self.add_input('a_r', val=np.zeros(nn), desc='perturbations in radial direction', units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('a_s', val=np.zeros(nn), desc='perturbations in tangent direction', units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('a_w', val=np.zeros(nn), desc='perturbations in normal direction', units=f'DU_{cb}/TU_{cb}**2')

        # Outputs
        self.add_output('p_dot', val=np.ones(nn), desc='rate of change of semi-parameter', units=f'DU_{cb}/TU_{cb}')
        self.add_output('f_dot', val=np.ones(nn), units=f'1/TU_{cb}')
        self.add_output('g_dot', val=np.ones(nn), units=f'1/TU_{cb}')
        self.add_output('h_dot', val=np.ones(nn), units=f'1/TU_{cb}')
        self.add_output('k_dot', val=np.ones(nn), units=f'1/TU_{cb}')
        self.add_output('L_dot', val=np.ones(nn), units=f'rad/TU_{cb}')

        # partials
        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='p_dot', wrt='p', rows=ar, cols=ar)
        self.declare_partials(of='p_dot', wrt='f', rows=ar, cols=ar)
        self.declare_partials(of='p_dot', wrt='g', rows=ar, cols=ar)
        self.declare_partials(of='p_dot', wrt='L', rows=ar, cols=ar)
        self.declare_partials(of='p_dot', wrt='a_s', rows=ar, cols=ar)

        self.declare_partials(of='f_dot', wrt='p', rows=ar, cols=ar)
        self.declare_partials(of='f_dot', wrt='f', rows=ar, cols=ar)
        self.declare_partials(of='f_dot', wrt='g', rows=ar, cols=ar)
        self.declare_partials(of='f_dot', wrt='h', rows=ar, cols=ar)
        self.declare_partials(of='f_dot', wrt='k', rows=ar, cols=ar)
        self.declare_partials(of='f_dot', wrt='L', rows=ar, cols=ar)
        self.declare_partials(of='f_dot', wrt='a_r', rows=ar, cols=ar)
        self.declare_partials(of='f_dot', wrt='a_s', rows=ar, cols=ar)
        self.declare_partials(of='f_dot', wrt='a_w', rows=ar, cols=ar)

        self.declare_partials(of='g_dot', wrt='p', rows=ar, cols=ar)
        self.declare_partials(of='g_dot', wrt='f', rows=ar, cols=ar)
        self.declare_partials(of='g_dot', wrt='g', rows=ar, cols=ar)
        self.declare_partials(of='g_dot', wrt='h', rows=ar, cols=ar)
        self.declare_partials(of='g_dot', wrt='k', rows=ar, cols=ar)
        self.declare_partials(of='g_dot', wrt='L', rows=ar, cols=ar)
        self.declare_partials(of='g_dot', wrt='a_r', rows=ar, cols=ar)
        self.declare_partials(of='g_dot', wrt='a_s', rows=ar, cols=ar)
        self.declare_partials(of='g_dot', wrt='a_w', rows=ar, cols=ar)

        self.declare_partials(of='h_dot', wrt='p', rows=ar, cols=ar)
        self.declare_partials(of='h_dot', wrt='f', rows=ar, cols=ar)
        self.declare_partials(of='h_dot', wrt='g', rows=ar, cols=ar)
        self.declare_partials(of='h_dot', wrt='h', rows=ar, cols=ar)
        self.declare_partials(of='h_dot', wrt='k', rows=ar, cols=ar)
        self.declare_partials(of='h_dot', wrt='L', rows=ar, cols=ar)
        self.declare_partials(of='h_dot', wrt='a_w', rows=ar, cols=ar)

        self.declare_partials(of='k_dot', wrt='p', rows=ar, cols=ar)
        self.declare_partials(of='k_dot', wrt='f', rows=ar, cols=ar)
        self.declare_partials(of='k_dot', wrt='g', rows=ar, cols=ar)
        self.declare_partials(of='k_dot', wrt='h', rows=ar, cols=ar)
        self.declare_partials(of='k_dot', wrt='k', rows=ar, cols=ar)
        self.declare_partials(of='k_dot', wrt='L', rows=ar, cols=ar)
        self.declare_partials(of='k_dot', wrt='a_w', rows=ar, cols=ar)

        self.declare_partials(of='L_dot', wrt='p', rows=ar, cols=ar)
        self.declare_partials(of='L_dot', wrt='f', rows=ar, cols=ar)
        self.declare_partials(of='L_dot', wrt='g', rows=ar, cols=ar)
        self.declare_partials(of='L_dot', wrt='h', rows=ar, cols=ar)
        self.declare_partials(of='L_dot', wrt='k', rows=ar, cols=ar)
        self.declare_partials(of='L_dot', wrt='L', rows=ar, cols=ar)
        self.declare_partials(of='L_dot', wrt='a_w', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']

        a_r = inputs['a_r']
        a_s = inputs['a_s']
        a_w = inputs['a_w']

        cL = np.cos(L)
        sL = np.sin(L)

        w = 1 + f * cL + g * sL
        s_sq = 1 + h ** 2 + k ** 2
        sqrt_p_mu = np.sqrt(p)

        outputs['p_dot'] = 2 * p * a_s * sqrt_p_mu / w
        outputs['f_dot'] = sqrt_p_mu * (a_r * sL + ((w + 1) * cL + f) * a_s / w - (h * sL - k * cL) * g * a_w / w)
        outputs['g_dot'] = sqrt_p_mu * (-a_r * cL + ((w + 1) * sL + g) * a_s / w + (h * sL - k * cL) * f * a_w / w)
        outputs['h_dot'] = sqrt_p_mu * s_sq * a_w * cL / (2 * w)
        outputs['k_dot'] = sqrt_p_mu * s_sq * a_w * sL / (2 * w)
        outputs['L_dot'] = sqrt_p_mu * (h * sL - k * cL) * a_w / w + np.sqrt(p) * (w / p)**2

    def compute_partials(self, inputs, partials):

        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']

        a_r = inputs['a_r']
        a_s = inputs['a_s']
        a_w = inputs['a_w']

        cL = np.cos(L)
        sL = np.sin(L)
        sqrt_p_mu = np.sqrt(p)

        w = 1 + f * cL + g * sL
        s_sq = 1 + h ** 2 + k ** 2

        partials['p_dot', 'p'] = 3 * sqrt_p_mu * a_s / w
        partials['p_dot', 'f'] = -2 * p * a_s * sqrt_p_mu / (w ** 2) * cL
        partials['p_dot', 'g'] = -2 * p * a_s * sqrt_p_mu / (w ** 2) * sL
        partials['p_dot', 'L'] = -2 * p * a_s * sqrt_p_mu / w ** 2 * (g * cL - f * sL)
        partials['p_dot', 'a_s'] = 2 * p * sqrt_p_mu / w

        partials['f_dot', 'p'] = np.sqrt(1 / p) * (a_r * sL + ((w + 1) * cL + f) * a_s / w -
                                                                        (h * sL - k * cL) * g * a_w / w) / 2
        partials['f_dot', 'f'] = sqrt_p_mu * (a_s * sL * (g + sL) + (h * sL - k * cL) * g * a_w * cL) / (w ** 2)
        partials['f_dot', 'g'] = sqrt_p_mu * (-a_s * sL * (f + cL) - (h * sL - k * cL) * a_w * (1 + f * cL)) / (w ** 2)
        partials['f_dot', 'h'] = -sqrt_p_mu * sL * g * a_w / w
        partials['f_dot', 'k'] = sqrt_p_mu * cL * g * a_w / w
        partials['f_dot', 'L'] = sqrt_p_mu * (a_r * cL + a_s * (w * (cL * (g * cL - f * sL) - sL * (1 + w)) -
                                                                (g * cL - f * sL) * ((1 + w) * cL + f)) / (w ** 2)
                                                            - a_w * g * (f * h + g * k + h * cL + k * sL) / (w ** 2))
        partials['f_dot', 'a_r'] = sqrt_p_mu * sL
        partials['f_dot', 'a_s'] = sqrt_p_mu * ((w + 1) * cL + f) / w
        partials['f_dot', 'a_w'] = -sqrt_p_mu * (h * sL - k * cL) * g / w

        partials['g_dot', 'p'] = np.sqrt(1 / p) * (-a_r * cL + ((w + 1) * sL + f) * a_s / w +
                                                                        (h * sL - k * cL) * g * a_w / w) / 2
        partials['g_dot', 'f'] = sqrt_p_mu * ((h * sL - k * cL) * a_w * (w - cL) - a_s * cL * (g + sL)) / (w ** 2)
        partials['g_dot', 'g'] = sqrt_p_mu * ((cL * (f + cL)) * a_s - (h * sL - k * cL) * a_w * f * sL) / (w ** 2)
        partials['g_dot', 'h'] = sqrt_p_mu * sL * f * a_w / w
        partials['g_dot', 'k'] = -sqrt_p_mu * cL * f * a_w / w
        partials['g_dot', 'L'] = sqrt_p_mu * (a_r * sL + a_s * (w * (sL * (g * cL - f * sL) + cL * (1 + w)) -
                                                                (g * cL - f * sL) * ((1 + w) * sL + g)) / (w ** 2)
                                                            + a_w * g * (f * h + g * k + h * cL + k * sL) / (w ** 2))
        partials['g_dot', 'a_r'] = -sqrt_p_mu * cL
        partials['g_dot', 'a_s'] = sqrt_p_mu * ((w + 1) * sL + g) / w
        partials['g_dot', 'a_w'] = sqrt_p_mu * (h * sL - k * cL) * f / w

        partials['h_dot', 'p'] = np.sqrt(1 / p) * s_sq * a_w * cL / (4 * w)
        partials['h_dot', 'f'] = -0.5 * sqrt_p_mu * s_sq * a_w * cL * cL / (w ** 2)
        partials['h_dot', 'g'] = -0.5 * sqrt_p_mu * s_sq * a_w * cL * sL / (w ** 2)
        partials['h_dot', 'h'] = sqrt_p_mu * h * a_w * cL / w
        partials['h_dot', 'k'] = sqrt_p_mu * k * a_w * cL / w
        partials['h_dot', 'L'] = -0.5 * sqrt_p_mu * s_sq * a_w * (g + sL) / (w ** 2)
        partials['h_dot', 'a_w'] = sqrt_p_mu * s_sq * cL / (2 * w)

        partials['k_dot', 'p'] = np.sqrt(1 / p) * s_sq * a_w * sL / (4 * w)
        partials['k_dot', 'f'] = -0.5 * sqrt_p_mu * s_sq * a_w * sL * cL / (w ** 2)
        partials['k_dot', 'g'] = -0.5 * sqrt_p_mu * s_sq * a_w * sL * sL / (w ** 2)
        partials['k_dot', 'h'] = sqrt_p_mu * h * a_w * sL / w
        partials['k_dot', 'k'] = sqrt_p_mu * k * a_w * sL / w
        partials['k_dot', 'L'] = 0.5 * sqrt_p_mu * s_sq * a_w * (f + cL) / (w ** 2)
        partials['k_dot', 'a_w'] = sqrt_p_mu * s_sq * sL / (2 * w)

        partials['L_dot', 'p'] = 0.5 * np.sqrt(1 / p) * (h * sL - k * cL) * a_w / w\
            - 1.5 * np.sqrt(1 / p**5) * w**2
        partials['L_dot', 'f'] = -sqrt_p_mu * (h * sL - k * cL) * a_w * cL / w**2 + 2 * np.sqrt(p) * w * cL / p**2
        partials['L_dot', 'g'] = -sqrt_p_mu * (h * sL - k * cL) * a_w * sL / w**2 + 2 * np.sqrt(p) * w * sL / p**2
        partials['L_dot', 'h'] = sqrt_p_mu * sL * a_w / w
        partials['L_dot', 'k'] = -sqrt_p_mu * cL * a_w / w
        partials['L_dot', 'L'] = sqrt_p_mu * ((h * cL + k * sL) - (h * sL - k * cL) * (g * cL - f * sL) / w) * a_w / w\
            + 2 * np.sqrt(p) * w * (g * cL - f * sL) / p**2
        partials['L_dot', 'a_w'] = sqrt_p_mu * (h * sL - k * cL) / w


if __name__ == '__main__':
    p = om.Problem(om.Group())
    p.model.add_subsystem('mee', TwoBodyDynamicsComp())

    p.setup(force_alloc_complex=True)
    p.run_model()

    cpd = p.check_partials(method='cs')  # , step=1E-20)
    assert_check_partials(cpd)

