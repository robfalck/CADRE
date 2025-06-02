import numpy as np
import openmdao.api as om
import CADRE.orbital_equations


class MEEToCart(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('central_body', values=('earth', 'sun', 'moon'))

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        self.add_input('p', val=np.ones(nn), units=f'DU_{cb}')
        self.add_input('f', val=np.ones(nn))
        self.add_input('g', val=np.ones(nn))
        self.add_input('h', val=np.ones(nn))
        self.add_input('k', val=np.ones(nn))
        self.add_input('L', val=np.ones(nn), units='rad')

        self.add_output('x', val=np.ones(nn), units=f'DU_{cb}')
        self.add_output('y', val=np.ones(nn), units=f'DU_{cb}')
        self.add_output('z', val=np.ones(nn), units=f'DU_{cb}')
        self.add_output('vx', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')
        self.add_output('vy', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')
        self.add_output('vz', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs', show_summary=True)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']

        cL = np.cos(L)
        sL = np.sin(L)

        alpha_sq = h ** 2 - k ** 2
        s_sq = 1 + h ** 2 + k ** 2
        w = 1 + f * cL + g * sL
        r = p / w

        outputs['x'] = r * (cL + alpha_sq * cL + 2 * h * k * sL) / s_sq
        outputs['y'] = r * (sL - alpha_sq * sL + 2 * h * k * cL) / s_sq
        outputs['z'] = 2 * r * (h * sL - k * cL) / s_sq

        outputs['vx'] = -np.sqrt(1 / p) * (
                sL + alpha_sq * sL - 2 * h * k * cL + g - 2 * f * h * k + alpha_sq * g) / s_sq
        outputs['vy'] = -np.sqrt(1 / p) * (
                -cL + alpha_sq * cL + 2 * h * k * sL - f + 2 * g * h * k + alpha_sq * f) / s_sq
        outputs['vz'] = 2 * np.sqrt(1 / p) * (h * cL + k * sL + f * h + g * k) / s_sq

