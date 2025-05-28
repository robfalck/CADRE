import numpy as np
import openmdao.api as om
import CADRE.orbital_equations
from openmdao.utils.cs_safe import arctan2


class MEEToKep(om.ExplicitComponent):
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

        self.add_output('sma', val=np.ones(nn), units=f'DU_{cb}')
        self.add_output('ecc', val=np.ones(nn), units=f'DU_{cb}')
        self.add_output('inc', val=np.ones(nn), units=f'DU_{cb}')
        self.add_output('aop', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')
        self.add_output('raan', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')
        self.add_output('ta', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs', show_summary=True)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']

        outputs['sma'] = p / (1 - f**2 - g**2)

        outputs['ecc'] = np.sqrt(f**2 + g**2)

        outputs['inc'] = arctan2(2*np.sqrt(h**2+k**2), 1-h**2-k**2)

        outputs['raan'] = arctan2(k, h)

        outputs['aop'] = arctan2(g*h-f*k, f*h+g*k)

        outputs['ta'] = L - arctan2(g, f)
