import numpy as np
import openmdao.api as om

import CADRE.orbital_equations


class CartesianToMEE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('central_body', values=('earth', 'sun', 'moon'))

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        self.add_input('sma', val=np.ones(nn), units=f'DU_{cb}')
        self.add_input('ecc', val=np.ones(nn), units=f'DU_{cb}')
        self.add_input('inc', val=np.ones(nn), units=f'DU_{cb}')
        self.add_input('aop', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')
        self.add_input('raan', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')
        self.add_input('ta', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')

        self.add_output('p', val=np.ones(nn), units=f'DU_{cb}')
        self.add_output('f', val=np.ones(nn))
        self.add_output('g', val=np.ones(nn))
        self.add_output('h', val=np.ones(nn))
        self.add_output('k', val=np.ones(nn))
        self.add_output('L', val=np.ones(nn), desc='true longitude', units='rad')

        ar = np.arange(nn)
        self.declare_partials(of='p', wrt=['sma', 'ecc'], rows=ar, cols=ar)
        self.declare_partials(of=['f', 'g'], wrt=['ecc', 'aop', 'raan'], rows=ar, cols=ar)
        self.declare_partials(of=['h', 'k'], wrt=['ecc', 'inc'], rows=ar, cols=ar)
        self.declare_partials(of='L', wrt=['raan', 'aop', 'ta'], rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        a = inputs['sma']
        e = inputs['ecc']
        i = inputs['inc']
        w = inputs['aop']
        Om = inputs['raan']
        f = inputs['TA']

        outputs['p'] = a*(1 - e**2)
        outputs['f'] = e*np.cos(w + Om)
        outputs['g'] = e*np.sin(w + Om)
        outputs['h'] = np.tan(i/2)*np.cos(Om)
        outputs['k'] = np.tan(i/2)*np.sin(Om)
        outputs['L'] = Om + w + f

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        a = inputs['sma']
        e = inputs['ecc']
        i = inputs['inc']
        w = inputs['aop']
        Om = inputs['raan']
        f = inputs['TA']

        partials['p', 'sma'] = 1 - e**2
        partials['p', 'ecc'] = -2*a*e

        partials['f', 'ecc'] = np.cos(w + Om)
        partials['f', 'aop'] = -e*np.sin(w + Om)
        partials['f', 'raan'] = -e*np.sin(w + Om)

        partials['g', 'ecc'] = np.sin(w + Om)
        partials['g', 'aop'] = e*np.cos(w + Om)
        partials['g', 'raan'] = e*np.cos(w + Om)

        partials['h', 'inc'] = 0.5*np.cos(Om)/(np.cos(i/2)**2)
        partials['h', 'raan'] = -np.tan(i/2)*np.sin(Om)

        partials['k', 'inc'] = 0.5*np.sin(Om)/(np.cos(i/2)**2)
        partials['k', 'raan'] = np.tan(i/2)*np.cos(Om)
