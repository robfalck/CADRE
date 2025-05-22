import numpy as np
import openmdao.api as om
import CADRE.orbit


class RComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('central_body', types=str, values=('earth', 'sun', 'moon'))

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        self.add_input('p', val=np.ones(nn), desc='semi-latus rectum', units=f'DU_{cb}')
        self.add_input('f', val=np.ones(nn))
        self.add_input('g', val=np.ones(nn))
        self.add_input('L', val=np.ones(nn), desc='true longitude', units='rad')

        self.add_output('r', val=np.ones(nn), desc='geocentric radius', units=f'DU_{cb}')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='r', wrt=['p', 'f', 'g'], rows=ar, cols=ar)
        self.declare_partials(of='r', wrt='L', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        L = inputs['L']

        outputs['r'] = p / (1 + f * np.cos(L) + g * np.sin(L))

    def compute_partials(self, inputs, partials):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        L = inputs['L']

        partials['r', 'p'] = 1 / (1 + f * np.cos(L) + g * np.sin(L))
        partials['r', 'f'] = -p * np.cos(L) / ((1 + f * np.cos(L) + g * np.sin(L)) ** 2)
        partials['r', 'g'] = -p * np.sin(L) / ((1 + f * np.cos(L) + g * np.sin(L)) ** 2)
        partials['r', 'L'] = p * (f * np.sin(L) - g * np.cos(L)) / ((1 + f * np.cos(L) + g * np.sin(L)) ** 2)


