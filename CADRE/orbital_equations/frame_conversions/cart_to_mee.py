import numpy as np
import openmdao.api as om
import CADRE.orbital_equations
from openmdao.utils.cs_safe import arctan2


class CartesianToMEE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('central_body', values=('earth', 'sun', 'moon'))

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        self.add_input('x', val=np.ones(nn), units=f'DU_{cb}')
        self.add_input('y', val=np.ones(nn), units=f'DU_{cb}')
        self.add_input('z', val=np.ones(nn), units=f'DU_{cb}')
        self.add_input('vx', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')
        self.add_input('vy', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')
        self.add_input('vz', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}')

        self.add_output('p', val=np.ones(nn), units=f'DU_{cb}')
        self.add_output('f', val=np.ones(nn))
        self.add_output('g', val=np.ones(nn))
        self.add_output('h', val=np.ones(nn))
        self.add_output('k', val=np.ones(nn))
        self.add_output('L', val=np.ones(nn), desc='true longitude', units='rad')

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs', show_summary=True)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        vx = inputs['vx']
        vy = inputs['vy']
        vz = inputs['vz']

        hx = y*vz - z*vy
        hy = z*vx - x*vz
        hz = x*vy - y*vx
        h_norm = np.sqrt(hx**2 + hy**2 + hz**2)
        r = np.sqrt(x**2 + y**2 + z**2)
        v = np.sqrt(vx**2 + vy**2 + vz**2)

        e_vec = ((v**2 - 1 / r) * r - (x*vx + y*vy + z*vz) * v)
        e = np.linalg.norm(e_vec)

        i = np.arccos(hz/h_norm)
        Omega = arctan2(hx, -hy)
        omega = arctan2(e_vec[2], e_vec[0] * np.cos(Omega) + e_vec[1] * np.sin(Omega))

        outputs['p'] = h_norm**2
        outputs['f'] = e * np.cos(omega + Omega)
        outputs['g'] = e * np.sin(omega + Omega)
        outputs['h'] = np.tan(i/2) * np.cos(Omega)
        outputs['k'] = np.tan(i/2) * np.sin(Omega)
        outputs['L'] = arctan2(y, x)
