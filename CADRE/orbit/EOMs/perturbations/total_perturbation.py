import openmdao.api as om
import numpy as np


class TotalPerturbation(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('central_body', types=str, values=('earth', 'sun', 'moon'))

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        self.add_input('T_r', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('T_s', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('T_w', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')

        self.add_input('J2_r', val=np.zeros(nn),
                       desc='zonal gravity perturbation in the radial direction', units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('J2_s', val=np.zeros(nn),
                       desc='zonal gravity perturbation in the tangential direction', units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('J2_w', val=np.zeros(nn),
                       desc='zonal gravity perturbation in the normal direction', units=f'DU_{cb}/TU_{cb}**2')

        self.add_input('D_r', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('D_s', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')

        self.add_input('sb_r', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('sb_s', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')
        self.add_input('sb_w', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')

        self.add_output('a_r', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')
        self.add_output('a_s', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')
        self.add_output('a_w', val=np.zeros(nn), units=f'DU_{cb}/TU_{cb}**2')

        ar = np.arange(nn)

        self.declare_partials(of='a_r', wrt='T_r', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='a_r', wrt='J2_r', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='a_r', wrt='D_r', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='a_r', wrt='sb_r', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='a_s', wrt='T_s', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='a_s', wrt='J2_s', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='a_s', wrt='D_s', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='a_s', wrt='sb_s', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='a_w', wrt='T_w', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='a_w', wrt='J2_w', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='a_w', wrt='sb_w', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        T_r = inputs['T_r']
        T_s = inputs['T_s']
        T_w = inputs['T_w']

        J2_r = inputs['J2_r']
        J2_s = inputs['J2_s']
        J2_w = inputs['J2_w']

        sb_r = inputs['sb_r']
        sb_s = inputs['sb_s']
        sb_w = inputs['sb_w']

        D_r = inputs['D_r']
        D_s = inputs['D_s']

        outputs['a_r'] = T_r + J2_r + D_r + sb_r
        outputs['a_s'] = T_s + J2_s + D_s + sb_s
        outputs['a_w'] = T_w + J2_w + sb_w
