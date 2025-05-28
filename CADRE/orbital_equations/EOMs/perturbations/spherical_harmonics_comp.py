import openmdao.api as om
import numpy as np


class SphericalHarmonicsComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('central_body', values=('earth', 'sun', 'moon'))

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        self.add_input('h', val=np.ones(nn))
        self.add_input('k', val=np.ones(nn))
        self.add_input('L', val=np.ones(nn), desc='true longitude', units='rad')
        self.add_input('r', val=np.ones(nn), desc='geocentric radius', units=f'DU_{cb}')

        self.add_output('J2_r', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}**2',
                        desc='zonal gravity perturbation in the radial direction')
        self.add_output('J2_s', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}**2',
                        desc='zonal gravity perturbation in the tangential direction')
        self.add_output('J2_w', val=np.ones(nn), units=f'DU_{cb}/TU_{cb}**2',
                        desc='zonal gravity perturbation in the normal direction')

        # partials
        self.declare_coloring(wrt='*', tol=1.0E-12, show_sparsity=False, show_summary=False)
        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        J2 = 202.7e-6

        r = inputs['r']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']
        s_sq = 1 + h**2 + k**2

        outputs['J2_r'] = -3 * J2 * (1 - 12 * ((h * np.sin(L) - k * np.cos(L)) / s_sq) ** 2) / (2 * r**4)
        outputs['J2_s'] = -12 * J2 * ((h * np.sin(L) - k * np.cos(L)) * (
                h * np.cos(L) + k * np.sin(L)) / (s_sq ** 2)) / (r**4)
        outputs['J2_w'] = -6 * J2 * ((1 - h**2 - k**2) * (
                h * np.sin(L) - k * np.cos(L)) / (s_sq ** 2)) / (r**4)


if __name__ == '__main__':
    p = om.Problem()
    p.model.add_subsystem('harmonics', SphericalHarmonicsComp(num_nodes=3, num_quadrature_nodes=9, mu=3.98E5),
                          promotes_inputs=['*'], promotes_outputs=['*'])

    p.setup()

    num_node = 3
    p.set_val('h', np.ones(num_node,))
    p.set_val('k', np.ones(num_node,))
    p.set_val('L', np.ones(nun_quad,))
    p.set_val('r', np.ones((num_node, nun_quad)))

    p.run_model()
    print(p.get_val('J2_r'))


