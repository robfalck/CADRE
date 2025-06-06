import jax.numpy as jnp
import openmdao.api as om

class MEEToCart(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('central_body', values=('earth', 'sun', 'moon'))

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        self.add_input('p', shape=(nn,), units=f'DU_{cb}')
        self.add_input('f', shape=(nn,), units='unitless')
        self.add_input('g', shape=(nn,), units='unitless')
        self.add_input('h', shape=(nn,), units='unitless')
        self.add_input('k', shape=(nn,), units='unitless')
        self.add_input('L', shape=(nn,), units='rad')

        self.add_output('x', shape=(nn,), units=f'DU_{cb}')
        self.add_output('y', shape=(nn,), units=f'DU_{cb}')
        self.add_output('z', shape=(nn,), units=f'DU_{cb}')
        self.add_output('vx', shape=(nn,), units=f'DU_{cb}/TU_{cb}')
        self.add_output('vy', shape=(nn,), units=f'DU_{cb}/TU_{cb}')
        self.add_output('vz', shape=(nn,), units=f'DU_{cb}/TU_{cb}')

    def compute_primal(self, p, f, g, h, k, L):
        cL = jnp.cos(L)
        sL = jnp.sin(L)

        alpha_sq = h ** 2 - k ** 2
        s_sq = 1 + h ** 2 + k ** 2
        w = 1 + f * cL + g * sL
        r = p / w

        x = r * (cL + alpha_sq * cL + 2 * h * k * sL) / s_sq
        y = r * (sL - alpha_sq * sL + 2 * h * k * cL) / s_sq
        z = 2 * r * (h * sL - k * cL) / s_sq

        sqrt_oop = jnp.sqrt(1. / p)

        vx = -sqrt_oop * (sL + alpha_sq * sL - 2 * h * k * cL + g - 2 * f * h * k + alpha_sq * g) / s_sq
        vy = -sqrt_oop * (-cL + alpha_sq * cL + 2 * h * k * sL - f + 2 * g * h * k + alpha_sq * f) / s_sq
        vz = 2 * sqrt_oop * (h * cL + k * sL + f * h + g * k) / s_sq

        return x, y, z, vx, vy, vz
