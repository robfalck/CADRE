"""Definition of the Vector Magnitude Component."""


from six import string_types

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class VectorUnitizeComp(ExplicitComponent):
    """
    Computes the unitized vector

    math::
        \hat{a} = \bar{a} / np.sqrt(np.dot(a, a))

    where a is of shape (vec_size, n)

    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the vector magnitude is computed')
        self.options.declare('length', types=int, default=3,
                             desc='The length of the input vector at each point')
        self.options.declare('in_name', types=string_types, default='a',
                             desc='The variable name for input vector.')
        self.options.declare('units', types=string_types, default=None, allow_none=True,
                             desc='The units for the input and output vector.')
        self.options.declare('out_name', types=string_types, default='a_mag',
                             desc='The variable name for output unitized vector.')

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the vector magnitude component.
        """
        opts = self.options
        vec_size = opts['vec_size']
        m = opts['length']

        self.add_input(name=opts['in_name'],
                       shape=(vec_size, m),
                       units=opts['units'])

        self.add_output(name=opts['out_name'],
                        val=np.zeros(shape=(vec_size, m)),
                        units=opts['units'])

        row_idxs = np.repeat(np.arange(vec_size * m, dtype=int), m)
        temp = np.reshape(np.arange(vec_size * m, dtype=int), newshape=(vec_size, m))
        col_idxs = np.repeat(temp, m, axis=0).ravel()
        self.declare_partials(of=opts['out_name'], wrt=opts['in_name'],
                              rows=row_idxs, cols=col_idxs, val=1.0)

    def compute(self, inputs, outputs):
        """
        Compute the vector magnitude of input.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        opts = self.options
        a = inputs[opts['in_name']]
        a_mag = np.sqrt(np.einsum('ni,ni->n', a, a))
        outputs[opts['out_name']] = a / a_mag[:, np.newaxis]

    def compute_partials(self, inputs, partials):
        """
        Compute the sparse partials for the vector magnitude w.r.t. the inputs.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        opts = self.options
        m = opts['length']
        a = inputs[opts['in_name']]
        a_mag = np.sqrt(np.einsum('ni,ni->n', a, a))
        f = a[0, :]
        g = a_mag
        gp = a.ravel() / np.sqrt(np.einsum('ni,ni->n', a, a))
        g_squared = np.einsum('ni,ni->n', a, a)

        print(a_mag - (a * a[0] / a_mag)[0] / (a_mag ** 2))

        # print()
        # print(f)
        # # print(fp)
        # print(g)
        # print(gp)
        # print()

        partials[opts['out_name'], opts['in_name']] = 88

        print()
        print('gp')
        print(gp.shape)
        print('f')
        print(f.shape)

        print((g - gp[0] * f[0]) / g_squared, end='')
        print((g - gp[1] * f[0]) / g_squared, end='')
        print((g - gp[2] * f[0]) / g_squared)

        print((g - gp[0] * f[1]) / g_squared, end='')
        print((g - gp[1] * f[1]) / g_squared, end='')
        print((g - gp[2] * f[1]) / g_squared)

        print((g - gp[0] * f[2]) / g_squared, end='')
        print((g - gp[1] * f[2]) / g_squared, end='')
        print((g - gp[2] * f[2]) / g_squared)

        diag = (g - gp * f) * g_squared



    #
    #     "f' * g - g' * f / 2g"
    #     fp = 1
    #     g = np.sqrt(np.einsum('ni,ni->n', a, a))[:, np.newaxis]
    #     gp = a.ravel() / np.repeat(np.sqrt(np.einsum('ni,ni->n', a, a)), opts['length'])
    #     f = a
    #
    #     # Use the following for sparse partials
    #     partials[opts['out_name'], opts['in_name']] = a.ravel() / np.repeat(np.sqrt(np.einsum('ni,ni->n', a, a)), opts['length'])
