"""Definition of the Vector Magnitude Component."""


from six import string_types

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class RowwiseDivideComp(ExplicitComponent):
    """
    Divides vector a at each node by quantity b at each node

    math::
        \bar{c}_i = \bar{a}_i / b_i

    where a is of shape (vec_size, n) and b is of shape (vec_size,)

    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the vector magnitude is computed')
        self.options.declare('length', types=int, default=3,
                             desc='The length of the input vector at each point')
        self.options.declare('a_name', types=string_types, default='a',
                             desc='The variable name for input vector a.')
        self.options.declare('a_units', types=string_types, default=None, allow_none=True,
                             desc='The units for the input vector a.')
        self.options.declare('b_name', types=string_types, default='b',
                             desc='The variable name for input scalar b.')
        self.options.declare('b_units', types=string_types, default=None, allow_none=True,
                             desc='The units for the input scalar b.')
        self.options.declare('c_name', types=string_types, default='c',
                             desc='The variable name for output vector c.')
        self.options.declare('c_units', types=string_types, default=None, allow_none=True,
                             desc='The units for the output vector c.')

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the vector magnitude component.
        """
        opts = self.options
        vec_size = opts['vec_size']
        m = opts['length']

        self.add_input(name=opts['a_name'],
                       shape=(vec_size, m),
                       units=opts['a_units'])

        self.add_input(name=opts['b_name'],
                       shape=(vec_size,),
                       units=opts['b_units'])

        self.add_output(name=opts['c_name'],
                        val=np.zeros(shape=(vec_size, m)),
                        units=opts['c_units'])

        ar = np.arange(vec_size * m, dtype=int)

        self.declare_partials(of=opts['c_name'], wrt=opts['a_name'], rows=ar, cols=ar)

        rs = ar
        cs = np.repeat(np.arange(vec_size, dtype=int), m)

        self.declare_partials(of=opts['c_name'], wrt=opts['b_name'], rows=rs, cols=cs)

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
        a = inputs[opts['a_name']]
        b = inputs[opts['b_name']]
        outputs[opts['c_name']] = a / b[:, np.newaxis]

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
        a = inputs[opts['a_name']]
        b = inputs[opts['b_name']]

        partials[opts['c_name'], opts['a_name']] = np.repeat((1.0 / b[:, np.newaxis]).ravel(), m)
        partials[opts['c_name'], opts['b_name']] = (-a / b[:, np.newaxis]**2).ravel()
