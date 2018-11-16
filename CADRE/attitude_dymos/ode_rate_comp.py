from __future__ import print_function, division

from six import string_types, iteritems

import numpy as np
from scipy.sparse import csr_matrix

from openmdao.api import ExplicitComponent

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units


class ODERateComp(ExplicitComponent):
    """
    Compute the approximated rates of a variable in the ODE.

    Currently this component only works in the ODE of Radau Pseudospectral phases since it
    requires input values to be provided at all nodes.

    Notes
    -----
    .. math::

        \\dot{p}_a = \\frac{d\\tau_s}{dt} \\left[ D \\right] p_a

    where
    :math:`p_a` are the values of the variable at all nodes,
    :math:`\\dot{p}_a` are the time-derivatives of the variable at all nodes,
    :math:`D` is the Lagrange differentiation matrix,
    and :math:`\\frac{d\\tau_s}{dt}` is the ratio of segment duration in segment tau space
    [-1 1] to segment duration in time.
    """

    def initialize(self):
        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        # Save the names of the dynamic controls/parameters
        self._dynamic_names = []
        self._input_names = {}
        self._output_rate_names = {}
        self._var_options = {}

    def _setup_variables(self):
        num_nodes = self.num_nodes
        time_units = self.options['time_units']

        for name, options in iteritems(self._var_options):
            self._input_names[name] = name
            self._output_rate_names[name] = 'dYdt:{0}'.format(name)
            shape = options['shape']
            input_shape = (num_nodes,) + shape
            output_shape = (num_nodes,) + shape

            print(name, input_shape, output_shape)

            units = options['units']
            rate_units = get_rate_units(units, time_units)

            self._dynamic_names.append(name)

            self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)

            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)

            size = np.prod(shape)
            self.rate_jacs[name] = np.zeros((num_nodes, size, num_nodes, size))
            for i in range(size):
                self.rate_jacs[name][:, i, :, i] = self.D
            self.rate_jacs[name] = self.rate_jacs[name].reshape((num_nodes * size,
                                                                 num_nodes * size),
                                                                order='C')
            self.rate_jac_rows[name], self.rate_jac_cols[name] = \
                np.where(self.rate_jacs[name] != 0)

            self.sizes[name] = size

            cs = np.tile(np.arange(num_nodes, dtype=int), reps=size)
            rs = np.concatenate([np.arange(0, num_nodes * size, size, dtype=int) + i
                                 for i in range(size)])

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt=self._input_names[name],
                                  rows=self.rate_jac_rows[name], cols=self.rate_jac_cols[name])

    def setup(self):
        num_nodes = self.options['grid_data'].num_nodes
        time_units = self.options['time_units']
        gd = self.options['grid_data']

        self.add_input('dt_dstau', shape=num_nodes, units=time_units)

        # self._var_options = {}
        self.rate_jacs = {}
        self.rate_jac_rows = {}
        self.rate_jac_cols = {}
        self.sizes = {}
        self.num_nodes = num_nodes

        # The control interpolation matrix L is the product of M_index_to_disc and the nominal
        # pseudospectral interpolation matrix from the discretization nodes to all nodes.
        _, self.D = gd.phase_lagrange_matrices('all', 'all')

        self._setup_variables()

        self.set_check_partial_options('*', method='cs')

    def add_rate(self, name, shape, units):
        self._var_options[name] = {'shape': shape,
                                   'units': units}

    def compute(self, inputs, outputs):
        var_options = self._var_options

        for name, options in iteritems(var_options):

            u = inputs[self._input_names[name]]

            a = np.tensordot(self.D, u, axes=(1, 0)).T

            # divide each "row" by dt_dstau or dt_dstau**2
            outputs[self._output_rate_names[name]] = (a / inputs['dt_dstau']).T

    def compute_partials(self, inputs, partials):
        var_options = self._var_options
        num_nodes = self.options['grid_data'].subset_num_nodes['all']

        for name, options in iteritems(var_options):
            control_name = self._input_names[name]

            size = self.sizes[name]

            rate_name = self._output_rate_names[name]

            # Unroll matrix-shaped controls into an array at each node
            u_d = np.reshape(inputs[control_name], (num_nodes, size))

            dt_dstau = inputs['dt_dstau']
            dt_dstau_tile = np.tile(dt_dstau, size)

            partials[rate_name, 'dt_dstau'] = \
                (-np.dot(self.D, u_d).ravel(order='F') / dt_dstau_tile ** 2)

            dt_dstau_x_size = np.repeat(dt_dstau, size)[:, np.newaxis]

            r_nz, c_nz = self.rate_jac_rows[name], self.rate_jac_cols[name]
            partials[rate_name, control_name] = \
                (self.rate_jacs[name] / dt_dstau_x_size)[r_nz, c_nz]