import openmdao.api as om
from jax.experimental import sparse as jsp
# import jax.debug
import jax.numpy as jnp


def _compute_OdotBI(O_BI, D, dt_dstau):
    """
    Return a the transformation matrix from the rolled frame to the inertial frame.

    Parameters
    ----------
    O_BI : array-like (n, 3, 3)
        The transformation matrix from the body to the inertial frame at n points in time.
    D : array_like (n, n)
        A differentiation matrix which, when multiplied by a value at n points in time,
        provides the rate of change of the value with respect to non-dimensional time.
    dt_dstau : array_like (n,)
        The conversion factor which provides the ratio of time span for each point in time
        (different nodes may be on different segments) to the nondimensional time span of
        the segment.

    Returns
    -------
    O_RI : array-like
        Transformation matrix to transform from the rolled frame to the inertial frame.
    """
    nn = O_BI.shape[0]
    obi_flat = jnp.reshape(O_BI, (nn, 9))
    rate = D @ obi_flat / dt_dstau[:, jnp.newaxis]
    return jnp.reshape(rate, (nn, 3, 3))


class OdotBIComp(om.JaxExplicitComponent):
    """
    Calculates time derivative of body frame orientation matrix.

    This is accomplished by passing in O_BI at all points, and multiplying
    it by the differentiation matrix (giving the rate of change in dimensionless time, tau)
    and then multilying that by the ratio of the time span of each segment with respect
    to its dimensionless time span. Since this technique relies upon being able to compute
    the full differentiation matrix, this model WILL NOT WORK with GaussLobatto collocation,
    where multiple ODEs each handle a subset of the nodes.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('time_units', types=str, default='s')
        self.options.declare('grid_data')

    def get_self_statics(self):
        # return value must be hashable
        return self._D,

    def setup(self):
        nn = self.options['num_nodes']

        _, self._D = self.options['grid_data'].phase_lagrange_matrices('all', 'all', sparse=False)
        self._D = jsp.BCOO.fromdense(self._D)

        # Inputs
        self.add_input('O_BI', shape=(nn, 3, 3), units='unitless',
                       desc='Rotation matrix from body-fixed frame to Earth-centered '
                            'inertial frame over time')

        self.add_input('dt_dstau', shape=(nn,), units=self.options['time_units'])

        # Outputs
        self.add_output('Odot_BI', shape=(nn, 3, 3), units=f'1/{self.options["time_units"]}',
                        desc='First time derivative of O_BI over time')

    def compute_primal(self, O_BI, dt_dstau):
        return _compute_OdotBI(O_BI, self._D, dt_dstau)
