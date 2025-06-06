import openmdao.api as om
from jax.experimental import sparse as jsp
# import jax.debug
import jax.numpy as jnp


def _compute_wdot_B(w_B, D, dt_dstau):
    """
    Return a the transformation matrix from the rolled frame to the inertial frame.

    Parameters
    ----------
    w_dot : array-like (n, 3)
        Angular velocity of the body frame wrt the inertial frame at n points in time.
    D : array_like (n, n)
        A differentiation matrix which, when multiplied by a value at n points in time,
        provides the rate of change of the value with respect to non-dimensional time.
    dt_dstau : array_like (n,)
        The conversion factor which provides the ratio of time span for each point in time
        (different nodes may be on different segments) to the nondimensional time span of
        the segment.

    Returns
    -------
    wdot_B : array-like
        Angular acceleration of the body frame wrt the inertial frame.
    """
    rate = D @ w_B / dt_dstau[:, jnp.newaxis]
    return rate


class AngularAccelerationComp(om.JaxExplicitComponent):
    """
    Calculates time derivative of the body frame angular velocity wrt the inertial frame.

    This is accomplished by passing in O_BI at all points, and multiplying
    it by the differentiation matrix (giving the rate of change in dimensionless time, tau)
    and then multilying that by the ratio of the time span of each segment with respect
    to its dimensionless time span. Since this technique relies upon being able to compute
    the full differentiation matrix, this model WILL NOT WORK with GaussLobatto collocation,
    where multiple ODEs each handle a subset of the nodes.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('grid_data')

    def get_self_statics(self):
        # return value must be hashable
        return self._D,

    def setup(self):
        nn = self.options['num_nodes']

        _, self._D = self.options['grid_data'].phase_lagrange_matrices('all', 'all', sparse=False)
        self._D = jsp.BCOO.fromdense(self._D)

        # Inputs
        self.add_input('w_B', shape=(nn, 3), units='rad/s',
                       desc='Angular velocity of the body frame wrt the inertial frame over time.')

        self.add_input('dt_dstau', shape=(nn,), units='s')

        # Outputs
        self.add_output('wdot_B', shape=(nn, 3), units='rad/s**2',
                        desc='First time derivative of w_B over time')

    def compute_primal(self, w_B, dt_dstau):
        return _compute_wdot_B(w_B, self._D, dt_dstau)
