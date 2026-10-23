import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

from CADRE.orbital_equations import CentralBody, set_body_units

_UNITS = {'x': 'DU', 'y': 'DU', 'z': 'DU', 'vx': 'DU/TU', 'vy': 'DU/TU', 'vz': 'DU/TU',
          'p': 'DU', 'f': 'unitless', 'g': 'unitless', 'h': 'unitless', 'k': 'unitless',
          'L': 'rad'}


def cart_to_mee(x, y, z, vx, vy, vz):
    """
    Return the modified equinoctial elements for a Cartesian position and velocity (mu = 1).

    Parameters
    ----------
    x, y, z, vx, vy, vz : array-like (n,)
        Cartesian position and velocity.

    Returns
    -------
    tuple of array-like (n,)
        ``(p, f, g, h, k, L)``.
    """
    r_vec = jnp.stack((x, y, z))
    v_vec = jnp.stack((vx, vy, vz))
    h_vec = jnp.cross(r_vec, v_vec, axis=0)
    hx, hy, hz = h_vec
    h_norm = jnp.sqrt(jnp.sum(h_vec**2, axis=0))
    r = jnp.sqrt(jnp.sum(r_vec**2, axis=0))
    v_sq = jnp.sum(v_vec**2, axis=0)

    e_vec = (v_sq - 1 / r) * r_vec - jnp.sum(r_vec * v_vec, axis=0) * v_vec

    # The direction of the ascending node, and the in-plane unit vector 90 degrees ahead of it.
    hk_denom = h_norm + hz
    h = -hy / hk_denom
    k = hx / hk_denom

    # The equinoctial frame's in-plane unit vectors f_hat and g_hat.
    s_sq = 1 + h**2 + k**2
    f_hat = jnp.stack((1 - k**2 + h**2, 2 * h * k, -2 * k)) / s_sq
    g_hat = jnp.stack((2 * h * k, 1 + k**2 - h**2, 2 * h)) / s_sq

    p = h_norm**2
    f = jnp.sum(e_vec * f_hat, axis=0)
    g = jnp.sum(e_vec * g_hat, axis=0)
    L = jnp.arctan2(jnp.sum(r_vec * g_hat, axis=0), jnp.sum(r_vec * f_hat, axis=0))
    return p, f, g, h, k, L


class CartesianToMEE(JaxExplicitComp):
    """
    Convert a Cartesian position and velocity to modified equinoctial elements.

    The original version of this component computed the eccentricity vector from the position
    and speed magnitudes rather than the vectors, and normed it over all nodes at once. This
    version uses the standard equinoctial-frame projections, so that it inverts MEEToCart.
    """
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    inputs: dict[str, InputVar] = {
        name: InputVar(shape=('num_nodes',), val=1.0) for name in ('x', 'y', 'z', 'vx', 'vy', 'vz')
    }

    outputs: dict[str, OutputVar] = {
        name: OutputVar(shape=('num_nodes',)) for name in ('p', 'f', 'g', 'h', 'k', 'L')
    }

    def model_post_init(self, context, /):
        super().model_post_init(context)
        set_body_units(self, _UNITS)
        self.outputs['L'].desc = 'true longitude'

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        mee = cart_to_mee(*(inputs[name] for name in ('x', 'y', 'z', 'vx', 'vy', 'vz')))
        for name, val in zip(('p', 'f', 'g', 'h', 'k', 'L'), mee):
            outputs[name] = val
