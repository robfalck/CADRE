import jax.numpy as jnp
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

from CADRE.orbital_equations import CentralBody, set_body_units


def mee_to_cart(p, f, g, h, k, L):
    """
    Return the Cartesian position and velocity for modified equinoctial elements (mu = 1).

    Parameters
    ----------
    p, f, g, h, k, L : array-like (n,)
        Modified equinoctial elements.

    Returns
    -------
    tuple of array-like (n,)
        ``(x, y, z, vx, vy, vz)``.
    """
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


_UNITS = {'p': 'DU', 'f': 'unitless', 'g': 'unitless', 'h': 'unitless', 'k': 'unitless',
          'L': 'rad', 'x': 'DU', 'y': 'DU', 'z': 'DU', 'vx': 'DU/TU', 'vy': 'DU/TU',
          'vz': 'DU/TU'}


class MEEToCart(JaxExplicitComp):
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    inputs: dict[str, InputVar] = {
        name: InputVar(shape=('num_nodes',), val=1.0) for name in ('p', 'f', 'g', 'h', 'k', 'L')
    }

    outputs: dict[str, OutputVar] = {
        name: OutputVar(shape=('num_nodes',)) for name in ('x', 'y', 'z', 'vx', 'vy', 'vz')
    }

    def model_post_init(self, context, /):
        super().model_post_init(context)
        set_body_units(self, _UNITS)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        cart = mee_to_cart(*(inputs[name] for name in ('p', 'f', 'g', 'h', 'k', 'L')))
        for name, val in zip(('x', 'y', 'z', 'vx', 'vy', 'vz'), cart):
            outputs[name] = val
