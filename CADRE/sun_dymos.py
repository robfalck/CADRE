"""
Sun discipline for CADRE, on om4.

Rotation matrices follow ``x_Y = O_XY x_X`` with column vectors, so O_BI takes body vectors into
the inertial frame (see ai/ARCHITECTURE.md, ADL-005).
"""
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

#: Degrees to radians.
d2r = np.pi / 180.


def _sun_direction_eci(T):
    """
    Return the low-precision solar ephemeris: the unit vector from Earth to Sun in ECI.

    Parameters
    ----------
    T : array-like (n,)
        Days since the epoch of the ephemeris.

    Returns
    -------
    r_e2s_I : array-like (3, n)
        Unit vector from Earth to Sun in the Earth-centered inertial frame.
    """
    L = d2r*280.460 + d2r*0.9856474*T
    g = d2r*357.528 + d2r*0.9856003*T
    Lambda = L + d2r*1.914666*jnp.sin(g) + d2r*0.01999464*jnp.sin(2*g)
    eps = d2r*23.439 - d2r*3.56e-7*T
    return jnp.stack((jnp.cos(Lambda), jnp.sin(Lambda)*jnp.cos(eps), jnp.sin(Lambda)*jnp.sin(eps)))


class SunPositionECIComp(JaxExplicitComp):
    """
    Compute the direction from Earth to Sun in Earth-centered inertial frame.

    The original declared this unit vector in km.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'LD': InputVar(shape=(1,), val=5000.0, units='d',
                       desc='Launch date, in days since the epoch of the ephemeris.',
                       tags={'dymos.static_target'}),
        't': InputVar(shape=('num_nodes',), val=0.0, units='s', desc='Time since launch',
                      tags={'dymos.time_target'}),
    }

    outputs: dict[str, OutputVar] = {
        'r_e2s_I': OutputVar(shape=(3, 'num_nodes'), units='unitless',
                             desc='Unit vector from Earth to Sun in Earth-centered '
                                  'inertial frame over time.'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['r_e2s_I'] = _sun_direction_eci(inputs['LD'] + inputs['t'] / 86400.)


class SunLOSComp(JaxExplicitComp):
    """
    Compute the Satellite to sun line of sight.

    LOS is 1 on the sunlit side of the Earth, 0 in the umbra (within r1 of the Earth-Sun line on
    the far side), and blends between with a smoothstep out to the Earth's radius, r2.
    """
    num_nodes: int = Field(default=1)

    # Earth's radius is 6378 km. 0.85 is the alpha in John Hwang's paper
    r1: float = Field(default=6378.137 * 0.85)
    r2: float = Field(default=6378.137)

    inputs: dict[str, InputVar] = {
        'r_e2b_I': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from Earth to satellite in Earth-centered '
                                 'inertial frame over time.'),
        'r_e2s_I': InputVar(shape=(3, 'num_nodes'), units='unitless',
                            desc='Unit vector from Earth to Sun in Earth-centered '
                                 'inertial frame over time.'),
    }

    outputs: dict[str, OutputVar] = {
        'LOS': OutputVar(shape=('num_nodes',), units='unitless',
                         desc='Satellite to sun line of sight over time'),
    }

    def get_self_statics(self):
        return {'r1': self.r1, 'r2': self.r2}

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        r1 = statics['r1']
        r2 = statics['r2']
        r_b = inputs['r_e2b_I']
        r_s = inputs['r_e2s_I']

        dot = jnp.sum(r_b * r_s, axis=0)
        # Squared distance of the satellite from the Earth-Sun line.
        dist_sq = jnp.sum(jnp.cross(r_b, r_s, axis=0)**2, axis=0)

        # Only take the square root inside the penumbra band, so that no branch jnp.where
        # discards has an infinite derivative (sqrt at 0) to poison the gradient.
        in_band = (dist_sq > r1**2) & (dist_sq < r2**2)
        dist = jnp.sqrt(jnp.where(in_band, dist_sq, r1**2))
        x = (dist - r1) / (r2 - r1)
        blend = 3*x**2 - 2*x**3

        shadowed = jnp.where(dist_sq <= r1**2, 0.0, jnp.where(dist_sq >= r2**2, 1.0, blend))
        outputs['LOS'] = jnp.where(dot >= 0.0, 1.0, shadowed)


class SunPositionBodyComp(JaxExplicitComp):
    """
    Direction from Earth to Sun in body-fixed frame.

    The original applied O_BI rather than its transpose (ADL-005), and declared the unit vector
    in km.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'O_BI': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from body-fixed frame to Earth-centered '
                              'inertial frame over time'),
        'r_e2s_I': InputVar(shape=(3, 'num_nodes'), units='unitless',
                            desc='Unit vector from Earth to Sun in Earth-centered '
                                 'inertial frame over time.'),
    }

    outputs: dict[str, OutputVar] = {
        'r_e2s_B': OutputVar(shape=(3, 'num_nodes'), units='unitless',
                             desc='Unit vector from Earth to Sun in body-fixed frame over time.'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['r_e2s_B'] = jnp.einsum('jin,jn->in', inputs['O_BI'], inputs['r_e2s_I'])


class SunPositionSphericalComp(JaxExplicitComp):
    """
    Compute the azimuth and elevation angles of the Sun in the body-fixed frame.

    Azimuth is measured in the body x-y plane from +x, in [0, 2 pi). Elevation is the angle from
    the body +z axis, in [0, pi]. Both are singular when the Sun is along the body z axis.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'r_e2s_B': InputVar(shape=(3, 'num_nodes'), units='unitless',
                            desc='Direction from Earth to Sun in body-fixed frame over time.'),
    }

    outputs: dict[str, OutputVar] = {
        'azimuth': OutputVar(shape=('num_nodes',), units='rad',
                             desc='Azimuth angle of the Sun in the body-fixed frame over time.'),
        'elevation': OutputVar(shape=('num_nodes',), units='rad',
                               desc='Elevation angle of the Sun in the body-fixed frame '
                                    'over time.'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        x, y, z = inputs['r_e2s_B']
        r = jnp.sqrt(x**2 + y**2 + z**2)
        outputs['azimuth'] = jnp.mod(jnp.arctan2(y, x), 2 * np.pi)
        outputs['elevation'] = jnp.arccos(z / r)
