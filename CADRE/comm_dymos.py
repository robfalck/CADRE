"""
Communication discipline for CADRE, on om4.

Rotation matrices follow ``x_Y = O_XY x_X`` with column vectors (ai/ARCHITECTURE.md, ADL-005).
The legacy comm code, like its O_BI, built each matrix as the transpose of what its name says
under that convention (it applied ``r_e2g_I = O_IE r_e2g_E`` and ``r_b2g_A = O_AB r_b2g_B``).
Here the names keep their meaning and the matrices are transposed: O_IE takes inertial vectors to
Earth-fixed, O_AB takes antenna vectors to body.

The legacy RK4-integrated data downloaded (``Comm_DataDownloaded``) is a dymos4 state ``Data``
here, whose rate is the bit rate ``Dr`` (ADL-006).

Not ported yet: ``Comm_GainPattern``, which interpolates the antenna gain table and is blocked by
ai/OM4_NEEDS.md #3. ``CommBitRateComp`` takes the gain as an input.
"""
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from om4.api import InputVar, JaxExplicitComp, OutputVar

from CADRE.attitude_dymos.quaternion_attitude_comp import _dcm_from_quaternion
from CADRE.sun_dymos import azimuth_elevation

#: Earth's radius, km.
Re = 6378.137


def _static(units, desc, val=0.0):
    return InputVar(shape=(1,), val=val, units=units, desc=desc, tags={'dymos.static_target'})


class CommGSPosEarthComp(JaxExplicitComp):
    """
    Returns position of the ground station in Earth frame.

    The original declared the latitude and longitude in radians but converted them from degrees.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'lon': _static('rad', 'Longitude of ground station in Earth-fixed frame'),
        'lat': _static('rad', 'Latitude of ground station in Earth-fixed frame'),
        'alt': _static('km', 'Altitude of ground station in Earth-fixed frame'),
    }

    outputs: dict[str, OutputVar] = {
        'r_e2g_E': OutputVar(shape=(3, 'num_nodes'), units='km',
                             desc='Position vector from earth to ground station in '
                                  'Earth-fixed frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        lat = inputs['lat']
        lon = inputs['lon']
        r_GS = Re + inputs['alt']
        r = jnp.concatenate((r_GS * jnp.cos(lat) * jnp.cos(lon),
                             r_GS * jnp.cos(lat) * jnp.sin(lon),
                             r_GS * jnp.sin(lat)))
        outputs['r_e2g_E'] = jnp.broadcast_to(r[:, jnp.newaxis], (3, self.num_nodes))


class CommEarthsSpinComp(JaxExplicitComp):
    """
    Returns the Earth quaternion as a function of time.

    The Earth turns once per 24 hours (a solar day, as in the original).
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        't': InputVar(shape=('num_nodes',), val=0.0, units='s', desc='Time',
                      tags={'dymos.time_target'}),
    }

    outputs: dict[str, OutputVar] = {
        'q_E': OutputVar(shape=(4, 'num_nodes'), units='unitless',
                         desc='Quaternion matrix in Earth-fixed frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        theta = np.pi / 3600.0 / 24.0 * inputs['t']
        z = jnp.zeros_like(theta)
        outputs['q_E'] = jnp.stack((jnp.cos(theta), z, z, -jnp.sin(theta)))


class CommEarthsSpinMtxComp(JaxExplicitComp):
    """
    Quaternion to rotation matrix for the earth spin.

    O_IE takes inertial vectors into the Earth-fixed frame; it is the transpose of the original's.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'q_E': InputVar(shape=(4, 'num_nodes'), units='unitless',
                        desc='Quaternion matrix in Earth-fixed frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'O_IE': OutputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                          desc='Rotation matrix from Earth-centered inertial frame to '
                               'Earth-fixed frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['O_IE'] = _dcm_from_quaternion(inputs['q_E'])


class CommGSPosECIComp(JaxExplicitComp):
    """
    Convert time history of ground station position from earth frame to inertial frame.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'O_IE': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from Earth-centered inertial frame to '
                              'Earth-fixed frame over time'),
        'r_e2g_E': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from earth to ground station in '
                                 'Earth-fixed frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'r_e2g_I': OutputVar(shape=(3, 'num_nodes'), units='km',
                             desc='Position vector from earth to ground station in '
                                  'Earth-centered inertial frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['r_e2g_I'] = jnp.einsum('jin,jn->in', inputs['O_IE'], inputs['r_e2g_E'])


class CommVectorECIComp(JaxExplicitComp):
    """
    Determine vector between satellite and ground station.

    Takes the 3-vector position ``r_e2b_I`` rather than the original's 6-vector.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'r_e2g_I': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from earth to ground station in '
                                 'Earth-centered inertial frame over time'),
        'r_e2b_I': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from earth to satellite in Earth-centered '
                                 'inertial frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'r_b2g_I': OutputVar(shape=(3, 'num_nodes'), units='km',
                             desc='Position vector from satellite to ground station '
                                  'in Earth-centered inertial frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['r_b2g_I'] = inputs['r_e2g_I'] - inputs['r_e2b_I']


class CommVectorBodyComp(JaxExplicitComp):
    """
    Transform the satellite to ground station vector from the inertial to the body frame.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'r_b2g_I': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from satellite to ground station '
                                 'in Earth-centered inertial frame over time'),
        'O_BI': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from body-fixed frame to Earth-centered '
                              'inertial frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'r_b2g_B': OutputVar(shape=(3, 'num_nodes'), units='km',
                             desc='Position vector from satellite to ground station '
                                  'in body-fixed frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['r_b2g_B'] = jnp.einsum('jin,jn->in', inputs['O_BI'], inputs['r_b2g_I'])


class CommAntRotationComp(JaxExplicitComp):
    """
    Fixed antenna angle to time history of the quaternion.

    The antenna is rotated by antAngle about the body axis (1, -1, 0) / sqrt(2).
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'antAngle': _static('rad', 'Fixed antenna angle'),
    }

    outputs: dict[str, OutputVar] = {
        'q_A': OutputVar(shape=(4, 'num_nodes'), units='unitless',
                         desc='Quaternion matrix in antenna angle frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        half = inputs['antAngle'][0] / 2.
        rt2 = np.sqrt(2)
        q = jnp.stack((jnp.cos(half), jnp.sin(half) / rt2, -jnp.sin(half) / rt2, 0.0 * half))
        outputs['q_A'] = jnp.broadcast_to(q[:, jnp.newaxis], (4, self.num_nodes))


class CommAntRotationMtxComp(JaxExplicitComp):
    """
    Translate antenna angle into the body frame.

    O_AB takes antenna-frame vectors into the body frame; it is the transpose of the original's.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'q_A': InputVar(shape=(4, 'num_nodes'), units='unitless',
                        desc='Quaternion matrix in antenna angle frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'O_AB': OutputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                          desc='Rotation matrix from antenna angle to body-fixed frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['O_AB'] = _dcm_from_quaternion(inputs['q_A'])


class CommVectorAntComp(JaxExplicitComp):
    """
    Transform the satellite to ground station vector from the body to the antenna frame.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'r_b2g_B': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from satellite to ground station '
                                 'in body-fixed frame over time'),
        'O_AB': InputVar(shape=(3, 3, 'num_nodes'), units='unitless',
                         desc='Rotation matrix from antenna angle to body-fixed frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'r_b2g_A': OutputVar(shape=(3, 'num_nodes'), units='km',
                             desc='Position vector from satellite to ground station '
                                  'in antenna angle frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['r_b2g_A'] = jnp.einsum('jin,jn->in', inputs['O_AB'], inputs['r_b2g_B'])


class CommVectorSphericalComp(JaxExplicitComp):
    """
    Convert satellite-ground vector into Az-El, in the antenna frame.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'r_b2g_A': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from satellite to ground station '
                                 'in antenna angle frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'azimuthGS': OutputVar(shape=('num_nodes',), units='rad',
                               desc='Azimuth angle from satellite to ground station in the '
                                    'antenna frame over time'),
        'elevationGS': OutputVar(shape=('num_nodes',), units='rad',
                                 desc='Elevation angle from satellite to ground station in the '
                                      'antenna frame over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['azimuthGS'], outputs['elevationGS'] = azimuth_elevation(inputs['r_b2g_A'])


class CommDistanceComp(JaxExplicitComp):
    """
    Calculates distance from ground station to satellite.
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'r_b2g_A': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from satellite to ground station '
                                 'in antenna angle frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'GSdist': OutputVar(shape=('num_nodes',), units='km',
                            desc='Distance from ground station to satellite over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['GSdist'] = jnp.sqrt(jnp.sum(inputs['r_b2g_A']**2, axis=0))


class CommLOSComp(JaxExplicitComp):
    """
    Determines if the Satellite has line of sight with the ground stations.

    CommLOS is 0 while the satellite is below the ground station's horizon, 1 once it is more
    than Rb above it (measured as the projection of the satellite-to-station vector on the
    station's position, divided by Re), and blends between with a smoothstep.
    """
    num_nodes: int = Field(default=1)
    Rb: float = Field(default=100.0)

    inputs: dict[str, InputVar] = {
        'r_b2g_I': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from satellite to ground station '
                                 'in Earth-centered inertial frame over time'),
        'r_e2g_I': InputVar(shape=(3, 'num_nodes'), units='km',
                            desc='Position vector from earth to ground station in '
                                 'Earth-centered inertial frame over time'),
    }

    outputs: dict[str, OutputVar] = {
        'CommLOS': OutputVar(shape=('num_nodes',), units='unitless',
                             desc='Satellite to ground station line of sight over time'),
    }

    def get_self_statics(self):
        return {'Rb': self.Rb}

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        Rb = statics['Rb']
        proj = jnp.sum(inputs['r_b2g_I'] * inputs['r_e2g_I'], axis=0) / Re
        x = proj / -Rb
        blend = 3 * x ** 2 - 2 * x ** 3
        outputs['CommLOS'] = jnp.where(proj > 0, 0.0, jnp.where(proj < -Rb, 1.0, blend))


class CommBitRateComp(JaxExplicitComp):
    """
    Compute the data rate the satellite receives.

    The rate is tagged as the rate of the data downloaded, a dymos4 state ``Data``, which
    replaces the legacy ``Comm_DataDownloaded`` integrator (ADL-006).
    """
    # constants
    pi: float = 2 * np.arccos(0.)
    c: float = 299792458
    Gr: float = 10 ** (12.9 / 10.)
    Ll: float = 10 ** (-2.0 / 10.)
    f: float = 437e6
    k: float = 1.3806503e-23
    SNR: float = 10 ** (5.0 / 10.)
    T: float = 500.

    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'P_comm': InputVar(shape=('num_nodes',), val=0.0, units='W',
                           desc='Communication power over time'),
        'gain': InputVar(shape=('num_nodes',), val=0.0, units='unitless',
                         desc='Transmitter gain over time'),
        'GSdist': InputVar(shape=('num_nodes',), val=0.0, units='km',
                           desc='Distance from ground station to satellite over time'),
        'CommLOS': InputVar(shape=('num_nodes',), val=0.0, units='unitless',
                            desc='Satellite to ground station line of sight over time'),
    }

    outputs: dict[str, OutputVar] = {
        'Dr': OutputVar(shape=('num_nodes',), units='Gibyte/s', desc='Download rate over time',
                        tags={'dymos.state_rate_source:Data', 'dymos.state_units:Gibyte'}),
    }

    @property
    def alpha(self):
        return (self.c ** 2 * self.Gr * self.Ll / 16.0 / self.pi ** 2 / self.f ** 2 / self.k /
                self.SNR / self.T / 1e6)

    def get_self_statics(self):
        return {'alpha': self.alpha}

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        GSdist = inputs['GSdist']
        S2 = jnp.where(jnp.abs(GSdist) > 1e-10, GSdist * 1e3, 1e-10)
        outputs['Dr'] = (statics['alpha'] * inputs['P_comm'] * inputs['gain'] *
                         inputs['CommLOS'] / S2 ** 2)
