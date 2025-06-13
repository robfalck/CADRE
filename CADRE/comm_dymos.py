"""
Communications Discpline for CADRE
"""

import os
from six.moves import range

import openmdao.api as om
import jax
import jax.numpy as jnp
import interpax

import numpy as np
import scipy.sparse
from MBI import MBI

from CADRE.explicit import ExplicitComponent

from CADRE.kinematics import fixangles, computepositionspherical, \
    computepositionsphericaljacobian, computepositionrotd,\
    computepositionrotdjacobian

from CADRE import rk4

rt2 = jnp.sqrt(2)


def _single_q_A(ant_angle):
    """ Compute the instantaneous antenna quaternion based on its angle. """
    aao2 = ant_angle / 2.0
    saao2 = jnp.sin(aao2)
    caao2 = jnp.cos(aao2)
    return jnp.array([caao2, saao2 / rt2, -saao2 / rt2, 0.0])


_compute_q_A = jax.vmap(_single_q_A)


def _single_O_AB(q_A):
    """ Compute the instantaneous coordinate transformation matrix from the antenna frame to the body frame. """
    A = jnp.array([[ q_A[0],-q_A[3], q_A[2]],
                   [ q_A[3], q_A[0],-q_A[1]],
                   [-q_A[2], q_A[1], q_A[0]],
                   [ q_A[1], q_A[2], q_A[3]]])

    B = jnp.array([[ q_A[0], q_A[3],-q_A[2]],
                   [-q_A[3], q_A[0], q_A[1]],
                   [ q_A[2],-q_A[1], q_A[0]],
                   [ q_A[1], q_A[2], q_A[3]]])

    return jnp.dot(A.T, B)


_compute_O_AB = jax.vmap(_single_O_AB, in_axes=(0,))


def _single_O_IE(t, gha0, w_earth):
    """
    Compute the instantaneous transformation matrix from ECI to ECEF.

    Parameters
    ----------
    t : float
        Current simulation time (s).
    gha0 : float
        Initial Greenwich hour angle (rad).
    w_earth : float
        Earth (or central body) rotation rate (rad/s).
    """
    gha = gha0 + w_earth * t

    c = jnp.cos(gha)
    s = jnp.sin(gha)

    O_IE = jnp.array([[ c, s, 0],
                      [-s, c, 0],
                      [ 0, 0, 1]])
    return O_IE


_compute_O_IE = jax.vmap(_single_O_IE, in_axes=(0, None, None))


def _single_data_rate(P_comm, gain, gs_dist, comm_los, alpha):
    """
    Compute the instantaneous data download rate.

    Parameters
    ----------
    P_comm : float
        Transmitter input power.
    gain : float
        Transmitter gain
    gs_dist : float
        Distance to ground receiver.
    comm_los : float
        Line-of-sight indicator to ground receiver.
    alpha : float

    Returns
    -------
    data_rate : float
        The data download rate at the current state.
    """
    return alpha * P_comm * gain * comm_los / gs_dist ** 2


_compute_data_rate = jax.vmap(_single_data_rate, in_axes=(0, 0, 0, 0, None))


def _single_gs_distance(r_b2g_A):
    """
    Compute the instantatneous distance from the vehicle to the ground station.

    Parameters
    ----------
    r_b2g_A : array-like
        The cartesian position vector from the body to the ground in the antenna frame.
    """
    return jnp.linalg.norm(r_b2g_A)


_compute_gs_distance = jax.vmap(_single_gs_distance, in_axes=(0,))

### For comm gain, we can just use interpax
_rawG_file = os.path.dirname(os.path.realpath(__file__)) + '/data/Comm/Gain.txt'
_rawGdata = jnp.genfromtxt(_rawG_file)
_az_data = np.linspace(0, 2 * jnp.pi, 361)
_el_data = np.linspace(0, 2 * jnp.pi, 361)
_rawG = (10 ** (_rawGdata / 10.0)).reshape((361, 361), order='F')

_compute_comm_gain = interpax.interp2d(_az_data, _el_data, _rawG, kind='cubic')


def _single_gs_loc(O_IE, lat, lon, alt, r_cb):
    """ Compute the location of the ground station in the ECI frame.

    This assumes a spherical Earth. (No difference between geodetic and geocentric coordinates)

    Parameters
    ----------
    O_IE : array-like
        A 3 x 3 transformation matrix from ECI to ECEF
    lat : float
        Latitude (rad)
    long : float
        Longitude (rad)
    alt : float
        Altitude (m)
    r_cb : float
        Central body radius (m)
    """
    clat = jnp.cos(lat)
    slat = jnp.sin(lat)
    clon = jnp.cos(lon)
    slon = jnp.sin(lon)

    r_GS = r_cb + alt

    r_gs_E = r_GS * jnp.array([[clat * clon, clat * slon, slat]])
    r_gs_I = jnp.einsum('ij,j->i', O_IE.T, r_gs_E)

    return r_gs_E, r_gs_I


_compute_gs_loc = jax.vmap(_single_gs_loc, in_axes=(0, None, None, None, None))


def _single_comm_los(r_b2g_I, r_e2g_I, r_cb, k=20):
    proj = jnp.dot(r_b2g_I, r_e2g_I) / r_cb
    return 0.5 * (1 - jnp.tanh(k * proj))


_compute_comm_los = jax.vmap(_single_comm_los, in_axes=(0, 0, None, None))


class CommAntQuaternionComp(om.JaxExplicitComponent):
    """
    Build quaternion to translate from antenna frame to body frame.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('ant_angle', 0.0, units='rad')
        self.add_output('q_A', np.zeros((nn, 4)), units='unitless',
                        desc='Quarternion for antenna angle frame over time')

    def compute_primal(self, ant_angle):
        q_A = _compute_q_A(ant_angle)
        return q_A


class CommAntRotationMatrixComp(om.JaxExplicitComponent):
    """
    Build O_AB to translate from antenna frame to body frame.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('q_A', np.zeros((nn, 4)), units='unitless',
                        desc='Quarternion for antenna angle frame over time')
        self.add_output('O_AB', np.zeros((nn, 3, 3)), units='unitless',
                        desc='Quarternion for antenna angle frame over time')

    def compute_primal(self, ant_angle):
        O_AB = _compute_O_AB(ant_angle)
        return O_AB


class CommDataRateComp(om.JaxExplicitComponent):
    """
    Compute the data rate the satellite receives.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('P_comm', shape=(nn,), units='W',
                       desc='Communication power over time')

        self.add_input('gain', shape=(nn,), units=None,
                       desc='Transmitter gain over time')

        self.add_input('gs_dist', shape=(nn,), units='m',
                       desc='Distance from ground station to satellite over time')

        self.add_input('comm_los', shape=(nn,), units=None,
                       desc='Satellite to ground station line of sight over time')

        # Outputs
        self.add_output('data_rate', shape=(nn,), units='Gibyte/s',
                        desc='Download rate over time')

        # Compute the efficiency of the transmitter.
        c = 299792458
        Gr = 10 ** (12.9 / 10.)
        Ll = 10 ** (-2.0 / 10.)
        f = 437e6
        k = 1.3806503e-23
        SNR = 10 ** (5.0 / 10.)
        T = 500.
        self._alpha = c ** 2 * Gr * Ll / 16.0 / jnp.pi ** 2 / f ** 2 / k / SNR / T / 1e6

    def get_self_statics(self):
        # return value must be hashable
        return self._alpha,

    def compute_primal(self, P_comm, gain, gs_dist, comm_los):
        data_rate = _compute_data_rate(P_comm, gain, gs_dist, comm_los, self._alpha)
        return data_rate


class CommGSDistanceComp(om.JaxExplicitComponent):
    """
    Build O_AB to translate from antenna frame to body frame.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('r_b2g_A', shape=(nn, 3), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')
        self.add_output('gs_dist', shape=(nn,), units='km',
                        desc='Distance from ground station to satellite over time')

    def compute_primal(self, r_b2g_A):
        gs_dist = _compute_gs_distance(r_b2g_A)
        return gs_dist


class OIEComp(om.JaxExplicitComponent):
    """
    Build O_IE to translate Earth-centered inertial
    frame (ECI) to Earth-centered Earth-fixed (ECEF).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('gha0', types=float, default=0.0,
                             desc='Greenwich hour angle at the initial time.')
        self.options.declare('w_earth', types=float, default=2 * jnp.pi / 86400.,
                             desc='Earth rotation rate (rad/s)')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('time', shape=(nn,), units='s',
                       desc='Current simulation time')
        self.add_output('O_IE', shape=(nn, 3, 3), units='unitless',
                        desc='ECI to ECEF transformation matrix.')

    def compute_primal(self, time):
        O_IE = _compute_O_IE(time)
        return O_IE


class CommGainComp(om.JaxExplicitComponent):
    """
    Compute the comm transmitter gain based on azimuth and elevation.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('azimuthGS', shape=(nn,), units='rad',
                       desc='Comm azimuth angle.')
        self.add_input('elevationGS', shape=(nn,), units='rad',
                       desc='Comm elevation angle.')
        self.add_output('gain', shape=(nn,), units='unitless',
                        desc='Transmitter gain.')

    def compute_primal(self, azimuthGS, elevationGS):
        gain= _compute_comm_gain(azimuthGS, elevationGS)
        return gain


class CommGSLocComp(om.JaxExplicitComponent):
    """
    Compute the psition of the ground station in the ECI frame.

    This assumes a spherical Earth (no difference between geodetic and geocentric altitude).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('lat', types=float, default=42.2708, desc='Ground station latitude (deg)')
        self.options.declare('lon', types=float, default=-83.7264, desc='Ground station longitude (deg)')
        self.options.declare('alt', types=float, default=256., desc='Ground station altitude above mean sea level (m).')
        self.options.declare('r_cb', types=float, default=6378.137E3, desc='Earth radius (m)')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('O_IE', shape=(nn, 3, 3), units='unitless',
                       desc='ECI to ECF transformation matrix')
        self.add_output('r_e2g_E', shape=(nn, 3), units='m',
                        desc='Ground station ECEF position vector.')
        self.add_output('r_e2g_I', shape=(nn, 3), units='m',
                        desc='Ground station ECI position vector.')

    def get_self_statics(self):
        # return value must be hashable
        return self.options['lat'], self.options['lon'], self.options['alt'], self.options['r_cb'],

    def compute_primal(self, O_IE):
        lat = self.options['lat']
        lon = self.options['lon']
        alt = self.options['alt']
        r_cb = self.options['r_cb']
        r_e2g_E, r_e2g_I = _compute_gs_loc(O_IE, lat, lon, alt, r_cb)
        return r_e2g_E, r_e2g_I


class CommLOSComp(om.JaxExplicitComp):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('k', types=float, default=100., desc='Sharpness factor for activation function.')
        self.options.declare('r_cb', types=float, default=6378.137E3, desc='Earth radius (m)')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('r_b2g_I', shape=(nn, 3), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('r_e2g_I', shape=(nn, 3), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('CommLOS', shape=(nn,), units='unitless',
                        desc='Satellite to ground station line of sight over time')

    def get_self_statics(self):
        # return value must be hashable
        return self.options['k'], self.options['r_cb'],

    def compute_primal(self, r_b2g_I, r_e2g_I):
        k = self.options['k']
        r_cb = self.options['r_cb']
        r_e2g_E, r_e2g_I = _compute_comm_los(r_b2g_I, r_e2g_I, r_cb, k)
        return r_e2g_E, r_e2g_I


class Comm_VectorAnt(ExplicitComponent):
    """
    Transform from antenna to body frame
    """
    def __init__(self, n):
        super(Comm_VectorAnt, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_B', np.zeros((3, n)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in body-fixed frame over time')

        self.add_input('O_AB', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from antenna angle to body-fixed '
                            'frame over time')

        # Outputs
        self.add_output('r_b2g_A', np.zeros((3, n)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in antenna angle frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_b2g_A'] = computepositionrotd(self.n, inputs['r_b2g_B'],
                                                 inputs['O_AB'])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.J1, self.J2 = computepositionrotdjacobian(self.n, inputs['r_b2g_B'],
                                                       inputs['O_AB'])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_b2g_A = d_outputs['r_b2g_A']

        if mode == 'fwd':
            for k in range(3):
                if 'O_AB' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            dr_b2g_A[k, :] += self.J1[:, k, u, v] * \
                                d_inputs['O_AB'][u, v, :]
                if 'r_b2g_B' in d_inputs:
                    for j in range(3):
                        dr_b2g_A[k, :] += self.J2[:, k, j] * \
                            d_inputs['r_b2g_B'][j, :]
        else:
            for k in range(3):
                if 'O_AB' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            d_inputs['O_AB'][u, v, :] += self.J1[:, k, u, v] * \
                                dr_b2g_A[k, :]
                if 'r_b2g_B' in d_inputs:
                    for j in range(3):
                        d_inputs['r_b2g_B'][j, :] += self.J2[:, k, j] * \
                            dr_b2g_A[k, :]


class Comm_VectorBody(ExplicitComponent):
    """
    Transform from body to inertial frame.
    """
    def __init__(self, n):
        super(Comm_VectorBody, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_I', np.zeros((3, n)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('O_BI', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered'
                            'inertial frame over time')

        # Outputs
        self.add_output('r_b2g_B', np.zeros((3, n)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in body-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_b2g_I = inputs['r_b2g_I']
        O_BI = inputs['O_BI']
        r_b2g_B = outputs['r_b2g_B']

        for i in range(0, self.n):
            r_b2g_B[:, i] = np.dot(O_BI[:, :, i], r_b2g_I[:, i])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_b2g_I = inputs['r_b2g_I']
        O_BI = inputs['O_BI']

        self.J1 = np.zeros((self.n, 3, 3, 3))

        for k in range(0, 3):
            for v in range(0, 3):
                self.J1[:, k, k, v] = r_b2g_I[v, :]

        self.J2 = np.transpose(O_BI, (2, 0, 1))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_b2g_B = d_outputs['r_b2g_B']

        if mode == 'fwd':
            for k in range(3):
                if 'O_BI' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            dr_b2g_B[k, :] += self.J1[:, k, u, v] * \
                                d_inputs['O_BI'][u, v, :]
                if 'r_b2g_I' in d_inputs:
                    for j in range(3):
                        dr_b2g_B[k, :] += self.J2[:, k, j] * \
                            d_inputs['r_b2g_I'][j, :]
        else:
            for k in range(3):
                if 'O_BI' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            d_inputs['O_BI'][u, v, :] += self.J1[:, k, u, v] * \
                                dr_b2g_B[k, :]
                if 'r_b2g_I' in d_inputs:
                    for j in range(3):
                        d_inputs['r_b2g_I'][j, :] += self.J2[:, k, j] * \
                            dr_b2g_B[k, :]


class Comm_VectorECI(ExplicitComponent):
    """
    Determine vector between satellite and ground station.
    """
    def __init__(self, n):
        super(Comm_VectorECI, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2g_I', np.zeros((3, n)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        self.add_input('r_e2b_I', np.zeros((6, n)), units=None,
                       desc='Position and velocity vector from earth to satellite '
                            'in Earth-centered inertial frame over time')

        # Outputs
        self.add_output('r_b2g_I', np.zeros((3, n)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in Earth-centered inertial frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_b2g_I'] = inputs['r_e2g_I'] - inputs['r_e2b_I'][:3, :]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_b2g_I = d_outputs['r_b2g_I']

        if mode == 'fwd':
            if 'r_e2g_I' in d_inputs:
                dr_b2g_I += d_inputs['r_e2g_I']
            if 'r_e2b_I' in d_inputs:
                dr_b2g_I += -d_inputs['r_e2b_I'][:3, :]

        else:
            if 'r_e2g_I' in d_inputs:
                d_inputs['r_e2g_I'] += dr_b2g_I
            if 'r_e2b_I' in d_inputs:
                dr_e2b_I = np.zeros(d_inputs['r_e2b_I'].shape)
                dr_e2b_I[:3, :] += -dr_b2g_I
                d_inputs['r_e2b_I'] += dr_e2b_I


class Comm_VectorSpherical(ExplicitComponent):
    """
    Convert satellite-ground vector into Az-El.
    """
    def __init__(self, n):
        super(Comm_VectorSpherical, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_A', np.zeros((3, n)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')

        # Outputs
        self.add_output('azimuthGS', np.zeros(n), units='rad',
                        desc='Azimuth angle from satellite to ground station in '
                             'Earth-fixed frame over time')

        self.add_output('elevationGS', np.zeros(n), units='rad',
                        desc='Elevation angle from satellite to ground station '
                             'in Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        azimuthGS, elevationGS = computepositionspherical(self.n, inputs['r_b2g_A'])
        outputs['azimuthGS'] = azimuthGS
        outputs['elevationGS'] = elevationGS

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.Ja1, self.Ji1, self.Jj1, self.Ja2, self.Ji2, self.Jj2 = \
            computepositionsphericaljacobian(self.n, 3 * self.n, inputs['r_b2g_A'])

        self.J1 = scipy.sparse.csc_matrix((self.Ja1, (self.Ji1, self.Jj1)),
                                          shape=(self.n, 3 * self.n))
        self.J2 = scipy.sparse.csc_matrix((self.Ja2, (self.Ji2, self.Jj2)),
                                          shape=(self.n, 3 * self.n))
        self.J1T = self.J1.transpose()
        self.J2T = self.J2.transpose()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 'r_b2g_A' in d_inputs:
                r_b2g_A = d_inputs['r_b2g_A'].reshape((3 * self.n), order='F')
                if 'azimuthGS' in d_outputs:
                    d_outputs['azimuthGS'] += self.J1.dot(r_b2g_A)
                if 'elevationGS' in d_outputs:
                    d_outputs['elevationGS'] += self.J2.dot(r_b2g_A)
        else:
            if 'r_b2g_A' in d_inputs:
                if 'azimuthGS' in d_outputs:
                    az_GS = d_outputs['azimuthGS']
                    d_inputs['r_b2g_A'] += (self.J1T.dot(az_GS)).reshape((3, self.n), order='F')
                if 'elevationGS' in d_outputs:
                    el_GS = d_outputs['elevationGS']
                    d_inputs['r_b2g_A'] += (self.J2T.dot(el_GS)).reshape((3, self.n), order='F')
