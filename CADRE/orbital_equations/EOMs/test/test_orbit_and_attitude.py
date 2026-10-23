"""
Orbit propagation with a prescribed pointing law and the reaction wheel it requires.

The attitude is a quaternion control, so the transcription supplies its first and second time
derivatives, from which the body rates, accelerations and the torque required follow pointwise.
Pointing is not imposed on the quaternion directly: path constraints hold the body z axis along
the rolled frame's (anti-velocity) axis and keep the quaternion unit-norm. The roll about that
axis is left to the optimizer, which here minimizes the reaction wheel energy.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined
from scipy.spatial.transform import Rotation

from om4.api import Connection, Group, InputVar, JaxExplicitComp, OutputVar, Problem
from om4.api import PyOptSparseOptimizer
from om4.core.system import System
from om4.derivatives.coloring import ColoringSpec
from om4.utils.polymorphic import Polymorphic
from dymos4.grid_data import GaussLobattoGrid
from dymos4.options import Control, ObjectiveSpec, PathConstraint, State, Time
from dymos4.phase.phase import Phase
from dymos4.transcriptions.picard import PicardShooting

from CADRE.attitude_dymos import (AngularAccelerationComp, AngularVelocityComp,
                                  AttitudeTorqueComp, BodyVelComp, ORIComp, PointingComp,
                                  QuaternionAttitudeComp)
from CADRE.attitude_dymos.ori_comp import _compute_O_RI
from CADRE.orbital_equations import DU_earth_m, TU_earth_s
from CADRE.orbital_equations.EOMs import TwoBodyDynamicsComp
from CADRE.orbital_equations.frame_conversions import MEEToCart, StateMuxComp
from CADRE.orbital_equations.frame_conversions.mee_to_cart import mee_to_cart
from CADRE.reactionwheel_dymos import ReactionWheel
from CADRE.testing_utils import assert_partials_ok, wrap

MEE = ('p', 'f', 'g', 'h', 'k', 'L')

#: The semi-latus rectum of a circular orbit with an 8 TU period.
P_CIRC = (4 / np.pi)**(2 / 3)


class WheelEnergyRateComp(JaxExplicitComp):
    """The total reaction wheel power, as the rate of the wheel energy state ``E``."""
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'P_RW': InputVar(shape=(3, 'num_nodes'), units='W'),
    }
    outputs: dict[str, OutputVar] = {
        'E_dot': OutputVar(shape=('num_nodes',), units='W',
                           tags={'dymos.state_rate_source:E', 'dymos.state_units:J'}),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['E_dot'] = jnp.sum(inputs['P_RW'], axis=0)


class OrbitAttitudeODE(Group):
    num_nodes: int = Field(default=1)
    central_body: str = Field(default='earth')

    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    connections: list[Connection] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode='before')
    @classmethod
    def _build_from_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        nn = data.get('num_nodes', 1)
        cb = data.get('central_body', 'earth')
        prom = {'promotes_inputs': ['*'], 'promotes_outputs': ['*']}
        data['subsystems'] = {
            # Orbit propagation
            'eom': TwoBodyDynamicsComp(num_nodes=nn, central_body=cb, **prom),
            'mee_to_cart': MEEToCart(num_nodes=nn, central_body=cb, **prom),
            'state_mux_comp': StateMuxComp(num_nodes=nn, **prom),
            # Attitude
            'ori_comp': ORIComp(num_nodes=nn, **prom),
            'attitude_comp': QuaternionAttitudeComp(num_nodes=nn, **prom),
            'pointing_comp': PointingComp(num_nodes=nn, **prom),
            'omega_comp': AngularVelocityComp(num_nodes=nn, **prom),
            'omega_dot_comp': AngularAccelerationComp(num_nodes=nn, **prom),
            'body_vel_comp': BodyVelComp(num_nodes=nn, **prom),
            'torque_req_comp': AttitudeTorqueComp(num_nodes=nn, **prom),
            # Reaction wheel
            'reactionwheel': ReactionWheel(num_nodes=nn, **prom),
            'wheel_energy': WheelEnergyRateComp(num_nodes=nn, **prom),
        }
        data['connections'] = [Connection(src='T_req', tgt='T_RW')]
        return data


def _quaternion_from_dcm(O):
    """Scalar-first quaternions for body-to-inertial matrices O (3, 3, n), kept sign-continuous."""
    q_xyzw = Rotation.from_matrix(np.moveaxis(O, -1, 0)).as_quat()
    q = np.vstack((q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]))
    for i in range(1, q.shape[1]):
        if np.dot(q[:, i], q[:, i - 1]) < 0:
            q[:, i] = -q[:, i]
    return q


def pointing_guess(num_samples=33):
    """
    Return the zero-roll pointing quaternion on the circular orbit, sampled in phase fraction.
    """
    frac = np.linspace(0, 1, num_samples)
    z = np.zeros_like(frac)
    cart = np.asarray(mee_to_cart(np.full_like(frac, P_CIRC), z, z, z, z, 2 * np.pi * frac))
    r = cart[:3] * DU_earth_m / 1000.
    v = cart[3:] * DU_earth_m / TU_earth_s / 1000.
    return frac, _quaternion_from_dcm(np.asarray(_compute_O_RI(r, v)))


def build_problem(num_segments=8, nodes_per_seg=8):
    phase = Phase(
        ode=OrbitAttitudeODE(),
        transcription=PicardShooting(grid_data=GaussLobattoGrid(num_segments=num_segments,
                                                                nodes_per_seg=nodes_per_seg)),
        time=Time(units='TU_earth', opt_initial=False, opt_duration=False, duration_val=8.0),
        states={
            **{name: State(opt_initial=False) for name in MEE},
            'w_RW': State(opt_initial=False),
            'E': State(opt_initial=False),
        },
        controls={'q': Control(units='unitless', shape=(4,), lower=-1.1, upper=1.1)},
        path_constraints=[PathConstraint(name='ode.pointing_err', equals=0.0),
                          PathConstraint(name='ode.q_norm_err', equals=0.0)],
        objective=ObjectiveSpec(name='states_E', loc='final'),
    )
    phase.optimizer = PyOptSparseOptimizer(
        optimizer='IPOPT', print_results=False,
        opt_settings={'print_level': 0, 'tol': 1.0e-6, 'linear_solver': 'mumps',
                      'hessian_approximation': 'limited-memory'})
    phase.coloring = ColoringSpec()

    prob = Problem(model=phase)

    phase.set_time_val(t_initial=0.0, t_duration=8.0)
    phase.set_state_val('p', P_CIRC)
    for name in ('f', 'g', 'h', 'k'):
        phase.set_state_val(name, 0.0)
    phase.set_state_val('L', [0.0, 2 * np.pi])
    phase.set_state_val('w_RW', [[0, 0, 0], [0, 0, 0]])
    phase.set_state_val('E', 0.0)

    frac, q = pointing_guess()
    phase.set_control_val('q', q.T, time_vals=frac)
    return prob


class TestOrbitAttitudeODE(unittest.TestCase):

    def test_partials(self):
        n = 4
        frac = np.linspace(0, 0.9, n)
        _, q = pointing_guess(n)
        prob = wrap(OrbitAttitudeODE(num_nodes=n))
        prob.set_val('p', np.full(n, P_CIRC))
        for name in ('f', 'g', 'h', 'k'):
            prob.set_val(name, np.full(n, 0.01))
        prob.set_val('L', 2 * np.pi * frac)
        prob.set_val('q', q)
        prob.set_val('q_rate', 1e-3 * np.ones((4, n)))
        prob.set_val('q_rate2', 1e-6 * np.ones((4, n)))
        prob.set_val('w_RW', 10 * np.ones((3, n)))
        prob.run_model()
        assert_partials_ok(prob.check_partials(differentiator='cs'))


class TestOrbitAndAttitude(unittest.TestCase):

    # Blocked: the rate units of the unitless orbital elements and of the unitless quaternion
    # control are composed as 'unitless/TU_earth', which om4 will not connect to '1/TU_earth'
    # or '1/s'. See ai/OM4_NEEDS.md #1 and ai/DYMOS4_NEEDS.md #1.
    @unittest.expectedFailure
    def test_propagation(self):
        prob = build_problem()
        result = prob.run_optimizer()
        self.assertTrue(result.converged)

        p = prob.get_val('states_p')
        L = prob.get_val('states_L')
        assert_allclose(p, P_CIRC * np.ones_like(p), rtol=1e-6)
        assert_allclose(L[-1], 2 * np.pi, rtol=1e-6)
        for c in ('x', 'y', 'z', 'vx', 'vy', 'vz'):
            val = prob.get_val(f'ode.{c}')
            assert_allclose(val[-1], val[0], atol=1e-6)

        assert_allclose(prob.get_val('ode.pointing_err'), 0.0, atol=1e-6)
        assert_allclose(prob.get_val('ode.q_norm_err'), 0.0, atol=1e-6)

        # The body rotates once per orbit about the orbit normal, which lies in the body x-y
        # plane, where the inertia is isotropic, so no gyroscopic torque is required.
        n_orbit = 2 * np.pi / (8.0 * TU_earth_s)
        w_B = prob.get_val('ode.w_B')
        assert_allclose(np.linalg.norm(w_B, axis=0), n_orbit, rtol=1e-3)
        # The orbit is prograde equatorial, so w_B is the inertial +z axis seen from the body.
        z_I_in_B = np.einsum('jin,j->in', prob.get_val('ode.O_BI'), np.array([0., 0., 1.]))
        assert_allclose(w_B, n_orbit * z_I_in_B, rtol=1e-3, atol=1e-9)
        assert_allclose(prob.get_val('ode.T_req'), 0.0, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
