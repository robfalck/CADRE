import unittest

import numpy as np
from numpy.testing import assert_allclose
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

from om4.api import Group, Problem
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic
from dymos4.grid_data import GaussLobattoGrid
from dymos4.options import State, Time
from dymos4.phase.phase import Phase
from dymos4.transcriptions.picard import PicardShooting

from CADRE.orbital_equations.EOMs import TwoBodyDynamicsComp, TwoBodyEOM
from CADRE.orbital_equations.frame_conversions import MEEToCart

MEE = ('p', 'f', 'g', 'h', 'k', 'L')

#: The semi-latus rectum of a circular orbit with an 8 TU period.
P_CIRC = (4 / np.pi)**(2 / 3)


class TwoBodyODE(Group):
    """Unperturbed two-body motion in MEE, with the Cartesian state as an output."""
    num_nodes: int = Field(default=1)
    central_body: str = Field(default='earth')

    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode='before')
    @classmethod
    def _build_from_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        nn = data.get('num_nodes', 1)
        cb = data.get('central_body', 'earth')
        data['subsystems'] = {
            'eom': TwoBodyDynamicsComp(num_nodes=nn, central_body=cb,
                                       promotes_inputs=['*'], promotes_outputs=['*']),
            'mee_to_cart': MEEToCart(num_nodes=nn, central_body=cb,
                                     promotes_inputs=['*'], promotes_outputs=['*']),
        }
        return data


def _orbit_phase(ode, num_segments=8, nodes_per_seg=8):
    return Phase(
        ode=ode,
        transcription=PicardShooting(grid_data=GaussLobattoGrid(num_segments=num_segments,
                                                                nodes_per_seg=nodes_per_seg)),
        time=Time(units='TU_earth', opt_initial=False, opt_duration=False, duration_val=8.0),
        states={name: State(opt_initial=False) for name in MEE},
    )


def _set_circular_orbit(phase):
    phase.set_time_val(t_initial=0.0, t_duration=8.0)
    phase.set_state_val('p', P_CIRC)
    for name in ('f', 'g', 'h', 'k'):
        phase.set_state_val(name, 0.0)
    phase.set_state_val('L', [0.0, 2 * np.pi])


class TestOrbitProp(unittest.TestCase):

    def test_propagation(self):
        phase = _orbit_phase(TwoBodyODE())
        prob = Problem(model=phase)
        _set_circular_orbit(phase)
        prob.run_model()

        p = prob.get_val('states_p')
        L = prob.get_val('states_L')
        x = prob.get_val('ode.x')
        y = prob.get_val('ode.y')
        z = prob.get_val('ode.z')
        vx = prob.get_val('ode.vx')
        vy = prob.get_val('ode.vy')
        vz = prob.get_val('ode.vz')

        assert_allclose(p, P_CIRC * np.ones_like(p), rtol=1e-6)
        for name in ('f', 'g', 'h', 'k'):
            assert_allclose(prob.get_val(f'states_{name}'), np.zeros_like(p), atol=1e-6)
        assert_allclose(L[-1], 2 * np.pi, rtol=1e-6)

        # One period later the spacecraft is back where it started.
        for c in (x, y, z, vx, vy, vz):
            assert_allclose(c[-1], c[0], atol=1e-6)

    def test_j2_propagation(self):
        """
        J2 on an equatorial orbit is purely radial.

        So p (driven only by a_s) and the plane (h, k, driven only by a_w) do not change, while
        the extra inward pull speeds the spacecraft up.
        """
        phase = _orbit_phase(TwoBodyEOM())
        prob = Problem(model=phase)
        _set_circular_orbit(phase)
        prob.run_model()

        p = prob.get_val('states_p')
        assert_allclose(p, P_CIRC * np.ones_like(p), rtol=1e-8)
        for name in ('h', 'k'):
            assert_allclose(prob.get_val(f'states_{name}'), np.zeros_like(p), atol=1e-12)
        # J2 speeds up an equatorial orbit, so it completes the circuit slightly early.
        self.assertGreater(prob.get_val('states_L')[-1], 2 * np.pi)


if __name__ == '__main__':
    unittest.main()
