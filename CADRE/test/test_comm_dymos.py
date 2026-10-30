import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from om4.api import Group, Problem
from dymos4.grid_data import GaussLobattoGrid
from dymos4.options import State, Time
from dymos4.phase.phase import Phase
from dymos4.transcriptions.picard import PicardShooting

from CADRE.comm_dymos import (CommAntRotationComp, CommAntRotationMtxComp, CommBitRateComp,
                              CommDistanceComp, CommEarthsSpinComp, CommEarthsSpinMtxComp,
                              CommGSPosECIComp, CommGSPosEarthComp, CommLOSComp,
                              CommVectorAntComp, CommVectorBodyComp, CommVectorECIComp,
                              CommVectorSphericalComp, GainPatternGroup, Re)
from CADRE.test.test_tables import CASE, LEGACY, assert_close_to_range
from CADRE.testing_utils import assert_partials_ok, wrap

N = 4


def _run(comp, vals):
    p = wrap(comp)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


def _rotations(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([Rotation.from_rotvec(rng.normal(size=3)).as_matrix() for _ in range(n)],
                    axis=-1)


def _legacy_quat_to_mtx(q):
    """The legacy quaternion-to-matrix (A^T B), at one node."""
    A = np.array([[q[0], -q[3], q[2]], [q[3], q[0], -q[1]], [-q[2], q[1], q[0]],
                  [q[1], q[2], q[3]]])
    B = np.array([[q[0], q[3], -q[2]], [-q[3], q[0], q[1]], [q[2], -q[1], q[0]],
                  [q[1], q[2], q[3]]])
    return A.T @ B


class TestGroundStation(unittest.TestCase):

    def test_position_earth(self):
        p = _run(CommGSPosEarthComp(num_nodes=2), {'lat': np.pi / 2, 'lon': 0.3, 'alt': 1.0})
        assert_allclose(p.get_val('r_e2g_E'), [[0, 0], [0, 0], [Re + 1, Re + 1]], atol=1e-9)
        p = _run(CommGSPosEarthComp(num_nodes=2), {'lat': 0.4, 'lon': -1.2, 'alt': 0.3})
        r = p.get_val('r_e2g_E')
        assert_allclose(np.linalg.norm(r, axis=0), Re + 0.3)
        assert_allclose(np.arctan2(r[1], r[0]), -1.2)
        assert_partials_ok(p.check_partials(), tol=1e-5)

    def test_earth_spin(self):
        t = np.array([0.0, 6.0, 12.0, 24.0]) * 3600.
        p = _run(CommEarthsSpinComp(num_nodes=N), {'t': t})
        q_E = p.get_val('q_E')
        assert_partials_ok(p.check_partials())

        p = _run(CommEarthsSpinMtxComp(num_nodes=N), {'q_E': q_E})
        O_IE = p.get_val('O_IE')
        for i in range(N):
            assert_allclose(O_IE[..., i], _legacy_quat_to_mtx(q_E[:, i]).T, atol=1e-15)
        # Quadratic in q, which a forward FD reference resolves only to its step size.
        assert_partials_ok(p.check_partials(differentiator='cs'))

        # A station on the prime meridian turns eastward, a quarter turn every 6 hours.
        r_E = np.tile([[Re], [0.], [0.]], (1, N))
        p = _run(CommGSPosECIComp(num_nodes=N), {'O_IE': O_IE, 'r_e2g_E': r_E})
        assert_allclose(p.get_val('r_e2g_I'),
                        Re * np.array([[1, 0, -1, 1], [0, 1, 0, 0], [0, 0, 0, 0]]), atol=1e-9)
        assert_partials_ok(p.check_partials())


class TestVectors(unittest.TestCase):

    def test_vector_eci(self):
        rng = np.random.default_rng(1)
        r_g, r_b = 7000 * rng.normal(size=(3, N)), 7000 * rng.normal(size=(3, N))
        p = _run(CommVectorECIComp(num_nodes=N), {'r_e2g_I': r_g, 'r_e2b_I': r_b})
        assert_allclose(p.get_val('r_b2g_I'), r_g - r_b)
        assert_partials_ok(p.check_partials())

    def test_vector_body(self):
        O_BI = _rotations()
        r_I = np.random.default_rng(2).normal(size=(3, N))
        p = _run(CommVectorBodyComp(num_nodes=N), {'O_BI': O_BI, 'r_b2g_I': r_I})
        for i in range(N):
            assert_allclose(O_BI[..., i] @ p.get_val('r_b2g_B')[:, i], r_I[:, i])
        assert_partials_ok(p.check_partials())

    def test_antenna(self):
        angle = 0.7
        p = _run(CommAntRotationComp(num_nodes=N), {'antAngle': angle})
        q_A = p.get_val('q_A')
        assert_partials_ok(p.check_partials())

        p = _run(CommAntRotationMtxComp(num_nodes=N), {'q_A': q_A})
        O_AB = p.get_val('O_AB')
        axis = np.array([1., -1., 0.]) / np.sqrt(2)
        expected = Rotation.from_rotvec(angle * axis).as_matrix()
        for i in range(N):
            assert_allclose(O_AB[..., i], expected, atol=1e-15)
            assert_allclose(O_AB[..., i], _legacy_quat_to_mtx(q_A[:, i]).T, atol=1e-15)
        assert_partials_ok(p.check_partials(differentiator='cs'))

        r_B = np.random.default_rng(3).normal(size=(3, N))
        p = _run(CommVectorAntComp(num_nodes=N), {'O_AB': O_AB, 'r_b2g_B': r_B})
        for i in range(N):
            assert_allclose(O_AB[..., i] @ p.get_val('r_b2g_A')[:, i], r_B[:, i])
        assert_partials_ok(p.check_partials())

    def test_spherical_and_distance(self):
        r = np.array([[100., 0., 0.], [0., -50., 0.], [0., 0., 30.], [3., 4., 0.]]).T
        p = _run(CommVectorSphericalComp(num_nodes=N), {'r_b2g_A': r})
        assert_allclose(p.get_val('azimuthGS'), [0., 3 * np.pi / 2, 0., np.arctan2(4, 3)])
        assert_allclose(p.get_val('elevationGS'), [np.pi / 2, np.pi / 2, 0., np.pi / 2])
        p = _run(CommDistanceComp(num_nodes=N), {'r_b2g_A': r})
        assert_allclose(p.get_val('GSdist'), [100., 50., 30., 5.])
        assert_partials_ok(p.check_partials())

        r = np.array([[100., 20., 10.], [-30., -50., 40.], [10., 5., 30.], [3., 4., 2.]]).T
        p = _run(CommVectorSphericalComp(num_nodes=N), {'r_b2g_A': r})
        assert_partials_ok(p.check_partials(), tol=1e-5)


class TestCommLOS(unittest.TestCase):

    def test_regions_and_partials(self):
        r_g = np.tile([[Re], [0.], [0.]], (1, N))
        # Satellite below the horizon, far above it, and in the 100 km blend band.
        r_b = np.array([[Re - 10, 0, 0], [Re + 500, 0, 0], [Re + 50, 0, 0], [Re + 20, 300, 0]]).T
        p = _run(CommLOSComp(num_nodes=N), {'r_b2g_I': r_g - r_b, 'r_e2g_I': r_g})
        los = p.get_val('CommLOS')
        assert_allclose(los[:3], [0.0, 1.0, 0.5])
        x = 0.2
        assert_allclose(los[3], 3 * x**2 - 2 * x**3)
        assert_partials_ok(p.check_partials(), tol=1e-5)


class TestBitRate(unittest.TestCase):

    def test_values_and_partials(self):
        comp = CommBitRateComp(num_nodes=N)
        vals = {'P_comm': np.array([1., 2., 3., 4.]), 'gain': np.array([1., 2., 0.5, 3.]),
                'GSdist': np.array([500., 1000., 2000., 800.]),
                'CommLOS': np.array([1., 1., 0.5, 0.])}
        p = _run(comp, vals)
        assert_allclose(p.get_val('Dr'), comp.alpha * vals['P_comm'] * vals['gain'] *
                        vals['CommLOS'] / (vals['GSdist'] * 1e3)**2)
        assert_partials_ok(p.check_partials(differentiator='cs'))

    def test_data_downloaded(self):
        """With a constant link, the data downloaded grows linearly."""
        duration = 600.0
        phase = Phase(
            ode=CommBitRateComp(),
            transcription=PicardShooting(grid_data=GaussLobattoGrid(num_segments=2,
                                                                    nodes_per_seg=5)),
            time=Time(units='s', opt_initial=False, opt_duration=False, duration_val=duration),
            states={'Data': State(opt_initial=False)},
        )
        prob = Problem(model=phase)
        phase.set_time_val(t_initial=0.0, t_duration=duration)
        phase.set_state_val('Data', 0.0)
        for name, val in (('P_comm', 2.0), ('gain', 1.5), ('GSdist', 900.), ('CommLOS', 1.0)):
            prob.set_val(f'ode.{name}', val)
        prob.run_model()

        rate = CommBitRateComp().alpha * 2.0 * 1.5 / (900e3)**2
        assert_allclose(prob.get_val('states_Data'), rate * prob.get_val('t'), rtol=1e-12)


class TestCommGeometryChain(unittest.TestCase):

    def test_overhead_pass(self):
        """A satellite 500 km straight above the station, at t = 0, is in view 500 km away."""
        prom = {'promotes_inputs': ['*'], 'promotes_outputs': ['*']}
        model = Group(subsystems={
            'gs_earth': CommGSPosEarthComp(num_nodes=1, **prom),
            'spin': CommEarthsSpinComp(num_nodes=1, **prom),
            'spin_mtx': CommEarthsSpinMtxComp(num_nodes=1, **prom),
            'gs_eci': CommGSPosECIComp(num_nodes=1, **prom),
            'vec_eci': CommVectorECIComp(num_nodes=1, **prom),
            'vec_body': CommVectorBodyComp(num_nodes=1, **prom),
            'ant': CommAntRotationComp(num_nodes=1, **prom),
            'ant_mtx': CommAntRotationMtxComp(num_nodes=1, **prom),
            'vec_ant': CommVectorAntComp(num_nodes=1, **prom),
            'dist': CommDistanceComp(num_nodes=1, **prom),
            'los': CommLOSComp(num_nodes=1, **prom),
        })
        p = Problem(model=model)
        for name, val in (('lat', 0.0), ('lon', 0.0), ('alt', 0.0), ('t', 0.0),
                          ('antAngle', 0.3), ('r_e2b_I', [[Re + 500.], [0.], [0.]]),
                          ('O_BI', np.eye(3)[..., np.newaxis])):
            p.set_val(name, val)
        p.run_model()
        assert_allclose(p.get_val('GSdist'), [500.])
        assert_allclose(p.get_val('CommLOS'), [1.0])



class TestGainPatternGroup(unittest.TestCase):

    def _run_legacy(self):
        nodes = slice(0, None, 100)
        vals = {k: LEGACY[CASE + k][nodes] for k in ('azimuthGS', 'elevationGS')}
        return _run(GainPatternGroup(num_nodes=vals['azimuthGS'].size), vals), nodes

    def test_matches_legacy(self):
        p, nodes = self._run_legacy()
        assert_close_to_range(p.get_val('gain'), LEGACY[CASE + 'gain'][nodes], 5e-4)

    def test_partials(self):
        assert_partials_ok(self._run_legacy()[0].check_partials(differentiator='cs'))


if __name__ == '__main__':
    unittest.main()
