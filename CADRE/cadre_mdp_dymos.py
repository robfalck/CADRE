"""
The CADRE multidisciplinary design problem (MDP) on dymos4.

This replaces the legacy ``CADRE_MDP_Group`` (``CADRE/CADRE_mdp.py``). Each design point is a
12-hour ``Phase`` of ``CADREODE`` starting from a launch condition in
``data/Launch/launch1.dat``, transcribed by Radau collocation. The optimizer chooses the
controls (attitude ``q``, panel currents ``Isetpt``, comm power ``P_comm``), each point's
initial state of charge, and the design parameters shared by all points (``cellInstd``,
``finAngle``, ``antAngle``); the objective is the total data downloaded. With
``solve_segments=True`` each segment's states are solved by Newton instead, which evaluates a
given design (a simulation) rather than optimizing it.

Picard shooting is not used: the reaction wheel speeds rotate at the body rate
(``alpha_RW`` has ``-w_B x w_RW``), about 6e-3 rad/s, and plain Picard iteration diverges on
segments longer than about 170 s (ai/DYMOS4_NEEDS.md #3 also blocks it).

Differences from the legacy MDP:

- The controls are dymos4 controls, with a value at every node, where the legacy model
  shaped each time history with an m-point B-spline (``BsplineParameters``). dymos4 has no
  B-spline control (ai/DYMOS4_NEEDS.md #3).
- The attitude is the quaternion ``q`` with pointing path constraints, which subsumes the
  legacy roll angle ``gamma`` (ADL-002).
- The battery limits are path constraints rather than KS aggregates (ADL-006).
- The orbit has J2 only; the legacy model also had J3 and J4, which move the orbit by up to
  3 km in 12 hours.
"""
import os

import numpy as np
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from om4.api import (Connection, Constraint, DesignVar, Group, InputVar, JaxExplicitComp,
                     Objective, OutputVar)
from om4.core.system import System
from om4.transforms import AffineTransform
from om4.utils.polymorphic import Polymorphic
from om4.utils.promotion import normalize_promotes_list
from dymos4.grid_data import BirkhoffGrid, RadauGrid
from dymos4.options import Control, ObjectiveSpec, Parameter, PathConstraint, State, Time
from dymos4.phase.phase import Phase
from dymos4.transcriptions.birkhoff import Birkhoff
from dymos4.transcriptions.radau import Radau

from CADRE.attitude_dymos.obr_comp import _compute_O_BR
from CADRE.attitude_dymos.ori_comp import _compute_O_RI
from CADRE.battery_dymos import battery_path_constraints
from CADRE.cadre_ode import CADREODE
from CADRE.orbital_equations import DU_earth_m, TU_earth_s
from CADRE.orbital_equations.frame_conversions.cart_to_mee import cart_to_mee

#: Length of each design point, s.
DURATION = 43200.0

#: The ground station, Ann Arbor, and its altitude in km.
LAT, LON, ALT = np.radians(42.2708), np.radians(-83.7264), 0.256

MEE = ('p', 'f', 'g', 'h', 'k', 'L')

#: Reference magnitudes of the states, which scale them and their defects for the optimizer.
STATE_REF = {'p': 1.0, 'f': 1.0, 'g': 1.0, 'h': 1.0, 'k': 1.0, 'L': 10.0, 'w_RW': 1.0,
             'temperature': 100.0, 'SOC': 1.0, 'Data': 1000.0}

#: Each state's rate output in ``CADREODE``, and its units per second.
STATE_RATE = {'p': ('p_dot', 'DU_earth/s'), 'f': ('f_dot', '1/s'), 'g': ('g_dot', '1/s'),
              'h': ('h_dot', '1/s'), 'k': ('k_dot', '1/s'), 'L': ('L_dot', 'rad/s'),
              'w_RW': ('alpha_RW', 'rad/s**2'), 'temperature': ('temperature_dot', 'degK/s'),
              'SOC': ('SOC_dot', '1/s'), 'Data': ('Dr', 'Gibyte/s')}

#: The design parameters shared by every design point, with the legacy MDP's bounds.
SHARED = {
    'cellInstd': dict(val=np.ones((7, 12)), units='unitless', lower=0.0, upper=1.0),
    'finAngle': dict(val=np.pi / 4, units='rad', lower=0.0, upper=np.pi / 2),
    'antAngle': dict(val=0.0, units='rad', lower=-np.pi / 4, upper=np.pi / 4),
}

#: The Earth's gravitational parameter (km**3/s**2), radius (km) and J2, as the legacy orbit had.
_MU, _RE, _J2 = 398600.44, 6378.137, 1.08264e-3


def launch_points(filename=None):
    """
    Return the legacy MDP's design points: the days since J2000 and the initial position and
    velocity (km, km/s), from every other row of ``data/Launch/launch1.dat``.
    """
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'Launch',
                                'launch1.dat')
    data = np.loadtxt(filename)[1::2]
    return [{'LD': row[0] - 2451545, 'r0': row[1:4], 'v0': row[4:7]} for row in data]


def _cartesian_j2_rate(t, s):
    r = s[:3]
    rr = np.sqrt(r @ r)
    z2 = (r[2] / rr)**2
    a = -_MU * r / rr**3 - 1.5 * _MU * _J2 * _RE**2 * r / rr**5 * (1 - 5 * z2)
    a[2] -= 3.0 * _MU * _J2 * _RE**2 * r[2] / rr**5
    return np.r_[s[3:], a]


def orbit_guess(r0, v0, frac):
    """
    The orbit from ``r0``, ``v0`` (km, km/s) at the phase fractions ``frac``, under J2.

    Returns the MEE (``p`` in DU, ``L`` unwrapped) and the Cartesian position and velocity.
    """
    sol = solve_ivp(_cartesian_j2_rate, (0.0, DURATION), np.r_[r0, v0], t_eval=frac * DURATION,
                    rtol=1e-10, atol=1e-8)
    r, v = sol.y[:3], sol.y[3:]
    du, du_tu = DU_earth_m / 1000., DU_earth_m / TU_earth_s / 1000.
    mee = np.asarray(cart_to_mee(*(r / du), *(v / du_tu)))
    mee[5] = np.unwrap(mee[5])
    return dict(zip(MEE, mee)), r, v


def quaternion_from_dcm(O):
    """Scalar-first quaternions of body-to-inertial matrices ``O`` (3, 3, n), sign-continuous."""
    q_xyzw = Rotation.from_matrix(np.moveaxis(O, -1, 0)).as_quat()
    q = np.vstack((q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]))
    for i in range(1, q.shape[1]):
        if np.dot(q[:, i], q[:, i - 1]) < 0:
            q[:, i] = -q[:, i]
    return q


def roll_quaternion(r, v, gamma):
    """The attitude quaternion that points the body z axis anti-velocity, rolled by ``gamma``."""
    O_RI = np.asarray(_compute_O_RI(r, v))
    O_BR = np.asarray(_compute_O_BR(np.broadcast_to(gamma, r.shape[1:]).astype(float)))
    return quaternion_from_dcm(np.einsum('ijn,jkn->ikn', O_RI, O_BR))


def cadre_phase(LD, mee0, num_segments=60, nodes_per_seg=4, table_method='lagrange3',
                solve_segments=False, objective=False, transcription='radau', num_nodes=None):
    """
    One design point: a 12-hour phase of ``CADREODE`` from the initial MEE ``mee0``.

    Parameters
    ----------
    LD : float
        Launch day, days since J2000.
    mee0 : dict
        Initial ``p`` (DU), ``f``, ``g``, ``h``, ``k`` and ``L`` (rad).
    num_segments, nodes_per_seg : int
        The Radau grid.
    table_method : str
        Interpolation method of the tables.
    solve_segments : bool
        Solve each segment's states by Newton (a simulation of the given design), rather than
        leaving them to the optimizer.
    objective : bool
        Maximize this point's data downloaded (for a single-point problem).
    transcription : str
        ``'radau'`` (on ``num_segments`` x ``nodes_per_seg``) or ``'birkhoff'`` (one segment of
        ``num_nodes`` CGL nodes).
    """
    if transcription == 'radau':
        tx = Radau(grid_data=RadauGrid(num_segments=num_segments, nodes_per_seg=nodes_per_seg),
                   solve_segments=solve_segments)
    elif transcription == 'birkhoff':
        tx = Birkhoff(grid_data=BirkhoffGrid(num_nodes=num_nodes), solve_segments=solve_segments)
    else:
        raise ValueError(f"transcription must be 'radau' or 'birkhoff', not {transcription!r}")

    def scaled(**kwargs):
        name = kwargs.pop('name')
        ref = AffineTransform(scale=1.0 / STATE_REF[name])
        return State(transform=ref, defect_transform=ref.model_copy(), **kwargs)

    phase = Phase(
        ode=CADREODE(table_method=table_method),
        transcription=tx,
        time=Time(units='s', opt_initial=False, opt_duration=False, duration_val=DURATION),
        states={
            **{name: scaled(name=name, opt_initial=False, val=float(mee0[name])) for name in MEE},
            'w_RW': scaled(name='w_RW', opt_initial=False, val=np.zeros(3)),
            'temperature': scaled(name='temperature', opt_initial=False, val=273.0 * np.ones(5)),
            # The legacy MDP's initial state of charge bounds.
            'SOC': scaled(name='SOC', opt_initial=True, val=0.5, lower=0.2, upper=1.0),
            'Data': scaled(name='Data', opt_initial=False, val=0.0),
        },
        controls={
            'q': Control(units='unitless', shape=(4,), lower=-1.1, upper=1.1),
            'Isetpt': Control(units='A', shape=(12,), val=0.2, lower=0.0, upper=0.4),
            'P_comm': Control(units='W', val=0.1, lower=0.0, upper=25.0),
        },
        parameters={
            'LD': Parameter(val=LD, units='d', static_targets=True),
            'lat': Parameter(val=LAT, units='rad', static_targets=True),
            'lon': Parameter(val=LON, units='rad', static_targets=True),
            'alt': Parameter(val=ALT, units='km', static_targets=True),
            **{name: Parameter(val=spec['val'], units=spec['units'], static_targets=True,
                               shape=np.shape(spec['val']) or None)
               for name, spec in SHARED.items()},
        },
        path_constraints=[PathConstraint(name='ode.pointing_err', equals=0.0),
                          PathConstraint(name='ode.q_norm_err', equals=0.0),
                          *battery_path_constraints()],
        objective=(ObjectiveSpec(name='states_Data', loc='final', units='Gibyte')
                   if objective else None),
    )
    return phase


def set_initial_guess(phase, r0, v0, gamma=np.pi / 4, Isetpt=0.2, P_comm=0.1, SOC0=0.5,
                      num_samples=721):
    """
    Seed a design point: the J2 orbit from ``r0``, ``v0``, and the legacy MDP's initial design
    (roll ``gamma`` = pi/4, ``Isetpt`` = 0.2 A, ``P_comm`` = 0.1 W, ``SOC0`` = 0.5).
    """
    frac = np.linspace(0.0, 1.0, num_samples)
    mee, r, v = orbit_guess(r0, v0, frac)
    phase.set_time_val(t_initial=0.0, t_duration=DURATION)
    for name in MEE:
        phase.set_state_val(name, mee[name], time_vals=frac)
    phase.set_state_val('w_RW', np.zeros((2, 3)))
    phase.set_state_val('temperature', 273.0 * np.ones((2, 5)))
    phase.set_state_val('SOC', [SOC0, SOC0])
    phase.set_state_val('Data', [0.0, 0.0])
    phase.set_control_val('q', roll_quaternion(r, v, gamma).T, time_vals=frac)
    phase.set_control_val('Isetpt', np.broadcast_to(Isetpt, (2, 12)))
    phase.set_control_val('P_comm', np.broadcast_to(P_comm, (2,)))


def simulate(LD, r0, v0, num_segments=60, nodes_per_seg=4, table_method='lagrange3', **design):
    """
    Simulate one design point (Newton on each segment) and return its problem.

    ``design`` is passed to ``set_initial_guess`` (``gamma``, ``Isetpt``, ``P_comm``, ``SOC0``).
    """
    from om4.api import Problem

    mee, _, _ = orbit_guess(r0, v0, np.zeros(1))
    phase = cadre_phase(LD, {k: v[0] for k, v in mee.items()}, num_segments, nodes_per_seg,
                        table_method, solve_segments=True)
    prob = Problem(model=phase)
    set_initial_guess(phase, r0, v0, **design)
    prob.run_model()
    return prob


def seed_states(phase, sim):
    """
    Set ``phase``'s states from the simulation ``sim`` (a problem from ``simulate``),
    interpolated in time.
    """
    t = sim.get_val('t')
    t_unique, first = np.unique(t, return_index=True)
    for name in phase.states:
        vals = sim.get_val(f'states_{name}').reshape(-1, t.size)[:, first]
        phase.set_state_val(name, vals.T if vals.shape[0] > 1 else vals[0],
                            time_vals=t_unique / DURATION)


def seed_state_rates(prob, phase, prefix=''):
    """
    Set a Birkhoff phase's state rates from its own ODE at its current states.

    Interpolating rates from another grid misses where they switch (eclipse, ground-station
    passes); evaluating the ODE at this grid's nodes does not. Runs the model.
    """
    prob.run_model()
    for name in phase.states:
        source, units = STATE_RATE[name]
        # The state rates are per unit tau of the one segment: dt/dtau = duration / 2.
        phase.set_val(f'state_rates_{name}',
                      prob.get_val(f'{prefix}ode.{source}', units=units) * 0.5 * DURATION)


class MDPComp(JaxExplicitComp):
    """
    The MDP objective, the negative total data downloaded, and each point's state-of-charge
    periodicity residual ``SOC0 - SOCf`` (the legacy ``con5``).
    """
    npts: int = Field(default=1)

    def model_post_init(self, context, /):
        for i in range(self.npts):
            self.inputs[f'Data_{i}'] = InputVar(shape=(1,), val=0.0, units='Gibyte')
            self.inputs[f'SOC0_{i}'] = InputVar(shape=(1,), val=0.0, units='unitless')
            self.inputs[f'SOCf_{i}'] = InputVar(shape=(1,), val=0.0, units='unitless')
            self.outputs[f'SOC_periodic_{i}'] = OutputVar(shape=(1,), units='unitless')
        self.outputs['obj'] = OutputVar(shape=(1,), units='Gibyte')
        super().model_post_init(context)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['obj'] = -sum(inputs[f'Data_{i}'] for i in range(self.npts))
        for i in range(self.npts):
            outputs[f'SOC_periodic_{i}'] = inputs[f'SOC0_{i}'] - inputs[f'SOCf_{i}']


def _mdp_kwargs(npts, num_segments, nodes_per_seg, table_method, transcription, num_nodes):
    points = launch_points()[:npts]
    subsystems, connections = {}, []
    for i, pt in enumerate(points):
        mee, _, _ = orbit_guess(pt['r0'], pt['v0'], np.zeros(1))
        phase = cadre_phase(pt['LD'], {k: v[0] for k, v in mee.items()}, num_segments,
                            nodes_per_seg, table_method, transcription=transcription,
                            num_nodes=num_nodes)
        phase.promotes_inputs = normalize_promotes_list([(f'parameters:{name}', name) for name in SHARED])
        subsystems[f'pt{i}'] = phase
        connections += [Connection(src=f'pt{i}.final_states_Data', tgt=f'mdp.Data_{i}'),
                        Connection(src=f'pt{i}.states_SOC', tgt=f'mdp.SOC0_{i}', src_indices=[0]),
                        Connection(src=f'pt{i}.final_states_SOC', tgt=f'mdp.SOCf_{i}')]
    subsystems['mdp'] = MDPComp(npts=npts)
    design_vars = {name: DesignVar(lower=spec['lower'], upper=spec['upper'])
                   for name, spec in SHARED.items()}
    constraints = {f'mdp.SOC_periodic_{i}': Constraint(equals=0.0) for i in range(npts)}
    # The total is thousands of Gibyte.
    objectives = {'mdp.obj': Objective(transform=AffineTransform(scale=1e-2))}
    return dict(subsystems=subsystems, connections=connections, design_vars=design_vars,
                constraints=constraints, objectives=objectives)


class CADREMDP(Group):
    """
    The CADRE MDP: ``npts`` design points (``pt0``, ``pt1``, ...) and the ``mdp`` objective.

    ``cellInstd``, ``finAngle`` and ``antAngle`` are promoted from every point to one shared
    design variable each. ``transcription`` is ``'radau'`` (on ``num_segments`` x
    ``nodes_per_seg``) or ``'birkhoff'`` (``num_nodes``); the seeding simulation always uses
    the Radau grid.
    """
    npts: int = Field(default=6, ge=1, le=6)
    num_segments: int = Field(default=60)
    nodes_per_seg: int = Field(default=4)
    table_method: str = Field(default='lagrange3')
    transcription: str = Field(default='radau')
    num_nodes: int | None = Field(default=None, description='Birkhoff node count.')

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    connections: list[Connection] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode='before')
    @classmethod
    def _build_from_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        kwargs = _mdp_kwargs(data.get('npts', 6), data.get('num_segments', 60),
                             data.get('nodes_per_seg', 4), data.get('table_method', 'lagrange3'),
                             data.get('transcription', 'radau'), data.get('num_nodes'))
        for key in ('design_vars', 'constraints', 'objectives'):
            kwargs[key] = {**kwargs[key], **data.get(key, {})}
        data.update(kwargs)
        return data

    def set_initial_guess(self, simulate_states=True, prob=None, **design):
        """
        Seed every design point with its orbit and the legacy initial design.

        With ``simulate_states`` the states are those of a simulation of that design, so the
        optimizer starts with every defect (nearly) satisfied. A Birkhoff grid also needs its
        state rates, from its own ODE, for which ``prob`` (this group's problem) is required.
        """
        for i, pt in enumerate(launch_points()[:self.npts]):
            phase = self.subsystems[f'pt{i}']
            set_initial_guess(phase, pt['r0'], pt['v0'], **design)
            if simulate_states:
                sim = simulate(pt['LD'], pt['r0'], pt['v0'], self.num_segments, self.nodes_per_seg,
                               self.table_method, **design)
                if self.transcription == 'radau':
                    # The same grid: copy the states exactly.
                    for name in phase.states:
                        for var in (f'states_{name}', f'initial_states_{name}', f'final_states_{name}'):
                            phase.set_val(var, sim.get_val(var))
                else:
                    seed_states(phase, sim)
        if simulate_states and self.transcription == 'birkhoff':
            if prob is None:
                raise ValueError('Seeding Birkhoff state rates needs the problem: pass prob.')
            for i in range(self.npts):
                seed_state_rates(prob, self.subsystems[f'pt{i}'], prefix=f'pt{i}.')
