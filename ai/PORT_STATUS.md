# Port status — dymos 1.x components to om4 / dymos4

`[x]` ported and tested, `[-]` removed, `[ ]` not started. Tests are values plus
`check_partials` unless noted.

## Attitude (`CADRE/attitude_dymos/`)

- [x] `OBRComp` — roll matrix, body to rolled (the transpose of the original, ADL-005). Not used by the quaternion ODE (ADL-002), kept for the roll-angle form.
- [x] `ORIComp` — rolled frame from r, v.
- [x] `OBIComp` — `O_BR O_RI`. Not used by the quaternion ODE (ADL-002).
- [x] `QuaternionAttitudeComp` — **new**: `O_BI` and its first and second rates from `q` and its control rates.
- [x] `PointingComp` — **new**: pointing and unit-norm path-constraint residuals.
- [x] `AngularVelocityComp` — `[w_B x] = O_BI^T Odot_BI`; the original's sign was wrong (ADL-005).
- [x] `AngularAccelerationComp` — now pointwise, from `O_BI`, `Odot_BI` and `Oddot_BI`.
- [-] `OdotBIComp` — replaced by `QuaternionAttitudeComp` (ADL-002).
- [x] `BodyVelComp` — now `O_BI^T v_I` (ADL-005); the original's TODO about the missing `w x r` term is kept.
- [x] `AttitudeTorqueComp`

## Reaction wheel

- [x] `ReactionWheel` (`CADRE/reactionwheel_dymos.py`), plus a Picard propagation test against the
  closed-form spin-up.

## Orbit (`CADRE/orbital_equations/`)

- [x] `TwoBodyDynamicsComp` (analytic partials; one partial bug fixed, ADL-004)
- [x] `MEEToCart`
- [x] `StateMuxComp` (component-major `(3, n)` outputs; unit conversion DU/TU to km tested)
- [x] `CartesianToMEE` (rewritten; round trip with `MEEToCart`)
- [x] `KepToMEE` (renamed from the misnamed `CartesianToMEE` in `kep_to_mee.py`; round trip with `MEEToKep`)
- [x] `MEEToKep`
- [x] `RComp`
- [x] `SphericalHarmonicsComp`
- [x] `TotalPerturbation`
- [x] `Perturbations` (group)
- [x] `TwoBodyEOM` (group; the unused `sc_params` option is dropped)

## Blocked

Every component above is ported and passes its own tests. What is blocked is the dymos4
integration tests, which are marked `expectedFailure`:

| Test | Blocked by | Verified past the blocker? |
|---|---|---|
| `orbital_equations/EOMs/test/test_orbit_prop.py::test_propagation` | `OM4_NEEDS.md` #1 / `DYMOS4_NEEDS.md` #1 (unitless rate units) | Yes: with explicit rate units, Picard propagates the orbit and closes it after one period |
| `orbital_equations/EOMs/test/test_orbit_prop.py::test_j2_propagation` | same | Yes |
| `orbital_equations/EOMs/test/test_orbit_and_attitude.py::test_propagation` | same (orbit states and the quaternion control) | Yes: IPOPT converges in 4 iterations, pointing error 1e-13, `w_B` is the orbit normal in the body frame at the orbital rate, `T_req` ~ 0 |

## Not started

The legacy OpenMDAO 2.x model (`CADRE/*.py` at the top level: battery, comm, power, solar, sun,
thermal, kinematics, orbit, rk4, the CADRE group and MDP) and its tests in `CADRE/test/`. Its
RK4 integration is to be replaced by dymos4 phases. **When porting it, apply `O_BI^T`**: the
legacy code uses `O_BI` as inertial to body (`r_b2g_B = O_BI r_b2g_I` in comm, `r_e2s_B` in sun),
while the ported components use body to inertial (ADL-005).

The CI workflow (`.github/workflows/CADRE_test_workflow.yml`) still installs dymos 1.x and
OpenMDAO with conda; om4 and dymos4 are not published, so CI cannot run the ported tests yet.
