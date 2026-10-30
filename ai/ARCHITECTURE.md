# Architecture Decision Record — CADRE on om4 / dymos4

**Project:** CADRE (the CubeSat design problem), being ported from OpenMDAO 2.x / dymos 1.x to
om4 (OpenMDAO 4) and dymos4.
**Status:** Living document, updated as decisions are made.
**Python target:** 3.12+

> Everything here is provisional. Decisions are recorded so they are explicit and reversible.
> If one is getting in the way, say so.

The om4 and dymos4 architecture documents (`../om4.git/ai/ARCHITECTURE.md`,
`../dymos4.git/ai/ARCHITECTURE.md`) apply here too. This file records only what is specific to
CADRE.

---

## Layout

| Path | What it is |
|---|---|
| `CADRE/attitude_dymos/` | Attitude kinematics and the pointing constraint (om4, JAX) |
| `CADRE/orbital_equations/` | Orbit EOMs in modified equinoctial elements, frame conversions, perturbations (om4) |
| `CADRE/reactionwheel_dymos.py` | Reaction wheel power and dynamics (om4, JAX) |
| `CADRE/sun_dymos.py` | Sun direction, line of sight and body-frame angles (om4, JAX) |
| `CADRE/battery_dymos.py` | Battery state-of-charge rate, current, and path constraints (om4, JAX) |
| `CADRE/thermal_dymos.py` | Fin and body temperature rates (om4, JAX) |
| `CADRE/power_dymos.py` | Solar and battery power (om4, JAX) |
| `CADRE/comm_dymos.py` | Ground-station geometry, line of sight and bit rate (om4, JAX) |
| `CADRE/cadre_ode.py` | `CADREODE`, the full pointwise ODE of one design point (replaces the legacy `CADRE` group) |
| `CADRE/cadre_mdp_dymos.py` | A design point as a dymos4 `Phase`, its simulation, and `CADREMDP`, the multi-point design problem (ADL-008) |
| `CADRE/solar_dymos.py` | Exposed area of the solar cells, interpolated (om4) |
| `CADRE/tables.py` | The solar exposed area, cell voltage and antenna gain tables, smoothed as the legacy MBI fits and sampled for grid interpolation (ADL-007) |
| `CADRE/testing_utils.py` | `wrap` and `assert_partials_ok` for component tests |
| `CADRE/*.py`, `CADRE/test/` (other than the above) | The legacy OpenMDAO 2.x model with its own RK4 integrator. Not ported yet, not run by `pixi run test` |
| `ai/PORT_STATUS.md` | Per-component port status and the list of blocked items |
| `ai/OM4_NEEDS.md`, `ai/DYMOS4_NEEDS.md` | Upstream gaps hit by the port |

## Environment

pixi, mirroring dymos4: `pixi install`, `pixi run test`, `pixi run lint`. om4 and dymos4 are
installed editable from sibling checkouts: dymos4 from `../dymos4.git` (`main`), and om4, until
OpenMDAO/om4 PR #404 merges, from the worktree `../om4-pr404.git` (branch `main-pr404`: om4
`main` with the PR merged). To refresh it, in `../om4.git`: `git fetch upstream main
pull/404/head:pr-404`, then in `../om4-pr404.git`: `git merge upstream/main pr-404`. When the PR
merges, point `pixi.toml` back at `../om4.git` and remove the worktree.

---

## Conventions

- Components are om4 pydantic models: `inputs`/`outputs` dicts of `InputVar`/`OutputVar`,
  `num_nodes: int = Field(default=1)` and parameterized shapes `('num_nodes',)`.
- **Component-major** (dymos4 ADL-005): the node axis is last. A vector is `(3, num_nodes)`, a
  rotation matrix `(3, 3, num_nodes)`, a quaternion `(4, num_nodes)`.
- Prefer `JaxExplicitComp`. Components that already had analytic partials keep them, declared
  with `PartialsSpec(..., diagonal=True)` and verified by `check_partials`.
- JAX components must be complex-step safe (no `jnp.linalg.norm`/`abs`; use
  `sqrt(sum(x**2))`), so a whole ODE's partials can be checked with `differentiator='cs'`.
- **Rotation matrices: `O_XY` takes vectors in frame X into frame Y**, `x_Y = O_XY x_X`, so its
  columns are X's axes expressed in Y and `O_XZ = O_YZ O_XY`. `O_BI` is body to inertial (ADL-005).
  Vectors are **column vectors**, so a matrix is postmultiplied by the vector it rotates
  (`einsum('ijn,jn->in', O, x)`). Under a row-vector convention (`x_Y = x_X O`) the same name
  would mean the transpose.
- Orbit components declare units with `DU`/`TU` placeholders, resolved from their
  `central_body` field by `CADRE.orbital_equations.set_body_units` in `model_post_init`.
- Group ODEs derive `subsystems`/`connections` from their fields in a `mode='before'`
  validator (the dymos4 `RaceCarODE` pattern), so dymos4 can rebuild them at any `num_nodes`.
- ODEs tag their state targets and rate sources (`dymos.state_target:<s>`,
  `dymos.state_rate_source:<s>`, `dymos.state_units:<u>`, `dymos.control_*target:<c>`).
- Tests live in `test/` beside the code: a values test against a closed form or an independent
  calculation, and a `check_partials` test, for every component.

---

## Architecture Decisions Log

| ID | Decision | Rationale | Alternatives considered |
|----|----------|-----------|-------------------------|
| ADL-001 | **The dymos-1.x-era components are ported in place**, keeping their module paths. The legacy OpenMDAO 2.x model is left untouched until it is ported. | These modules were already the dymos-compatible rewrite; there is no reason for a third copy. | A new package beside them. |
| ADL-002 | **The attitude is prescribed through a quaternion control `q`.** dymos4's control interpolation supplies `q_rate` and `q_rate2`, so the ODE computes `O_BI`, `Odot_BI`, `Oddot_BI` (`QuaternionAttitudeComp`, by JAX jvps of the DCM along the control rates), then `w_B`, `wdot_B` and the torque required, pointwise. Pointing is a pair of path constraints (`PointingComp.pointing_err`: the body z axis has no component along the rolled frame's x and y axes) plus `q_norm_err = 0`. **The roll is subsumed by `q`**: it is not a separate control, and the optimizer chooses it. `OdotBIComp` is removed. | CADRE insists on the pointing and works out the torque needed to achieve it, so the attitude is an input, not a state. The original differentiated `O_BI` and `w_B` along the node axis with the grid's differentiation matrix inside the ODE: the ODE was then not pointwise, needed the grid and `dt_dstau` passed in, and differentiated the interpolant. Making the attitude a control moves that differentiation into the transcription, where it belongs. Quaternions avoid Euler-angle singularities and the nine-entry over-parameterization of `O_BI`. | (a) The differentiation matrix in the ODE, as in dymos 1.x: rejected, above. (b) Attitude states driven by torque controls: models a different problem (attitude dynamics rather than a pointing law). (c) Analytic derivatives of the pointing law from the orbit's acceleration and jerk: exact, but needs the force model duplicated in the kinematics. (d) Keep roll `gamma` as the design control and constrain `O_BI(q) = O_BR(gamma) O_RI`: rejected in favor of letting `q` subsume it. |
| ADL-003 | **Upstream gaps are recorded, not worked around.** A test that an om4 or dymos4 gap blocks is written in its intended form and marked `unittest.expectedFailure`, citing the `*_NEEDS.md` entry. It becomes an unexpected success when the gap is fixed. | Keeps CADRE free of workarounds that would have to be found and removed later, and turns each gap into a concrete upstream request. Each blocked test is also checked in a scratch script past the blocker, so a later, hidden blocker is not mistaken for the recorded one. | Work around in CADRE (e.g. explicit `rate_units` on every unitless state). |
| ADL-004 | **Bugs found in the original components are fixed during the port** and noted in the class docstring: a wrong analytic partial in `TwoBodyDynamicsComp` (`g_dot` wrt `f`, `p` and `L`), `CartesianToMEE`'s scalar eccentricity vector, the misnamed `KepToMEE` (and its units, input name and partials), `MEEToKep`'s units, the sign of `w_B` (ADL-005), the legacy thermal state Jacobian, and `SphericalHarmonicsComp`'s J2, which was the Moon's (202.7e-6) instead of the Earth's (1.08264e-3). | Each is verified by a test (round trips, `check_partials`, a spin about body z, J2 against the Cartesian acceleration). | Port bug-for-bug. |
| ADL-005 | **`O_BI` is body to inertial, as its name and description say, and every rotation matrix follows `x_Y = O_XY x_X`, with column vectors (the matrix is postmultiplied by the vector).** So `O_RI`'s columns are the rolled axes in the inertial frame, `O_BR` is the transpose of the original's, `O_BI = O_RI O_BR`, `v_B = O_BI^T v_I`, the quaternion DCM is the active rotation of `q`, and `[w_B x] = O_BI^T Odot_BI`. | The legacy model named and described `O_BI` as body to inertial but used it as inertial to body everywhere (`O_RI` built from rows, `O_BI = O_BR O_RI`, `x_B = O_BI x_I` in sideslip, comm and sun), and its `w_B` formula gave the negative of the body rate under that reading. Rob chose to make the math match the name. Note that reading `O_XY` as "from X to Y" is common but not universal: Schaub & Junkins' `[BN]` and Groves' `C^b_n` read to/from (`v_B = [BN] v_N`), which is how the legacy math was written. Verified: on the circular orbit `w_B` is the orbit normal expressed in the body frame. | Keep the legacy inertial-to-body math, fixing only descriptions and the `w_B` sign: a smaller change, and the legacy sun and comm components would port unchanged. |
| ADL-006 | **The legacy RK4-integrated quantities become dymos4 states, and its KS-aggregated time-history constraints become path constraints.** `BatterySOC` and `ThermalTemperature` were `RK4` subclasses; their rates are now ODE outputs tagged as state rate sources (`SOC_dot`, `temperature_dot`). `BatteryConstraints` aggregated the current and SOC limits with KS functions; these are now `PathConstraint`s from `battery_path_constraints()`. | dymos4 does the integration (Picard here), and path constraints at the nodes are dymos's idiom for a limit over the trajectory. A KS aggregate existed so an RK4 model of thousands of steps would not need a constraint per step, and it couples all nodes in one output. | Keep the KS aggregates as an ODE output or a post-phase `KSComp`: fewer constraints, but conservative and not pointwise. |
| ADL-007 | **The interpolated tables are the legacy MBI fits, pre-smoothed and sampled on knot-aligned grids, not the raw data.** `CADRE/tables.py` reproduces MBI's least-squares tensor B-spline with SciPy (`MBIFit`: exposed area [4, 10, 8] control points, cubic; cell voltage [6, 6, 15], quadratic; gain [15, 15], cubic), and samples each fit with a grid point at every knot and 8 (16 for the quadratic voltage fit) intervals per knot span. A grid interpolator (om4's `StructuredMetaModelComp`, `'cubic'` or the complex-step-safe `'lagrange3'`) then matches the fit to a few 1e-4 of the range. | The legacy model never interpolated the raw tables: MBI fit and smoothed them, and the fits differ from the raw data by up to 25% (exposed area) and 50% (panel voltage) of the range, so interpolating the raw tables would change the model. `MBIFit` matches the legacy MBI outputs stored in `CADRE/test/data1346.pkl` to MBI's CG tolerance (about 1e-4). The fit's highest derivative jumps at its knots; a grid that misses them stalls near 1e-3. The tables are built on first use (about a second) rather than stored, so no generated binary is committed. | (a) Interpolate the raw tables: rejected, above. (b) Ask om4 for a least-squares fit method: the smoothing is a property of CADRE's data, not of the interpolator. (c) Commit the sampled tables as `.npz` files: 22 MB, and redundant with the script that makes them. |

| ADL-008 | **The MDP is one Radau `Phase` per design point in a `CADREMDP` group, with dymos4 controls at the nodes.** Each point integrates the MEE orbit, `w_RW`, `temperature`, `SOC` and `Data` over 12 hours from its launch condition. The controls are `q` (ADL-002), `Isetpt` and `P_comm`, with the legacy bounds; `cellInstd`, `finAngle` and `antAngle` are promoted from every phase to one shared design variable each; an `MDPComp` gives the objective (minus the total data) and each point's `SOC0 - SOCf` (the legacy `con5`). The battery limits are path constraints (ADL-006). For optimization the states are left to the optimizer (`solve_segments=False`), scaled by reference magnitudes (`STATE_REF`), and seeded from a simulation of the starting design (`solve_segments=True`, Newton on each segment), so every defect starts satisfied. The cell voltage table extrapolates. The orbit has J2 only. | Picard shooting diverges here: the wheel speeds rotate at the body rate (`alpha_RW` has `-w_B x w_RW`, about 6e-3 rad/s), so plain Picard needs segments under about 170 s, and its first sweep also discards the guess (`DYMOS4_NEEDS.md` #3). Radau with Newton-solved segments simulates any design; with optimizer-driven states it is the usual dymos formulation. dymos4 has no B-spline control (`DYMOS4_NEEDS.md` #4), so the controls have a value per control node (3 per segment), and a grid fine enough to carry the legacy design is about 300 segments: there, the legacy optimized design gives 30,950 Gb against the legacy 30,892 Gb, and the initial design's `con5` matches to 2e-4. A seeded start matters: the legacy initial design drains the battery to SOC -13, which a flat state guess hides from the optimizer. Radau extrapolates each segment's controls to its end node, which can take `Isetpt` below 0, the edge of the cell voltage table (the legacy MBI also extended its tables past their edges). J3 and J4, which the legacy orbit had, move the orbit by under 3 km in 12 hours. | (a) Picard shooting, so the optimizer sees only controls: rejected, above. (b) One phase per orbit pass, or a `Trajectory`: the points are independent and share only parameters, so a group of phases needs no linkages. (c) Add J3 and J4 to the MEE perturbations: deferred, as the effect is small. |
---

## Open Questions

- [x] ~~Angular velocity sign convention.~~ **Resolved (ADL-005):** `O_BI` is body to inertial
      and `[w_B x] = O_BI^T Odot_BI`.
- [ ] Which pointing law should CADRE use going forward? The rolled frame holds the body z axis
      anti-velocity (as the original did); the roll that maximizes solar power needs the solar
      and power models ported.

> **AI INSTRUCTION:** If your task touches an open question above, stop and ask before
> proceeding.

---

## Session Startup Checklist for AI

- Read this file, `ai/PORT_STATUS.md`, `ai/OM4_NEEDS.md` and `ai/DYMOS4_NEEDS.md`.
- When something in om4 or dymos4 gets in the way, add it to the right `*_NEEDS.md` (what hit
  it, what CADRE does meanwhile, the fix) instead of working around it (ADL-003).
- Write the tests alongside the implementation.
