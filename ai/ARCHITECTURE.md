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
| `CADRE/testing_utils.py` | `wrap` and `assert_partials_ok` for component tests |
| `CADRE/*.py`, `CADRE/test/` (other than the above) | The legacy OpenMDAO 2.x model with its own RK4 integrator. Not ported yet, not run by `pixi run test` |
| `ai/PORT_STATUS.md` | Per-component port status and the list of blocked items |
| `ai/OM4_NEEDS.md`, `ai/DYMOS4_NEEDS.md` | Upstream gaps hit by the port |

## Environment

pixi, mirroring dymos4: `pixi install`, `pixi run test`, `pixi run lint`. om4 and dymos4 are
installed editable from the sibling checkouts `../om4.git` and `../dymos4.git`.

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
| ADL-004 | **Bugs found in the original components are fixed during the port** and noted in the class docstring: a wrong analytic partial in `TwoBodyDynamicsComp` (`g_dot` wrt `f`, `p` and `L`), `CartesianToMEE`'s scalar eccentricity vector, the misnamed `KepToMEE` (and its units, input name and partials), `MEEToKep`'s units, and the sign of `w_B` (ADL-005). | Each is verified by a test (round trips, `check_partials`, a spin about body z). | Port bug-for-bug. |
| ADL-005 | **`O_BI` is body to inertial, as its name and description say, and every rotation matrix follows `x_Y = O_XY x_X`, with column vectors (the matrix is postmultiplied by the vector).** So `O_RI`'s columns are the rolled axes in the inertial frame, `O_BR` is the transpose of the original's, `O_BI = O_RI O_BR`, `v_B = O_BI^T v_I`, the quaternion DCM is the active rotation of `q`, and `[w_B x] = O_BI^T Odot_BI`. | The legacy model named and described `O_BI` as body to inertial but used it as inertial to body everywhere (`O_RI` built from rows, `O_BI = O_BR O_RI`, `x_B = O_BI x_I` in sideslip, comm and sun), and its `w_B` formula gave the negative of the body rate under that reading. Rob chose to make the math match the name. Note that reading `O_XY` as "from X to Y" is common but not universal: Schaub & Junkins' `[BN]` and Groves' `C^b_n` read to/from (`v_B = [BN] v_N`), which is how the legacy math was written. Verified: on the circular orbit `w_B` is the orbit normal expressed in the body frame. | Keep the legacy inertial-to-body math, fixing only descriptions and the `w_B` sign: a smaller change, and the legacy sun and comm components would port unchanged. |

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
