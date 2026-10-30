# dymos4 features needed by the CADRE port

The dymos4 counterpart of `OM4_NEEDS.md`: gaps in dymos4 that the CADRE port has hit. These
are tracked here instead of being worked around in CADRE. Each entry says what hits it, what
CADRE does in the meantime, and what the fix looks like.

## 1. Rate units are composed by string concatenation

No longer blocking: with `OM4_NEEDS.md` #1 fixed, `unitless/TU_earth` connects to `1/TU_earth`,
and the tests it hit pass. Kept because rate units built by concatenation are still fragile
(e.g. for compound or offset units).

A state's or control's rate units default to `f"{units}/{time_units}"`, so a `unitless` state
gets `unitless/TU_earth`, which om4 refuses to connect to an ODE output in `1/TU_earth`
(see `OM4_NEEDS.md` #1). The same happens for a unitless control's `_rate` and `_rate2`.

- **Hit by:** `test_orbit_prop.py` (states `f`, `g`, `h`, `k`), `test_orbit_and_attitude.py`
  (those states, and the quaternion control `q`, whose rates feed `q_rate` in `1/s`),
  `test_battery_dymos.py` (the state of charge `SOC`).
- **Fix:** derive rate units with om4's unit algebra (or special-case unitless/None), so a
  unitless quantity's rate is `1/<time units>`. `OM4_NEEDS.md` #1 fixes this more generally.

## 2. `set_state_val` / `set_control_val` / `set_time_val` / `set_parameter_val` take no units

dymos 1.x's guess-setting methods accept `units=`; dymos4's raise `TypeError` for it. Values
must be given in the variable's declared units, and a caller with values in other units has to
convert them by hand.

- **Hit by:** `test_orbit_prop.py`, `test_orbit_and_attitude.py` (the original tests set
  `t_duration` in `TU_earth` and `p` in `DU_earth` by name), `test_reactionwheel_dymos.py`
  (the original set the torque control with `units='N*m'`).
- **Meanwhile:** the tests pass values in the declared units.
- **Fix:** accept `units=` and convert with `om4.utils.units.convert_units` before setting.

## 3. Picard's first sweep integrates the ODE's default outputs, not the guess

`picard_phase_parts` orders the Picard cycle `picard_update_comp`, `ms_update_comp`, `ode`, so
the first NonlinearBlockGS sweep integrates every segment from the anchors' and the ODE rates'
*default* values (1.0) before the ODE has been evaluated at the states the user seeded. The
seeded guess is overwritten by `anchor + integral of 1`: a 720 s segment moves each state by
720 in its units. CADRE's temperatures jump by 720 K, the T**4 radiation term blows up and the
iteration diverges from an exact orbit guess. The orbit tests did not notice, because their
default rates are mild in TU units. Separately, the Picard NonlinearBlockGS has `iprint=-1` and
does not raise on non-convergence, so a diverged or unconverged solve returns garbage silently
(see also `OM4_NEEDS.md` #4).

- **Hit by:** a CADRE design point (`cadre_phase`) under `PicardShooting`.
- **Meanwhile:** CADRE uses Radau instead (ARCHITECTURE ADL-008), which it needs anyway: the
  reaction wheel speeds rotate at the body rate (`-w_B x w_RW`), so plain Picard diverges on
  segments longer than about 170 s even from a consistent start.
- **Fix:** order the cycle `ode`, `ms_update_comp`, `picard_update_comp`, so the first sweep
  evaluates the ODE at the guess and the anchors from it. Checked in a scratch copy: the CADRE
  phase then converges in 17 sweeps on a problem where it diverged. Also raise (or at least
  warn) when the Picard solve does not converge.

## 4. No B-spline control parameterization

The legacy model shaped each control time history with an m-point B-spline
(`BsplineParameters`: 300 control points over 1500 time points, for the roll, the comm power
and the 12 panel currents). dymos4 controls have a value at every control input node, and
`Control.order` (polynomial controls) is declared but not wired (dymos4 ADL-023), so there is
no way to give a control fewer degrees of freedom than the grid has nodes, or to make it
smoother than the per-segment Lagrange interpolant.

- **Hit by:** the CADRE MDP (`CADRE/cadre_mdp_dymos.py`), whose `q`, `Isetpt` and `P_comm`
  controls have 3 values per Radau segment: on a 300-segment grid, 900 per curve against the
  legacy 300, and on coarser grids too few to follow the legacy design (60 segments give 1.3 times
  the legacy data downloaded at the legacy design, from comm power the grid cannot resolve).
- **Meanwhile:** plain dymos4 controls, with value and rate continuity.
- **Fix:** a control option for a B-spline (or other basis) parameterization: `num_cp` control
  points as the design variables, evaluated (with rates) at the nodes by a fixed sparse basis
  matrix, independent of the grid. om4's `SplineComp` with `BSplineCurve` (om4 PR #405, the
  counterpart of OpenMDAO 3's `bsplines`) is the interpolation it would need.
