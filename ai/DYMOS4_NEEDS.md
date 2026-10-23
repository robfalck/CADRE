# dymos4 features needed by the CADRE port

The dymos4 counterpart of `OM4_NEEDS.md`: gaps in dymos4 that the CADRE port has hit. These
are tracked here instead of being worked around in CADRE. Each entry says what hits it, what
CADRE does in the meantime, and what the fix looks like.

## 1. Rate units are composed by string concatenation

A state's or control's rate units default to `f"{units}/{time_units}"`, so a `unitless` state
gets `unitless/TU_earth`, which om4 refuses to connect to an ODE output in `1/TU_earth`
(see `OM4_NEEDS.md` #1). The same happens for a unitless control's `_rate` and `_rate2`.

- **Hit by:** `test_orbit_prop.py` (states `f`, `g`, `h`, `k`), `test_orbit_and_attitude.py`
  (those states, and the quaternion control `q`, whose rates feed `q_rate` in `1/s`).
- **Meanwhile:** the affected tests are `expectedFailure`. Explicit `State.rate_units` would
  avoid it for states, but that is a workaround that every unitless state would need.
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
