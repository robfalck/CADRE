# om4 features needed by the CADRE port

Gaps in om4 that the CADRE port to om4/dymos4 has hit. These are tracked here instead of
being worked around in CADRE. Each entry says what hits it, what CADRE does in the meantime
(usually: the affected test is marked `expectedFailure`, so it flips to an unexpected success
once fixed), and what the fix looks like.

Versions in use: om4 branch `arch_udpate` (7946a83a), dymos4 branch `transition` (d467322).

## 1. `'unitless/X'` is not compatible with `'1/X'`

    >>> from om4.utils.units import is_compatible
    >>> is_compatible('1/s', 'unitless/s')
    False

`unitless` is accepted as a unit on its own, but in a compound expression it does not reduce to
1, so `unitless/TU_earth` and `1/TU_earth` are considered incompatible and the connection is
refused:

    RuntimeError: Can't connect 'ode.f_dot' to 'f_computed_f': units '1/TU_earth' of
    'ode.f_dot' are incompatible with units 'unitless/TU_earth' of 'f_computed_f'.

- **Hit by:** every phase with a unitless state or control. In CADRE: the modified equinoctial
  elements `f`, `g`, `h`, `k` (`test_orbit_prop.py`, `test_orbit_and_attitude.py`) and the
  attitude quaternion control `q` (`test_orbit_and_attitude.py`).
- **Interacts with:** `DYMOS4_NEEDS.md` #1 (dymos4 builds rate units by string concatenation).
  Fixing either one unblocks CADRE; fixing this one is the more general fix.
- **Meanwhile:** the affected tests are `expectedFailure`. Verified in a scratch script that
  with explicit `State(rate_units='1/TU_earth')` the orbit propagates correctly under Picard,
  so this is the only blocker for those tests' orbit part.
- **Fix:** treat `unitless` as the dimensionless unit 1 when parsing compound units.

## 2. om4 declares no dependencies, and jax 0.12 breaks asdex

om4's `pyproject.toml` has no `dependencies`, so each consumer's environment has to supply and
bound them. A fresh solve picks jax/jaxlib 0.12.0.1, and asdex 0.5.2 (the latest, which om4's
`JaxDifferentiator` uses for sparsity detection) then fails on every JAX component's
`setup_partials`:

    ImportError: cannot import name 'ClosedJaxpr' from 'jax._src.core'

- **Hit by:** every JAX component's `check_partials` and every linearization.
- **Meanwhile:** CADRE's `pixi.toml` pins `jax`/`jaxlib` to `>=0.11,<0.12` (dymos4's lockfile
  happens to have 0.11.2, so it did not see this).
- **Fix:** declare om4's runtime dependencies (numpy, scipy, networkx, pydantic, rich,
  structlog, jax, asdex, ...) with bounds in om4's `pyproject.toml`, and upper-bound jax until
  asdex supports 0.12.

## 3. No structured-grid (N-D table) interpolation component

CADRE's solar model interpolates a table of exposed area on a 3-D grid (fin angle 10 x azimuth
73 x elevation 37) for 84 outputs (7 cells on each of 12 panels), with derivatives with respect
to all three inputs. The legacy model used MBI (a separate least-squares B-spline package that
is not in the environment); OpenMDAO 3 would use `MetaModelStructuredComp`. om4 has no
equivalent on `main` or `arch_udpate`. The upstream `feat/metamodel_1..4` stack adds
interpolation methods (`interp_methods.py`), a 1-D `SplineComp` and an unstructured
`MetaModelComp`, but no N-D structured-grid component. (A local `metamodels` branch's
`MetaModelStructuredComp` design, in `METAMODEL_PLAN.md`, is the shape of what is needed.)

- **Hit by:** `Solar_ExposedArea` (`CADRE/solar.py`); comm's antenna gain (`comm.py`, a 2-D
  table) and the power model's solar-cell curve (`power.py`, a 3-D table) will need it too.
- **Meanwhile:** the solar exposed-area component is not ported (blocked).
- **Fix:** a `MetaModelStructuredComp` that takes several outputs on one grid (or one output with
  extra trailing dimensions), smooth (cubic or B-spline) methods, vectorized over `num_nodes`,
  and with node-diagonal partials.
