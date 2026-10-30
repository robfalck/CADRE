# om4 features needed by the CADRE port

Gaps in om4 that the CADRE port to om4/dymos4 has hit. These are tracked here instead of
being worked around in CADRE. Each entry says what hits it, what CADRE does in the meantime
(usually: the affected test is marked `expectedFailure`, so it flips to an unexpected success
once fixed), and what the fix looks like.

Versions in use: om4 branch `arch_udpate` at 3dc1ebd1 or later (needed for #1), dymos4 branch
`transition` (d467322).

## 1. ~~`'unitless/X'` is not compatible with `'1/X'`~~ — FIXED

**Fixed** on om4 `arch_udpate` in 3dc1ebd1 (2026-10-23, committed, not yet pushed): the cause was
`unit_library.ini` defining `unitless` as its own base dimension, so `unitless/s` had powers
`1/s` does not. It is now `unitless: m/m` under `[units]` (so `percent` and `drag_count`, built
from it, are dimensionless too), with regression tests in `om4/utils/tests/test_units.py`. OM3
has the same bug. With the fix, all four CADRE tests that were `expectedFailure` on it pass and
their marks are removed. CADRE needs om4 at 3dc1ebd1 or later. Side effects: `unitless/s` is
now equivalent to `Hz`; `rad` stays its own dimension, so `rad/s` and `Hz` remain incompatible.

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
  attitude quaternion control `q` (`test_orbit_and_attitude.py`), and the battery state of charge
  `SOC` (`test_battery_dymos.py`).
- **Interacts with:** `DYMOS4_NEEDS.md` #1 (dymos4 builds rate units by string concatenation).
  Fixing either one unblocks CADRE; fixing this one is the more general fix.
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

## 3. No structured-grid (N-D table) interpolation component — met by om4 PR #404 (in review)

CADRE's solar model interpolates a table of exposed area on a 3-D grid (fin angle 10 x azimuth
73 x elevation 37) for 84 outputs (7 cells on each of 12 panels), with derivatives with respect
to all three inputs. The legacy model used MBI (a separate least-squares B-spline package that
is not in the environment); OpenMDAO 3 would use `MetaModelStructuredComp`. om4 has no
equivalent on `main` or `arch_udpate`. The upstream `feat/metamodel_1..4` stack adds
interpolation methods (`interp_methods.py`), a 1-D `SplineComp` and an unstructured
`MetaModelComp`, but no N-D structured-grid component. (A local `metamodels` branch's
`MetaModelStructuredComp` design, in `METAMODEL_PLAN.md`, is the shape of what is needed.)

- **Hit by:** `Solar_ExposedArea` (`CADRE/solar.py`, a 3-D table with 84 outputs),
  `Comm_GainPattern` (`CADRE/comm.py`, a 2-D table of antenna gain in azimuth and elevation) and
  `Power_CellVoltage` (`CADRE/power.py`, a 3-D table of cell voltage in temperature, exposed area
  and current).
- **Meanwhile:** CADRE works against om4 `main` with PR #404 merged, in the worktree
  `../om4-pr404.git` (ARCHITECTURE, Environment), on the expectation that the PR merges soon.
  The three are ported as `ExposedAreaGroup`, `CellVoltageGroup` and `GainPatternGroup`,
  interpolating the tables of `CADRE/tables.py` (the legacy MBI fits sampled for grid
  interpolation, ADL-007), and their tests are not marked `expectedFailure`. On plain om4 `main`
  their modules (`solar_dymos`, `power_dymos`, `comm_dymos`) do not import.
- **In review:** OpenMDAO/om4 PR #404 (dwmunster, `feat/metamodel_1`, open as of 2026-10-29)
  adds `StructuredMetaModelComp`, which meets this need: N-D rectilinear grids, a trailing value
  shape on each output, `vec_size` points with node-diagonal partials, and `cubic`, `lagrange3`,
  `akima` and other methods. Checked against its branch head 80688338 with the `CADRE/tables.py`
  tables: all three reproduce the legacy MBI outputs to within 2e-4 to 6e-4 of their range, and
  the partials agree with finite difference and complex step. Notes for the port: `cubic` is
  SciPy's B-spline and rejects complex step, which an ODE-level `differentiator='cs'` check
  needs (CADRE's convention), so `lagrange3` (2e-4 to 6e-4) is the complex-step-safe choice;
  `akima` and `pchip` are slow on the 84-output solar table (about 1 s per evaluation at 1500
  nodes, against 5 ms for `cubic`); outputs are node-major, `(vec_size, *value_shape)`, against
  CADRE's component-major convention; and every input must have shape `(vec_size,)`, so the
  scalar fin angle must be broadcast.
- **Fix:** a `MetaModelStructuredComp` that takes several outputs on one grid (or one output with
  extra trailing dimensions), smooth (cubic or B-spline) methods, vectorized over `num_nodes`,
  and with node-diagonal partials.

## 4. Block solvers swallow `AnalysisError` and can report false convergence

`BlockNonlinearSolver.single_iteration` catches `AnalysisError` from a subsystem
(`reraise_child_analysiserror` defaults to False), abandons the rest of the sweep, and sets the
residual to the change in outputs over the sweep. When the error comes before anything has
changed, that residual is exactly zero and the solver reports `converged`, with the outputs
downstream of the failing component left at stale or default values.

- **Hit by:** a CADRE design point under Picard (`DYMOS4_NEEDS.md` #3): a table component raised
  `AnalysisError` for a temperature below its grid, and NonlinearBlockGS printed
  `abs=0.00e+00 ... converged in 19 iterations` with inconsistent multiple-shooting anchors.
- **Meanwhile:** nothing; CADRE no longer uses Picard. Diagnosed with
  `reraise_child_analysiserror=True`.
- **Fix:** treat an `AnalysisError` in a sweep as a failed iteration (never a zero residual):
  re-raise by default, or mark the solve non-converged.

## 5. `Problem.set_val` on a missing variable raises an opaque `TypeError`

Setting a name that does not exist raises `TypeError: 'NoneType' object is not subscriptable`
from `execnode.set_val` (`src_node[0]` on a failed lookup), rather than an error naming the
variable.

- **Hit by:** a CADRE test that set `h` and `k` on `RComp`, which has neither.
- **Fix:** raise a `KeyError`/`ValueError` naming the variable, as `get_val` does.

## Nice to have

Not gaps that block CADRE, but things the port kept working around by hand.

- **Table meta-model adapters.** `StructuredMetaModelComp` (PR #404) takes every input as
  `(vec_size,)` and puts the node axis first on its outputs, so CADRE needs adapter components:
  `FinAngleNodesComp` broadcasts the static fin angle to the nodes, `ExposedAreaTransposeComp`
  moves the node axis last, and `CellVoltagePointsComp`/`CellVoltageSumComp` flatten the cell
  points and sum them. An option for node-last outputs, and for inputs that are scalars rather than
  per point, would remove most of them.
- **Shape expressions.** `shape=('7*12*num_nodes',)` is not accepted; CADRE adds a `num_cells`
  property and refers to that.
- **Promotes assigned after construction.** Assigning `system.promotes_inputs = [...]` skips the
  validator, so tuples must be passed through `normalize_promotes_list` by hand (dymos4 does the
  same in `picard_iter_group`).
- **Ambiguous input defaults, one at a time.** Inputs promoted together whose components disagree
  on a default raise one error per name; `CADREODE` needed several rounds to find them all. Listing
  every ambiguous name at once would help.
- **Log noise.** Every CADRE run filters asdex `DenseColoringWarning`, the table extrapolation
  warning (its text lists the points, so it changes on every evaluation and is never deduplicated)
  and the "no coupling to resolve" INFO message.
- **Why an optimization stalled.** Finding the constraints IPOPT and Uno could not satisfy needed a
  hand-written report of the worst violation of each constraint and where it was. A built-in
  post-optimization report of the worst constraints (and design variables at their bounds) would
  have found it at once.
