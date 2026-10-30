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

OpenMDAO 3 has the same option and default but differs in two ways. Its first iteration runs in
`_iter_initialize` (`NonlinearBlockGS._run_apply`), which does not catch the error, so a failure on
the first sweep is raised. Later iterations go through `Solver._gs_iter`, where the `try` is inside
the loop over subsystems, so only the failing subsystem is skipped and the rest of the sweep runs.
om4's `try` wraps the whole sweep (`self._run_subsystems(exec_node)`), on every iteration. In a
cycle `y1 = 0.5 y2 + 1`, `y2 = 0.5 y1` (and a three-component variant with a feedback into the
component after the failing one), with A raising:

| A raises | om4 | OpenMDAO 3.43 |
|---|---|---|
| on the first sweep | "converged in 1 iterations" at the starting point | raises `AnalysisError` |
| once, later, in the two-component loop | false convergence | false convergence too, at the same point (B's input has not changed, so neither does its output) |
| once, later, with a pending update after A | false convergence | converges to the solution |

So OpenMDAO 3 is not immune, but it raises on the first sweep and recovers whenever the rest of the
sweep has work to do; om4 reports convergence in all three.

**Reproduction.** Each script runs on its own; the om4 ones need only om4 (checked on `main` at
333db9d4), the OpenMDAO 3 ones only OpenMDAO (checked on 3.43.0).

Two components; `python om4_nlbgs_analysiserror.py first` or `... later`:

```python
"""
om4: an AnalysisError swallowed by NonlinearBlockGS reports false convergence.

A and B form a cycle, y1 = 0.5*y2 + 1 and y2 = 0.5*y1, solved by y1 = 4/3, y2 = 2/3, under
NonlinearBlockGS with the default reraise_child_analysiserror=False. A raises AnalysisError:

- scenario "first": while y2 < 0 (as a table does outside its grid), starting from y2 = -1, so A
  fails on the first sweep;
- scenario "later": once, on its second evaluation, starting from a valid y2 = 1.

om4 abandons the whole sweep when A raises, so B does not run, the outputs do not change, the
residual (the change in outputs over the sweep) is exactly 0, and the solver reports convergence
at a point that is not a solution.

Run: python om4_nlbgs_analysiserror.py first|later
"""
import sys

import numpy as np

import om4.api as om
from om4.core.analysis_error import AnalysisError
from om4.core.explicitcomp import ExplicitComponent
from om4.derivatives.partials_spec import PartialsSpec

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "first"
calls = {"A": 0}


class A(ExplicitComponent):
    inputs: dict[str, om.InputVar] = {"y2": om.InputVar(shape=(1,), val=0.0)}
    outputs: dict[str, om.OutputVar] = {"y1": om.OutputVar(shape=(1,), val=1.0)}
    partials: list[PartialsSpec] = [PartialsSpec(of="y1", wrt="y2", val=0.5)]

    def compute_outputs(self, inputs, outputs):
        calls["A"] += 1
        if SCENARIO == "first" and np.any(inputs["y2"] < 0.0):
            raise AnalysisError(f"A: y2 = {inputs['y2']} is below 0")
        if SCENARIO == "later" and calls["A"] == 2:
            raise AnalysisError("A: failed on its second evaluation")
        outputs["y1"] = 0.5 * inputs["y2"] + 1.0


class B(ExplicitComponent):
    inputs: dict[str, om.InputVar] = {"y1": om.InputVar(shape=(1,), val=0.0)}
    outputs: dict[str, om.OutputVar] = {"y2": om.OutputVar(shape=(1,), val=1.0)}
    partials: list[PartialsSpec] = [PartialsSpec(of="y2", wrt="y1", val=0.5)]

    def compute_outputs(self, inputs, outputs):
        outputs["y2"] = 0.5 * inputs["y1"]


prob = om.Problem(model=om.Group(
    subsystems={"A": A(), "B": B()},
    connections=[om.Connection(src="A.y1", tgt="B.y1"), om.Connection(src="B.y2", tgt="A.y2")],
    nonlinear_solver=om.NonlinearBlockGS(maxiter=50, iprint=2, atol=1e-10, rtol=1e-10),
))
prob.set_val("B.y2", -1.0 if SCENARIO == "first" else 1.0)
prob.run_model()
print(f"om4, scenario {SCENARIO!r}: y1 = {prob.get_val('A.y1')[0]:.6f}, "
      f"y2 = {prob.get_val('B.y2')[0]:.6f} (solution y1 = 1.333333, y2 = 0.666667)")
```

```
$ python om4_nlbgs_analysiserror.py first
NonlinearBlockGS on root started atol=1e-10, rtol=1e-10
iter   1 | abs=0.00e+00 rel=0.00e+00
NonlinearBlockGS on root converged in 1 iterations
om4, scenario 'first': y1 = 1.000000, y2 = -1.000000 (solution y1 = 1.333333, y2 = 0.666667)

$ python om4_nlbgs_analysiserror.py later
iter   1 | abs=5.59e-01 rel=1.00e+00
iter   2 | abs=0.00e+00 rel=0.00e+00
NonlinearBlockGS on root converged in 2 iterations
om4, scenario 'later': y1 = 1.500000, y2 = 0.750000 (solution y1 = 1.333333, y2 = 0.666667)
```

Three components, with a feedback into the component after the failing one (solution
`y1, y2, y3 = 1.166667, 0.666667, 0.333333`):

```python
"""om4: A fails once (2nd call) in a cycle A -> B -> C, with B also fed back by C."""
import om4.api as om
from om4.core.analysis_error import AnalysisError
from om4.core.explicitcomp import ExplicitComponent
from om4.derivatives.partials_spec import PartialsSpec

calls = {"A": 0}


class A(ExplicitComponent):
    inputs: dict[str, om.InputVar] = {"y3": om.InputVar(shape=(1,), val=0.0)}
    outputs: dict[str, om.OutputVar] = {"y1": om.OutputVar(shape=(1,), val=0.0)}
    partials: list[PartialsSpec] = [PartialsSpec(of="y1", wrt="y3", val=0.5)]

    def compute_outputs(self, inputs, outputs):
        calls["A"] += 1
        if calls["A"] == 2:
            raise AnalysisError("A: failed on its second evaluation")
        outputs["y1"] = 0.5 * inputs["y3"] + 1.0


class B(ExplicitComponent):
    inputs: dict[str, om.InputVar] = {"y1": om.InputVar(shape=(1,), val=0.0),
                                      "y3": om.InputVar(shape=(1,), val=0.0)}
    outputs: dict[str, om.OutputVar] = {"y2": om.OutputVar(shape=(1,), val=0.0)}
    partials: list[PartialsSpec] = [PartialsSpec(of="y2", wrt="y1", val=0.5),
                                    PartialsSpec(of="y2", wrt="y3", val=0.25)]

    def compute_outputs(self, inputs, outputs):
        outputs["y2"] = 0.5 * inputs["y1"] + 0.25 * inputs["y3"]


class C(ExplicitComponent):
    inputs: dict[str, om.InputVar] = {"y2": om.InputVar(shape=(1,), val=0.0)}
    outputs: dict[str, om.OutputVar] = {"y3": om.OutputVar(shape=(1,), val=0.0)}
    partials: list[PartialsSpec] = [PartialsSpec(of="y3", wrt="y2", val=0.5)]

    def compute_outputs(self, inputs, outputs):
        outputs["y3"] = 0.5 * inputs["y2"]


prob = om.Problem(model=om.Group(
    subsystems={"A": A(), "B": B(), "C": C()},
    connections=[om.Connection(src="A.y1", tgt="B.y1"), om.Connection(src="C.y3", tgt="B.y3"),
                 om.Connection(src="B.y2", tgt="C.y2"), om.Connection(src="C.y3", tgt="A.y3")],
    nonlinear_solver=om.NonlinearBlockGS(maxiter=50, iprint=2, atol=1e-10, rtol=1e-10),
))
prob.run_model()
y = [prob.get_val(n)[0] for n in ("A.y1", "B.y2", "C.y3")]
print("om4: y1, y2, y3 = " + ", ".join(f"{v:.6f}" for v in y))
```

```
$ python om4_three.py
iter   1 | abs=1.15e+00 rel=1.00e+00
iter   2 | abs=0.00e+00 rel=0.00e+00
NonlinearBlockGS on root converged in 2 iterations
om4: y1, y2, y3 = 1.000000, 0.500000, 0.250000
```

The same problems in OpenMDAO 3, for comparison:

```python
"""
OpenMDAO 3 version of om4_nlbgs_analysiserror.py: the same cycle, scenarios and solver.

- scenario "first": the first iteration runs inside _iter_initialize/_run_apply, which does not
  catch AnalysisError, so it is raised to the caller.
- scenario "later": Solver._gs_iter catches the error per subsystem, so only A is skipped that
  sweep, B still runs, and the solver goes on to the true solution.

Run: python om3_nlbgs_analysiserror.py first|later
"""
import sys

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.core.analysis_error import AnalysisError

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "first"
calls = {"A": 0}


class A(om.ExplicitComponent):
    def setup(self):
        self.add_input("y2", 0.0)
        self.add_output("y1", 1.0)
        self.declare_partials("y1", "y2", val=0.5)

    def compute(self, inputs, outputs):
        calls["A"] += 1
        if SCENARIO == "first" and np.any(inputs["y2"] < 0.0):
            raise AnalysisError(f"A: y2 = {inputs['y2']} is below 0")
        if SCENARIO == "later" and calls["A"] == 2:
            raise AnalysisError("A: failed on its second evaluation")
        outputs["y1"] = 0.5 * inputs["y2"] + 1.0


class B(om.ExplicitComponent):
    def setup(self):
        self.add_input("y1", 0.0)
        self.add_output("y2", 1.0)
        self.declare_partials("y2", "y1", val=0.5)

    def compute(self, inputs, outputs):
        outputs["y2"] = 0.5 * inputs["y1"]


prob = om.Problem()
prob.model.add_subsystem("A", A())
prob.model.add_subsystem("B", B())
prob.model.connect("A.y1", "B.y1")
prob.model.connect("B.y2", "A.y2")
prob.model.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, atol=1e-10, rtol=1e-10)
prob.setup()
prob.set_val("B.y2", -1.0 if SCENARIO == "first" else 1.0)
try:
    prob.run_model()
except AnalysisError as err:
    print(f"OpenMDAO {openmdao.__version__}, scenario {SCENARIO!r}: run_model raised "
          f"AnalysisError: {str(err).splitlines()[-1]}")
else:
    print(f"OpenMDAO {openmdao.__version__}, scenario {SCENARIO!r}: y1 = {prob.get_val('A.y1')[0]:.6f}, "
          f"y2 = {prob.get_val('B.y2')[0]:.6f} (solution y1 = 1.333333, y2 = 0.666667)")
```

```
$ python om3_nlbgs_analysiserror.py first
OpenMDAO 3.43.0, scenario 'first': run_model raised AnalysisError: 'A' <class A>: Error calling compute(), A: y2 = [-1.] is below 0

$ python om3_nlbgs_analysiserror.py later
NL: NLBGS 1 ; 0.559016994 1
NL: NLBGS 2 ; 0 0
NL: NLBGS Converged
OpenMDAO 3.43.0, scenario 'later': y1 = 1.500000, y2 = 0.750000 (solution y1 = 1.333333, y2 = 0.666667)
```

```python
"""OpenMDAO 3 version of om4_three.py."""
import openmdao
import openmdao.api as om
from openmdao.core.analysis_error import AnalysisError

calls = {"A": 0}


class A(om.ExplicitComponent):
    def setup(self):
        self.add_input("y3", 0.0); self.add_output("y1", 0.0); self.declare_partials("y1", "y3", val=0.5)

    def compute(self, inputs, outputs):
        calls["A"] += 1
        if calls["A"] == 2:
            raise AnalysisError("A: failed on its second evaluation")
        outputs["y1"] = 0.5 * inputs["y3"] + 1.0


class B(om.ExplicitComponent):
    def setup(self):
        self.add_input("y1", 0.0); self.add_input("y3", 0.0); self.add_output("y2", 0.0)
        self.declare_partials("y2", "y1", val=0.5); self.declare_partials("y2", "y3", val=0.25)

    def compute(self, inputs, outputs):
        outputs["y2"] = 0.5 * inputs["y1"] + 0.25 * inputs["y3"]


class C(om.ExplicitComponent):
    def setup(self):
        self.add_input("y2", 0.0); self.add_output("y3", 0.0); self.declare_partials("y3", "y2", val=0.5)

    def compute(self, inputs, outputs):
        outputs["y3"] = 0.5 * inputs["y2"]


prob = om.Problem()
for name, comp in (("A", A()), ("B", B()), ("C", C())):
    prob.model.add_subsystem(name, comp)
prob.model.connect("A.y1", "B.y1"); prob.model.connect("C.y3", "B.y3")
prob.model.connect("B.y2", "C.y2"); prob.model.connect("C.y3", "A.y3")
prob.model.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, atol=1e-10, rtol=1e-10)
prob.setup()
prob.run_model()
y = [prob.get_val(n)[0] for n in ("A.y1", "B.y2", "C.y3")]
print(f"OpenMDAO {openmdao.__version__}: y1, y2, y3 = " + ", ".join(f"{v:.6f}" for v in y))
```

```
$ python om3_three.py
NL: NLBGS 17 ; 4.36557457e-10 3.81058589e-10
NL: NLBGS 18 ; 1.09139364e-10 9.52646472e-11
NL: NLBGS Converged
OpenMDAO 3.43.0: y1, y2, y3 = 1.166667, 0.666667, 0.333333
```

- **Hit by:** a CADRE design point under Picard (`DYMOS4_NEEDS.md` #3): a table component raised
  `AnalysisError` for a temperature below its grid, and NonlinearBlockGS printed
  `abs=0.00e+00 ... converged in 19 iterations` with inconsistent multiple-shooting anchors.
- **Meanwhile:** nothing; CADRE no longer uses Picard. Diagnosed with
  `reraise_child_analysiserror=True`.
- **Fix:** treat a swallowed `AnalysisError` as a failed iteration, so it can never count as
  convergence (OpenMDAO 3 lacks this too), or re-raise by default; and catch the error per
  subsystem, as OpenMDAO 3 does, so the rest of the sweep still runs.

## 5. `Problem.set_val` on a missing variable raises an opaque `TypeError`

Setting a name that does not exist raises `TypeError: 'NoneType' object is not subscriptable`
from `execnode.set_val` (`src_node[0]` on a failed lookup), rather than an error naming the
variable.

- **Hit by:** a CADRE test that set `h` and `k` on `RComp`, which has neither.
- **Fix:** raise a `KeyError`/`ValueError` naming the variable, as `get_val` does.

## Nice to have

Not gaps that block CADRE, but things the port kept working around by hand.

- **Node-last outputs from the table meta-models.** `StructuredMetaModelComp` (PR #404) always
  puts the vectorized (point) axis first on its outputs, `(vec_size, *value_shape)`. dymos4 and
  CADRE are component-major (node axis last, dymos4 ADL-005), so CADRE adds
  `ExposedAreaTransposeComp` to turn the `(num_nodes, 7, 12)` exposed area into the
  `(7, 12, num_nodes)` its consumers take. Reordering the training data cannot help: its shape is
  the grid axes followed by `value_shape`, and has no point axis. The point-first layout is assumed
  in three places in `om4/components/table_meta_model.py`, which an option such as
  `vec_axis='last'` would change together:
  - `_build_table_io` declares each output as `(vec_size, *value_shape)`;
  - `_write_outputs` writes `value[:, columns].reshape(shape)`, with one row of `value` per point;
  - the partials pattern, `rows = arange(vec_size * output_size)` with
    `cols = repeat(arange(vec_size), output_size)`, assumes the flattened output runs point by point.

  (A scalar input needs nothing: a `(1,)` source promoted to a `(vec_size,)` input with
  `src_indices=[0] * vec_size` broadcasts it, and dymos4 still sees a static target.)
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
