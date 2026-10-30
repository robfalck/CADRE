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

## Sun (`CADRE/sun_dymos.py`, from `CADRE/sun.py`)

- [x] `SunPositionECIComp` (`Sun_PositionECI`); its output is a unit vector, which the original declared in km.
- [x] `SunLOSComp` (`Sun_LOS`); takes the 3-vector `r_e2b_I` rather than the original's 6-vector, and guards the square root so the derivative stays finite on the Earth-Sun line.
- [x] `SunPositionBodyComp` (`Sun_PositionBody`); now `O_BI^T r_e2s_I` (ADL-005).
- [x] `SunPositionSphericalComp` (`Sun_PositionSpherical`)

## Solar (`CADRE/solar_dymos.py`, from `CADRE/solar.py`)

- [x] `ExposedAreaGroup` (`Solar_ExposedArea`): interpolates `exposed_area_table()` (ADL-007) with om4's `StructuredMetaModelComp`, from om4 PR #404 (`OM4_NEEDS.md` #3); values checked against the legacy outputs. The static `finAngle` reaches the table's per-point input through `src_indices=[0] * num_nodes`, and `ExposedAreaTransposeComp` moves the node axis last (`OM4_NEEDS.md`, Nice to have).

## Battery (`CADRE/battery_dymos.py`, from `CADRE/battery.py`)

- [x] `BatterySOCComp` (`BatterySOC`): the rate of an `SOC` state, in place of the legacy RK4.
- [x] `BatteryPowerComp` (`BatteryPower`)
- [x] `battery_path_constraints()` replaces `BatteryConstraints`: the KS aggregates over the time history become path constraints on `I_bat` and `SOC` (ADL-006).

## Thermal (`CADRE/thermal_dymos.py`, from `CADRE/thermal_temperature.py`)

- [x] `ThermalTemperatureComp` (`ThermalTemperature`): the rate of a 5-vector `temperature` state, in place of the legacy RK4. Checked against the legacy rate, verbatim; the legacy analytic state Jacobian was wrong (ADL-004). Picard propagation tested against the closed-form radiative cooling.

## Power (`CADRE/power_dymos.py`, from `CADRE/power.py`)

- [x] `CellVoltageGroup` (`Power_CellVoltage`): interpolates `cell_voltage_table()` (ADL-007) at one point per cell (`CellVoltagePointsComp`) and sums each panel's cells (`CellVoltageSumComp`). Values checked against the legacy outputs.
- [x] `SolarPowerComp` (`Power_SolarPower`)
- [x] `PowerTotalComp` (`Power_Total`)

## Comm (`CADRE/comm_dymos.py`, from `CADRE/comm.py`)

The legacy comm matrices are transposed so their names keep the from/to meaning (ADL-005): `O_IE` is inertial to Earth-fixed and `O_AB` is antenna to body.

- [x] `CommGSPosEarthComp` (`Comm_GSposEarth`); the original declared lat/lon in rad but converted them from degrees.
- [x] `CommEarthsSpinComp`, `CommEarthsSpinMtxComp` (`Comm_EarthsSpin`, `Comm_EarthsSpinMtx`); tested to turn the station eastward a quarter turn per 6 h.
- [x] `CommGSPosECIComp` (`Comm_GSposECI`)
- [x] `CommVectorECIComp` (`Comm_VectorECI`); takes the 3-vector `r_e2b_I`.
- [x] `CommVectorBodyComp` (`Comm_VectorBody`); now `O_BI^T r_b2g_I`.
- [x] `CommAntRotationComp`, `CommAntRotationMtxComp` (`Comm_AntRotation`, `Comm_AntRotationMtx`)
- [x] `CommVectorAntComp` (`Comm_VectorAnt`)
- [x] `CommVectorSphericalComp` (`Comm_VectorSpherical`)
- [x] `CommDistanceComp` (`Comm_Distance`)
- [x] `CommLOSComp` (`Comm_LOS`)
- [x] `CommBitRateComp` (`Comm_BitRate`); its `Dr` is the rate of a `Data` state, which replaces `Comm_DataDownloaded` (ADL-006). Picard propagation of `Data` tested.
- [x] `GainPatternGroup` (`Comm_GainPattern`): interpolates `antenna_gain_table()` (ADL-007). The legacy `fixangles` wrap is dropped: `CommVectorSphericalComp` already gives angles inside the table. Values checked against the legacy outputs.

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
- [x] `OrbitInitialComp` (`orbit_initial_comp.py`, from `Orbit_Initial` in `CADRE/orbit.py`): outputs the initial MEE rather than r, v; checked against the legacy r, v. The legacy `Orbit_Dynamics` RK4 is replaced by `TwoBodyEOM` in a phase.

## Blocked

Nothing is blocked. The three table components need om4 PR #404 (`OM4_NEEDS.md` #3), which
CADRE uses from the `../om4-pr404.git` worktree until it merges (ARCHITECTURE, Environment).
Every component above is ported and passes its own tests.

## Full model and MDP (`CADRE/cadre_ode.py`, `CADRE/cadre_mdp_dymos.py`)

- [x] `CADREODE` (`CADRE` in `CADRE/CADRE_group.py`): every discipline above, pointwise. The legacy RK4 components (`Orbit_Dynamics`, `ReactionWheel_Dynamics`, `ThermalTemperature`, `BatterySOC`, `Comm_DataDownloaded`) are dymos4 states; `BsplineParameters` is replaced by dymos4 controls (`DYMOS4_NEEDS.md` #4); `BatteryConstraints` by path constraints.
- [x] `cadre_phase`, `simulate` and `CADREMDP` (`CADRE_MDP_Group` in `CADRE/CADRE_mdp.py`): one Radau phase per launch point, shared `cellInstd`/`finAngle`/`antAngle`, the total-data objective and the SOC periodicity constraint (ADL-008). Checked against the legacy results: the initial design's `con5` to 2e-4 and its objective to 5% (`mdp_execute.pkl`), and design point 0 of the legacy optimized design (`data1346.pkl`) to 3% in data, 0.01 in SOC and 1.5 K in temperature; all six points of that design give 30,950 Gb against 30,892.
- [ ] Optimizing the MDP to the legacy optimum: not reached. On one design point (60x4 Radau grid, 7003 variables), seeded from a simulation of the legacy initial design, IPOPT stalls in restoration at about 132 Gb and Uno (`ipopt` preset) stops on a line-search failure at about 203 Gb, against the legacy 5066 Gb; both are left with a defect of `L` (5e-3) and `q_norm_err` (2e-4) unsatisfied. Uno's `filtersqp` preset does not finish its first iteration at this size. `cadre_phase` and `CADREMDP` also take `transcription='birkhoff'` (one segment of `num_nodes` nodes, states seeded from a Radau simulation and state rates from the phase's own ODE; needs dymos4 e58a2eb, `DYMOS4_NEEDS.md` #5). One-point Birkhoff optimizations at 241 nodes ran out of memory before their first iteration (two at once, on a 32 GB machine); a memory profile is next.

Differences from the legacy model: J2 only (the legacy orbit also had J3 and J4, under 3 km in 12 hours); controls at the nodes rather than 300-point B-splines; the roll subsumed by `q`.

`sun_dymos.azimuth_elevation` (used by `SunPositionSphericalComp` and `CommVectorSphericalComp`) uses `jnp.arctan2` and `jnp.mod`, which reject complex inputs, so those two components, and the whole ODE, are not complex-step safe as the conventions ask.

## Not started

The legacy model's own tests in `CADRE/test/` (`test_mdp.py`, `test_assembly.py`, ...) still need classic OpenMDAO and are not run.

The CI workflow (`.github/workflows/CADRE_test_workflow.yml`) still installs dymos 1.x and
OpenMDAO with conda; om4 and dymos4 are not published, so CI cannot run the ported tests yet.
