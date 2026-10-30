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

- [x] `ExposedAreaGroup` (`Solar_ExposedArea`): interpolates `exposed_area_table()` (ADL-007) with om4's `StructuredMetaModelComp`, from om4 PR #404 (`OM4_NEEDS.md` #3); values checked against the legacy outputs. Broadcasts the static `finAngle` to the nodes (`FinAngleNodesComp`) and moves the node axis last (`ExposedAreaTransposeComp`).

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

## Not started

The rest of the legacy OpenMDAO 2.x model (`CADRE/*.py` at the top level: `parameters.py`'s
`BsplineParameters`, which likely becomes dymos controls rather than a port, and the CADRE group
and MDP) and its tests in `CADRE/test/`. Its
RK4 integration is to be replaced by dymos4 phases. **When porting it, apply `O_BI^T`**: the
legacy code uses `O_BI` as inertial to body (`r_b2g_B = O_BI r_b2g_I` in comm, `r_e2s_B` in sun),
while the ported components use body to inertial (ADL-005).

The CI workflow (`.github/workflows/CADRE_test_workflow.yml`) still installs dymos 1.x and
OpenMDAO with conda; om4 and dymos4 are not published, so CI cannot run the ported tests yet.
