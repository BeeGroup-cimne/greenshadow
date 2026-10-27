# Rooftop PV: modules and strings on the roof planes

How `greenshadow.dimensionate.plan_pv_strings_in_buildings` fills each
detected roof plane with modules, values each module against the annual
irradiation, wires the survivors into strings and sums them per building -
once per mounting scenario, so the scenarios can be compared side by side.

```python
modules, summary = gs.plan_pv_strings_in_buildings(
    roof_planes,                                   # detect_roof_planes(_tiled)
    poa_basis_path=".../annual_solar_exposure_fourier_poa.tif",
    annual_energy_path=".../annual_solar_exposure_total.tif",
    dsm_path="dsm_no_vegetation_roofs.tif",        # refined roof DSM
    latitude_deg=41.67,
    output_path="pv_modules.gpkg",
    output_summary_path="pv_building_summary.csv")
best = summary[summary.scenario == "best"]
```

Inputs: the roof planes of [roof_inference.md](roof_inference.md) and the
irradiation rasters of [solar_energy.md](solar_energy.md), both built on the
DSM **with** vegetation, since the trees around a roof stay.

## 1. Two mounting regimes

Split by the plane's own tilt (`flat_slope_threshold_deg` = 5°):

| Regime | Planes | Orientation | Irradiation read from | Rows |
|---|---|---|---|---|
| **Coplanar** | inclined | the roof's own tilt and azimuth | `total` raster: it already measures direct + diffuse on that surface (Wh per pixel → Wh/m²) | edge to edge; never self-shade |
| **Racks** | flat | South (`flat_azimuth_deg`) at each tilt of `flat_tilts` (0, 5 ... 35°) | `fourier_poa` basis, reconstructed at the rack's tilt | pitched so no row shades the next at `row_shading_altitude_deg` |

The row-shading altitude defaults to `solar_min_altitude(latitude)`, the
sun at winter-solstice noon. The rack sweep deliberately includes 0°: flat
modules need no gap between rows, and on a small or awkward roof the extra
count can beat a steeper tilt that yields more per module but fits fewer.
Inclined planes facing outside `allowed_azimuth_range` (90-270°) are
skipped; None keeps every orientation.

## 2. Packing

* The plane polygon is shrunk by `setback_m` (access, wind uplift), and
  DSM pixels more than `obstacle_residual_m` (0.15 m) off the fitted plane
  are cut out as clutter (chimneys, aerials) when `dsm_path` is given.
* A grid of module cells, oriented along the plane's aspect (coplanar) or
  due South (racks), is laid fully inside the polygon. Its origin is swept
  over `phase_steps × phase_steps` sub-pitch offsets and the arrangement with
  the most modules wins - a half-cell shift is often worth a row.
* Each candidate size of `modules` (`DEFAULT_MODULES`: 2.30, 2.20 and
  1.78 × 1.14 m) is tried, and **one size per building** is kept, the one
  yielding most, so a building is not fitted with three models.
* Cells are ordered in serpentine, so consecutive modules of a string are
  neighbours.

## 3. Valuing, filtering, wiring

* Each module's annual POA is sampled under its footprint;
  energy = POA × area × `module_efficiency` (0.21) × `performance_ratio`
  (0.80); peak power from the spec or from area × efficiency at STC.
* **Shade filter.** A module is dropped when its POA is below `min_poa_ratio`
  (0.7) of the best module **on the whole building**, not of its own plane:
  one threshold per building removes the skirt a chimney darkens and the
  faces a neighbour overshadows all winter, instead of keeping the least bad
  modules of a hopeless plane.
* **Strings** of `string_size` (18) modules, the last one as short as
  `min_string_size` (1) allows. `string_size=None` wires each plane as one
  string, dropping the electrical grouping.
* `objective="energy"` (default) maximises annual kWh off the roof, which on
  a flat roof usually means laying modules flat; `"specific_yield"`
  maximises kWh/kWp, for when modules or the inverter are the constraint.

The Catalan pipeline asks for **roof capacity**, not a wiring design:
`string_size=None`, `min_poa_ratio=None`, `allowed_azimuth_range=None`,
`setback_m=0.05`, `phase_steps=5`. Override them with
`PipelineConfig.roof_pv_options`.

## 4. Scenarios and outputs

Every mounting costed survives as a **scenario**: `coplanar` (inclined
planes), `south_00` ... `south_35` (racks on flat planes), and `best`, the
per-plane optimum - the only scenario that covers a whole roof that has both
kinds of plane.

* `pv_modules.gpkg`: one layer per scenario (`poa_coplanar`, `poa_south_05`,
  ..., `poa_best`), one polygon per module with its plane, string,
  `poa_kwh_m2`, `energy_kwh` and `peak_kw`. The file is rewritten from
  scratch, so a scenario that disappears leaves no stale layer.
* `pv_building_summary.csv`: per building and scenario - `module_name`,
  `n_modules`, `n_strings`, `peak_kw`, `energy_kwh`,
  `specific_yield_kwh_kwp`, `roof_area_m2`, `buildable_area_m2`,
  `module_area_m2`, `coverage_ratio`. Areas are measured on the roof
  surface, so pitched roofs are not under-reported.

Both frames are long: a building appears once per scenario. **Filter to one
scenario** (usually `best`) before summing.

Alpicat, run of 2026-10-22 with the pipeline's capacity settings: 109,947
modules on 1,319 buildings in `best`, 47.9 MWp, 59.1 GWh/year
(1,235 kWh/kWp).

`n_workers` plans buildings in forked processes; the output does not
depend on it.

## 5. Limitations

* Coplanar modules take the plane's irradiation as sampled from the `total`
  raster; module temperature and row-to-row mismatch are inside the single
  `performance_ratio`.
* Racks face South only; East-West racks on flat roofs are not costed.
* No economics: this is a capacity and yield estimate. Parcels get NPV and
  the rest ([parcel_pv.md](parcel_pv.md)).
* Clutter removal depends on the refined roof DSM; regularized facets
  (`regularize_roof_planes`) are optional and not used by the pipeline.
