# Irradiance for trackers and vertical bifacial PV

**Status: implemented (2026-10-19).** This document describes how the annual
solar aggregation was extended so that `greenshadow.parcel_pv` can cost
single-axis trackers, dual-axis trackers and vertical bifacial fences, which
the `fourier_poa` basis cannot represent. § 7 lists where the implementation
departs from the original plan, and § 8 how it was validated. The parcel
planner as a whole is described in [parcel_pv.md](parcel_pv.md), and the
other irradiation modes in [solar_energy.md](solar_energy.md).

## 1. Why the fixed basis is not enough

`compute_surface_solar_energy(mode="fourier_poa")` loops over the discretised
sun positions *t* (azimuth θ from North clockwise, altitude φ). At each pixel
it adds the lit beam `E_t = DNI_t · lit_t` and the diffuse `D_t = DHI_t · svf`,
where `lit_t = hillshade_t / sin φ_t` recovers the shadow mask. It then
projects the beam onto three fixed axes, with the sun vector

    s = (s_e, s_n, s_u) = (cos φ sin θ, cos φ cos θ, sin φ)

    B_z = Σ E_t s_u     B_e = Σ E_t s_e     B_n = Σ E_t s_n
    D_sky = Σ D_t       GHI = Σ (E_t s_u + D_t)

A fixed panel with normal **n** gets `max(0, B · n)` of direct beam. That is
exact only if `s_t · n ≥ 0` at every sun position that counts. Two cases break
it:

* **Vertical fence.** The east face takes the morning beam and the west face
  the afternoon beam, and `B_e` nets the two against each other. On Alpicat a
  vertical east face came out with the same POA as a north face (521 kWh/m²).
* **Trackers.** The normal **n**(t) changes with every sun position, so the
  sum does not factorise.

Both are solved by accumulating more sums in the same loop over the same
hillshades. That costs one more pass over the stack, not a new shadow run.

## 2. The bases

Two new modes of `compute_surface_solar_energy`, both in Wh/m²:

**`mode="vertical_basis"`** (4 bands, `VERTICAL_BASIS_BANDS`): the lit beam on
a vertical panel facing each cardinal point.

| Band | Sum |
|---|---|
| `H_east` | Σ E_t · max(0, s_e) |
| `H_west` | Σ E_t · max(0, −s_e) |
| `H_north` | Σ E_t · max(0, s_n) |
| `H_south` | Σ E_t · max(0, −s_n) |

`reconstruct_vertical_poa(bands, d_sky, ghi, facing_azimuth_deg, albedo)`
adds half the sky (`D_sky / 2`) and half the ground (`GHI · albedo / 2`).
Both come from the `fourier_poa` basis built on the same hillshades.

**`mode="tracking_basis"`** (6 bands, `TRACKING_BASIS_BANDS`), with
`G_t = E_t s_u + D_t`:

| Band | Sum | Panel |
|---|---|---|
| `T2_dir` | Σ E_t | dual axis: normal incidence |
| `T2_sky` | Σ D_t (1 + s_u)/2 | tilt = zenith angle |
| `T2_gnd` | Σ G_t (1 − s_u)/2 | × albedo when read |
| `T1_dir` | Σ E_t cos AOI₁ | single axis, horizontal N-S |
| `T1_sky` | Σ D_t (1 + cos R)/2 | tilt = \|R\| |
| `T1_gnd` | Σ G_t (1 − cos R)/2 | × albedo when read |

For the single axis, `R = atan2(s_e, s_u)` is held within
`±tracker_max_rotation_deg` (60° by default, stored as a raster tag), and
`cos AOI₁ = s_e sin R + s_u cos R`.
`reconstruct_tracking_poa(bands, kind, albedo)` returns
`(direct, sky, ground)` for `"dual_axis"` or `"single_axis"`.

Both modes also write **`<raster>_sun.csv`** (`write_sun_table` /
`read_sun_table`). It holds each sun-position bin and its annual beam
(`dni_wh_m2`, normal to the sun) and diffuse (`dhi_wh_m2`) energy,
unobstructed. That is all the row-shading factors of § 3 need.

## 3. Shading between rows

The hillshades already include shading by terrain, buildings and trees.
Shading of one row by the next depends on the layout (its pitch or ground
cover ratio), not on the pixel. It is therefore computed once per layout over
the sun table and applied to the pixel's band as a factor
(`parcel_pv._layout_factors`, cached per system, direction and pitch):

* **Single axis.** The tracker backtracks, so no row shades the next, but each
  turns less than it would alone. The factors are the ratio of the backtracked
  to the free-tracking direct (Σ DNI·cos AOI) and sky diffuse
  (Σ DHI·(1 + cos tilt)/2), from pvlib's `tracking.singleaxis` (Anderson &
  Mikofski 2020) at the layout's ground cover ratio.
* **Dual axis.** For each sun position, the share of a tracker covered by its
  eight neighbours. The neighbours sunward of it are projected onto the plane
  facing the sun, and the rectangle overlaps are combined as
  `1 − Π(1 − cover)`.
* **Vertical fences.** On each face, the shadow the next fence casts up it.
  Its top is at `height − pitch · s_u / (s·n)`, measured against the panel
  between its clearance and its top, beam-weighted.

This separation assumes terrain shading and row shading are not concentrated
in the same hours. That is reasonable away from steep horizons; the error is
largest where both happen at low sun on a parcel with a high horizon.

## 4. Systems in `parcel_pv`

`MountingSystem` gained `mounting` (`"fixed"`, `"vertical"`,
`"single_axis"`, `"dual_axis"`), `bifaciality`, `ground_cover_ratio`,
`modules_along`, `tracker_spacing_m` and `tracker_max_rotation_deg`.
`DEFAULT_SYSTEMS` adds:

| Key | Kind | Geometry | Capex €/Wp | Opex €/kWp·yr |
|---|---|---|---|---|
| `pv_single_axis` | PV field | 1 portrait module across, GCR 0.35, N-S rows | 0.62 | 14 |
| `pv_dual_axis` | PV field | 5 × 6 landscape (≈ 30 modules), at least 21 m E-W × 23 m N-S, more until shading ≤ 2 % | 0.78 | 18 |
| `apv_single_axis` | agrivoltaic | axis 2.4 m, pitch 8–12 m set by the crop | 0.70 | 15 |
| `apv_vertical_bifacial` | agrivoltaic | 2 landscape modules high, bifaciality 0.75, pitch 8–12 m | 0.75 | 13 |

* Single-axis trackers are laid on North-South rows only. Dual-axis trackers
  are not laid in rows: each stands on a point of a lattice, its swept circle
  inside the setback, spaced at the least ground per tracker that keeps the
  neighbours' shading within `tracker_max_shading_loss` (2 %). Vertical
  fences go both ways: East/West faces on N-S rows, South/North faces on E-W
  rows.
* The crop under a vertical fence keeps `1 − 0.6 · height / pitch` of its
  light: about 75 % at 8 m and 80 % at 10 m, as in field trials.
* `plan_parcel_pv_scenarios` takes `vertical_basis_path=` and
  `tracking_basis_path=`. A system whose basis is not given is skipped with a
  warning, never costed on the wrong basis.
* The scenario table reports `mounting` and `row_shading_factor`.
* The pipeline ([pipelines.md](pipelines.md)) builds both bases on the
  no-vegetation hillshades and passes them in.

## 5. Memory

Each mode holds its own accumulators: 4 or 6 float32 rasters, against the 5 of
`fourier_poa`. On Alpicat (11,169 × 10,265 px) that is about 1.8 and 2.8 GB.

*Update 2026-10-20:* `compute_surface_solar_energies(modes=...)` now sums any
set of modes in one read of the hillshades, so the pipeline computes
`fourier_poa`, `vertical_basis` and `tracking_basis` together - 15 bands,
about 6.9 GB on Alpicat - on the GPU when they fit in its free memory and on
the CPU otherwise ([solar_energy.md](solar_energy.md#4-reading-the-hillshades-once)).

## 6. Elevation of the modules

Tracker axes (2–2.5 m) and overhead agrivoltaics (4–5 m) sit above the ground
the hillshades describe. The bases are built at ground level.

`apply_height_shadow_to_hillshades(height_m=...)` is **not** a raised-surface
hillshade: it blacks out every pixel standing more than `height_m` above the
DEM and keeps the shadows cast at ground level
([hillshades_and_sky_view.md](hillshades_and_sky_view.md#3-elevated-shadows-apply_height_shadow_to_hillshades)).
Shading at module height would need the shadow march to start `height_m`
above the ground, which is not implemented.

## 7. Departures from the original plan

* **Two modes, not one.** `vertical_basis` and `tracking_basis` are separate
  passes, which halves the peak memory (§ 5).
* **No East-West-axis single-axis bands (`T1ew_*`).** The plan marked them
  optional. Trackers are laid on N-S rows only; adding them is three more
  bands of the same form.
* **Row shading as factors only.** The fallback of per-GCR bands was not
  needed: the factors reproduce pvlib's backtracking exactly on open ground
  (§ 8), and the GCR stays a free scenario variable.

## 8. Validation

`test_tracking_basis.py` builds all three bases for an open, flat site: every
hillshade is `sin φ`, the SVF is 1, and the weather is a clear-sky (Ineichen)
year at Lleida. It then compares each panel with pvlib run over the same
discretised sun positions and energies, using the isotropic sky:

| Panel | Basis (kWh/m²) | pvlib (kWh/m²) | Difference |
|---|---|---|---|
| Fixed South 30° | 2,219.7 | 2,221.0 | −0.06 % |
| Dual axis | 3,050.4 | 3,050.4 | 0.00 % |
| Single axis, N-S, ±60° | 2,572.1 | 2,572.1 | 0.00 % |
| Vertical East | 1,092.0 | 1,092.0 | 0.00 % |
| Vertical West | 1,060.9 | 1,060.9 | 0.00 % |
| Vertical South | 1,575.1 | 1,575.1 | 0.00 % |
| Vertical North | 395.9 | 395.9 | 0.00 % |

The only gap is the fixed basis, whose `max(0, ·)` is applied to the annual
sum rather than to each hour. The East/West asymmetry comes from the 3°
azimuth flooring of the sun table, which pvlib was given too.

Gains of a lone panel over fixed South 30°: single axis +15.9 %, dual axis
+37.4 %. The literature figures are +15–25 % and +30–40 %.

At parcel level (`test_parcel_pv.py` § 10, same clear-sky year) and with
shading between rows included:

| | kWh/kWp | vs fixed | Row factor |
|---|---|---|---|
| Fixed South 30°, no-shading pitch | 1,776 | — | — |
| Single axis, GCR 0.35, backtracking | 1,973 | +11 % | 0.949 |
| Dual axis, 21 × 23 m | 2,393 | +35 % | 0.978 |
| Vertical bifacial, E/W faces, 8 m / 10 m | 1,452 / 1,471 | | |
| Vertical bifacial, S/N faces, 8 m / 10 m | 1,486 / 1,489 | | |

The crop between fences keeps 77 % / 82 % of its light at 8 / 10 m.

These are clear-sky figures. Under real weather (ERA5-Land, Lleida 2023), the
gains of a lone panel over fixed South 30° depend on the beam/sky split
chosen for the weather (`docs/irradiance_era5.md`):

| Split | Fixed South 30° (kWh/m²) | Single axis | Dual axis |
|---|---|---|---|
| `disc` (default) | 2,007 | +17.1 % | +35.9 % |
| `dirint` | 1,994 | +14.0 % | +31.6 % |
| `era5` | 2,029 | +18.9 % | +38.9 % |

Trackers work almost only on the beam, so a split that puts more energy in
the beam favours them. That is one more reason the split is checked against
PVGIS-SARAH3 before trackers are compared with fixed fields.

## 9. Still open

* East-West-axis and tilted-axis single-axis trackers.
* Trackers and vertical fences over existing crop rows (the `rows_*` systems
  are fixed-tilt only).
* View-factor reduction of the sky seen between dense rows. Diffuse is
  treated as unobstructed by neighbouring rows, which slightly overstates
  tight layouts.
* Capex and opex of the new systems are literature defaults and should be
  calibrated like the rest of `parcel_pv`.

## References

* Anderson, K., Mikofski, M. (2020). *Slope-Aware Backtracking for
  Single-Axis Trackers.* NREL/TP-5K00-76626.
* pvlib python 0.14: `tracking.singleaxis`,
  `irradiance.get_total_irradiance`.
* Next2Sun vertical bifacial agrivoltaics; field trials of vertical bifacial
  rows at 8–10 m (crops at ≥ 75 % of open-field irradiance).
* Rutgers Agrivoltaics Program: single-axis trackers, 2.4 m axis, 9.8–10.4 m
  row spacing.
