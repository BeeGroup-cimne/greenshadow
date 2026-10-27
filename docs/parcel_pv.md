# PV fields and agrivoltaics on rural parcels

How `greenshadow.parcel_pv.plan_parcel_pv_scenarios` lays out every PV and
agrivoltaic scenario on every agricultural parcel, costs it over the
project's life against what the parcel earns today, and picks each parcel's
best.

```python
scenarios, best, tables = gs.plan_parcel_pv_scenarios(
    sigpac_parcels,
    poa_basis_path=".../annual_solar_exposure_no_vegetation_fourier_poa.tif",
    vertical_basis_path=".../..._no_vegetation_vertical_basis.tif",
    tracking_basis_path=".../..._no_vegetation_tracking_basis.tif",
    crop_rows=crop_row_lines,                       # detect_crop_rows
    vegetated_basis_path=".../annual_solar_exposure_fourier_poa.tif",
    economics=gs.Economics(), criterion="npv", latitude_deg=41.67,
    output_path="parcel_pv.gpkg", output_summary_path="parcel_pv_scenarios.csv")
```

The module docstring of `greenshadow/parcel_pv.py` explains the reasoning in
full. The irradiation of trackers and vertical fences, and the shading
between their rows, are in
[pv_tracking_irradiance_plan.md](pv_tracking_irradiance_plan.md).

## 1. Inputs and which basis for what

| Input | Built on | Used for |
|---|---|---|
| `poa_basis_path` (`fourier_poa`) | DSM **without vegetation** | Every new layout: a field clears the parcel, so today's trees will not shade it |
| `vertical_basis_path`, `tracking_basis_path` | same hillshades | Vertical fences, single- and dual-axis trackers; a system whose basis is missing is skipped, never costed on the wrong one |
| `vegetated_basis_path` (`fourier_poa`) | DSM **with** vegetation | Tables down the alleys of an existing plantation, which stand among its trees |
| `crop_rows` | `detect_crop_rows` | Agrivoltaics built into the rows already planted |

Parcels are filtered to agricultural SIGPAC uses, identified by the SIGPAC
7-tuple (or `parcel_id_column`) and reprojected to the basis CRS. Parcels
under `min_parcel_area_m2` (2,000 m²) are not laid out.

## 2. Scenarios

A scenario is a mounting system, a row direction and a pitch - and, for
agrivoltaics, the crop that sets the pitch. Its name is the same on every
parcel (`pv_fixed_EW_8.9m`, `apv_overhead_NS_6.0m_forage`,
`pv_dual_axis_points_21x23m`, `apv_rows_overhead_k1` ...), so one layer
compares all parcels.

### Mounting systems (`DEFAULT_SYSTEMS`)

| Key | Kind | Mounting | Clearance | Capex €/Wp | Opex €/kWp·yr |
|---|---|---|---|---|---|
| `pv_fixed` | PV field | fixed, 2 portrait across; S 30° on E-W rows, E/W tent 10° on N-S rows | 0.8 m | 0.55 | 12 |
| `apv_grazing` | agrivoltaic | as `pv_fixed`, sheep under the panels | 1.2 m | 0.58 | 10 |
| `apv_overhead` | agrivoltaic | overhead, 1 across, S 20°, crop underneath, bifacial gain 7 % | 4.5 m | 0.95 | 15 |
| `apv_interspace` | agrivoltaic | fixed, 1 across, crops between rows | 0.8 m | 0.62 | 13 |
| `pv_single_axis` | PV field | horizontal N-S tracker, GCR 0.35, backtracking | 1.5 m | 0.62 | 14 |
| `pv_dual_axis` | PV field | 5 × 6 landscape per tracker, on a lattice ≥ 21 × 23 m | 1.0 m | 0.78 | 18 |
| `apv_single_axis` | agrivoltaic | N-S tracker over wide rows, pitch set by the crop | 2.4 m | 0.70 | 15 |
| `apv_vertical_bifacial` | agrivoltaic | vertical fence, 2 landscape high, bifaciality 0.75 | 0.8 m | 0.75 | 13 |
| `apv_rows_overhead` | agrivoltaic | overhead over every existing crop row | 4.5 m | 0.95 | 15 |
| `apv_rows_alley` | agrivoltaic | tables down every existing alley | 0.8 m | 0.62 | 13 |

* **Directions.** Synthetic rows run North-South (modules split East/West)
  and East-West (modules facing South), on the basis raster's own grid
  (grid North differs from true North by under 2° in Catalonia).
  Single-axis trackers are laid on N-S rows only. Dual-axis trackers stand
  on lattice points spaced until their neighbours cost them at most
  `tracker_max_shading_loss` (2 %) of the beam.
* **Pitch.** A PV field takes the closest pitch at which no row shades the
  next at winter-solstice noon; an agrivoltaic system takes the pitches its
  crop allows (machinery width, or light for overhead structures).
* **Existing rows.** On a parcel with detected rows, `rows_*` systems follow
  each block's own direction and spacing: over every crop row, or down every
  alley that still leaves `tractor_lane_m` (2 m) beside the table, its
  buffers and the canopy. The table is narrowed, turned to landscape and
  laid flatter until it fits.
* Rows stop `setback_m` (5 m) inside the parcel and pieces shorter than
  `min_row_length_m` (20 m) are dropped.

### Crops (`DEFAULT_CROPS`)

| Crop | Margin €/ha·yr | Shade threshold | Slope | Pitches (m) |
|---|---|---|---|---|
| Rainfed cereal | 300 | 0.10 | 0.9 | interspace, single axis, vertical 10-12; overhead 7 |
| Forage | 500 | 0.15 | 0.6 | interspace, single axis, vertical 8-10; overhead 6-8 |
| Leafy vegetables | 3,000 | 0.25 | 0.5 | interspace 6-8; overhead 5-7; single axis 8-10 |
| Berries | 6,000 | 0.30 | 0.4 | overhead 5-7 |
| Vineyard | 1,500 | 0.25 | 0.4 | interspace 7.5-9 |
| Sheep grazing | 80 | 0.30 | 0.3 | grazing under a field |

## 3. Energy

Each table's annual POA is reconstructed from the basis under it, for its
orientation; energy = POA × module area × `module_efficiency` (0.21) ×
`performance_ratio` (0.80), × (1 + bifacial gain), × the layout's
row-shading factor for trackers and fences. Albedo 0.2.

## 4. What the farm keeps

The crop's response to shade is piecewise linear (Laub et al. 2022): no
loss up to `shade_threshold` of radiation reduction, then `shade_slope` of
relative yield per unit beyond it. The reduction is a first-order estimate
from the share of ground under panels, not a ray trace:

| Layout | Cropped share of the rows' ground | Radiation reduction |
|---|---|---|
| Overhead, grazing | `1 − land_loss_frac` (posts) | ground cover of the panels |
| Interspace, trackers | strip between tables and their buffers | half the ground cover |
| Vertical fence | strip between fences and their buffers | `0.6 · height / pitch` (≈ 75 % light kept at 8 m) |

The crop keeps the parcel outside the buildable area too. The farm's
margin is `gross_margin × relative_yield × cropped area`, and existing-row
scenarios keep the crop already growing, with its shade response by SIGPAC
use (`SIGPAC_USE_SHADE_RESPONSE`).

## 5. Valuation and ranking

`project_cashflows` over `Economics` (real terms):

| Field | Default |
|---|---|
| `electricity_price_eur_mwh` | 40 (Spanish solar capture price, 2025) |
| `price_escalation` | 0 |
| `discount_rate` | 0.06 |
| `lifetime_years` | 30 |
| `degradation_per_year` | 0.005 |
| `fixed_project_cost_eur` | 0 (grid connection depends on the substation, not in the data) |

Year 0 is capex (peak × €/Wp + fixed cost); each later year sells the
degraded energy, pays opex, and adds the farm margin **minus the baseline**
the parcel earns today (`SIGPAC_USE_MARGIN_EUR_HA` by use, e.g. arable 300,
fruit 2,000, vineyard 1,500, olive 800 €/ha). Building over an orchard
therefore pays for the orchard. Results: `npv_eur`, `irr`, `roi`,
`payback_years`, `lcoe_eur_mwh`, and the land-equivalent ratio

    ler = energy / (best PV field's energy on the parcel) + crop yield share

`criterion` picks the best: `"npv"` (default), `"irr"`, `"roi"`,
`"energy"`, `"ler"` maximised; `"lcoe"`, `"payback"` minimised. Ranking by
energy always picks the densest field, which is rarely the best investment
once the displaced crop is counted.

## 6. Outputs

* `scenarios_df` / `parcel_pv_scenarios.csv`: one row per parcel and
  scenario - system, kind, mounting, crop, direction, pitch, `n_rows` /
  `n_trackers`, `n_modules`, `peak_kwp`, `energy_mwh`,
  `specific_yield_kwh_kwp`, `row_shading_factor`, crop area, radiation
  reduction, `relative_crop_yield`, capex, opex, revenue, farm and baseline
  margins, and the valuation columns.
* `parcels_gdf` / layer `parcel_pv_best`: one row per parcel with the best
  scenario overall (`best_*`) and the best of each kind (`pv_field_*`,
  `agrivoltaic_*`, `existing_rows_*`). **`best_viable`** says whether the best
  actually beats leaving the parcel as it is (NPV > 0): a ranking names a
  winner even among losers.
* `tables_gdf` / layers `ppv_<scenario>`: one polygon per table - each
  continuous stretch of modules down a row, at its plan width - or per
  dual-axis tracker, plus `ppv_best`, `ppv_best_pv_field`,
  `ppv_best_agrivoltaic` and `ppv_best_existing_rows` with each parcel's own
  winner (`tables="all"`; `"best_per_kind"`, `"best"` or None write less).

Like the rooftop planner, the frames are long: filter to one scenario before
summing.

## 7. Validation

`test_parcel_pv.py` runs every system on a synthetic basis. Under a
clear-sky year at Lleida, at parcel level with shading between rows:
fixed South 30° 1,776 kWh/kWp, single axis +11 %, dual axis +35 %; vertical
bifacial fences 1,452-1,489 kWh/kWp, with the crop keeping 77-82 % of its
light. Details in
[pv_tracking_irradiance_plan.md § 8](pv_tracking_irradiance_plan.md#8-validation).

## 8. Limitations

* **Every economic and agronomic default is a placeholder** - capex, opex,
  crop margins, `SIGPAC_USE_MARGIN_EUR_HA`, shade responses - to be
  calibrated against regional data; pass your own `systems`, `crops`,
  `economics` and `baseline_margin_eur_ha`.
* No grid-connection cost by distance to a substation.
* Single-axis trackers are horizontal and North-South only; East-West or
  tilted axes, and trackers or fences over existing crop rows, are not
  implemented.
* Crop shading is estimated from ground cover, not ray-traced; diffuse
  light between dense rows is treated as unobstructed.
* Built for SIGPAC parcels: other parcel data needs `uso_sigpac` (for the
  agricultural filter and the baseline margin) and an id column.
