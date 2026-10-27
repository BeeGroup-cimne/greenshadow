# Crop rows: how each rural parcel is planted

How `greenshadow.crop_rows.detect_crop_rows` reads the plantation geometry
of every agricultural parcel off the LiDAR canopy - row direction, spacing,
canopy width and the alley between rows - and, where there is no canopy to
read, the direction the field is worked. The agrivoltaic planner fits its
rows to this ([parcel_pv.md](parcel_pv.md)).

```python
rows, parcels = gs.detect_crop_rows(
    sigpac_parcels,                       # load_spanish_sigpac_parcels
    dsm_path="dsm.tif",
    dsm_no_vegetation_path="dsm_no_vegetation.tif",
    dsm_class_path="dsm_class.tif",       # strongly recommended
    dem_slope_path="dem_slope.tif", dem_aspect_path="dem_aspect.tif",
    output_path="crop_rows.gpkg")
```

The module docstring of `greenshadow/crop_rows.py` explains the method in
full; this document summarises it.

## 1. Which parcels, and what they are called

Only agricultural SIGPAC uses are kept (`filter_agricultural_parcels`):
roads, buildings, water, forest and urban zones never reach the output.
Each parcel gets a `row_pattern`:

| `row_pattern` | Meaning | Alpicat (2,077 agricultural parcels) |
|---|---|---|
| `rows` | woody crop planted in rows; row columns filled | 345 |
| `scattered` | canopy present but not in rows (or not a plantation) | 828 |
| `herbaceous` | some canopy, too little to sweep | 249 |
| `empty` | no canopy, or too small to analyse | 655 |

The sweep runs only on woody uses (`SIGPAC_SPACING_PRIORS`: vineyard,
olive, fruit, nuts, citrus and their associations) - or, with
`woody_only=True`, on any parcel whose canopy shows a plantation anyway
(cover ≥ `woody_min_cover_frac` = 0.15, median height 1-6 m), since SIGPAC
carries a young orchard as arable land for years. Every parcel also gets a
`working_axis_deg` from its outline's principal axis: the way the field is
worked, and so the way machinery has to keep moving.

## 2. Method

The canopy is `chm = dsm − dsm_no_vegetation`, kept between 0.8 and 25 m and,
with the class raster, only on vegetation classes (without it a farm shed
is a tree and two sheds are a row).

1. **Sweep.** For each direction (1° steps, refined to 0.25°), canopy pixels
   are projected onto the direction's normal into a profile - the
   **fraction** of canopy at each offset, which cancels the parcel's own
   shape. The spectrum proposes a spacing within `row_spacing_range_m`,
   narrowed by the use's prior (which keeps a 7 m olive grove from being
   read at its 3.5 m harmonic). The direction is chosen by **how much of the
   profile's variance folding at that period explains**, which, unlike the
   power of the fundamental, does not penalise the sharp profile of a
   well-kept orchard. A direction is eligible only if the spectrum has power
   at the claimed period (`min_spectral_purity`).
2. **Blocks.** A parcel can hold several plantations. Canopy pixels are
   filed by (nearest row, cell along the row) and connected cells form a
   block with its own azimuth and pitch; a run of empty cells across the
   lattice (a corridor) separates two blocks, a few dead trees do not. Blocks
   are peeled off and the remainder re-swept up to `max_row_blocks` times.
   A block needs `min_rows` (4) rows, `min_row_periodicity` (0.25) and
   `min_captured_fraction` (0.45).
3. **One direction unless earned.** Separate trees are in rows along both
   axes of their grid. A block pointing another way than a larger one is
   folded along the larger one's direction and follows it, unless its canopy
   is flat that way (periodicity below `hedgerow_max_periodicity` = 0.16): a
   hedge, or a planting that really crosses. On Alpicat, trellised fruit
   hedgerows fold to 0.02-0.13 along their rows and grids of separate olive
   or nut trees to 0.19-0.28; 0.16 sits in that gap.
4. **Superintensive hedgerows.** Olive and nut parcels are swept again down
   to 3 m (`HEDGEROW_MIN_SPACING_M`), below their traditional band, and that
   reading is kept only where the canopy really is a hedge.
5. **Rows as lines.** Each row is tracked on its own trees, as the median
   cross-row position about nodes at least `row_segment_m` (10 m) apart, so
   rows bend round terraces without staircasing. A row is laid from one side
   of the parcel to the other, set in by `parcel_margin_m` (2 m), and broken
   only by a real hole - bare ground at least `row_hole_min_length_m` (30 m)
   long across at least `row_hole_min_rows` (5) rows - not by dead trees.
6. **Clean-up.** No two rows closer than `min_row_separation_m` (2 m) or
   three quarters of their spacing: the one on fewer trees is deleted, or cut
   back where they only touch.

## 3. Outputs

`output_path` gets two layers, read straight back on the next call unless
`update=True` (pass the same flag that rebuilds the digital models):

* **`crop_rows`**: one line per row - `parcel_id`, `row_index`,
  `block_index`, and **that block's** `row_azimuth_deg`, `row_spacing_m`,
  `canopy_width_m`, plus `row_length_m`, canopy height percentiles and cover.
* **`crop_row_parcels`**: every agricultural parcel with `row_pattern`,
  `row_confidence`, `working_axis_deg`, and for planted ones the largest
  block's `row_azimuth_deg`, `row_spacing_m`, `canopy_width_m`,
  `alley_width_m`, `row_periodicity`; `n_rows` and `n_blocks` over the whole
  parcel; `row_azimuth_spread_deg` (how much its blocks disagree);
  `captured_fraction`; canopy height and cover; terrain `slope_deg` /
  `aspect_deg`; `min_pv_area_ok` (≥ `min_pv_area_m2`, a flag only); and the
  protected-area columns when `exclusions` is given (flagged, never
  removed).

`row_azimuth_deg` is in [0, 180) and is the direction the rows run; a panel
facing across them has azimuth `row_azimuth_deg + 90` (whichever of the two
falls in 90-270°). `working_axis_deg` is in [0, 90) and names a *pair* of
perpendicular directions: try both.

Alpicat: 8,866 rows over 345 planted parcels; median spacing 4.41 m, canopy
2.16 m tall and 2.27 m wide, alley 1.93 m; 21 planted parcels hold more than
one direction.

## 4. Evaluation

* `test_crop_rows.py` runs the detector on synthetic canopies with a known
  answer.
* `eval_crop_rows.py --truth rows.gpkg --parcels sigpac.gpkg --detected
  crop_rows.gpkg` scores detected rows against hand-drawn ones, parcel by
  parcel.

## 5. Limitations

* Needs a canopy the 0.5 m CHM resolves: vineyards under ~1.8 m spacing and
  young plantations are at the edge of what it sees.
* `hedgerow_max_periodicity` was set on Alpicat's orchards; check the gap
  between hedgerows and tree grids before using it elsewhere.
* Herbaceous parcels get only the working axis, not a planting geometry.
* The SIGPAC use decides the spacing prior; a misdeclared use narrows the
  search to the wrong band.
