# Digital models: DSM and DEM from classified LiDAR

How `greenshadow.dsm_dem_estimation` turns classified LAZ tiles into the
rasters everything downstream reads: a surface model with everything on it,
the same surface without vegetation, and the bare ground, each with a class
raster, slope and aspect.

```python
gs.build_digital_model_from_lidar_tiled(
    laz_paths=paths, out_path="dsm.tif",
    out_classification_path="dsm_class.tif",
    out_slope_path="dsm_slope.tif", out_aspect_path="dsm_aspect.tif",
    resolution=0.5, epsg=25831,
    classification=(1, 2, 3, 4, 5, 6, 8, 9, 17, 77),
    aggregation="max", fill_gaps="smart_general", crane_removal=True,
    aoi=aoi, aoi_epsg=4326, aoi_buffer_in_meters=500,
    classification_correction_buildings_gdf=buildings, ...)
```

The tuned calls for all three models are in
[`greenshadow/pipelines/catalonia.py`](../greenshadow/pipelines/catalonia.py)
(`CataloniaMunicipalityPipeline.build_terrain`).

## 1. The three models

| Model | LiDAR classes | Aggregation | Gap fill | Used by |
|---|---|---|---|---|
| `dsm.tif` | 1, 2, 3, 4, 5, 6, 8, 9, 17, 77 | `max` | `smart_general` | Hillshades with trees, SVF, canopy, crop rows |
| `dsm_no_vegetation.tif` | 1, 2, 6, 8, 9, 17, 77 | `max` | `smart_general` | Hillshades for PV fields, roof detection, crop rows |
| `dem.tif` | 2, 8, 9, 17 | `max` | `dem` | Canopy height, elevated shadows, bare-ground slope |

The DEM additionally drops every point that falls on a building pixel of
`dsm_class.tif` (`filter_raster_path`, `filter_points_from_class=[6, 17, 77]`,
`filter_mode="exclude"`) and the isolated points left behind
(`isolated_points_exclusion_ratio=0.9` in a 7-pixel window), so the ground
under a roof is interpolated rather than read off stray returns. The
canopy model needs the difference `dsm - dem`, and the crop rows the
difference `dsm - dsm_no_vegetation`, so each pair must come from the same
grid and the same class corrections.

## 2. What one build does, in order

`build_digital_model_from_lidar_pure` (one window) runs:

1. **Grid** aligned to `resolution` over `bounds` (or the LiDAR extent),
   and the AOI mask of `aoi` buffered by `aoi_buffer_in_meters`.
2. **Point reading**: only the `classification` classes, optionally
   filtered by a class raster (`filter_raster_path`) and by class priority
   (`lidar_class_priority` within `lidar_class_window_diameter`, e.g. to drop
   ground returns under a roof).
3. **Aggregation** per pixel: `"max"` (surfaces), `"min"`, `"mean"`,
   `"smart_for_roofs"` (median after a 3-MAD outlier cut) or `"tin"`
   (interpolated on the Delaunay triangulation of the returns, used for the
   refined roof DSM).
4. **Classification** by majority vote, counted only where the point also
   contributes to the elevation, so the class raster describes the surface
   that was kept.
5. **Isolated pixels** removed (`isolated_points_exclusion_ratio`).
6. **Crane removal** (`crane_removal`, § 4).
7. **Gap filling** of elevation, then of classes (§ 5).
8. **Classification correction** from building footprints (§ 3).
9. **Smoothing** per class group (§ 6).
10. **Slope and aspect** (3 × 3 central differences, **radians**),
    optionally smoothed by class (`slope_smooth_window`,
    `aspect_smooth_window`).
11. **Clip** to the buffered AOI - last, so the steps above still saw the
    data beyond the edge.

## 3. Classification correction

The building class of the LiDAR is not always where the buildings are, and
the DEM, the canopy and the roof detector all key on it. Cadastral
footprints (`classification_correction_buildings_gdf`, buffered by
`classification_correction_building_buffer_m`) correct it on the class
raster:

* `"enforce"`: pixels inside a footprint become
  `classification_correction_building_class_value` (6).
* `"strict"`: the same, and building-class pixels **outside** every
  footprint become unclassified. Used by the Catalan pipeline.

With `classification_correction_avoid_vegetation_classes=True` (default),
vegetation pixels inside a footprint keep their class, so a tree over a
courtyard is not turned into roof.

## 4. Crane removal

Construction cranes and masts win the `max` aggregation and cast long false
shadows. `crane_removal=True` removes an object only if it is **thin**
(`crane_width_m`), **tall** above its surroundings (`crane_height_m`,
`crane_min_clearance_m`), **long and elongated** (`crane_min_length_m`,
`crane_min_elongation`), **small** (`crane_max_area_m2`) and **seen
through** by the laser: the point cloud must show a second surface well
below the top (`crane_point_gap_m`) in enough of the object
(`crane_min_seeds`, `crane_min_seed_fraction`). Vegetation never votes an
object alive. Chimneys, lift shafts, antennas and spires fail at least one
test and stay; overhead cables pass them all and are removed too.

The removed pixels are restored from the returns that reached the roof
through the lattice, when those sit no higher than the surface around the
object (`crane_second_tolerance_m`), and interpolated otherwise
(`crane_fill="dw_plane"`) - before the general gap fill, which would flood a
60 m jib with the street level. `crane_exclusion_gdf` protects areas from
the filter. It runs only with `max`, `smart_for_roofs` or `tin` aggregation,
is designed for resolutions ≤ 0.5 m, and must run before smoothing.
`diagnose_crane.py` reports the decision for each object;
`test_crane_removal.py` checks the guarantees on synthetic scenes.

## 5. Gap filling

Pixels without returns are filled only where valid data is close
(`_interpolation_support_mask`), never extrapolated:

| `fill_gaps` | Method |
|---|---|
| `"smart_general"` | By hole size: ≤ 3 px bilinear, ≤ 200 px nearest, larger holes the **minimum** of their boundary; nothing filled below the lowest contour |
| `"dem"` | Fast DEM interpolation, for bare ground |
| `"smart_for_roofs"` | Roof-aware fill |
| `"nearest"`, `"bilinear"`, `"idw"`, `"median"`, `"harmonic"`, `"biharmonic"`, `"dw_plane"` | Classic methods |

`"smart_general"` floods any hole over 200 px with its lowest boundary
value. That is right for a gap in the ground and wrong for a large hole on a
roof, which becomes a trench; crane removal fills its own holes first for
that reason.

## 6. Smoothing

`smooth_classes` lists the classes to smooth, grouped as terrain,
vegetation and buildings (`terrain_classes`, `vegetation_classes`,
`building_classes`), and `smooth_mode` gives each group an operator -
`"gaussian"`, `"box"`, `"median"`, `"opening"` or None:

```python
smooth_mode={"terrain": "gaussian", "buildings": None, "vegetation": None}
```

Each operator averages only the valid pixels of its own group (normalised
convolution), so terrain is smoothed without dragging roof edges down into
it. `smooth_unary=True` smooths all `smooth_classes` as one group (the DEM).
The Catalan models use an 11-pixel window, one iteration.

## 7. Whole municipalities: `build_digital_model_from_lidar_tiled`

Same parameters, one tile per LAZ file. Each tile is built on a buffered
window (halo, `tile_buffer_pixels`, by default derived from the smoothing,
slope and gap-fill reach) from its own and its neighbours' points, and only
its core is written into the pre-allocated output, so the result has no
seams.

* **Workers.** `n_workers=None` runs as many forked tiles as free memory
  allows, at most half the cores; `1` is sequential. The workers decode LAZ
  with the serial `Lazrs` backend (see
  [data_sources.md](data_sources.md#1-icgc-lidar-greenshadowlidar)).
* **Speed** (Alpicat, 41 tiles at 0.5 m, 2026-10-22): about 13 min → 2.6 min,
  bit-identical output. Points are cropped to the tile window before the AOI
  test, each neighbour contributes a pre-cut halo strip (each LAZ decoded
  twice instead of about nine times), and only X, Y, Z and class are
  decompressed.
* **Precision.** Coordinates stay float64 and are shifted near the origin
  before triangulating. Casting a UTM northing near 4.6e6 m to float32 rounds
  it to 0.5 m; every DSM built before that fix (and its cached
  rasterizations) carries that error until rebuilt.

## 8. Other functions

* `remove_objects_from_dsm(dsm_path, objects, out_path, buffer_m)` masks
  geometries (e.g. buildings) as nodata; `build_dem_from_masked_dsm`
  interpolates such a DSM into a DEM. An alternative to the class-based DEM
  when only footprints are available.
* `reclassify_unclassified_points` reassigns class-1 returns to building,
  vegetation or ground from their height above ground and planarity, when
  more than `unclassified_global_fraction_threshold` of a tile is
  unclassified.

## 9. Limitations

* Tuned for the ICGC classes and 0.5 m. Other LiDAR needs its own class sets
  (`classification`, `building_classes`, `vegetation_classes`).
* The slope and aspect rasters are radians; the roof planner's per-plane
  columns are degrees. Do not mix them.
* A rebuilt model does not invalidate what was derived from it (SVF,
  hillshades, crop rows, roofs): rebuild those too.
