# Changelog

What changed in greenshadow, newest first. Dates are those of the commits;
"Unreleased", when present, is the working tree on top of the last one. The
documents in [docs/](docs/README.md) describe the current behaviour in
detail.

## 1.1.0 - 2026-11-09 - Roof imagery: orthophoto, SAM 3, shadows and terraces

### Changed
- **LiDAR roof obstacle detector off by default** (`ObstacleConfig.lidar`,
  `--lidar-obstacles` to turn it on): the roof obstacles come from the sky
  view factor, SAM 3 and the terraces; `planes_qc` then has no fit measures
  (`planes_qc_without_lidar`: `fit`/`method` "off", `needs_review` False).
  A `roof_obstacles.gpkg` made with the other setting is recomputed.
- **Cast shadows are no PV; little ones are obstacles**
  (`roof_obstacles.estimate_orthophoto_sun`, `detect_sam3_obstacles(...,
  dsm_path=)`): the sun of the orthophoto is estimated from its shadows per
  flight date (Alpicat 2022: 168 / 56 deg), the shade the DSM casts under it
  predicted, and a PV mask more than half in it is a shadow (8 of the 253
  Alpicat arrays: eave, parapet and dome shadows; no real array lost). A
  little compact shadow cast from the roof itself, or a little dark mask the
  DSM does not explain, becomes a `"shadow"` obstacle widened toward the sun
  over its caster; others are dropped. Dark masks of the other prompts are
  no longer dropped when little. `sun_*` columns in `coreg_log`; a
  `roof_obstacles.gpkg` without them is recomputed.
  ([docs/roof_obstacles.md](docs/roof_obstacles.md) §6)
- **Usable area keeps its corners**: the opening that removes what is
  narrower than a module runs on the pieces simplified by
  `ObstacleConfig.fragment_simplify_m` (0.2 m). On the pixel outlines of the
  planes it acted as an axis-aligned square and cut a triangle out of every
  corner of a roof turned against the grid. Alpicat: +2.5% usable area.
  ([docs/roof_obstacles.md](docs/roof_obstacles.md) §4)
- **SAM 3 prompts**: "roof hatch", "solar PV" (an extensive, array prompt
  with "solar panel(s)"), "HVAC system" and "antenna" added; the SAM 3
  window cache is keyed on the prompts, so every window is inferred again
  once.
- **Log of what is reused**: hillshades, elevated-shadow hillshades,
  videos, solar exposure rasters and SAM 3 windows kept from a previous run
  are logged as `Using existing ...`, with their path or the count kept,
  like the digital models. The roof obstacle detectors log one indented
  block each, and the transformers notes, the weight-loading bar and a
  stray rasterio warning no longer break the SAM 3 output.
  [docs/example_run_alpicat.md](docs/example_run_alpicat.md) times a rerun
  on what is on disk (25 min) and the roof obstacles (12.5 min from scratch).

### Added
- **Roof obstacles by their shadows** (`detect_shadow_obstacles`,
  `ObstacleConfig.shadows`, on with imagery): dark compact patches on each
  plane of the registered orthophoto, measured against the plane's own
  brightness (the darkest plane's along a seam), not cast from off the roof
  by the DSM, become `"shadow"` obstacles widened toward the sun over their
  casters, with the caster's height read from the shadow's length. Vent
  pipes and small chimneys that neither the LiDAR nor SAM 3 found. The sun
  is estimated once for SAM 3 and the shadows (`orthophoto_suns`).
  Alpicat: 6,659 shadows on 1,042 roofs in 10 s, −4.4% usable area.
  `n_shadows` in `coreg_log`; a `roof_obstacles.gpkg` without it is
  recomputed. ([docs/roof_obstacles.md](docs/roof_obstacles.md) §7)
- **Orthophotos** (`greenshadow.orthophoto`): the ICGC 25 cm RGB
  orthophoto of any edition read tile by tile from the territory-wide
  GeoPackages (HTTP range requests, no WMS) into a shared local cache, with
  the flight window of each sheet of the current edition
  (`icgc_orthophoto`, `icgc_orthophoto_sheets`,
  `nearest_icgc_orthophoto_year`); any other GDAL raster, e.g. PNOA sheets,
  through `local_orthophoto`; `read_orthophoto` crops onto the LiDAR's grid
  and CRS, undoing a displacement. ([docs/data_sources.md](docs/data_sources.md))
- **Orthophoto co-registration** (`roof_obstacles.coregister_orthophoto`):
  per roof, the normalised cross-correlation of the orthophoto's edges with
  the LiDAR's (DSM slope and plane boundaries) within +-5 m, steered by the
  consensus of the neighbouring roofs; `coreg_log` layer with the shift,
  score, status and the orthophoto and LiDAR dates (warning past
  `max_date_gap_years`). Pipeline: `ObstacleConfig(imagery=True)`,
  `--roof-imagery`, hook `SolarPipeline.load_orthophoto(aoi, planes)`;
  Catalonia reads the edition closest to the LiDAR flight
  (`CataloniaConfig.orthophoto_year`, `orthophoto_dir`,
  `--orthophoto-year`). ([docs/roof_obstacles.md](docs/roof_obstacles.md) §5)
- **SAM 3 on the orthophoto** (`roof_obstacles.detect_sam3_obstacles`,
  `greenshadow.sam3_worker`): text-prompted segmentation of each registered
  roof (chimney, skylight, roof window, solar panel, AC unit, vent, dormer)
  through the transformers port of Meta's SAM 3, in windows of the model's
  input upsampled 3x; score, area, on-roof, whole-roof and shadow filters,
  duplicates across prompts merged; outputs cached per window. Runs
  in-process or in another interpreter over a pipe
  (`ObstacleConfig.sam3_python`, `--sam3-python`), since torch's CUDA wheels
  would downgrade the CUDA 13 libraries numba-cuda uses. Without torch or
  the gated checkpoint the stage warns and continues on the LiDAR and SVF.
  `class` and `score` in `detections`, `classes` and `score` in `obstacles`,
  `n_sam3` in `coreg_log`. `--no-sam3`.
  ([docs/roof_obstacles.md](docs/roof_obstacles.md) §6)
- **Existing PV out of the usable area**: SAM 3 asks for "solar panel" and
  "solar panels" (the plural outlines whole arrays); their masks skip the
  area and shadow limits but must cover at most 0.8 of the roof and be dark
  (25th percentile of brightness <= 0.55), and are closed over 1 m gaps
  into one array per group (`sam3_extensive_prompts`,
  `sam3_extensive_max_roof_fraction`, `sam3_extensive_max_brightness`,
  `sam3_array_close_m`). First real SAM 3 run on all of Alpicat: 254 PV
  arrays, 8,510 m² on 150 roofs, 29 of 32 checked by eye correct.
- **Walkable terraces, balconies and patios out of the usable area**
  (`roof_obstacles.detect_terraces`, `ObstacleConfig.terraces`): a flat
  plane walled in by parapets, railings or higher floors (guard measured on
  the DSM without vegetation) is a terrace when enclosed (guard >= 0.75,
  <= 60 m²) or, with the registered orthophoto, tiled (guard >= 0.5 and
  half of it ceramic tile, garden or pool water). The whole plane comes
  out; `planes_qc` gains `terrace_guard`, `terrace_finish`, `terrace`.
  Alpicat: 1,605 planes, 31,700 m²; -4.2% modules, -2.7% energy (112,847
  modules, 60.52 GWh/year). ([docs/roof_obstacles.md](docs/roof_obstacles.md) §8)
- **Validation and debug figures** (`greenshadow.roof_obstacles_validation`):
  stratified labelling samples and a QGIS template, object precision/recall
  (IoU or distance matching), hit and area metrics per detector and fused,
  usable area and PV yield per variant, per-roof figures (orthophoto as
  published and registered, detections by source, labels, usable area);
  `ObstacleConfig.debug_pngs` / `debug_part_ids` in the pipeline. First
  Alpicat check against 38 provisional labels on 18 roofs: fused (LiDAR,
  SVF, SAM 3) recall 0.63, precision 0.29 (a lower bound: step walls and
  parapets count against it), yield within 4% of the labels'; SAM 3 alone
  0.80 / 0.21. (§8, §12)
- Tests: `test_orthophoto.py`, `test_sam3.py`, `test_obstacle_validation.py`.

### Changed
- `ObstacleConfig.obstacle_setback_m` and `edge_setback_m` 0.3 -> 0.1 m:
  +11% modules on Alpicat (117,819 modules, 62.23 GWh/year with SAM 3).

### Fixed
- Orthophoto reads over HTTP had no timeout: a stalled range request hung
  its tile download for good (seen on the current ICGC edition). GDAL now
  gives up after 120 s (30 s to connect) and retries.
- The ICGC tile map (`tiles.pdf`) was drawn through pyplot's GUI backend;
  on a desktop the Tcl interpreter it left behind was inherited by the
  forked roof-obstacle workers and aborted the run ("Tcl_AsyncDelete: async
  handler deleted by the wrong thread") on a municipality's first run. It
  is now drawn on a bare matplotlib Figure.

## 2026-10-27 - Roof obstacles: LiDAR residual detector, fusion and usable area (`01090ae`)

### Added
- **Roof obstacles** (`greenshadow.roof_obstacles`): chimneys, vents, AC
  units, parapets, skylights and light wells, found per roof plane in the
  LiDAR returns (signed residual to the plane refitted on them, MAD
  threshold, DBSCAN, concave hulls), in the LiDAR DSM where the returns are
  sparse, and in the sky view factor; fused, then cut out of each plane with
  an obstacle setback, a roof-edge setback and a module-width opening.
  `planes_qc` flags planes that need review. A pipeline stage between the
  roof planes and the rooftop PV, `roof_obstacles.gpkg`, configured by
  `PipelineConfig.roof_obstacles` (`ObstacleConfig`); new hook
  `SolarPipeline.roof_lidar()`. ([docs/roof_obstacles.md](docs/roof_obstacles.md))
- `plan_pv_strings_in_buildings(buildable_areas=...)` packs a usable area
  computed upstream instead of applying its own setback and DSM residual
  cut; `obstacle_residual_m=None` switches that cut off.
- Tests: `test_roof_obstacles.py`.

### Changed
- The Catalan pipeline's rooftop PV packs the usable area of the roof
  obstacle stage: fewer modules than before, where the obstacles and the
  0.3 m roof-edge setback are (`--no-roof-obstacles` restores the previous
  layout).

## 2026-10-27 - Pipelines, CUDA hillshades, one-pass irradiation and documentation (`6584fa3`)

### Added
- **Pipelines** (`greenshadow.pipelines`): the whole chain in one call per
  area. `SolarPipeline` holds the region-agnostic stages and
  `CataloniaMunicipalityPipeline` the Catalan sources and tuned parameters;
  `run_catalonia_municipalities(...)` and
  `python -m greenshadow.pipelines catalonia <code> ...`. A region without
  LiDAR only has to supply an area and a DEM. The example script is now a
  configuration plus one call. ([docs/pipelines.md](docs/pipelines.md))
- **CUDA hillshades**: `generate_multiple_hillshades_gdal(cuda=None)` uses
  the GPU when usable, marching each ray straight towards the sun. Alpicat,
  per sun position: opaque 38 s → 2 s, canopy 113 s → 1.7 s.
- **One-pass irradiation**: `compute_surface_solar_energies(modes=...)` sums
  any set of modes in one read of the hillshades, on CUDA accumulators when
  they fit in GPU memory and on the CPU otherwise, prefetching the next
  hillshade while the current one decodes.
- **Parallel roof detection**: `detect_roof_planes(n_workers=...)` segments
  building parts in forked processes; `plan_pv_strings_in_buildings` takes
  `n_workers` too. Results do not depend on the worker count.
- **Dual-axis trackers on a lattice**: spaced until the neighbours cost each
  tracker at most `tracker_max_shading_loss` (2 %) of its beam, instead of
  a fixed 21 × 23 m.
- **Documentation**: `docs/README.md` (index), `pipelines.md`,
  `data_sources.md`, `digital_models.md`, `hillshades_and_sky_view.md`,
  `vegetation_canopy.md`, `solar_energy.md`, `rooftop_pv.md`,
  `crop_rows.md`, `parcel_pv.md`; this changelog.
- Tests: `test_solar_energy_backends.py` (CPU and CUDA sums agree; one pass
  equals one call per mode), `test_pipelines.py` (DEM-only pipeline,
  offline).

### Changed
- **Hillshade storage**: ZSTD level 6 without predictor
  (`HILLSHADE_COMPRESSION`) instead of DEFLATE + floating-point predictor.
  Lossless; canopy hillshades write in 0.8 s instead of 4.4 s at less than
  half the size. Old hillshades still read.
- **Tiled DSM builder** about 5× faster on Alpicat (13 → 2.6 min, 41 tiles,
  bit-identical): points cropped to the tile window before the AOI test,
  halo strips cut once per LAZ, only X/Y/Z/class decompressed, and
  `n_workers=None` runs as many tiles as free memory allows. Workers decode
  LAZ with the serial `Lazrs` backend, since laspy's parallel backend hangs
  after `fork()`.
- **Irradiation rasters** are compressed with all cores (`num_threads`):
  a 5-band raster in 9 s instead of 40 s.
- **Progress bars** follow the structured logger's indentation, on stderr.
- `icgc_tiles_from_geojson(cache_path=...)` reuses the tiles found and the
  plot of a previous run.
- **CPU sky view factor** rewritten on the CUDA path's horizon model (every
  ray step up to `search_radius`, float32 angles, rays on the same pixels); the
  old one sampled every 2 % of the radius up to a height-angle cap.

### Fixed
- **Tiled roof detection**: every tile failed with "daemonic processes are
  not allowed to have children" once roof parts were detected in parallel;
  tile processes are no longer daemonic, and stopping a tile also stops its
  workers.
- **Incomplete rasters taken as done**: rasterio does not raise when GDAL
  fails to write (e.g. a full disk), so truncated hillshades were renamed
  into place and skipped by every later run. Hillshades, elevated-shadow
  hillshades and irradiation rasters are now checked block by block
  (`geotiff_is_complete`) after writing and before being reused, and the
  shadow videos are written under a temporary name.
  `apply_height_shadow_to_hillshades` raises, listing the files it could
  not process, instead of warning and returning fewer.

### Known issues
- The CPU hillshade path rotates the grid without reshaping it, so it drops
  the grid's corners (about 13 % at azimuth 150°) and truncates long low-sun
  rays. CPU hillshades made without an AOI keep that error until
  regenerated with `overwrite=True`; the CUDA path is not affected.

## 2026-10-19 - Parcel PV and agrivoltaic scenarios, crop rows, roofs and irradiance fixes (`04d08a0`)

### Added
- **Crop rows** (`detect_crop_rows`): row direction, spacing, canopy width
  and alley per block of each planted parcel, read off `dsm −
  dsm_no_vegetation`; working axis for the rest. Blocks of separate trees
  follow the larger planting's direction unless their canopy is flat that
  way (`hedgerow_max_periodicity`); superintensive olive and nut hedgerows
  down to 3 m. ([docs/crop_rows.md](docs/crop_rows.md))
- **Parcel PV and agrivoltaics** (`plan_parcel_pv_scenarios`): fixed,
  single- and dual-axis PV fields; solar grazing, overhead, interspace,
  single-axis and vertical bifacial agrivoltaics, pitch set by the crop;
  agrivoltaics built into existing crop rows; NPV, IRR, ROI, LCOE, payback
  and LER against the current-use margin; best scenario by a selectable
  criterion; one layer per scenario. ([docs/parcel_pv.md](docs/parcel_pv.md))
- **Tracker and vertical bases**: `vertical_basis` and `tracking_basis`
  modes with a sun table for the shading between rows, validated against
  pvlib. ([docs/pv_tracking_irradiance_plan.md](docs/pv_tracking_irradiance_plan.md))
- **`fourier_poa` basis** and `reconstruct_poa_from_basis`: exact annual
  POA for any fixed panel orientation.
- **Rooftop PV** (`plan_pv_strings_in_buildings`, module `dimensionate`,
  formerly `pv`): modules and strings on the roof planes, one layer per
  mounting scenario plus `best`. ([docs/rooftop_pv.md](docs/rooftop_pv.md))
- **Roof inference**: Delaunay-smoothed LiDAR, TIN-refined roof DSM,
  segmentation repairs (recover, coplanar merge, skirt), optional clutter
  removal from the SVF, `detect_roof_planes_tiled` for whole
  municipalities, `filter_roof_planes_by_aoi`, `regularize_roof_planes`
  (wireframe / intersections), `roof_planes_3d` WebGL viewer.
  ([docs/roof_inference.md](docs/roof_inference.md))
- **Semi-transparent vegetation**: tree crowns from the canopy height,
  Beer-Lambert attenuation with path-length bands in the hillshades,
  seasonal leaf table, growth from the LiDAR flight date
  (`icgc_lidar_acquisition_date`, `vegetation_growth`).
  ([docs/vegetation_canopy.md](docs/vegetation_canopy.md))
- **ERA5 direct radiation**: `load_ERA5Land_with_direct_radiation` joins
  ERA5's `fdir` and `tcc` point series to ERA5-Land.
- **Crane removal** in the DSM builder (`crane_removal=True`), with
  `diagnose_crane.py`.
- `filter_agricultural_parcels` and `sigpac_parcel_id` (FEGA use groups,
  stable SIGPAC id).
- Evaluation scripts `eval_irradiance_pvgis.py`, `eval_crop_rows.py`; tests
  for crane removal, crop rows, parcel PV and tracking bases.

### Changed
- **Irradiance split**: ERA5 values are moved to the middle of their hour
  (−30 min) and GHI is split into DNI/DHI at the table's own sun position;
  `irradiance_decomposition` `"disc"` (default), `"dirint"`, `"erbs"` or
  `"era5"`, compared against PVGIS-SARAH3.
  ([docs/irradiance_era5.md](docs/irradiance_era5.md))
- LiDAR coordinates kept in float64 and shifted before triangulating
  (float32 rounded UTM northings to 0.5 m).
- The annual weather download always includes the analysed year.

### Fixed
- **`dt_hours`**: daylight-only tables counted each day's last hour for the
  whole night, overstating annual energy by about a third and leaning it
  West. Energy rasters from before this date must be recomputed; hillshades
  are unaffected.

### Removed
- `greenshadow.pv` (replaced by `dimensionate`), `buildings_infer_v2` and
  the point-native roof detector `detect_roof_planes_v2`.

## 2026-07-01 - Digital models for Barcelona (`ebd0250`)

### Added
- `build_digital_model_from_lidar_tiled`: seam-free tiled DSM/DEM with a
  halo per tile.
- Classification correction from cadastral footprints
  (`"enforce"`, `"strict"`), class-aware smoothing, gap-filling strategies,
  `reclassify_unclassified_points`, `force_raster_value_based_on_geometries`.
- `compute_surface_solar_energy` (`total`, `monthly`,
  `fourier_decomposition`), `pv_potential_from_fourier_raster`,
  `aggregate_pv_by_polygons`, `generate_daily_shadow_mp4`,
  `load_hillshade`, hybrid CUDA sky view factor.
- First roof plane detectors (`buildings_infer`, `buildings_infer_v2`) and a
  PV layout engine (`pv`).
- `rural_parcels`: SIGPAC recintos, Natura 2000 / ENP protected areas.
- `satellite`: Sentinel-2 NDVI and Landsat LST.
- `logging_utils`: structured, indented logging.
- `usecase.md` and the municipality example.

### Removed
- `build_dem_from_dsm_obstacle_fast`, `build_dem_from_dsm_height_disruption`,
  `remove_buildings_from_dsm`, `refine_dem_with_buffered_interpolation`;
  the DEM is built from the LiDAR ground classes instead.

## 2026-01-22 - Weather data ready (`32c7fb4`)

### Added
- ERA5-Land download and loading.
- `calculate_sky_view_factor` (CUDA) and single hillshades
  (`generate_hillshade_gdal`).

## 2026-01-16 - Initial commit (`2c7c22f`)

- ICGC LiDAR tile discovery and download, DSM/DEM from point clouds,
  solar positions by date and by year, batch hillshades with cast shadows
  and QGIS temporal metadata, elevated shadows.
