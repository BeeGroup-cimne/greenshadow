# Changelog

What changed in greenshadow, newest first. Dates are those of the commits;
"Unreleased" is the working tree on top of the last one. The documents in
[docs/](docs/README.md) describe the current behaviour in detail.

## Unreleased (2026-10-19 → 2026-10-22)

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
