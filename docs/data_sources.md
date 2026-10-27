# Data sources

Where each input comes from, how it is fetched and cached, and what the
library assumes about it. Everything is downloaded on first use and kept on
disk; repeat calls are offline.

| Input | Provider | Coverage | Function | Cache |
|---|---|---|---|---|
| Classified LiDAR (LAZ) | ICGC, territorial LiDAR | Catalonia | `icgc_tiles_from_geojson`, `icgc_download_from_geojson` | `laz_dir` |
| LiDAR flight dates | ICGC metadata | Catalonia | `icgc_lidar_metadata`, `icgc_lidar_acquisition_date` | `laz_dir` |
| Municipal boundary, buildings | Spanish cadastre and INE census tracts, via `hypercadaster_ES` | Spain | `hc.download`, `hc.mergers.*` | `hypercadaster_dir` |
| Agricultural parcels | SIGPAC recintos (FEGA) | Spain | `load_spanish_sigpac_parcels` | `{area}/rural_parcels` |
| Protected areas | Natura 2000 and ENP (MITECO) | Spain | `load_spanish_preserved_regions` | `preserved_areas_dir` |
| Hourly weather | ERA5-Land (Copernicus), via `hypercadaster_ES` | Global | `load_ERA5Land` | `weather_data_dir` |
| Direct radiation, cloud cover | ERA5 single levels (Copernicus) | Global | `load_ERA5_radiation` | `weather_data_dir` |
| NDVI, land surface temperature | Sentinel-2 L2A, Landsat 8/9 (Planetary Computer) | Global | `get_ndvi`, `get_lst` | none |

## 1. ICGC LiDAR (`greenshadow.lidar`)

The ICGC territorial LiDAR is published as 1 km tiles of classified LAZ
(index: `datacloud.icgc.cat/.../lidar-territorial-tall.json`).

* `icgc_tiles_from_geojson(aoi, aoi_epsg, aoi_buffer_in_meters=...)` returns
  the tiles intersecting the buffered AOI. `cache_path` keeps them in a
  GeoJSON reused while the AOI, buffer and `centroid_only` match, so a
  re-run neither fetches the index nor redraws `pdf_plot_path`.
* `icgc_download_from_geojson(out_dir, tiles, ...)` downloads in parallel
  (`max_workers`) and skips files already present. Keep one `laz_dir` for
  every area: neighbouring municipalities share tiles.
* The classes follow the ICGC specification
  (`lidar-territorial-v3r1`): 1 unclassified, 2 ground, 3/4/5 low, medium
  and high vegetation, 6 building, 8 key points, 9 water, 17 bridge, 77
  other constructions, among others. The tuned class sets of each digital
  model are in [digital_models.md](digital_models.md).
* `icgc_lidar_metadata(cache_dir)` downloads the per-tile metadata once
  (flight window `DATA_INICI`/`DATA_FI`, density `DENSITAT`, class list).
  `icgc_lidar_acquisition_date(aoi, cache_dir)` reduces each tile to the
  midpoint of its flight window and combines the tiles covering the AOI,
  area-weighted (`statistic="mean"`, or `"median"`, `"min"`, `"max"`). It
  feeds the canopy growth ([vegetation_canopy.md](vegetation_canopy.md)).

LAZ decoding: laspy's default backend decodes in a rayon thread pool, which
does not survive `fork()`. A forked worker that decodes LAZ after its parent
did hangs. The tiled DSM builder therefore switches its workers to the
serial `Lazrs` backend; any new fork-based worker that reads LAZ must do the
same.

## 2. Cadastre and census tracts (`hypercadaster_ES`)

The Catalan pipeline uses `hypercadaster_ES` for two things:

* **The municipality boundary**: the union of its INE census tracts,
  selected by cadastral code (`hc.utils.cadaster_to_ine_codes`).
* **Building footprints** (`hc.mergers.join_cadaster_building`), which
  correct the LiDAR building class before the DSM is rasterised
  ([digital_models.md](digital_models.md#3-classification-correction)).
  They are fetched only when a digital model has to be built.

`hc.download` refreshes its DEM and postal-code layers on every call.

## 3. SIGPAC parcels (`greenshadow.rural_parcels`)

`load_spanish_sigpac_parcels(municipality_codes, year, workdir, target_crs)`
downloads the year's recintos from the FEGA ATOM service
(`fega.gob.es/atom/<year>/rec_<year>/`) for the given 5-digit codes.
`harmonize_fields=True` adds English translations of the use and incidence
codes (`harmonize_sigpac_fields`).

* **Identity.** SIGPAC has no single key: a recinto is the 7-tuple
  provincia/municipio/agregado/zona/poligono/parcela/recinto.
  `sigpac_parcel_id` joins it into strings like `"25-23-0-0-12-45-3"`; every
  downstream join uses it.
* **Agricultural uses.** `filter_agricultural_parcels` keeps the uses FEGA
  counts as agricultural - arable (`IV`, `TA`, `TH`), permanent crops
  (fruit, nuts, olive, citrus, vineyard and their associations) and pasture
  (`PA`, `PR`, `PS`) - and drops water, roads, buildings, forest, scrub,
  unproductive land, urban zones, landscape elements, the "other" zones
  (`ZC`, `ZV`) and parcels with no use. A missing `uso_sigpac` column raises
  rather than letting roads reach the planners. Crop rows and parcel PV both
  apply it.

## 4. Protected areas

`load_spanish_preserved_regions(data_dir, datasets=["natura2000", "enp"])`
downloads the MITECO layers, optionally dissolved per resource.
`tag_parcels_with_preserved_areas(parcels, preserved, min_intersection_ratio)`
adds `is_preserved`, `preserved_resource`, `preserved_area_m2` and
`preserved_area_ratio`. Protected parcels are **flagged, never removed**:
protection constrains a project rather than ruling it out. The Catalan
pipeline does this only with `tag_preserved_areas=True`.

## 5. Weather (`greenshadow.weather`)

* **ERA5-Land** (0.1°, hourly): loaded by `hypercadaster_ES` at one point
  (`use_point_location=True`), through the ARCO Zarr store when the
  `weather` extra is installed and the CDS queue otherwise. Its `ssrd` is
  the all-sky global horizontal irradiance.
* **ERA5 single levels** (0.25°): `ssrd`, `fdir` (all-sky direct) and `tcc`
  (cloud cover) as a point time series
  (`reanalysis-era5-single-levels-timeseries`), snapped to the 0.25° grid
  and cached as `era5_radiation_<start>_<end>_<lat>_<lon>.csv`. ERA5 runs
  about five days behind real time.
* `load_ERA5Land_with_direct_radiation` joins the two; if the ERA5 download
  fails it returns ERA5-Land alone with a warning.

Both need a Copernicus CDS account and `~/.cdsapirc`. How the stamps are
aligned and the global split into beam and diffuse is
[irradiance_era5.md](irradiance_era5.md).

## 6. Satellite indices (`greenshadow.satellite`)

Optional (`pystac-client`, `planetary-computer`, `odc-stac`, `rioxarray`):

* `get_ndvi(polygon, start_date, end_date, composition="median" | "best_scene")`
  from Sentinel-2 L2A at 10 m, with the post-2022-01-25 reflectance offset.
* `get_lst(...)` from Landsat 8/9 band 10, single-channel algorithm of
  Jiménez-Muñoz & Sobrino (2014).
* `get_vegetation_surface_indices` returns both;
  `search_sentinel2_scenes` / `search_landsat_scenes` list scenes.

They are not used by the solar chain.
