# Pipelines: from an area to irradiation and PV layouts in one call

`greenshadow.pipelines` chains every computation of the library for one area
and caches each product, so a call is cheap to repeat and a run that stops
part-way resumes where it stopped.

```python
import datetime as dt
import greenshadow as gs

results = gs.run_catalonia_municipalities(
    {"Alpicat": "25023"},                 # output name: cadastral code
    wd="/mnt/data/greenshadow",
    year=2023,                            # annual irradiation + PV layouts
    days=[dt.date(2023, 12, 22)],         # daily shadows, videos, irradiation
)
alpicat = results["Alpicat"]
alpicat.parcel_pv_best                    # best scenario per parcel
alpicat.annual["fourier_poa"]             # path of the annual POA basis
```

From the shell:

```bash
python -m greenshadow.pipelines catalonia 25023 --name Alpicat \
    --wd /mnt/data/greenshadow --year 2023 --day 2023-12-22
```

`--no-roofs`, `--no-parcels`, `--no-weather`, `--no-videos`,
`--opaque-vegetation`, `--recalculate-digital-models` and
`--recalculate-solar-exposure` switch the corresponding parts off or force
them; everything else is a field of the configuration below.

## 1. How it is split

| Part | Where | What it decides |
|---|---|---|
| `SolarPipeline` | `pipelines/base.py` | The region-agnostic stages: weather, sky view factor, crop rows, canopy, daily and annual shadows and irradiation, rooftop and parcel PV |
| `PipelineConfig` | `pipelines/base.py` | Settings shared by every region |
| `CataloniaMunicipalityPipeline` | `pipelines/catalonia.py` | Where a Catalan municipality's area, terrain, buildings, parcels and roofs come from, and the tuned parameters of each |
| `CataloniaConfig` | `pipelines/catalonia.py` | `PipelineConfig` plus the Catalan data directories |

A region subclasses `SolarPipeline` and implements its hooks:

| Hook | Returns | Default |
|---|---|---|
| `area()` | AOI, shapely geometry in EPSG:4326 | required |
| `build_terrain(aoi)` | `TerrainModels` (paths of the rasters) | required |
| `load_parcels(aoi)` | parcels for PV fields and agrivoltaics | None |
| `detect_roofs(aoi, terrain)` | `RoofPlanes` (facets + their DSM) | None |
| `acquisition_date(aoi)` | survey date, to age the canopy | None |
| `svf_options(class_path)` | extra `calculate_sky_view_factor` arguments | `{}` |
| `load_weather(aoi)` | hourly weather (GHI, DHI, DNI ...) | ERA5-Land + ERA5 `fdir` at the centroid |

## 2. Stages and their conditions

`run()` executes, in order:

1. **Area** and its centroid.
2. **Weather** (`include_weather`). A failure is reported and the run
   continues without it; the irradiation and PV stages then step aside with
   a warning, since there is nothing to weight the hillshades with.
3. **Parcels** (`parcels`).
4. **Terrain** (`build_terrain`), then the **sky view factors**: `svf.tif`
   on the DSM, and `svf_no_vegetation.tif` on the DSM without vegetation
   when the terrain has one.
5. **Crop rows**, when there are parcels, `crop_rows=True` and the terrain
   has a separate surface without vegetation (the canopy is their
   difference).
6. **Roof planes** (`roofs`).
7. **Canopy**, when `semitransparent_vegetation` and the terrain carries a
   class raster and a surface without vegetation; aged from
   `acquisition_date()` to the simulated year when
   `grow_vegetation_to_simulated_year`.
8. **Each day in `days`**: sun table, hillshades with the canopy, the same
   without vegetation, elevated shadows at `elevated_height_m` (skipped when
   DSM and DEM are the same raster), videos (`daily_videos`), and the daily
   irradiation.
9. **The `year`**: sun table, hillshades with the canopy, irradiation modes
   `total`, `fourier_poa` and `fourier_decomposition` in one pass, rooftop
   PV, then - when there are parcels - hillshades without vegetation, the
   `fourier_poa`, `vertical_basis` and `tracking_basis` modes on them, and
   the parcel PV and agrivoltaic scenarios.
10. **Cleanup** of the GDAL `*.xml` side files under the area's directory
    (`cleanup_xml`).

A terrain without a separate surface without vegetation (a bare DEM in both
`dsm` and `dem`) runs one set of hillshades and computes every mode on it.

## 3. Configuration

`PipelineConfig` (all fields have defaults except `wd`; at least one of
`year` and `days` is required):

| Field | Default | Meaning |
|---|---|---|
| `wd` | - | Working directory; outputs go to `{wd}/{name}` |
| `year`, `days` | None, `()` | Annual and daily analyses |
| `tz_local` | `Europe/Madrid` | Time zone of the sun tables |
| `include_weather`, `weather_data_dir` | True, `{wd}/weather` | Weather loading and cache |
| `irradiance_decomposition` | `"disc"` | Beam/sky split, see [irradiance_era5.md](irradiance_era5.md) |
| `resolution` | 0.5 | Raster resolution, m |
| `aoi_buffer_m` | 500 | Terrain beyond the AOI, for shadows cast from outside it |
| `svf_search_radius_m`, `svf_angular_resolution_deg` | 250, 10 | Sky view factor |
| `daily_granularity`, `daily_solar_step_deg` | `PT15M`, 0.1 | Daily sun table |
| `annual_granularity`, `annual_solar_step_deg` | `PT1H`, 3 | Annual sun table |
| `elevated_height_m` | 1.5 | Elevated shadows; None skips them |
| `daily_videos` | True | MP4 per hillshade set |
| `semitransparent_vegetation`, `vegetation_leaf_type` | True, 0 | Canopy model; 0 deciduous, 1 evergreen |
| `grow_vegetation_to_simulated_year`, `vegetation_*_growth_m_per_year` | True, 0.25 / 0.20 | Canopy growth since the survey |
| `roofs`, `parcels`, `crop_rows` | True | Which plans to make |
| `economics`, `criterion` | `Economics()`, `"npv"` | Parcel scenario valuation and ranking |
| `roof_pv_options`, `parcel_pv_options` | `{}` | Passed to the planners over the pipeline's defaults |
| `recalculate_digital_models` | False | Rebuild DSMs, DEM, SVF, crop rows and roofs |
| `recalculate_solar_exposure` | False | Rebuild the irradiation rasters from the existing hillshades |
| `cleanup_xml` | True | Remove `*.xml` side files at the end |

`CataloniaConfig` adds `hypercadaster_dir` (`{wd}/hypercadaster_ES`),
`laz_dir` (`{wd}/icgc`, shared by municipalities), `preserved_areas_dir`,
`rural_parcels_year` (2025), `tag_preserved_areas` (False), `epsg` (25831),
`download_workers` (8) and `render_roof_3d_view` (False).

## 4. Outputs

Under `{wd}/{name}/`:

| Path | Product |
|---|---|
| `dsm.tif`, `dsm_class.tif`, `dsm_slope.tif`, `dsm_aspect.tif` | DSM with everything, its classes, slope and aspect (radians) |
| `dsm_no_vegetation*.tif` | The same without vegetation |
| `dsm_no_vegetation_*_roofs.tif` | Refined roof DSM and its derivatives (roof detector) |
| `dem.tif`, `dem_slope.tif`, `dem_aspect.tif` | Bare ground |
| `svf.tif`, `svf_no_vegetation.tif` | Sky view factors |
| `tiles.geojson`, `tiles.pdf` | LiDAR tiles found, and their map |
| `rural_parcels/` | SIGPAC download cache |
| `crop_rows.gpkg` | Layers `crop_rows` and `crop_row_parcels` |
| `roofs_rasterised.gpkg`, `roofs_3d.html` | Roof facets; optional 3D view |
| `YYYYMMDD/hillshades*/`, `YYYYMMDD/*.mp4` | Daily hillshades and videos |
| `YYYYMMDD/daily_solar_energy_total.tif` | Daily irradiation |
| `YYYY/hillshades/`, `YYYY/hillshades_no_vegetation/` | Annual hillshades |
| `YYYY/annual_solar_exposure_<mode>.tif` | Annual irradiation per mode |
| `YYYY/annual_solar_exposure_no_vegetation_<mode>.tif` (+ `_sun.csv`) | Bases for the parcel PV |
| `pv_modules.gpkg`, `pv_building_summary.csv` | Rooftop PV, one layer per scenario |
| `parcel_pv.gpkg`, `parcel_pv_scenarios.csv` | Parcel PV and agrivoltaics, one layer per scenario |

`run()` returns a `PipelineResult` holding those paths and the frames
(`parcels`, `crop_row_lines`, `roofs`, `canopy`, `roof_pv_summary`,
`parcel_pv_scenarios`, `parcel_pv_best`, `parcel_pv_tables` ...).

## 5. Adding a region

A region needs an area and a terrain; the rest is optional:

```python
from greenshadow.pipelines import SolarPipeline, PipelineConfig, TerrainModels

class MyDEMPipeline(SolarPipeline):
    def area(self):
        return my_polygon_wgs84

    def build_terrain(self, aoi):
        dem = self.path("dem.tif")
        ...                                  # write or fetch the DEM there
        return TerrainModels(dsm=dem, dem=dem)

MyDEMPipeline("site", PipelineConfig(wd="/data", year=2023)).run()
```

With a bare DEM in both `dsm` and `dem`, the canopy, crop rows, roofs,
elevated shadows and the hillshades without vegetation are skipped on their
own; `test_pipelines.py` runs exactly that case. Two things are still
needed for PV fields on such a region:

* **Parcels without SIGPAC.** `plan_parcel_pv_scenarios` keeps only parcels
  whose `uso_sigpac` is agricultural and identifies them by the SIGPAC
  7-tuple, unless `parcel_id_column` is given. Other parcels need those
  columns, or an adapter.
* **Units at coarse resolution.** The `total` raster is Wh per pixel, so its
  values scale with the pixel area: a 30 m DEM gives values 3,600 times
  those of a 0.5 m one for the same irradiation. The `fourier_poa` and
  tracking bases are per m² and do not.

## 6. Limitations

* The weather is one point, the AOI centroid. Large or mountainous areas
  vary more than that.
* A rebuilt digital model does not invalidate the products derived from it;
  pass `recalculate_digital_models` and `recalculate_solar_exposure`.
* Roofs are detected over the whole municipality even for a days-only run;
  pass `roofs=False` when only the shadows are wanted.
