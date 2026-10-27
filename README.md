# greenshadow

Solar irradiation and PV potential at sub-metre resolution, from classified
LiDAR and reanalysis weather. greenshadow builds surface and terrain models
from the point cloud, casts the shadows of terrain, buildings and
semi-transparent trees for every sun position, sums them with the weather
into irradiation rasters, and lays out PV on the roofs and PV fields and
agrivoltaics on the rural parcels - costed and ranked against what each
parcel earns today.

It is built and validated on Catalan municipalities (ICGC LiDAR, Spanish
cadastre, SIGPAC), and the solar part is region-agnostic.

## What it does

- **Digital models from LiDAR**: DSM, DSM without vegetation and DEM at
  0.5 m, with class correction from cadastral footprints, crane removal,
  class-aware smoothing and seam-free tiling of whole municipalities.
- **Cast shadows** for every sun position, on the GPU when available, with
  trees as a semi-transparent medium (tree crowns, seasonal leaves, growth
  since the LiDAR flight).
- **Sky view factor** for the diffuse irradiance.
- **Weather**: ERA5-Land with ERA5's own direct radiation, aligned to the
  middle of each hour and split into beam and diffuse at the exact sun
  position, checked against PVGIS-SARAH3.
- **Irradiation rasters**: on the surface itself, or as exact bases for any
  fixed panel orientation, vertical bifacial fences and single- and
  dual-axis trackers (validated against pvlib).
- **Roof planes** segmented from the LiDAR, and **rooftop PV**: modules and
  strings per building, one scenario per mounting.
- **Crop rows**: direction, spacing, canopy and alley of every plantation.
- **PV fields and agrivoltaics** on every agricultural parcel - fixed,
  trackers, solar grazing, overhead, interspace, vertical bifacial, and
  agrivoltaics built into existing crop rows - with NPV, IRR, LCOE, payback
  and land-equivalent ratio.
- **Pipelines**: the whole chain in one call per area, cached on disk.

## Quick start

```python
import datetime as dt
import greenshadow as gs

results = gs.run_catalonia_municipalities(
    {"Alpicat": "25023"},              # output name: cadastral code
    wd="/path/to/working/directory",
    year=2023,                         # annual irradiation + PV layouts
    days=[dt.date(2023, 12, 22)],      # daily shadows, videos, irradiation
)
alpicat = results["Alpicat"]
alpicat.roof_pv_summary                # rooftop PV per building and scenario
alpicat.parcel_pv_best                 # best PV / agrivoltaic scenario per parcel
```

or from the shell:

```bash
python -m greenshadow.pipelines catalonia 25023 --name Alpicat \
    --wd /path/to/working/directory --year 2023 --day 2023-12-22
```

Every product lands under `{wd}/{name}/` and is reused on the next run.
[`example/get_solar_energy_from_municipality.py`](example/get_solar_energy_from_municipality.py)
shows every setting; [docs/pipelines.md](docs/pipelines.md) lists them, the
outputs, and how to add a region (a bare DEM is enough for PV fields).

## Documentation

One document per computation, in [docs/](docs/README.md):

| | |
|---|---|
| [Pipelines](docs/pipelines.md) | The chain in one call; configuration, outputs, adding a region |
| [Data sources](docs/data_sources.md) | ICGC LiDAR, cadastre, SIGPAC, protected areas, ERA5, satellite |
| [Digital models](docs/digital_models.md) | DSM / DEM: classes, correction, cranes, gap filling, smoothing, tiling |
| [Hillshades and sky view](docs/hillshades_and_sky_view.md) | Sun tables, cast shadows (CPU / CUDA), elevated shadows, videos, SVF |
| [Vegetation canopy](docs/vegetation_canopy.md) | Semi-transparent trees, crowns, leaves, growth |
| [Irradiance from ERA5](docs/irradiance_era5.md) | Time alignment and beam/sky split, validated against PVGIS-SARAH3 |
| [Irradiation rasters](docs/solar_energy.md) | Modes, units, POA bases, the one-pass reader |
| [Trackers and vertical PV](docs/pv_tracking_irradiance_plan.md) | Vertical and tracking bases, shading between rows, pvlib validation |
| [Roof inference](docs/roof_inference.md) | Roof planes from LiDAR, regularization, accuracy |
| [Rooftop PV](docs/rooftop_pv.md) | Modules and strings on the roofs |
| [Crop rows](docs/crop_rows.md) | Plantation geometry of the parcels |
| [Parcel PV and agrivoltaics](docs/parcel_pv.md) | Scenarios, crops, economics, ranking |

[CHANGELOG.md](CHANGELOG.md) lists what changed and when;
[usecase.md](usecase.md) walks through Alpicat step by step.

## Modules

| Module | Main functions |
|---|---|
| `pipelines` | `run_catalonia_municipalities`, `CataloniaMunicipalityPipeline`, `CataloniaConfig`, `SolarPipeline`, `PipelineConfig`, `TerrainModels` |
| `lidar` | `icgc_tiles_from_geojson`, `icgc_download_from_geojson`, `icgc_lidar_metadata`, `icgc_lidar_acquisition_date` |
| `dsm_dem_estimation` | `build_digital_model_from_lidar_tiled`, `build_digital_model_from_lidar_pure`, `remove_objects_from_dsm`, `build_dem_from_masked_dsm` |
| `shader` | `generate_azimuth_altitude_pairs_by_date` / `_by_year`, `generate_multiple_hillshades_gdal`, `apply_height_shadow_to_hillshades`, `generate_daily_shadow_mp4`, `calculate_sky_view_factor`, `compute_surface_solar_energy` / `_energies`, `reconstruct_poa_from_basis`, `reconstruct_vertical_poa`, `reconstruct_tracking_poa`, `build_canopy_model_from_rasters`, `vegetation_growth`, `load_hillshade` |
| `weather` | `load_ERA5Land`, `load_ERA5Land_with_direct_radiation`, `load_ERA5_radiation` |
| `buildings_infer` | `detect_roof_planes`, `detect_roof_planes_tiled`, `filter_roof_planes_by_aoi`, `regularize_roof_planes` |
| `roofs_3d` | `roof_planes_3d` (WebGL view of the planes over the LiDAR) |
| `dimensionate` | `plan_pv_strings_in_buildings`, `PVModuleSpec`, `DEFAULT_MODULES` |
| `rural_parcels` | `load_spanish_sigpac_parcels`, `filter_agricultural_parcels`, `sigpac_parcel_id`, `load_spanish_preserved_regions`, `tag_parcels_with_preserved_areas` |
| `crop_rows` | `detect_crop_rows` |
| `parcel_pv` | `plan_parcel_pv_scenarios`, `project_cashflows`, `MountingSystem`, `CropProfile`, `Economics`, `DEFAULT_SYSTEMS`, `DEFAULT_CROPS` |
| `satellite` | `get_ndvi`, `get_lst`, `get_vegetation_surface_indices` |
| `utils` | `force_raster_value_based_on_geometries`, `has_cuda`, `cuda_info` |

Most are exported at the package level (`gs.<function>`); the rest through
their module, e.g. `gs.dsm_dem_estimation.build_digital_model_from_lidar_pure`
or `gs.shader.load_hillshade`.

## Installation

GDAL first:

```bash
# Ubuntu / Debian
sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
sudo apt update && sudo apt install gdal-bin libgdal-dev python3-dev
# macOS
brew install gdal
```

Then the package, with pip, in a virtual environment:

```bash
git clone https://github.com/BeeGroup-cimne/greenshadow.git
cd greenshadow
python -m venv venv && source venv/bin/activate
pip install -e ".[weather,cuda]"
```

| Extra | Brings | Needed for |
|---|---|---|
| `weather` | `hypercadaster_ES`, `zarr`, `fsspec`, `aiohttp` | Weather, municipality boundaries, cadastre (and the Catalan pipeline) |
| `cuda` | `numba-cuda[cu13]` (CUDA toolkit and libNVVM as wheels) | GPU hillshades, SVF and irradiation; everything has a CPU path |
| `viz3d` | `plotly` | Optional 3D outputs |
| `pdal`, `solar` | PDAL, `pysolar`, `sunpy` | Optional |
| `dev` | `pytest`, `black`, `flake8` | Development |

Install the CUDA toolkit through the `cuda` extra rather than conda: numba
then finds the pip libNVVM. `gs.utils.cuda_info()` says whether the GPU is
usable.

The satellite indices need `pystac-client planetary-computer odc-stac
rioxarray`.

**Weather credentials.** ERA5 and ERA5-Land come from the Copernicus
Climate Data Store: create an account at
[cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) and put the
API URL and key in `~/.cdsapirc`.

## Tests

The test and evaluation scripts are at the repository root and run as
`python test_<name>.py`; [docs/README.md](docs/README.md#tests-and-evaluation-scripts)
lists what each one checks.

## Citation

If you use greenshadow in your research, please cite:

```bibtex
@software{greenshadow2025,
  author = {Broto, Jose Manuel and Mor, Gerard and Sellart, Maite},
  title = {greenshadow: Environmental shading analysis and solar energy assessment using LiDAR data},
  year = {2025},
  url = {https://github.com/BeeGroup-cimne/greenshadow},
  version = {1.0.0}
}
```

## 📄 License

This project is licensed under the **EUPL v1.2**. See the [license](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12) for details.

## Authors

- **Jose Manuel Broto** - Lead Developer - jmbroto@cimne.upc.edu
- **Gerard Mor** - Co-developer - gmor@cimne.upc.edu  
- **Maite Sellart** - Co-developer - tsellart@cimne.upc.edu

*BeeGroup - Centre Internacional de Mètodes Numèrics en Enginyeria (CIMNE)*

## Acknowledgments

- **ICGC** for providing open access to high-quality LiDAR data
- **Copernicus Climate Change Service** for ERA5Land weather data
- **BeeGroup-CIMNE** for supporting this research
- The open-source geospatial community