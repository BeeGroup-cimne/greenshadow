# Use Case: Municipal Solar Energy Assessment with greenshadow

*From classified LiDAR, cadastre, SIGPAC and reanalysis weather to rooftop
PV and agrivoltaic plans for a whole municipality: Alpicat (Lleida,
Catalonia), year 2023.*

## Overview

This use case follows the analysis of Alpicat, an agricultural municipality
on the Lleida plain with orchards, olive groves, arable land and a compact
town. The whole chain runs in one call; the sections below open each step,
show the function behind it and what it produced on Alpicat. Every figure
quoted was measured on this run unless said otherwise.

```python
import datetime as dt
import greenshadow as gs

config = gs.CataloniaConfig(
    wd="/mnt/data/greenshadow",
    year=2023,                          # annual irradiation and PV layouts
    days=[dt.date(2023, 12, 22)],       # winter solstice: shadows and video
    rural_parcels_year=2025,
    irradiance_decomposition="disc",
    semitransparent_vegetation=True,
)
results = gs.run_catalonia_municipalities({"Alpicat": "25023"}, config)
```

*[Caption for Screenshot/Video 1: "One configuration and one call: the pipeline starting on Alpicat, with its sections and progress bars."]*

## Workflow Summary

1. **Area and weather** - municipal boundary, ERA5-Land weather at its centroid
2. **LiDAR, buildings and parcels** - ICGC tiles, cadastral footprints, SIGPAC recintos
3. **Digital models** - DSM, DSM without vegetation, DEM
4. **Sky view factor**
5. **Crop rows** - the plantation geometry of every agricultural parcel
6. **Roof planes** - every roof facet of the municipality
7. **Semi-transparent canopy** - the trees, aged to 2023
8. **Daily analysis** - hillshades, videos and irradiation of the solstice
9. **Annual irradiation** - the year's hillshades and irradiation bases
10. **Rooftop PV** - modules and strings on every roof
11. **PV fields and agrivoltaics** - every scenario on every parcel, ranked

The detail of each computation is in [docs/](docs/README.md).

## Detailed Workflow

### 1. Area and weather

The boundary is the union of Alpicat's INE census tracts, found from its
cadastral code through `hypercadaster_ES`. The weather is ERA5-Land at the
boundary's centroid, with ERA5's own direct radiation and cloud cover:

```python
weather = gs.load_ERA5Land_with_direct_radiation(
    data_dir=weather_dir, lat_range=[lat, lat], lon_range=[lon, lon],
    ym_range=[dt.datetime(2023, 1, 1), dt.datetime(2023, 12, 31)],
    tz_local="Europe/Madrid", use_point_location=True, mode="instantaneous")
```

8,760 hourly records. Each is moved to the middle of the hour it averages,
and its global irradiance is split into beam and diffuse at the exact sun
position with the DISC model. Against PVGIS-SARAH3 for 2023 the annual GHI
agrees within 1.1 % and the DNI within 2 %
([docs/irradiance_era5.md](docs/irradiance_era5.md)).

*[Caption for Screenshot/Video 2: "Weather loading: ERA5-Land and ERA5 direct radiation at the municipality centroid for 2023."]*

### 2. LiDAR, buildings and parcels

```python
tiles = gs.icgc_tiles_from_geojson(aoi=aoi, aoi_epsg=4326,
                                   aoi_buffer_in_meters=500,
                                   pdf_plot_path="tiles.pdf",
                                   cache_path="tiles.geojson")
paths = gs.icgc_download_from_geojson(out_dir=laz_dir, tiles=tiles, aoi=aoi,
                                      aoi_epsg=4326, max_workers=8)
parcels = gs.load_spanish_sigpac_parcels(municipality_codes="25023",
                                         year=2025, workdir=..., target_crs=25831)
```

41 LiDAR tiles cover the municipality and a 500 m buffer, so shadows cast
from beyond the boundary are counted. The cadastral footprints come from
`hypercadaster_ES`; the SIGPAC download holds 3,254 recintos, of which
2,077 are agricultural.

*[Caption for Screenshot/Video 3: "LiDAR tiles found for Alpicat and its 500 m buffer (tiles.pdf)."]*

*[Caption for Screenshot/Video 4: "SIGPAC recintos of Alpicat coloured by land use, over the cadastral building footprints."]*

### 3. Digital models

Three models at 0.5 m on a 10,265 × 11,169 px grid, built tile by tile with
`build_digital_model_from_lidar_tiled`:

| Model | Contents | Used for |
|---|---|---|
| `dsm.tif` | everything, trees included | shadows of the area as it is |
| `dsm_no_vegetation.tif` | terrain and buildings | PV fields that clear the parcel; roofs |
| `dem.tif` | bare ground | canopy height, terrain slope |

The building class is corrected from the cadastral footprints
(`classification_correction_buildings_mode="strict"`), construction cranes
are removed (`crane_removal=True`), and terrain is smoothed while building
edges are kept sharp. Each model takes about 2.6 minutes ([docs/digital_models.md](docs/digital_models.md)).

*[Caption for Screenshot/Video 5: "The DSM of Alpicat: town, orchards and terrain at 0.5 m."]*

*[Caption for Screenshot/Video 6: "Before and after crane removal: a construction crane's jib and its false shadow."]*

*[Caption for Screenshot/Video 7: "DSM with and without vegetation side by side over an orchard."]*

### 4. Sky view factor

```python
gs.calculate_sky_view_factor(dsm_path="dsm.tif", out_path="svf.tif",
                             search_radius=250.0, angular_resolution=10.0,
                             hybrid=True, class_path="dsm_class.tif",
                             building_classes=(6, 17, 76, 77),
                             vegetation_classes=(3, 4, 5))
```

The share of the sky each pixel sees, which weights the diffuse irradiance;
computed on the GPU at full resolution around buildings and trees and on a
coarser grid elsewhere. A second one on the DSM without vegetation serves
the PV fields ([docs/hillshades_and_sky_view.md](docs/hillshades_and_sky_view.md#5-sky-view-factor)).

*[Caption for Screenshot/Video 8: "Sky view factor: open fields near 1, streets and courtyards much lower."]*

### 5. Crop rows

```python
rows, row_parcels = gs.detect_crop_rows(parcels, dsm_path="dsm.tif",
    dsm_no_vegetation_path="dsm_no_vegetation.tif",
    dsm_class_path="dsm_class.tif", output_path="crop_rows.gpkg")
```

The canopy `dsm − dsm_no_vegetation` of a plantation is a comb whose
direction and pitch are the planting geometry. Of Alpicat's 2,077
agricultural parcels, 345 are planted in rows - 8,866 rows, median spacing
4.4 m, canopy 2.2 m tall and 2.3 m wide, alleys 1.9 m - and 21 of them hold
more than one planting direction ([docs/crop_rows.md](docs/crop_rows.md)).

*[Caption for Screenshot/Video 9: "Detected crop rows over the orchards, coloured by planting block."]*

### 6. Roof planes

```python
roofs = gs.detect_roof_planes_tiled(dsm_path="dsm_no_vegetation.tif", ...,
                                    laz_dir=laz_dir, laz_tiles=tiles,
                                    resolution_refined=0.20, aoi=aoi)
```

The LiDAR returns are denoised over their Delaunay graph, rasterised into a
0.20 m roof DSM and segmented building by building. On the whole
municipality: 13,627 planes on 2,386 roofs in 24 minutes, of which the
1,750 roofs touching Alpicat itself are kept. Against hand-digitised planes
the detector reaches a mean IoU of about 0.70
([docs/roof_inference.md](docs/roof_inference.md)).

*[Caption for Screenshot/Video 10: "Roof planes of the old town in the 3D viewer, over the LiDAR returns coloured by class."]*

### 7. Semi-transparent canopy

```python
flown = gs.icgc_lidar_acquisition_date(aoi, cache_dir=laz_dir)
canopy = gs.build_canopy_model_from_rasters(
    dsm_path="dsm.tif", dem_path="dem.tif", class_path="dsm_class.tif",
    under_canopy_path="dsm_no_vegetation.tif",
    growth=gs.vegetation_growth(flown, 2023))
```

15,645 trees are resolved into crowns sized from their height and aged from
the LiDAR flight to 2023. In the hillshades a tree attenuates the beam by
the length of crown it crosses, deciduous leaves thinning in winter, instead
of casting a black block ([docs/vegetation_canopy.md](docs/vegetation_canopy.md)).

*[Caption for Screenshot/Video 11: "Opaque versus semi-transparent trees: dappled shade under the crowns."]*

### 8. Daily analysis: the winter solstice

```python
day = gs.generate_azimuth_altitude_pairs_by_date(
    date=dt.date(2023, 12, 22), lat=lat, lon=lon, granularity="PT15M",
    solar_degrees_discretization=0.1, weather_df=weather.to_pandas())
gs.generate_multiple_hillshades_gdal(raster_dir=".../20231222/hillshades",
    dsm_path="dsm.tif", azimuth_altitude=day["azimuth_altitude_pairs"],
    aoi=aoi, canopy=canopy)
gs.generate_daily_shadow_mp4(solar_df=day["df"], ..., canopy_leaf_type=0)
gs.compute_surface_solar_energy(..., out_path=".../daily_solar_energy.tif")
```

One hillshade every 15 minutes of daylight, with and without vegetation,
plus a copy with every object over 1.5 m blacked out; a video of each; and
the day's direct and diffuse irradiation. On the GPU each hillshade of the
full grid takes about 2 seconds.

*[Caption for Video 12: "Shadows over Alpicat through 22 December 2023, with semi-transparent trees."]*

*[Caption for Video 13: "The same day without vegetation: the shadows a cleared parcel would see."]*

*[Caption for Screenshot/Video 14: "Irradiation of the winter solstice: direct and diffuse."]*

### 9. Annual irradiation

```python
year = gs.generate_azimuth_altitude_pairs_by_year(year=2023, lat=lat, lon=lon,
    granularity="PT1H", solar_degrees_discretization=3,
    weather_df=weather.to_pandas())
gs.generate_multiple_hillshades_gdal(raster_dir=".../2023/hillshades",
    dsm_path="dsm.tif", azimuth_altitude=year["azimuth_altitude_pairs"],
    canopy=canopy)
gs.compute_surface_solar_energies(solar_df=year["df"],
    hillshade_dir=".../2023/hillshades", svf_path="svf.tif",
    modes=("total", "fourier_poa", "fourier_decomposition"),
    out_path=".../2023/annual_solar_exposure.tif", canopy_leaf_type=0)
```

4,248 daylight hours fall on 461 distinct sun positions at 3°, so 461
hillshades describe the year. All three modes are summed in one read of
them: the irradiation on each surface as it is (mean annual direct about
940 kWh/m²), and the bases from which the irradiation of any panel
orientation is reconstructed exactly. The same is repeated on the DSM
without vegetation, adding the bases for vertical fences and trackers
([docs/solar_energy.md](docs/solar_energy.md)).

*[Caption for Screenshot/Video 15: "Annual irradiation of Alpicat, 2023: roofs, streets, orchards and fields."]*

### 10. Rooftop PV

```python
modules, summary = gs.plan_pv_strings_in_buildings(roofs,
    poa_basis_path=".../annual_solar_exposure_fourier_poa.tif",
    annual_energy_path=".../annual_solar_exposure_total.tif",
    dsm_path="dsm_no_vegetation_roofs.tif", latitude_deg=lat,
    output_path="pv_modules.gpkg", output_summary_path="pv_building_summary.csv")
```

Every roof plane is packed with modules - flush on pitched roofs, on South
racks swept from 0 to 35° on flat ones - and each mounting is kept as a
scenario. In the best scenario per plane, at roof capacity: 109,947 modules
on 1,319 buildings, 47.9 MWp and 59.1 GWh a year, 1,235 kWh/kWp
([docs/rooftop_pv.md](docs/rooftop_pv.md)).

*[Caption for Screenshot/Video 16: "Modules on the roofs of the old town, coloured by annual yield."]*

### 11. PV fields and agrivoltaics

```python
scenarios, best, tables = gs.plan_parcel_pv_scenarios(parcels,
    poa_basis_path=".../annual_solar_exposure_no_vegetation_fourier_poa.tif",
    vertical_basis_path=..., tracking_basis_path=...,
    crop_rows=rows, vegetated_basis_path=".../annual_solar_exposure_fourier_poa.tif",
    economics=gs.Economics(), criterion="npv",
    output_path="parcel_pv.gpkg", output_summary_path="parcel_pv_scenarios.csv")
```

On every agricultural parcel, every system - fixed, single- and dual-axis
PV fields; solar grazing, overhead, interspace, single-axis and vertical
bifacial agrivoltaics; agrivoltaics over or between the existing crop rows -
is laid out in both row directions, at each pitch its crop allows. Each is
costed over 30 years against what the parcel earns today, and the parcel's
best is chosen by net present value; `best_viable` says whether it beats
leaving the parcel as it is. The economic and agronomic defaults are
placeholders to be calibrated with regional data
([docs/parcel_pv.md](docs/parcel_pv.md)).

*[Caption for Screenshot/Video 17: "Best scenario per parcel: PV fields, agrivoltaics and parcels where nothing beats the current crop."]*

*[Caption for Screenshot/Video 18: "One parcel, several scenarios: fixed field, vertical bifacial fences and overhead agrivoltaics side by side."]*

## Results and Outputs

### Generated Products

Everything is written under `{wd}/Alpicat/` and reused on the next run:

**Digital models:** `dsm.tif`, `dsm_no_vegetation.tif`, `dem.tif`, each with
`*_class.tif`, `*_slope.tif`, `*_aspect.tif` (radians); `svf.tif`,
`svf_no_vegetation.tif`; the refined roof DSM `dsm_no_vegetation_roofs.tif`.

**Vector layers:** `crop_rows.gpkg` (rows and planted parcels),
`roofs_rasterised.gpkg` (roof planes).

**Daily analysis** (`20231222/`): `hillshades/`,
`hillshades_no_vegetation/`, `hillshades_dem1.5m/`, a video of each, and
`daily_solar_energy_total.tif`.

**Annual analysis** (`2023/`): `hillshades/`, `hillshades_no_vegetation/`,
`annual_solar_exposure_{total,fourier_poa,fourier_decomposition}.tif`,
`annual_solar_exposure_no_vegetation_{fourier_poa,vertical_basis,tracking_basis}.tif`
and the sun tables of the last two.

**PV plans:** `pv_modules.gpkg` and `pv_building_summary.csv` (rooftop, one
layer per mounting scenario), `parcel_pv.gpkg` and `parcel_pv_scenarios.csv`
(parcels, one layer per scenario plus the best of each kind).

*[Caption for Screenshot/Video 19: "The output directory of Alpicat and its layers open in QGIS."]*

### How the results are checked

- **Irradiance** against PVGIS-SARAH3 (`eval_irradiance_pvgis.py`).
- **Tracker and vertical bases** against pvlib on a clear-sky year
  (`test_tracking_basis.py`): within 0.06 %.
- **Roof planes** against 112 hand-digitised planes of an Alpicat window.
- **Crop rows** against hand-drawn rows (`eval_crop_rows.py`) and on
  synthetic canopies (`test_crop_rows.py`).
- **Crane removal** on synthetic scenes (`test_crane_removal.py`), with a
  per-object report (`diagnose_crane.py`).
- **CPU and GPU** paths against each other (`test_solar_energy_backends.py`).

## Technical Specifications

### Performance on Alpicat

Measured on a workstation with an NVIDIA RTX PRO 5000 (Blackwell) GPU:

| Step | Time |
|---|---|
| One digital model, 41 tiles at 0.5 m | about 2.6 min |
| Roof planes, whole municipality | 24 min (6 workers, peak 10 GB per tile) |
| Hillshade, full grid, one sun position (GPU) | about 2 s |
| Annual irradiation | bound by reading the 461 hillshades, not by computing |

**Hardware.** 32 GB of RAM or more for 0.5 m municipalities; an NVIDIA GPU
is optional - every step has a CPU path - but makes the hillshades
an order of magnitude faster (38 s → 2 s per sun position). Hillshades are the largest
product on disk (ZSTD-compressed, about 150 MB each with canopy bands).

### Data

- **LiDAR:** ICGC territorial LiDAR, classified, 1 km tiles; flight dates
  from the ICGC metadata.
- **Weather:** ERA5-Land (0.1°, hourly) and ERA5 (0.25°) from Copernicus,
  one point per municipality.
- **Buildings:** Spanish cadastre via `hypercadaster_ES`.
- **Parcels:** SIGPAC recintos (FEGA), 2025.

## Applications and Benefits

### Municipal Planning Applications

**Solar Energy Planning:**
- Rooftop PV capacity per building, with the mounting each roof suits
- Which agricultural parcels could host a PV field or agrivoltaics, and at
  what return against their current crop
- Shading impact of buildings and trees on any site, any day of the year

**Agricultural Planning:**
- Plantation geometry (row direction, spacing, alleys) of every orchard and
  vineyard
- Agrivoltaic layouts that fit the machinery and the crop's tolerance to
  shade

### Economic Analysis

**Investment Decision Support:**
- NPV, IRR, ROI, LCOE and payback of every scenario on every parcel
- Land-equivalent ratio of energy plus crop
- Every assumption (prices, costs, crop margins) replaceable

*[Caption for Screenshot/Video 20: "From the municipality to the parcel: rooftop PV and agrivoltaic potential of Alpicat."]*

## Conclusion

On Alpicat the chain runs from the raw point cloud to a rooftop PV plan
for every building and a ranked set of PV and agrivoltaic scenarios for
every agricultural parcel, with shadows from terrain, buildings and
semi-transparent trees, and weather checked against an independent
satellite product. The same call runs any other Catalan municipality, and
the solar part runs on any region that supplies an area and a terrain
model ([docs/pipelines.md](docs/pipelines.md)).
