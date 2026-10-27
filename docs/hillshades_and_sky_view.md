# Sun tables, hillshades with cast shadows, and the sky view factor

How `greenshadow.shader` decides where the sun is, which pixels it reaches,
and how much of the sky each pixel sees. These three products are what the
irradiation rasters ([solar_energy.md](solar_energy.md)) are summed from.

## 1. Sun tables

```python
day  = gs.generate_azimuth_altitude_pairs_by_date(
    date=dt.date(2023, 12, 22), lat=lat, lon=lon, tz_name="Europe/Madrid",
    granularity="PT15M", solar_degrees_discretization=0.1,
    weather_df=weather.to_pandas(), irradiance_decomposition="disc")
year = gs.generate_azimuth_altitude_pairs_by_year(
    year=2023, lat=lat, lon=lon, granularity="PT1H",
    solar_degrees_discretization=3, weather_df=..., ...)
```

Each returns `{"df": ..., "azimuth_altitude_pairs": [...]}`. `df` has one
row per time step (local and UTC time, exact azimuth and elevation, the
discretised `azimuth_altitude`, and with weather `GHI`, `DNI`, `DHI`, the
other weather columns and `clearness_index`); `azimuth_altitude_pairs` is
the list of **unique** discretised positions, which is what the hillshades
are computed for.

* **Discretisation.** Positions are floored to
  `solar_degrees_discretization` degrees, so many hours share one
  hillshade: 0.1° for a day (61 positions on 21 June at 15 min), 3° for a
  year (4,248 daylight hours → 461 hillshades at Alpicat). Flooring to 3°
  lowers the annual beam on the horizontal by about 2.4 % at Lleida; the
  hillshades and the energy use the same floored positions.
* **Weather.** Each weather record is moved to the middle of the hour it
  averages (`weather_time_offset_minutes=-30` for ERA5) and interpolated onto
  the table's times; GHI is then split into DNI and DHI at the table's exact
  sun position by `irradiance_decomposition` (`"disc"` default, `"dirint"`,
  `"erbs"`, `"era5"`, `"weather"`). Why, and how each option compares with
  PVGIS-SARAH3: [irradiance_era5.md](irradiance_era5.md).
* `only_daylight=True` keeps positions above the horizon.

## 2. Hillshades with cast shadows

```python
gs.generate_multiple_hillshades_gdal(
    raster_dir=".../hillshades", dsm_path=dsm_path,
    azimuth_altitude=year["azimuth_altitude_pairs"],
    aoi=aoi, canopy=canopy)          # canopy=None: opaque obstacles
```

One GeoTIFF per sun position, named `AAA_EE.tif` (azimuth, elevation in
whole degrees), holding the local hillshade in [0, 1] with every pixel in a
cast shadow set to 0. Existing files are kept unless `overwrite=True`, which
is what makes the `(azimuth, altitude)` stack a cache shared by every day of
the year. Pixels outside `aoi` are nodata.

| Path | When | How |
|---|---|---|
| CUDA | `cuda=None` and a usable GPU (default) | Rasters kept on the device for every position; each ray is marched straight towards the sun on the DSM grid. `max_workers` writer threads |
| CPU | otherwise, or `cuda=False` | GDAL hillshade plus a shadow sweep on the grid rotated to the sun's azimuth |

The two differ by a few nearest-neighbour pixels along shadow edges; at
azimuths 90/180/270°, where the rotation is an exact permutation, they agree
on ≥ 99.99 % of pixels. The CPU rotation (`reshape=False`) also drops the
grid's corners - about 13 % of the grid at azimuth 150° - and writes them
as 0, and truncates long low-sun rays. CPU hillshades made without an AOI
before 2026-10-19 carry that error and are only replaced with
`overwrite=True`.

Speed on the full Alpicat grid (10,265 × 11,169 px): opaque 38 s → 2 s per
position on the GPU; with the canopy 113 s → about 1.7 s, now bound by
writing the GeoTIFF.

**Storage.** Hillshades are written with `HILLSHADE_COMPRESSION` (ZSTD
level 6, no predictor). Their values sit on a few levels (k/255, whole ray
steps), which the floating-point predictor of DEFLATE turned into noise:
ZSTD writes a canopy hillshade in 0.8 s instead of 4.4 s at less than half
the size (346 → 152 MB), losslessly. Older DEFLATE hillshades still read
fine.

**Canopy hillshades** carry three more bands - the metres of crown the beam
crossed per LiDAR height class - and the tag `GREENSHADOW_CANOPY`. They must
be read with a month (`load_hillshade(..., month=, leaf_type=)`), which turns
the path lengths into a beam fraction; reading one without a month raises.
See [vegetation_canopy.md](vegetation_canopy.md).

`qgis_temporal=True` with `related_datetimes` adds the timestamp to the file
name for QGIS's temporal controller.

## 3. Elevated shadows: `apply_height_shadow_to_hillshades`

```python
gs.apply_height_shadow_to_hillshades(
    hillshade_dir=".../hillshades", dem_path=dem, dsm_path=dsm,
    out_dir=".../hillshades_dem1.5m", height_m=1.5, aoi=aoi)
```

It copies each hillshade with every pixel standing more than `height_m`
above the DEM (`dsm - dem > height_m`: roofs, trees, walls) set to 0, so
only open ground and low objects remain lit. The shadows themselves are
those cast at ground level; nothing is recomputed for a surface raised to
`height_m`. Only band 1 is written, so canopy path lengths are dropped.

## 4. Daily videos: `generate_daily_shadow_mp4`

Renders the day's hillshades in time order into an MP4 (`fps`,
`duration_seconds`, `zoom` around `center_xy`, `frame_size`), with the time
and the sun's altitude and azimuth on each frame. Canopy hillshades need
`canopy_leaf_type`; the month comes from each frame's timestamp.

## 5. Sky view factor

```python
gs.calculate_sky_view_factor(
    dsm_path=dsm, out_path="svf.tif", search_radius=250.0,
    angular_resolution=10.0, cuda=None, hybrid=True, subsample_factor=3,
    building_buffer_m=250, class_path=dsm_class,
    building_classes=(6, 17, 76, 77), vegetation_classes=(3, 4, 5))
```

The share of the sky hemisphere a pixel sees, which weights the diffuse
irradiance. From every pixel, rays are marched in `360 / angular_resolution`
directions up to `search_radius`; the horizon angle `h` of each direction
contributes `cos² h`, and the SVF is their mean, in [0, 1].

| Path | |
|---|---|
| CUDA, full resolution | every pixel |
| CUDA, `hybrid=True` | full resolution only near buildings and vegetation (`class_path` and the class lists, dilated by `building_buffer_m`; without a class raster, pixels more than `height_threshold` above the 10th height percentile), and on a grid subsampled by `subsample_factor` elsewhere, then upsampled |
| CPU (Numba) | same horizon model as CUDA, row block by row block; used when CUDA is not usable |

The Catalan pipeline computes two: on the DSM (`svf.tif`) for the rasters
with trees, and on the DSM without vegetation (`svf_no_vegetation.tif`) for
the PV fields, which clear the parcel and its trees.

`search_radius` must cover at least two pixels; on a coarse DEM raise it, or
the horizon is cut short.

## 6. Limitations

* Hillshades depend only on the DSM and the sun, so one stack serves every
  weather and every year with the same discretisation. A rebuilt DSM needs
  `overwrite=True`; nothing detects it.
* Diffuse light is weighted by the SVF only (isotropic sky); circumsolar
  and horizon brightening are not modelled.
