# Advanced Usage

## Working with Buildings

```python
import geopandas as gpd

# Load building footprints (example)
buildings_gdf = gpd.read_file("buildings.gpkg")

# Remove buildings from DSM to get clean terrain
clean_dsm_path = gs.remove_buildings_from_dsm(
    dsm_path=dsm_path,
    buildings=buildings_gdf,
    out_path=f"{working_dir}/dsm_no_buildings.tif",
    buildings_buffer_m=1.0,  # 1m buffer around buildings
    fill_method="spline",
    coarse_dem_path=f"{working_dir}/coarse_dem.tif"
)

# Generate DEM from clean DSM
terrain_dem = gs.build_dem_from_dsm_obstacle_fast(
    dsm_path=clean_dsm_path,
    out_dem_path=f"{working_dir}/terrain_dem.tif"
)

# Apply height-based shadows (e.g., for rooftop analysis at 2m height)
elevated_hillshades = gs.apply_height_shadow_to_hillshades(
    hillshade_dir=f"{working_dir}/hillshades",
    dem_path=terrain_dem,
    dsm_path=dsm_path,
    out_dir=f"{working_dir}/hillshades_2m",
    height_m=2.0,
    pattern="*.tif",
    recursive=True,
    mask_value=0.1
)
```

## Annual Solar Analysis

```python
# Generate solar positions for entire year
annual_solar = gs.generate_azimuth_altitude_pairs_by_year(
    year=2025,
    lat=41.390,
    lon=2.154,
    tz_name="Europe/Madrid",
    granularity="PT1H",  # Hourly
    solar_degrees_discretization=2.0,  # 2-degree precision
    only_daylight=True
)

print(f"Unique solar positions for year: {annual_solar['metadata']['unique_positions']}")

# Generate hillshades for all unique positions
annual_hillshades = gs.generate_multiple_hillshades_gdal(
    raster_dir=f"{working_dir}/annual_hillshades",
    dsm_path=dsm_path,
    azimuth_altitude=annual_solar['azimuth_altitude_pairs'],
    shadow_floor=0.0,  # Full shadows
    max_workers=4
)

# Create annual average
annual_average = gs.average_hillshades(
    hillshade_dir=f"{working_dir}/annual_hillshades",
    out_path=f"{working_dir}/annual_solar_exposure.tif",
    aggregation="mean"
)
```

## Custom Solar Calculations

```python
# Use custom solar position libraries
solar_positions = gs.generate_azimuth_altitude_pairs_by_date(
    date=dt.date(2025, 3, 20),  # Spring equinox
    lat=your_latitude,
    lon=your_longitude,
    tz_name="Your/Timezone"
)
```

## High-Resolution Processing

```python
# Sub-meter resolution processing
ultra_high_dsm = gs.build_dsm_from_lidar_pure(
    laz_paths=laz_paths,
    out_dsm="ultra_high_dsm.tif",
    resolution=0.25,  # 25cm pixels
    aggregation="max",  # Use maximum elevation in each pixel
    fill_gaps="nearest"
)
```

## Batch Processing

```python
# Process multiple dates
dates_to_process = [
    dt.date(2025, 3, 20),  # Spring equinox
    dt.date(2025, 6, 21),  # Summer solstice
    dt.date(2025, 9, 22),  # Autumn equinox
    dt.date(2025, 12, 21), # Winter solstice
]

for date in dates_to_process:
    solar_data = gs.generate_azimuth_altitude_pairs_by_date(date=date, ...)
    hillshades = gs.generate_multiple_hillshades_gdal(
        raster_dir=f"{working_dir}/{date.strftime('%Y%m%d')}",
        dsm_path=dsm_path,
        azimuth_altitude=solar_data['azimuth_altitude_pairs']
    )
    # Process results...
```

## Performance Tips

1. **Resolution vs. Speed**: Higher resolution (smaller pixel size) dramatically increases processing time
2. **Parallel Processing**: Use `max_workers` parameters for faster downloads and processing
3. **Memory Management**: For large areas, consider processing in tiles
4. **Caching**: Downloaded LAZ files and generated DSMs are cached for reuse
5. **GDAL Optimization**: Ensure GDAL is properly installed with multi-threading support

For module details and troubleshooting, see [Module & Data Reference](module-reference.md).
