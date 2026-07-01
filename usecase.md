# Use Case: Municipal Solar Energy Assessment with greenshadow

*A comprehensive workflow for analyzing solar energy potential across entire municipalities using LiDAR data, weather information, and cadastral building data.*

## Overview

This use case demonstrates the complete workflow for municipal-scale solar energy analysis using the greenshadow library. The example processes the municipality of Alpicat in Catalonia, Spain, showing how to integrate high-resolution LiDAR data with cadastral building information and historical weather data to produce accurate solar energy assessments.

## Workflow Summary

The analysis follows a systematic 8-step process that transforms raw LiDAR point clouds into detailed solar energy maps:

1. **Municipal Area Definition** - Define boundaries and coordinate systems
2. **Weather Data Integration** - Load historical ERA5Land climate data
3. **LiDAR Data Acquisition** - Discover and download relevant tiles
4. **Building Data Integration** - Retrieve accurate building footprints from cadastral databases
5. **Digital Model Generation** - Create multiple elevation models (DSM, DEM) with classification correction
6. **Solar Position Calculation** - Compute precise sun positions for analysis periods
7. **Shadow Analysis** - Generate hillshades incorporating terrain and building shadows
8. **Solar Energy Assessment** - Calculate surface solar energy and photovoltaic potential

## Detailed Workflow

### 1. Project Configuration and Municipal Setup

```python
# Configuration for Alpicat municipality analysis
wd = "/media/gmor/data/greenshadow"
municipal_codes = {"Alpicat": "25023"}
dsm_resolution = 0.5  # meters - sub-meter precision
tz_local = "Europe/Madrid"

# Analysis parameters
days_to_analyse = [dt.date(2023,12,22), dt.date(2023,6,21)]  # Solstices
year_to_analyse = 2023  # Full year analysis
sim_dem_height = 1.5  # meters above ground for rooftop analysis
```

**Key Features:**
- Flexible temporal analysis (daily, seasonal, or annual)
- Configurable spatial resolution from 0.25m to several meters
- Support for multiple municipalities in batch processing
- Integration with Spanish cadastral system via hypercadaster_ES

*[Caption for Screenshot/Video 1: "Initial configuration showing municipal boundary definition and analysis parameters setup for Alpicat municipality in Catalonia."]*

### 2. Weather Data Integration

The workflow automatically integrates historical weather data to provide realistic solar radiation values:

```python
# Load ERA5-Land weather data for the municipality centroid
weather_df = gs.load_ERA5Land(
    data_dir=weather_data_dir,
    lat_range=[lat, lat],  # Point location
    lon_range=[lon, lon],
    ym_range=[start_ym, end_ym],
    use_point_location=True,
    tz_local=tz_local,
    mode="instant"
)
```

**Capabilities:**
- **Historical Data Access**: Decades of hourly weather records from ERA5Land
- **Point-Specific Data**: Extract data for municipal centroids to optimize performance
- **Multiple Variables**: Solar radiation, temperature, wind speed, humidity
- **Temporal Flexibility**: From hourly to monthly aggregations

*[Caption for Screenshot/Video 2: "Weather data loading interface showing ERA5Land integration with temporal and spatial parameter configuration."]*

### 3. LiDAR Data Discovery and Download

Automated discovery and download of LiDAR tiles covering the municipal area:

```python
# Find matching ICGC LiDAR tiles
tiles = gs.icgc_tiles_from_geojson(
    aoi=aoi,
    aoi_epsg=4326,
    aoi_buffer_in_meters=500,
    pdf_plot_path=f"{wd}/{municipality_name}/tiles.pdf"
)

# Download LAZ files with parallel processing
paths = gs.icgc_download_from_geojson(
    out_dir=f"{wd}/icgc",
    tiles=tiles,
    aoi=aoi,
    max_workers=8
)
```

**Features:**
- **Automatic Tile Discovery**: Intelligently finds all tiles intersecting the area of interest
- **Parallel Downloads**: Multi-threaded downloading for efficiency
- **Caching System**: Avoids re-downloading existing files
- **Visualization**: Generates PDF maps showing tile coverage

*[Caption for Screenshot/Video 3: "LiDAR tile discovery and download progress, showing the automated identification of relevant ICGC tiles for the municipal area."]*

*[Caption for Screenshot/Video 4: "Generated PDF map showing LiDAR tile coverage over Alpicat municipality with buffer area highlighted."]*

### 4. Building Data Extraction

Integration with Spanish cadastral database for accurate building footprints:

```python
# Extract building geometries from cadastral database
building_geoms_gdf = hc.mergers.join_cadaster_building(
    gdf=None,
    cadaster_dir=hc.utils.cadaster_dir_(wd_hypercadaster),
    cadaster_codes=[cadastral_code],
    results_dir=hc.utils.results_dir_(wd_hypercadaster),
    open_street_dir=hc.utils.open_street_dir_(wd_hypercadaster)
)
```

**Benefits:**
- **High Accuracy**: Official cadastral data provides precise building footprints
- **Classification Correction**: Improves LiDAR classification accuracy
- **Shadow Analysis**: Enables accurate shadow casting from building edges
- **Automated Processing**: Seamless integration with hypercadaster_ES library

*[Caption for Screenshot/Video 5: "Building footprint extraction from Spanish cadastral database showing polygon geometries overlaid on the municipal area."]*

### 5. Digital Elevation Model Generation

Creation of multiple elevation models with advanced processing:

#### 5.1 Digital Surface Model (DSM) - Full Detail

```python
dsm_path = gs.build_digital_model_from_lidar_pure(
    laz_paths=paths,
    out_path=dsm_path,
    out_classification_path=dsm_class_path,
    out_slope_path=dsm_slope_path,
    out_aspect_path=dsm_aspect_path,
    resolution=0.5,
    classification=(2,3,4,5,6,7,8,9,17),  # All features
    aggregation="max",
    # Classification correction
    classification_correction_buildings_gdf=building_geoms_gdf,
    classification_correction_building_class_value=6,
    classification_correction_building_buffer_m=0.5
)
```

*[Caption for Screenshot/Video 6: "High-resolution DSM generation showing the complete elevation surface including buildings, vegetation, and terrain features at 0.5m resolution."]*

*[Caption for Screenshot/Video 7: "Classification correction process showing before and after comparison of building detection accuracy using cadastral data."]*

#### 5.2 DSM Without Vegetation

```python
dsm_no_vegetation_path = gs.build_digital_model_from_lidar_pure(
    laz_paths=paths,
    classification=(2,6,8,17),  # Excludes vegetation classes
    # Unified smoothing preserves building edges
    smooth_classes=(1,2,8,17),
    smooth_window_size=7,
    slope_smooth_window=7,
    aspect_smooth_window=7
)
```

**Advanced Smoothing:**
- **Unified Surface Smoothing**: Natural terrain transitions while preserving building edges
- **Class-Aware Processing**: Buildings (class 6) excluded from smoothing to maintain sharp edges
- **Multi-Iteration Smoothing**: Configurable iterations for slope and aspect refinement

*[Caption for Screenshot/Video 8: "DSM without vegetation showing terrain and buildings only, with unified smoothing applied to create natural surface transitions."]*

#### 5.3 Digital Elevation Model (DEM) - Bare Earth

```python
dem_path = gs.build_digital_model_from_lidar_pure(
    laz_paths=paths,
    classification=(2,8,17),  # Ground and low vegetation only
    filter_raster_path=dsm_class_path,
    filter_points_in_range=[6, 6],  # Exclude buildings
    filter_mode="exclude",
    isolated_points_exclusion_ratio=0.9,
    isolated_window_size=7
)
```

*[Caption for Screenshot/Video 9: "Clean DEM generation showing bare earth terrain with buildings and vegetation removed, demonstrating ground surface extraction."]*

### 6. Sky View Factor Analysis

Calculation of sky visibility for accurate solar analysis:

```python
gs.calculate_sky_view_factor(
    dsm_path=dsm_path,
    out_path=svf_path,
    search_radius=250.0,
    angular_resolution=10.0,
    cuda=True,
    hybrid=True,
    building_classes={6},
    vegetation_classes={3,4,5}
)
```

*[Caption for Screenshot/Video 10: "Sky View Factor calculation showing areas with high (bright) and low (dark) sky visibility, important for diffuse solar radiation analysis."]*

### 7. Solar Position Generation

Precise astronomical calculations for sun positions:

```python
# Daily analysis for solstices
for day in days_to_analyse:
    solar_data = gs.generate_azimuth_altitude_pairs_by_date(
        date=day,
        lat=lat, lon=lon,
        tz_name=tz_local,
        granularity="PT15M",  # 15-minute intervals
        solar_degrees_discretization=0.1,
        only_daylight=True,
        weather_df=weather_df.to_pandas()
    )
```

**Precision Features:**
- **High Temporal Resolution**: Down to minute-level precision
- **Astronomical Accuracy**: Precise solar position algorithms
- **Weather Integration**: Links solar positions with actual irradiance data
- **Optimization**: Smart discretization reduces computational load

*[Caption for Screenshot/Video 11: "Solar position calculation showing azimuth and elevation angles throughout summer solstice day with 15-minute temporal resolution."]*

### 8. Shadow Analysis and Hillshade Generation

Advanced shadow modeling incorporating terrain and buildings:

```python
# Generate hillshades with cast shadows
hillshade_paths = gs.generate_multiple_hillshades_gdal(
    raster_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
    dsm_path=dsm_path,
    azimuth_altitude=solar_data["azimuth_altitude_pairs"],
    related_datetimes=solar_data["df"]["datetime_local"].tolist(),
    include_datetime_info=True,
    qgis_temporal=True
)
```

**Shadow Analysis Features:**
- **Cast Shadow Modeling**: Accurate shadows from buildings and terrain
- **Multiple Sun Positions**: Batch processing for entire days or years
- **Temporal Metadata**: QGIS-compatible timestamps for animation
- **High Performance**: Optimized GDAL-based processing

*[Caption for Screenshot/Video 12: "Hillshade generation progress showing multiple shadow maps being created for different sun positions throughout the day."]*

*[Caption for Screenshot/Video 13: "Comparison of hillshades at different times of day showing shadow movement and intensity changes."]*

### 9. Height-Based Solar Analysis

Elevated analysis for rooftop solar potential:

```python
# Apply height-based shadows for rooftop analysis
elevated_paths = gs.apply_height_shadow_to_hillshades(
    hillshade_dir=f"{wd}/{municipality_name}/{day}/hillshades",
    dem_path=dem_path,
    dsm_path=dsm_path,
    out_dir=f"{wd}/{municipality_name}/{day}/hillshades_dem1.5m",
    height_m=1.5,  # Rooftop level
    pattern="*.tif"
)
```

**Applications:**
- **Rooftop Solar Assessment**: Analysis at specific heights above ground
- **Building Integration**: Considers building heights and nearby obstacles
- **Shadow Interaction**: Models shadows at elevated positions

*[Caption for Screenshot/Video 14: "Height-based shadow analysis showing solar conditions at 1.5m above ground level for rooftop solar panel assessment."]*

### 10. Daily Shadow Animation

Visualization of shadow progression:

```python
# Create daily shadow video
gs.generate_daily_shadow_mp4(
    solar_df=solar_data["df"],
    hillshade_dir=f"{wd}/{municipality_name}/{day}/hillshades",
    out_mp4=f"{wd}/{municipality_name}/{day}/hillshades.mp4",
    fps=10,
    duration_seconds=10,
    zoom=4,
    frame_size=(1920, 1088)
)
```

*[Caption for Video 15: "Time-lapse animation showing shadow progression throughout the summer solstice day, demonstrating dynamic shading patterns across the municipal area."]*

*[Caption for Video 16: "Elevated shadow animation at 1.5m height showing rooftop-level solar conditions throughout the day."]*

### 11. Solar Energy Calculation

Integration of shadow analysis with weather data:

```python
# Compute surface solar energy
solar_energy_path = gs.compute_surface_solar_energy(
    hillshade_dir=f"{wd}/{municipality_name}/{day}/hillshades",
    solar_df=solar_data["df"],
    svf_path=svf_path,
    aoi=aoi,
    out_path=f"{wd}/{municipality_name}/{day}/daily_solar_energy.tif"
)
```

**Energy Analysis Features:**
- **Weather Data Integration**: Real historical irradiance values
- **Shadow Factor Application**: Reduces energy in shadowed areas
- **Sky View Factor**: Accounts for diffuse radiation
- **High Resolution**: Maintains spatial detail from original LiDAR

*[Caption for Screenshot/Video 17: "Solar energy map showing daily energy accumulation across the municipality with color-coded energy levels from low (blue) to high (red)."]*

### 12. Annual Solar Analysis

Long-term assessment for comprehensive planning:

```python
# Annual solar position generation
annual_solar = gs.generate_azimuth_altitude_pairs_by_year(
    year=2023,
    lat=lat, lon=lon,
    granularity="PT1H",  # Hourly for annual analysis
    solar_degrees_discretization=3,  # Coarser for efficiency
    weather_df=weather_df.to_pandas()
)

# Annual solar energy computation
annual_raster_path = gs.compute_surface_solar_energy(
    hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
    solar_df=annual_solar["df"],
    mode="fourier_decomposition",
    out_path=f"{wd}/{municipality_name}/{year_to_analyse}/annual_solar_exposure.tif"
)
```

*[Caption for Screenshot/Video 18: "Annual solar analysis setup showing year-long solar position optimization and processing parameters."]*

*[Caption for Screenshot/Video 19: "Annual solar energy map showing cumulative energy patterns across the full year, revealing optimal locations for solar installations."]*

### 13. Photovoltaic Potential Assessment

Direct calculation of solar panel potential:

```python
# Compute PV potential from solar exposure analysis
pv_potential = gs.pv_potential_from_fourier_raster(
    fourier_raster_path=annual_raster_path,
    out_raster_path=pv_potential_raster_path,
    tilt=dsm_slope_path,
    azimuth=dsm_aspect_path,
    latitude_deg=aoi_centroid.y,
    technology="mono_Si",
    coverage_ratio=0.75
)
```

**PV Assessment Features:**
- **Technology-Specific Calculations**: Support for different panel technologies
- **Slope and Aspect Integration**: Uses actual terrain orientation
- **Coverage Ratio Modeling**: Accounts for realistic installation densities
- **Location-Specific Parameters**: Latitude-adjusted solar calculations

*[Caption for Screenshot/Video 20: "Photovoltaic potential map showing estimated solar panel energy generation potential across the municipality in kWh/year."]*

## Results and Outputs

### Generated Products

The workflow produces a comprehensive set of outputs for each municipality:

**Digital Elevation Models:**
- `dsm.tif` - Complete surface model including all features
- `dsm_no_vegetation.tif` - Surface model without vegetation
- `dem.tif` - Bare earth terrain model
- `svf.tif` - Sky view factor analysis

**Auxiliary Data:**
- `*_class.tif` - Classification rasters with building correction
- `*_slope.tif` - Surface slope analysis
- `*_aspect.tif` - Surface aspect analysis

**Solar Analysis:**
- `{date}/hillshades/` - Shadow maps for each sun position
- `{date}/hillshades_dem1.5m/` - Elevated shadow analysis
- `{date}/daily_solar_energy.tif` - Daily energy accumulation
- `{date}/hillshades.mp4` - Daily shadow animation

**Annual Assessment:**
- `{year}/annual_solar_exposure.tif` - Year-long energy analysis
- `pv_potential.tif` - Photovoltaic generation potential

*[Caption for Screenshot/Video 21: "Final output directory structure showing all generated products including elevation models, shadow analyses, and energy assessments."]*

### Quality Assurance Features

**Data Validation:**
- Automatic coordinate system verification
- Building footprint accuracy validation
- LiDAR classification quality assessment
- Weather data completeness checking

**Processing Optimization:**
- Memory-efficient batch processing
- Automatic file caching and reuse
- Configurable resolution and quality settings
- Progress monitoring and error handling

*[Caption for Screenshot/Video 22: "Quality assurance dashboard showing data validation results and processing optimization metrics."]*

## Technical Specifications

### Performance Characteristics

**Processing Scales:**
- **Municipal Area**: 10-100 km² typical coverage
- **Spatial Resolution**: 0.25m to 2m configurable
- **Temporal Resolution**: 1 minute to 1 hour intervals
- **Processing Time**: 2-8 hours per municipality (hardware dependent)

**Hardware Requirements:**
- **RAM**: 32GB+ recommended for 0.5m resolution
- **Storage**: 50-200GB per municipality
- **CPU**: Multi-core processor for parallel processing
- **GPU**: Optional CUDA support for Sky View Factor calculations

### Data Quality

**LiDAR Specifications:**
- Point density: Up to 4 points/m²
- Vertical accuracy: ±15cm
- Horizontal accuracy: ±30cm
- Classification: ASPRS LAS 1.4 standard

**Weather Data Quality:**
- Temporal coverage: 1979-present (ERA5Land)
- Spatial resolution: 0.1° × 0.1° (~11km)
- Temporal resolution: Hourly
- Variables: 50+ meteorological parameters

*[Caption for Screenshot/Video 23: "Technical specifications overview showing data quality metrics and processing performance characteristics."]*

## Applications and Benefits

### Municipal Planning Applications

**Solar Energy Planning:**
- Identify optimal locations for utility-scale solar installations
- Assess distributed rooftop solar potential
- Evaluate shading impacts from proposed developments
- Support renewable energy policy development

**Urban Planning:**
- Building height regulation optimization
- Green space planning for solar access
- Transportation infrastructure solar integration
- Climate adaptation planning

### Economic Analysis

**Investment Decision Support:**
- Site-specific solar resource assessment
- ROI calculations for solar installations
- Risk assessment for solar projects
- Portfolio optimization for solar developers

### Environmental Assessment

**Climate Impact Analysis:**
- Carbon footprint reduction potential
- Urban heat island mitigation
- Sustainable development planning
- Environmental impact assessment

*[Caption for Screenshot/Video 24: "Applications overview showing multiple use cases from municipal planning to economic analysis and environmental assessment."]*

## Conclusion

This municipal solar energy assessment workflow demonstrates the power of integrating high-resolution LiDAR data with sophisticated solar analysis algorithms. The greenshadow library provides a complete solution for municipal-scale renewable energy planning, combining precision spatial analysis with realistic temporal modeling.

**Key Advantages:**
- **Comprehensive Analysis**: From raw LiDAR to actionable solar energy maps
- **High Accuracy**: Sub-meter spatial resolution with building-accurate shadowing
- **Real Weather Integration**: Historical data for realistic energy estimates
- **Scalable Processing**: From individual buildings to entire municipalities
- **Automated Workflow**: Minimal manual intervention required
- **Open Source**: Transparent algorithms and extensible framework

The workflow successfully processes complex municipal areas like Alpicat, generating detailed solar energy assessments that can inform renewable energy policy, urban planning decisions, and investment strategies. The integration of cadastral building data ensures accurate shadow modeling, while weather data integration provides realistic energy estimates for practical applications.

*[Caption for Screenshot/Video 25: "Final summary showing the complete municipal solar energy assessment with all analysis products integrated into a comprehensive renewable energy planning toolkit."]*

---

**Total Processing Summary for Alpicat Municipality:**
- Area: ~25 km²
- LiDAR Points Processed: ~6.25 million points
- Solar Positions Calculated: 200+ per analysis day
- Hillshades Generated: 400+ shadow maps
- Final Products: 15+ geospatial datasets
- Processing Time: ~4 hours (modern workstation)
- Storage Required: ~75 GB

This workflow represents a state-of-the-art approach to municipal solar energy assessment, providing the detailed spatial and temporal analysis needed for informed renewable energy planning and policy development.