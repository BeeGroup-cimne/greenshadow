# greenshadow

A comprehensive Python library for environmental shading analysis and solar energy potential assessment using high-resolution LiDAR data. greenshadow combines advanced LiDAR processing, precise solar calculations, and sophisticated shadowing algorithms to provide accurate solar radiation analysis for rural and urban areas.

## Overview

greenshadow enables detailed solar energy analysis by processing LiDAR point clouds into accurate elevation models, computing solar positions throughout time, and generating hillshades that incorporate cast shadows from terrain and structures. The library is specifically optimized for high-resolution analysis and includes advanced features for building integration, classification correction, and temporal solar exposure assessment.

## Key Features

- **High-Resolution LiDAR Processing**: Convert LiDAR point clouds to DSM/DEM with sub-meter precision
- **Advanced Classification Management**: Intelligent building detection and classification correction using cadastral data
- **Unified Smoothing Algorithms**: Smart elevation surface smoothing that preserves building edges while creating natural terrain transitions
- **Precise Solar Calculations**: Accurate sun position computation for any date, time, and location
- **Cast Shadow Simulation**: Generate hillshades with realistic shadow casting from terrain and structures
- **Weather Data Integration**: ERA5Land weather data integration for realistic solar radiation modeling
- **Temporal Solar Analysis**: Daily, seasonal, and annual solar exposure assessment
- **Photovoltaic Potential**: Direct PV potential calculation from solar exposure analysis
- **Municipal-Scale Processing**: Tools for processing entire municipal areas efficiently

## Methodology of Solar Energy Calculation

### 1. Digital Model Generation
- **LiDAR Processing**: Point clouds are aggregated into high-resolution Digital Surface Models (DSM) and Digital Elevation Models (DEM)
- **Classification Correction**: Building footprints from cadastral data are used to correct LiDAR classification, ensuring accurate building representation
- **Unified Smoothing**: Natural terrain features are smoothed while preserving sharp building edges for accurate shadow casting

### 2. Solar Position Computation
- **Astronomical Calculations**: Precise solar positions (azimuth and elevation) are computed using established solar position algorithms
- **Temporal Resolution**: Supports flexible time granularity from minutes to hours across any time period
- **Optimization**: Solar positions are discretized to reduce computational load while maintaining accuracy

### 3. Shadow Analysis
- **Cast Shadow Modeling**: Advanced algorithms compute shadows cast by terrain features and buildings
- **Hillshade Generation**: Multiple hillshades are generated for each solar position incorporating both terrain shading and cast shadows
- **Height-Based Analysis**: Supports elevated analysis (e.g., rooftop solar potential at specific heights above ground)

### 4. Solar Energy Integration
- **Weather Data**: ERA5Land reanalysis data provides accurate solar radiation values
- **Surface Energy Computation**: Combines hillshade analysis with weather data to compute actual solar energy reaching each surface
- **PV Potential Assessment**: Direct calculation of photovoltaic potential using Fourier analysis of solar exposure patterns

## Use Case: Municipal Solar Energy Assessment

The library includes a comprehensive workflow for municipal-scale solar energy analysis:

### Workflow Overview
1. **Data Acquisition**: Automatic discovery and download of LiDAR tiles covering the municipal area
2. **Building Integration**: Retrieval of building geometries from cadastral databases
3. **Digital Model Generation**: Creation of multiple elevation models (DSM with/without vegetation, clean DEM)
4. **Classification Enhancement**: Correction of LiDAR classification using accurate building footprints
5. **Solar Position Generation**: Computation of solar positions for specified time periods
6. **Shadow Analysis**: Generation of hillshades incorporating cast shadows
7. **Weather Integration**: Processing of ERA5Land weather data for accurate solar radiation values
8. **Solar Energy Computation**: Calculation of surface solar energy and PV potential

### Key Capabilities
- **Multi-Scale Analysis**: From individual buildings to entire municipalities
- **Temporal Flexibility**: Daily snapshots, seasonal analysis, or full annual assessment
- **High Precision**: Sub-meter resolution analysis with building-accurate shadowing
- **Real Weather Data**: Integration with historical weather data for realistic energy estimates

## Main Functions and Modules

### Core Modules

#### `greenshadow.lidar`
LiDAR data access and management for the Catalonia region.

**Key Functions:**
- `icgc_tiles_from_geojson()`: Discover LiDAR tiles covering an area of interest
- `icgc_download_from_geojson()`: Download LAZ files with parallel processing
- `list_available_tiles()`: Query available LiDAR tiles
- `get_tile_info()`: Retrieve metadata for specific tiles

#### `greenshadow.dsm_dem_estimation`
Advanced digital elevation model generation with building integration.

**Key Functions:**
- `build_digital_model_from_lidar_pure()`: Main function for DSM/DEM generation with classification correction
- `build_dem_from_dsm_obstacle_fast()`: Convert DSM to DEM by removing obstacles
- `build_dem_from_dsm_height_disruption()`: Alternative DEM generation method
- `remove_buildings_from_dsm()`: Remove buildings from DSM using geometric data
- `refine_dem_with_buffered_interpolation()`: Improve DEM quality through interpolation

**Advanced Features:**
- **Classification Correction**: Automatic correction using cadastral building data
- **Unified Smoothing**: Smart smoothing that preserves building edges
- **Multi-Resolution Support**: From centimeter to meter-scale processing
- **Memory Optimization**: Efficient processing of large datasets

#### `greenshadow.shader`
Solar position calculation and shadow analysis.

**Key Functions:**
- `generate_azimuth_altitude_pairs_by_date()`: Solar positions for specific dates
- `generate_azimuth_altitude_pairs_by_year()`: Annual solar position sets
- `generate_multiple_hillshades_gdal()`: Batch hillshade generation with shadows
- `apply_height_shadow_to_hillshades()`: Elevated analysis for rooftop assessment
- `compute_surface_solar_energy()`: Solar energy calculation from hillshades and weather data
- `pv_potential_from_fourier_raster()`: PV potential assessment from solar exposure
- `calculate_sky_view_factor()`: Sky visibility analysis
- `generate_daily_shadow_mp4()`: Video generation of daily shadow progression

#### `greenshadow.weather`
Weather data integration for realistic solar radiation modeling.

**Key Functions:**
- `download_ERA5Land()`: Download ERA5Land reanalysis data
- `load_ERA5Land()`: Load and process weather data
- `ERA5Land_variables()`: Available weather variables

**Features:**
- **Historical Data**: Access to decades of weather records
- **High Resolution**: Up to hourly temporal resolution
- **Variable Selection**: Solar radiation, temperature, wind, humidity
- **Geographic Filtering**: Extract data for specific locations

#### `greenshadow.utils`
Utility functions for geometric operations and data management.

**Key Functions:**
- `force_raster_value_based_on_geometries()`: Apply geometric corrections to rasters
- Various coordinate system and projection utilities

### Example Usage

```python
import greenshadow as gs
import hypercadaster_ES as hc
import datetime as dt

# Configuration for a municipality
wd = "/path/to/working/directory" 
municipal_codes = {"Municipality": "cadastral_code"}
dsm_resolution = 0.5  # meters

# Process municipality
for municipality_name, cadastral_code in municipal_codes.items():
    # 1. Get building geometries for classification correction
    building_geoms_gdf = hc.mergers.join_cadaster_building(
        gdf=None,
        cadaster_dir=cadaster_dir,
        cadaster_codes=[cadastral_code],
        results_dir=results_dir,
        open_street_dir=open_street_dir
    )
    
    # 2. Generate high-quality DSM with building correction
    dsm_path = gs.build_digital_model_from_lidar_pure(
        laz_paths=laz_paths,
        out_path=f"{wd}/{municipality_name}/dsm.tif",
        out_classification_path=f"{wd}/{municipality_name}/dsm_class.tif",
        resolution=dsm_resolution,
        classification=(2,3,4,5,6,7,8,9,17),
        epsg=25831,
        # Classification correction
        classification_correction_buildings_gdf=building_geoms_gdf,
        classification_correction_building_class_value=6,
        classification_correction_building_buffer_m=0.5,
        aggregation="max"
    )
    
    # 3. Generate solar positions for analysis period
    solar_df = gs.generate_azimuth_altitude_pairs_by_date(
        date=dt.date(2023, 6, 21),  # Summer solstice
        lat=municipality_lat,
        lon=municipality_lon,
        tz_name="Europe/Madrid",
        granularity="PT15M",
        solar_degrees_discretization=0.1
    )
    
    # 4. Generate hillshades with shadows
    hillshade_paths = gs.generate_multiple_hillshades_gdal(
        raster_dir=f"{wd}/{municipality_name}/hillshades",
        dsm_path=dsm_path,
        azimuth_altitude=solar_df['azimuth_altitude_pairs'],
        related_datetimes=solar_df['df']['datetime_local'].tolist(),
        shadow_floor=0.1
    )
    
    # 5. Calculate solar energy with weather data
    solar_energy = gs.compute_surface_solar_energy(
        solar_df=solar_df['df'],
        hillshade_dir=f"{wd}/{municipality_name}/hillshades",
        weather_data_dir=weather_data_dir,
        out_dir=f"{wd}/{municipality_name}/solar_energy"
    )
    
    # 6. Assess PV potential
    pv_potential = gs.pv_potential_from_fourier_raster(
        fourier_raster_path=solar_energy,
        out_raster_path=f"{wd}/{municipality_name}/pv_potential.tif"
    )
```

## Installation

### Prerequisites

greenshadow requires GDAL and several Python packages. Install system dependencies first:

#### Ubuntu/Debian
```bash
sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
sudo apt update
sudo apt install gdal-bin libgdal-dev python3-dev
```

#### macOS
```bash
brew install gdal
```

### Python Installation

```bash
# Clone repository
git clone https://github.com/BeeGroup-cimne/greenshadow.git
cd greenshadow

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Optional: Install weather functionality
pip install pvlib polars cdsapi cfgrib xarray netCDF4
```

### Weather Data Setup

For weather data functionality:
1. Create account at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)
2. Create `~/.cdsapirc` file with your API credentials

## Data Sources

### LiDAR Data
- **Source**: Institut Cartogràfic i Geològic de Catalunya (ICGC)
- **Coverage**: Catalonia region
- **Resolution**: Up to 4 points/m²
- **Format**: LAZ (compressed LAS)
- **Classifications**: Ground, buildings, vegetation, water, etc.

### Weather Data
- **Source**: ERA5Land reanalysis from Copernicus
- **Coverage**: Global
- **Resolution**: Hourly, 0.1° × 0.1°
- **Variables**: Solar radiation, temperature, wind, humidity

### Building Data
- **Integration**: Spanish cadastral database via hypercadaster_ES
- **Purpose**: Classification correction and building detection
- **Coverage**: Spain

## Advanced Features

### Classification Correction
Automatic correction of LiDAR classification using accurate building footprints from cadastral data, ensuring precise building representation for shadow analysis.

### Unified Smoothing
Smart elevation surface smoothing that:
- Preserves sharp building edges for accurate shadow casting
- Creates natural transitions between different terrain types
- Handles nodata areas intelligently
- Applies different smoothing strategies to different land cover classes

### Temporal Analysis
Flexible temporal analysis supporting:
- Specific date analysis (solstices, equinoxes)
- Seasonal comparisons
- Full annual assessment
- Custom time period analysis

### Multi-Resolution Processing
Support for processing at various resolutions:
- High-resolution (0.25-0.5m) for detailed building analysis
- Medium-resolution (1-2m) for neighborhood studies
- Lower-resolution (5-10m) for regional assessments

## Performance and Optimization

### Memory Management
- Efficient processing of large LiDAR datasets
- Configurable memory limits and batch processing
- Optimized data structures for high-resolution analysis

### Parallel Processing
- Multi-threaded LiDAR downloading
- Parallel hillshade generation
- Concurrent processing of multiple time steps

### Caching and Storage
- Automatic caching of downloaded LiDAR files
- Efficient storage of intermediate results
- Reusable elevation models and classifications

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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