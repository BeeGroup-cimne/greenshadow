# greenshadow

A Python library for environmental shading analysis using LiDAR data and custom algorithms to simulate the solar shading of rural and urban areas in maximum detail. greenshadow uses hillshade techniques combined with cast shadow calculations to provide accurate solar radiation analysis.

## Features

- **LiDAR Data Processing**: Access and process LiDAR data from the Institut Cartogràfic i Geològic de Catalunya (ICGC)
- **DSM/DEM Generation**: Create Digital Surface Models (DSM) and Digital Elevation Models (DEM) from point clouds
- **Solar Position Calculation**: Compute accurate sun positions for any date, time, and location
- **Hillshade Generation**: Generate detailed hillshades with cast shadows for solar analysis
- **Building Integration**: Remove buildings from DSM to create clean terrain models
- **Temporal Analysis**: Analyze solar exposure over time periods (daily, seasonal, annual)
- **High-Resolution Output**: Support for sub-meter resolution analysis

## Installation

### Prerequisites

greenshadow requires GDAL to be installed on your system. Follow the installation instructions for your platform:

#### Ubuntu/Debian

```bash
# Add the unstable PPA for latest GDAL
sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo apt update

# Install GDAL and development headers
sudo apt install gdal-bin libgdal-dev

# Set environment variables for GDAL Python bindings
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# Install GDAL Python bindings matching system version
GDAL_VERSION=$(gdal-config --version)
pip install "GDAL==${GDAL_VERSION}.*"
```

#### macOS

```bash
# Using Homebrew
brew install gdal

# Install Python bindings
pip install GDAL==$(gdal-config --version)
```

#### Windows

Download and install GDAL from [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or use conda:

```bash
conda install -c conda-forge gdal
```

### Install greenshadow

1. Clone the repository:
```bash
git clone https://github.com/BeeGroup-cimne/greenshadow.git
cd greenshadow
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install setuptools wheel
python setup.py sdist
pip install dist/greenshadow-1.0.0.tar.gz
```

Or install in development mode:
```bash
pip install -e .
```

### Optional Dependencies

Install optional dependencies for enhanced functionality:

```bash
# For accurate solar position calculations
pip install pysolar sunpy

# For PDAL-based processing (advanced users)
pip install pdal

# For development
pip install pytest black flake8
```

## Quick Start

### Basic Usage

```python
import greenshadow as gs
from shapely.geometry import Point
import datetime as dt

# Define area of interest (Barcelona example)
aoi = Point(2.154, 41.390).buffer(0.01)  # ~1km buffer around Barcelona center
working_dir = "/path/to/your/data"

# 1. Find and download LiDAR tiles
tiles = gs.icgc_tiles_from_geojson(
    aoi=aoi,
    aoi_epsg=4326,
    aoi_buffer_in_meters=500,
    pdf_plot_path=f"{working_dir}/tiles_plot.pdf"
)

# Download LAZ files
laz_paths = gs.icgc_download_from_geojson(
    out_dir=f"{working_dir}/lidar_data",
    tiles=tiles,
    aoi=aoi,
    aoi_epsg=4326,
    max_workers=8
)

# 2. Create Digital Surface Model (DSM)
dsm_path = gs.build_dsm_from_lidar_pure(
    laz_paths=laz_paths,
    out_dsm=f"{working_dir}/dsm.tif",
    resolution=0.5,  # 0.5m resolution
    classification=(0,),  # Use unclassified points
    epsg=25831,  # UTM Zone 31N for Barcelona
    fill_gaps="nearest"
)

# 3. Create Digital Elevation Model (DEM) by removing obstacles
dem_path = gs.build_dem_from_dsm_obstacle_fast(
    dsm_path=dsm_path,
    out_dem_path=f"{working_dir}/dem.tif",
    method="multiscale_harmonic",
    levels=3,
    relax_iters=50
)

# 4. Generate solar positions for summer solstice
summer_solstice = dt.date(2025, 6, 21)
solar_data = gs.generate_azimuth_altitude_pairs_by_date(
    date=summer_solstice,
    lat=41.390,
    lon=2.154,
    tz_name="Europe/Madrid",
    granularity="PT15M",  # Every 15 minutes
    solar_degrees_discretization=1.0,
    only_daylight=True
)

print(f"Generated {len(solar_data['azimuth_altitude_pairs'])} solar positions")

# 5. Generate hillshades with cast shadows
hillshade_paths = gs.generate_multiple_hillshades_gdal(
    raster_dir=f"{working_dir}/hillshades",
    dsm_path=dsm_path,
    azimuth_altitude=solar_data['azimuth_altitude_pairs'],
    related_datetimes=solar_data['df']['datetime_local'].tolist(),
    shadow_floor=0.1,  # 10% illumination in shadow areas
    zFactor=1.0,
    include_datetime_info=True,
    qgis_temporal=True
)

# 6. Create average hillshade for the day
average_path = gs.average_hillshades(
    hillshade_dir=f"{working_dir}/hillshades",
    out_path=f"{working_dir}/summer_solstice_average.tif",
    aggregation="mean"
)

print(f"Analysis complete! Average hillshade saved to: {average_path}")
```

### Working with Buildings

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

### Annual Solar Analysis

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

## Module Structure

greenshadow is organized into specialized modules:

### `greenshadow.lidar`
- LiDAR tile discovery and download from ICGC
- Tile management and visualization
- Core functions: `icgc_tiles_from_geojson()`, `icgc_download_from_geojson()`

### `greenshadow.dsm_dem_estimation`
- DSM/DEM generation from point clouds
- Building removal and terrain reconstruction
- Core functions: `build_dsm_from_lidar_pure()`, `build_dem_from_dsm_obstacle_fast()`, `remove_buildings_from_dsm()`

### `greenshadow.shader`
- Solar position calculations
- Hillshade generation with cast shadows
- Temporal analysis tools
- Core functions: `generate_hillshade_gdal()`, `generate_azimuth_altitude_pairs_by_date()`

### `greenshadow.utils`
- Utility functions for geometry, projections, and file handling
- HTTP session management
- Validation and environment detection

## Data Sources

### ICGC LiDAR Data

greenshadow provides built-in access to high-resolution LiDAR data from the Institut Cartogràfic i Geològic de Catalunya, covering the Catalonia region with:

- **Resolution**: Up to 4 points/m²
- **Coverage**: All of Catalonia (~32,000 km²)
- **Format**: LAZ (compressed LAS)
- **Coordinate System**: EPSG:25831 (ETRS89 / UTM zone 31N)
- **Classifications**: Ground, buildings, vegetation, water, etc.

### Other Regions

For areas outside Catalonia, you can use greenshadow with your own LiDAR data:

1. Prepare LAZ/LAS files in a supported format
2. Use the DSM/DEM generation functions directly
3. Follow the same workflow for solar analysis

## Advanced Features

### Custom Solar Calculations

```python
# Use custom solar position libraries
solar_positions = gs.generate_azimuth_altitude_pairs_by_date(
    date=dt.date(2025, 3, 20),  # Spring equinox
    lat=your_latitude,
    lon=your_longitude,
    tz_name="Your/Timezone"
)
```

### High-Resolution Processing

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

### Batch Processing

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

## Troubleshooting

### Common Issues

1. **GDAL Import Errors**:
   ```bash
   # Check GDAL installation
   gdalinfo --version
   python -c "from osgeo import gdal; print(gdal.__version__)"
   ```

2. **Memory Issues with Large Datasets**:
   - Reduce resolution or process smaller areas
   - Use `chunk_size` parameters where available

3. **Coordinate System Issues**:
   - Ensure all inputs use consistent coordinate systems
   - ICGC data is in EPSG:25831 by default

4. **Download Failures**:
   - Check internet connection
   - Some tiles may be temporarily unavailable

### Getting Help

- **Issues**: Report bugs at [GitHub Issues](https://github.com/BeeGroup-cimne/greenshadow/issues)
- **Documentation**: See example scripts in the `examples/` directory
- **Contact**: For research collaborations, contact the authors

## Citation

If you use greenshadow in your research, please cite:

```
@software{greenshadow2025,
  author = {Broto, Jose Manuel and Mor, Gerard and Sellart, Maite},
  title = {greenshadow: Environmental shading analysis using LiDAR data},
  year = {2025},
  url = {https://github.com/BeeGroup-cimne/greenshadow},
  version = {1.0.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Jose Manuel Broto** - *Lead Developer* - [jmbroto@cimne.upc.edu](mailto:jmbroto@cimne.upc.edu)
- **Gerard Mor** - *Co-developer* - [gmor@cimne.upc.edu](mailto:gmor@cimne.upc.edu)
- **Maite Sellart** - *Co-developer* - [tsellart@cimne.upc.edu](mailto:tsellart@cimne.upc.edu)

## Acknowledgments

- **ICGC** (Institut Cartogràfic i Geològic de Catalunya) for providing open LiDAR data
- **BeeGroup-CIMNE** for supporting this research
- The open-source geospatial community for the foundational tools

---

**Copyright (c) 2025 Jose Manuel Broto, Gerard Mor, Maite Sellart**