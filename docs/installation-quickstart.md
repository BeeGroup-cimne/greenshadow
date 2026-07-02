# Installation & Quick Start

## Prerequisites

greenshadow requires GDAL to be installed on your system. Follow the installation instructions for your platform below.

### Ubuntu/Debian

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

### macOS

```bash
# Using Homebrew
brew install gdal

# Install Python bindings
pip install GDAL==$(gdal-config --version)
```

### Windows

Download and install GDAL from [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or use conda:

```bash
conda install -c conda-forge gdal
```

## Install greenshadow

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

Once you're comfortable with the basic workflow, move on to [Advanced Usage](advanced-usage.md) for building integration, annual analyses, and performance tuning.
