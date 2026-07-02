# greenshadow

A Python library for environmental shading analysis using LiDAR data and custom algorithms to simulate the solar shading of rural and urban areas in maximum detail. greenshadow uses hillshade techniques combined with cast shadow calculations to provide accurate solar radiation analysis.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-EUPL%20v1.2-blue.svg)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/BeeGroup-cimne/greenshadow)

### Key Features

- 🛰️ **LiDAR Data Processing**: Access and process LiDAR data from the Institut Cartogràfic i Geològic de Catalunya (ICGC)
- 🏔️ **DSM/DEM Generation**: Create Digital Surface Models (DSM) and Digital Elevation Models (DEM) from point clouds
- ☀️ **Solar Position Calculation**: Compute accurate sun positions for any date, time, and location
- 🌗 **Hillshade Generation**: Generate detailed hillshades with cast shadows for solar analysis
- 🏗️ **Building Integration**: Remove buildings from DSM to create clean terrain models
- 📅 **Temporal Analysis**: Analyze solar exposure over time periods (daily, seasonal, annual)
- 🔬 **High-Resolution Output**: Support for sub-meter resolution analysis

## 📚 Documentation

### Getting Started
- [**Installation & Quick Start**](docs/installation-quickstart.md) - GDAL prerequisites, installation methods, and your first solar analysis

### Library Reference
- [**Advanced Usage**](docs/advanced-usage.md) - Building integration, annual solar analysis, custom solar calculations, batch processing, and performance tips
- [**Module & Data Reference**](docs/module-reference.md) - Module structure, data sources, and troubleshooting

## 🚀 Quick Start

```python
import greenshadow as gs
from shapely.geometry import Point
import datetime as dt

# Define area of interest (Barcelona example)
aoi = Point(2.154, 41.390).buffer(0.01)
working_dir = "/path/to/your/data"

# 1. Find and download LiDAR tiles
tiles = gs.icgc_tiles_from_geojson(aoi=aoi, aoi_epsg=4326, aoi_buffer_in_meters=500)
laz_paths = gs.icgc_download_from_geojson(out_dir=f"{working_dir}/lidar_data", tiles=tiles, aoi=aoi, aoi_epsg=4326)

# 2. Build a DSM from the point cloud
dsm_path = gs.build_dsm_from_lidar_pure(laz_paths=laz_paths, out_dsm=f"{working_dir}/dsm.tif", resolution=0.5, epsg=25831)

# 3. Compute solar positions and generate hillshades
solar_data = gs.generate_azimuth_altitude_pairs_by_date(date=dt.date(2025, 6, 21), lat=41.390, lon=2.154, tz_name="Europe/Madrid")
hillshade_paths = gs.generate_multiple_hillshades_gdal(raster_dir=f"{working_dir}/hillshades", dsm_path=dsm_path, azimuth_altitude=solar_data['azimuth_altitude_pairs'])

# 4. Aggregate into a single average exposure map
average_path = gs.average_hillshades(hillshade_dir=f"{working_dir}/hillshades", out_path=f"{working_dir}/summer_solstice_average.tif")
```

For GDAL setup, detailed installation steps, and a fully worked example, see [Installation & Quick Start](docs/installation-quickstart.md).

## 🎯 Key Applications

- **Urban Solar Studies**: Rooftop solar potential, façade shading, urban heat island analysis
- **Rural & Terrain Analysis**: Land suitability, agricultural solar exposure, landscape planning
- **Energy Modeling**: Input generation for building energy simulation and photovoltaic sizing
- **Academic Research**: Environmental science, geography, and remote sensing applications

## 📊 Data Coverage

- **Native Support**: Institut Cartogràfic i Geològic de Catalunya (ICGC) LiDAR, covering all of Catalonia (~32,000 km²) at up to 4 points/m²
- **Extensible**: Works with any LAZ/LAS point cloud data for other regions

## 🏗️ Advanced Features

- **Terrain Reconstruction**: Multiscale harmonic DEM generation and building removal from DSMs
- **Flexible Temporal Resolution**: From single dates to full annual analyses with configurable granularity
- **Height-Based Shadows**: Apply shadow analysis at custom elevations (e.g., rooftop height)
- **Parallelized Processing**: Multi-worker downloads and hillshade generation for large areas

See [Advanced Usage](docs/advanced-usage.md) for detailed examples, and [Module & Data Reference](docs/module-reference.md) for troubleshooting and module details.

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

## 📄 License

This project is licensed under the **EUPL v1.2**. See the [license](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12) for details.

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
