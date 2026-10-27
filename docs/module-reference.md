# Module & Data Reference

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
