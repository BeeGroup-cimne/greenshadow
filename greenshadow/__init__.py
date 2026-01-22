# -*- coding: utf-8 -*-
"""
greenshadow: Environmental shading analysis using LiDAR data

A Python library for analyzing environmental shading in rural and urban areas using
high-resolution LiDAR data and custom hillshade algorithms.

For detailed usage examples, see the README.md and example/get_lidar_from_polygon.py
"""

__version__ = "1.0.0"
__author__ = "Jose Manuel Broto, Gerard Mor, Maite Sellart"
__email__ = "jmbroto@cimne.upc.edu"

# Import main modules
from . import lidar
from . import dsm_dem_estimation  
from . import shader
from . import utils

# Import commonly used functions for convenience
from .lidar import (
    icgc_tiles_from_geojson,
    icgc_download_from_geojson,
    list_available_tiles,
    get_tile_info
)

from .dsm_dem_estimation import (
    build_digital_model_from_lidar_pure,
    build_dem_from_dsm_obstacle_fast,
    build_dem_from_dsm_height_disruption,
    remove_buildings_from_dsm,
    refine_dem_with_buffered_interpolation
)

from .shader import (
    calculate_sky_view_factor,
    generate_multiple_hillshades_gdal,
    generate_azimuth_altitude_pairs_by_date,
    generate_azimuth_altitude_pairs_by_year,
    apply_height_shadow_to_hillshades,
    average_hillshades,
    delete_matching_files
)


# Import weather module if it exists and dependencies are available
try:
    from . import weather
    from .weather import (
        load_ERA5Land,
        download_ERA5Land,
        ERA5Land_variables
    )
    _has_weather = True
except ImportError as e:
    _has_weather = False
    # Optionally show which dependency is missing for debugging
    # print(f"Weather module unavailable: {e}")

# Create stub functions if weather is not available
if not _has_weather:
    def load_ERA5Land(*args, **kwargs):
        raise ImportError("Weather functionality requires additional dependencies: pip install pvlib polars cdsapi cfgrib xarray netCDF4"
                          "Then, to be able to download data from COPERNICUS Climate Data Store open and login in "
                          "https://cds.climate.copernicus.eu/profile?tab=profile, copy the API url and key,"
                          "and paste it in ~/.cdsapirc file")
    
    def download_ERA5Land(*args, **kwargs):
        raise ImportError("Weather functionality requires additional dependencies: pip install pvlib polars cdsapi cfgrib xarray netCDF4"
                          "Then, to be able to download data from COPERNICUS Climate Data Store open and login in "
                          "https://cds.climate.copernicus.eu/profile?tab=profile, copy the API url and key,"
                          "and paste it in ~/.cdsapirc file"
                          )
    
    def ERA5Land_variables(*args, **kwargs):
        raise ImportError("Weather functionality requires additional dependencies: pip install pvlib polars cdsapi cfgrib xarray netCDF4"
                          "Then, to be able to download data from COPERNICUS Climate Data Store open and login in "
                          "https://cds.climate.copernicus.eu/profile?tab=profile, copy the API url and key,"
                          "and paste it in ~/.cdsapirc file"
                          )

# Define what gets imported with "from greenshadow import *"
__all__ = [
    # Modules
    'lidar',
    'dsm_dem_estimation', 
    'shader',
    'utils',
    
    # LiDAR functions
    'icgc_tiles_from_geojson',
    'icgc_download_from_geojson',
    'list_available_tiles',
    'get_tile_info',
    
    # DSM/DEM functions
    'build_digital_model_from_lidar_pure',
    'build_dem_from_dsm_obstacle_fast',
    'build_dem_from_dsm_height_disruption',
    'remove_buildings_from_dsm',
    'refine_dem_with_buffered_interpolation',
    
    # Shader functions
    'calculate_sky_view_factor',
    'generate_multiple_hillshades_gdal',
    'generate_azimuth_altitude_pairs_by_date',
    'generate_azimuth_altitude_pairs_by_year',
    'apply_height_shadow_to_hillshades',
    'average_hillshades',
    'delete_matching_files'
]

# Always add weather functions (either real or stub)
__all__.extend([
    'load_ERA5Land',
    'download_ERA5Land', 
    'ERA5Land_variables'
])

# Add weather module if available
if _has_weather:
    __all__.append('weather')