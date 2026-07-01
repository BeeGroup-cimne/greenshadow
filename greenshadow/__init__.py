# -*- coding: utf-8 -*-
"""
greenshadow: Environmental shading analysis using LiDAR data

A Python library for analyzing environmental shading in rural and urban areas using
high-resolution LiDAR data and custom hillshade algorithms.

For detailed usage examples, see the README.md and example/get_solar_energy_from_municipality.py
"""

__version__ = "1.0.0"
__author__ = "Jose Manuel Broto, Maite Sellart, Gerard Mor"
__email__ = "gmor@cimne.upc.edu"

# Import main modules
from . import lidar
from . import dsm_dem_estimation
from . import shader
from . import utils
from . import rural_parcels
from . import buildings_infer
from . import buildings_infer_v2

# Import commonly used functions for convenience
from .lidar import (
    icgc_tiles_from_geojson,
    icgc_download_from_geojson,
    list_available_tiles,
    get_tile_info
)

from .dsm_dem_estimation import (
    build_digital_model_from_lidar_tiled
)

from .shader import (
    calculate_sky_view_factor,
    generate_multiple_hillshades_gdal,
    generate_azimuth_altitude_pairs_by_date,
    generate_azimuth_altitude_pairs_by_year,
    apply_height_shadow_to_hillshades,
    compute_surface_solar_energy,
    delete_matching_files,
    generate_daily_shadow_mp4,
    pv_potential_from_fourier_raster
)

from .utils import (
    force_raster_value_based_on_geometries
)

from .rural_parcels import (
    load_spanish_sigpac_parcels,
    load_spanish_preserved_regions,
    tag_parcels_with_preserved_areas,
    harmonize_sigpac_fields
)

from .buildings_infer import (
    detect_roof_planes
)

from .buildings_infer_v2 import (
    detect_roof_planes_v2
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

# Import satellite module if dependencies are available
try:
    from . import satellite
    from .satellite import (
        get_ndvi,
        get_lst,
        get_vegetation_surface_indices,
        search_sentinel2_scenes,
        search_landsat_scenes
    )
    _has_satellite = True
except ImportError as e:
    _has_satellite = False

# Create stub functions if satellite is not available
if not _has_satellite:
    def get_ndvi(*args, **kwargs):
        raise ImportError("Satellite functionality requires additional dependencies: pip install pystac-client planetary-computer odc-stac rioxarray shapely rasterio numpy")

    def get_lst(*args, **kwargs):
        raise ImportError("Satellite functionality requires additional dependencies: pip install pystac-client planetary-computer odc-stac rioxarray shapely rasterio numpy")

    def get_vegetation_surface_indices(*args, **kwargs):
        raise ImportError("Satellite functionality requires additional dependencies: pip install pystac-client planetary-computer odc-stac rioxarray shapely rasterio numpy")

    def search_sentinel2_scenes(*args, **kwargs):
        raise ImportError("Satellite functionality requires additional dependencies: pip install pystac-client planetary-computer odc-stac rioxarray shapely rasterio numpy")

    def search_landsat_scenes(*args, **kwargs):
        raise ImportError("Satellite functionality requires additional dependencies: pip install pystac-client planetary-computer odc-stac rioxarray shapely rasterio numpy")

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
    'build_digital_model_from_lidar_tiled',
    
    # Shader functions
    'calculate_sky_view_factor',
    'generate_multiple_hillshades_gdal',
    'generate_azimuth_altitude_pairs_by_date',
    'generate_azimuth_altitude_pairs_by_year',
    'apply_height_shadow_to_hillshades',
    'compute_surface_solar_energy',
    'delete_matching_files',
    'generate_daily_shadow_mp4',
    'pv_potential_from_fourier_raster',

    # Utils
    'force_raster_value_based_on_geometries',

    # Rural parcels
    'load_spanish_sigpac_parcels',
    'load_spanish_preserved_regions',
    'tag_parcels_with_preserved_areas',
    'harmonize_sigpac_fields',

    # Building inference
    'detect_roof_planes',
    'detect_roof_planes_v2'
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

# Always add satellite functions (either real or stub)
__all__.extend([
    'get_ndvi',
    'get_lst',
    'get_vegetation_surface_indices',
    'search_sentinel2_scenes',
    'search_landsat_scenes'
])

# Add satellite module if available
if _has_satellite:
    __all__.append('satellite')