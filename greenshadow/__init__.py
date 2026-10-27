# -*- coding: utf-8 -*-
"""
greenshadow: Environmental shading analysis using LiDAR data

A Python library for analyzing environmental shading in rural and urban areas using
high-resolution LiDAR data and custom hillshade algorithms.

For detailed usage examples, see the README.md and example/get_solar_energy_from_municipality.py;
the whole chain runs in one call through greenshadow.pipelines.
"""

__version__ = "1.0.0"
__author__ = "Jose Manuel Broto, Maite Sellart, Gerard Mor"
__email__ = "gmor@cimne.upc.edu"

# Import main modules
# utils first: it calls gdal.UseExceptions() at import time, which must happen
# before any other module issues its first GDAL call (otherwise GDAL emits a
# FutureWarning about exception handling not being configured).
from . import utils
from . import lidar
from . import dsm_dem_estimation
from . import shader
from . import rural_parcels
from . import buildings_infer
from . import roofs_3d
from . import dimensionate
from . import crop_rows
from . import parcel_pv
from . import pipelines

# Import commonly used functions for convenience
from .lidar import (
    icgc_tiles_from_geojson,
    icgc_download_from_geojson,
    list_available_tiles,
    get_tile_info,
    icgc_lidar_metadata,
    icgc_lidar_acquisition_date
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
    compute_surface_solar_energies,
    delete_matching_files,
    generate_daily_shadow_mp4,
    pv_potential_from_fourier_raster,
    reconstruct_poa_from_basis,
    reconstruct_vertical_poa,
    reconstruct_tracking_poa,
    read_sun_table,
    build_canopy_model,
    build_canopy_model_from_rasters,
    detect_tree_crowns,
    vegetation_growth,
    vegetation_extinction,
    apply_canopy_transmittance,
    CanopyModel,
    VegetationGrowth,
    NO_GROWTH
)

from .utils import (
    force_raster_value_based_on_geometries
)

from .crop_rows import (
    detect_crop_rows
)
from .parcel_pv import (
    plan_parcel_pv_scenarios,
    project_cashflows,
    MountingSystem,
    CropProfile,
    Economics,
    DEFAULT_SYSTEMS,
    DEFAULT_CROPS,
    SIGPAC_USE_MARGIN_EUR_HA,
    SIGPAC_USE_SHADE_RESPONSE
)

from .pipelines import (
    run_catalonia_municipalities,
    CataloniaConfig,
    CataloniaMunicipalityPipeline
)

from .rural_parcels import (
    load_spanish_sigpac_parcels,
    sigpac_parcel_id,
    load_spanish_preserved_regions,
    tag_parcels_with_preserved_areas,
    harmonize_sigpac_fields,
    filter_agricultural_parcels,
    SIGPAC_AGRICULTURAL_USES,
    SIGPAC_NON_AGRICULTURAL_USES,
)

from .buildings_infer import (
    detect_roof_planes,
    detect_roof_planes_tiled,
    filter_roof_planes_by_aoi,
    regularize_roof_planes
)

from .roofs_3d import (
    roof_planes_3d
)

from .dimensionate import (
    plan_pv_strings_in_buildings,
    write_scenario_layers,
    PVModuleSpec,
    DEFAULT_MODULES,
    solar_min_altitude
)

# Import weather module if it exists and dependencies are available
try:
    from . import weather
    from .weather import (
        load_ERA5Land,
        download_ERA5Land,
        ERA5Land_variables,
        download_ERA5_radiation,
        load_ERA5_radiation,
        load_ERA5Land_with_direct_radiation
    )
    _has_weather = True
except ImportError as e:
    _has_weather = False
    # Optionally show which dependency is missing for debugging
    # print(f"Weather module unavailable: {e}")

# Create stub functions if weather is not available
if not _has_weather:
    def load_ERA5Land(*args, **kwargs):
        raise ImportError("Weather functionality requires hypercadaster_ES: pip install hypercadaster_ES. "
                          "Then, to download data from the Copernicus Climate Data Store, log in at "
                          "https://cds.climate.copernicus.eu/profile?tab=profile, copy the API url and key, "
                          "and paste them in ~/.cdsapirc")
    
    def download_ERA5Land(*args, **kwargs):
        raise ImportError("Weather functionality requires hypercadaster_ES: pip install hypercadaster_ES. "
                          "Then, to download data from the Copernicus Climate Data Store, log in at "
                          "https://cds.climate.copernicus.eu/profile?tab=profile, copy the API url and key, "
                          "and paste them in ~/.cdsapirc")
    
    def ERA5Land_variables(*args, **kwargs):
        raise ImportError("Weather functionality requires hypercadaster_ES: pip install hypercadaster_ES. "
                          "Then, to download data from the Copernicus Climate Data Store, log in at "
                          "https://cds.climate.copernicus.eu/profile?tab=profile, copy the API url and key, "
                          "and paste them in ~/.cdsapirc")

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
    'icgc_lidar_metadata',
    'icgc_lidar_acquisition_date',
    
    # DSM/DEM functions
    'build_digital_model_from_lidar_tiled',
    
    # Shader functions
    'calculate_sky_view_factor',
    'generate_multiple_hillshades_gdal',
    'generate_azimuth_altitude_pairs_by_date',
    'generate_azimuth_altitude_pairs_by_year',
    'apply_height_shadow_to_hillshades',
    'compute_surface_solar_energy',
    'compute_surface_solar_energies',
    'delete_matching_files',
    'generate_daily_shadow_mp4',
    'pv_potential_from_fourier_raster',
    'reconstruct_poa_from_basis',

    # Semi-transparent vegetation
    'build_canopy_model',
    'build_canopy_model_from_rasters',
    'detect_tree_crowns',
    'vegetation_growth',
    'vegetation_extinction',
    'apply_canopy_transmittance',
    'CanopyModel',
    'VegetationGrowth',
    'NO_GROWTH',

    # Utils
    'force_raster_value_based_on_geometries',

    # Rural parcels
    'load_spanish_sigpac_parcels',
    'sigpac_parcel_id',
    'crop_rows',
    'detect_crop_rows',
    'load_spanish_preserved_regions',
    'tag_parcels_with_preserved_areas',
    'harmonize_sigpac_fields',
    'filter_agricultural_parcels',
    'SIGPAC_AGRICULTURAL_USES',
    'SIGPAC_NON_AGRICULTURAL_USES',

    # Building inference
    'detect_roof_planes',
    'detect_roof_planes_tiled',
    'filter_roof_planes_by_aoi',
    'regularize_roof_planes',
    'roof_planes_3d',

    # PV layout
    'dimensionate',
    'plan_pv_strings_in_buildings',
    'write_scenario_layers',
    'PVModuleSpec',
    'DEFAULT_MODULES',
    'solar_min_altitude',

    # Pipelines
    'pipelines',
    'run_catalonia_municipalities',
    'CataloniaConfig',
    'CataloniaMunicipalityPipeline'
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