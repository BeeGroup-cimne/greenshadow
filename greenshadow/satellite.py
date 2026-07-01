# -*- coding: utf-8 -*-
"""
Satellite data processing for vegetation indices and surface temperature.

This module provides functions to:
- Calculate Normalized Difference Vegetation Index (NDVI) from Sentinel-2 data
- Compute Land Surface Temperature (LST) from Landsat data using the Jiménez-Muñoz & Sobrino algorithm
- Query and download satellite imagery via STAC catalogs (Planetary Computer, USGS)

Requirements:
    pip install pystac-client planetary-computer odc-stac rioxarray shapely rasterio numpy

For detailed usage examples, see example/get_solar_energy_from_municipality.py
"""

import numpy as np
import planetary_computer
import pystac_client
import odc.stac
import rioxarray
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping, Polygon
from shapely.geometry.base import BaseGeometry
from typing import List, Union, Optional, Tuple, Dict
from datetime import datetime
import urllib.request
import re

from .logging_utils import info, warning, error, section, subsection


# ============================================================
# NDVI FUNCTIONS
# ============================================================


def get_ndvi(
    polygon: Union[str, dict, BaseGeometry],
    start_date: str = "2024-06-01",
    end_date: str = "2024-09-30",
    max_cloud_cover: int = 20,
    composition: str = "median",
    l2a_offset: int = -1000,
    output_geotiff: Optional[str] = None,
) -> rioxarray.DataArray:
    """
    Calculate NDVI from Sentinel-2 L2A data (10 m resolution).

    Args:
        polygon: Area of interest as WKT string, GeoJSON dict, or Shapely geometry (EPSG:4326)
        start_date: Start date for imagery search (YYYY-MM-DD)
        end_date: End date for imagery search (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage (0-100)
        composition: Temporal composition method - 'median' or 'best_scene'
        l2a_offset: DN to reflectance offset (-1000 for scenes after 2022-01-25, 0 for earlier)
        output_geotiff: Optional path to save output as GeoTIFF

    Returns:
        xarray.DataArray with NDVI values (-1 to 1) clipped to the input polygon

    Raises:
        ValueError: If no scenes found with the given criteria

    Example:
        >>> from shapely.geometry import Polygon
        >>> aoi = Polygon([(2.15, 41.38), (2.16, 41.38), (2.16, 41.39), (2.15, 41.39)])
        >>> ndvi = get_ndvi(aoi, start_date="2024-06-01", end_date="2024-08-31")
        >>> print(f"Mean NDVI: {float(ndvi.mean()):.3f}")
    """
    section("NDVI Calculation")

    # Convert polygon to Shapely geometry
    if isinstance(polygon, str):
        geom = shapely_wkt.loads(polygon)
    elif isinstance(polygon, dict):
        from shapely.geometry import shape
        geom = shape(polygon)
    else:
        geom = polygon

    bbox = geom.bounds
    subsection(f"Area bounds: {bbox}")

    # Search Sentinel-2 L2A scenes
    info("Searching Sentinel-2 L2A scenes...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())

    if not items:
        raise ValueError(
            f"No scenes found for criteria. Try extending date range or increasing max_cloud_cover."
        )

    info(f"Found {len(items)} scenes")

    # Load bands: Red (B04), NIR (B08), Scene Classification (SCL)
    info("Loading Sentinel-2 bands...")
    data = odc.stac.load(
        items,
        bands=["B04", "B08", "SCL"],
        bbox=bbox,
        resolution=10,
        crs="utm",
        chunks={},
        groupby="solar_day",
    )

    # Create cloud and shadow mask using SCL
    # Classes: 4=vegetation, 5=soil, 6=water, 7=unclassified, 11=snow
    subsection("Applying cloud mask...")
    valid_mask = data["SCL"].isin([4, 5, 6, 7, 11])

    # Convert DN to reflectance and compute NDVI
    info("Computing NDVI...")
    red = (data["B04"].where(valid_mask).astype("float32") + l2a_offset) / 10000.0
    nir = (data["B08"].where(valid_mask).astype("float32") + l2a_offset) / 10000.0
    ndvi = (nir - red) / (nir + red)
    ndvi = ndvi.where(np.isfinite(ndvi)).clip(-1, 1)

    # Apply temporal composition
    subsection(f"Applying {composition} composition...")
    if composition == "median":
        ndvi = ndvi.median(dim="time", skipna=True)
    elif composition == "best_scene":
        best_idx = min(
            range(len(items)), key=lambda i: items[i].properties["eo:cloud_cover"]
        )
        ndvi = ndvi.isel(time=best_idx)
    else:
        raise ValueError("composition must be 'median' or 'best_scene'")

    # Clip to exact polygon boundary
    info("Clipping to area of interest...")
    ndvi = ndvi.rio.write_crs(data.odc.crs)
    ndvi = ndvi.rio.clip([mapping(geom)], crs="EPSG:4326", drop=True)
    ndvi.name = "NDVI"

    # Save to GeoTIFF if requested
    if output_geotiff:
        info(f"Saving to {output_geotiff}...")
        ndvi.rio.to_raster(output_geotiff, compress="deflate")
        info("Saved successfully")

    return ndvi


# ============================================================
# LST FUNCTIONS
# ============================================================


def get_lst(
    polygon: Union[str, dict, BaseGeometry],
    start_date: str = "2024-07-01",
    end_date: str = "2024-07-31",
    max_cloud_cover: int = 20,
    water_vapor: float = 1.8,
    output_geotiff: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate Land Surface Temperature using Landsat 8/9 data.

    Uses the Jiménez-Muñoz & Sobrino (2014) single-channel algorithm for
    atmospheric correction applied to Landsat TIRS Band 10.

    Args:
        polygon: Area of interest as WKT string, GeoJSON dict, or Shapely geometry (EPSG:4326)
        start_date: Start date for imagery search (YYYY-MM-DD)
        end_date: End date for imagery search (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage (0-100)
        water_vapor: Precipitable water content in g/cm² (default: 1.8 for summer)
        output_geotiff: Optional path to save output as GeoTIFF

    Returns:
        Tuple of (LST_array in Celsius, metadata_dict) containing:
            - LST values as numpy array
            - Metadata with crs, transform, bounds, statistics

    Raises:
        ValueError: If no suitable Landsat scenes found

    Example:
        >>> from shapely.geometry import Polygon
        >>> aoi = Polygon([(0.55, 41.58), (0.72, 41.58), (0.72, 41.67), (0.55, 41.67)])
        >>> lst, meta = get_lst(aoi, start_date="2024-07-01", end_date="2024-07-31")
        >>> print(f"Mean LST: {np.nanmean(lst):.1f}°C")
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    section("LST Calculation")

    # Convert polygon to Shapely geometry
    if isinstance(polygon, str):
        geom = shapely_wkt.loads(polygon)
    elif isinstance(polygon, dict):
        from shapely.geometry import shape
        geom = shape(polygon)
    else:
        geom = polygon

    bbox = geom.bounds
    subsection(f"Area bounds: {bbox}")

    # Search Landsat Collection 2 Level-2 scenes
    info("Searching Landsat scenes...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {"lt": max_cloud_cover},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )
    items = sorted(search.items(), key=lambda i: i.properties["eo:cloud_cover"])

    if not items:
        raise ValueError(
            f"No Landsat scenes found for criteria. Try extending date range or increasing max_cloud_cover."
        )

    item = items[0]
    info(f"Selected: {item.id} (cloud={item.properties['eo:cloud_cover']}%)")

    # Helper function to read COG window
    def read_cog(href: str, bbox_wgs84: Tuple[float, float, float, float]):
        with rasterio.open(href) as src:
            bounds_transformed = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
            win = from_bounds(*bounds_transformed, transform=src.transform)
            arr = src.read(1, window=win).astype(np.float32)
            return arr, src.window_transform(win), src.crs

    # Load optical bands for NDVI calculation
    subsection("Loading optical bands...")
    red, transform, crs = read_cog(item.assets["red"].href, bbox)
    nir, _, _ = read_cog(item.assets["nir08"].href, bbox)

    # Convert L2 Surface Reflectance DN to reflectance
    SR_SCALE, SR_OFFSET = 2.75e-5, -0.2
    red = red * SR_SCALE + SR_OFFSET
    nir = nir * SR_OFFSET + SR_OFFSET
    ndvi = (nir - red) / (nir + red + 1e-10)

    # Get Level-1 Band 10 DN from USGS STAC
    subsection("Fetching Level-1 thermal data...")
    l1_catalog = pystac_client.Client.open("https://landsatlook.usgs.gov/stac-server")

    # Try to find corresponding L1 scene
    l1_id = item.id.replace("_L2SP_", "_L1TP_").replace("_SR", "")
    l1_search = l1_catalog.search(
        collections=["landsat-c2l1"],
        ids=[l1_id],
    )
    l1_items = list(l1_search.items())

    if not l1_items:
        # Fallback: search by time and bbox
        l1_search = l1_catalog.search(
            collections=["landsat-c2l1"],
            bbox=bbox,
            datetime=item.properties["datetime"][:10],
        )
        l1_items = list(l1_search.items())

    if not l1_items:
        raise ValueError("Could not find corresponding Level-1 scene for thermal data.")

    l1_item = l1_items[0]

    # Read Band 10 DN and metadata
    b10_href = l1_item.assets["tirs1"].href
    mtl_href = l1_item.assets["mtl.txt"].href

    dn10, _, _ = read_cog(b10_href, bbox)
    dn10[dn10 == 0] = np.nan

    # Parse L1 metadata coefficients
    info("Parsing metadata coefficients...")
    mtl = urllib.request.urlopen(mtl_href).read().decode()

    def _parse_mtl_float(key: str) -> float:
        match = re.search(rf"{key}\s*=\s*([-\d.Ee+]+)", mtl)
        if not match:
            raise ValueError(f"Could not parse {key} from metadata")
        return float(match.group(1))

    ML = _parse_mtl_float("RADIANCE_MULT_BAND_10")
    AL = _parse_mtl_float("RADIANCE_ADD_BAND_10")
    K1 = _parse_mtl_float("K1_CONSTANT_BAND_10")
    K2 = _parse_mtl_float("K2_CONSTANT_BAND_10")

    # Calculate radiance and brightness temperature
    subsection("Computing brightness temperature...")
    L = ML * dn10 + AL  # W/(m²·sr·μm)
    Tb = K2 / np.log(K1 / L + 1.0)  # Kelvin

    # Calculate emissivity using NDVI threshold method
    info("Computing emissivity...")
    Pv = ((ndvi - 0.2) / (0.5 - 0.2)) ** 2
    emissivity = np.where(
        ndvi < 0, 0.991,
        np.where(
            ndvi < 0.2, 0.966,
            np.where(
                ndvi > 0.5, 0.973,
                0.004 * Pv + 0.986
            )
        )
    )

    # Jiménez-Muñoz & Sobrino single-channel algorithm
    subsection("Applying JM-S atmospheric correction...")

    # Atmospheric function coefficients for Landsat 8 Band 10
    C_B10 = np.array([
        [0.04019, 0.02916, 1.01523],
        [-0.38333, -1.50294, 0.20324],
        [0.00918, 1.36072, -0.27514],
    ])

    w_vec = np.array([water_vapor ** 2, water_vapor, 1.0])
    psi1, psi2, psi3 = C_B10 @ w_vec

    info(f"Atmospheric parameters: ψ1={psi1:.4f}, ψ2={psi2:.4f}, ψ3={psi3:.4f} (w={water_vapor} g/cm²)")

    # Gamma and delta
    b_gamma = 1324.0  # K, Band 10
    gamma = (Tb ** 2) / (b_gamma * L)
    delta = Tb - (Tb ** 2) / b_gamma

    # Compute LST
    LST_K = gamma * ((psi1 * L + psi2) / emissivity + psi3) + delta
    LST_C = LST_K - 273.15

    # Statistics
    subsection("Results")
    info(f"Min LST: {np.nanmin(LST_C):.1f}°C")
    info(f"Mean LST: {np.nanmean(LST_C):.1f}°C")
    info(f"Max LST: {np.nanmax(LST_C):.1f}°C")

    # Prepare metadata
    metadata = {
        "crs": crs,
        "transform": transform,
        "bounds": bbox,
        "min_lst": float(np.nanmin(LST_C)),
        "mean_lst": float(np.nanmean(LST_C)),
        "max_lst": float(np.nanmax(LST_C)),
        "water_vapor": water_vapor,
        "scene_id": item.id,
        "date": item.properties["datetime"][:10],
    }

    # Save to GeoTIFF if requested
    if output_geotiff:
        info(f"Saving to {output_geotiff}...")
        with rasterio.open(
            output_geotiff,
            "w",
            driver="GTiff",
            height=LST_C.shape[0],
            width=LST_C.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(LST_C.astype(np.float32), 1)
        info("Saved successfully")

    return LST_C, metadata


# ============================================================
# COMBINED ANALYSIS
# ============================================================


def get_vegetation_surface_indices(
    polygon: Union[str, dict, BaseGeometry],
    ndvi_start_date: str = "2024-06-01",
    ndvi_end_date: str = "2024-09-30",
    lst_start_date: str = "2024-07-01",
    lst_end_date: str = "2024-07-31",
    max_cloud_cover: int = 20,
    water_vapor: float = 1.8,
) -> Dict[str, Union[rioxarray.DataArray, np.ndarray, Dict]]:
    """
    Retrieve both NDVI and LST for an area of interest.

    This is a convenience function that combines NDVI (from Sentinel-2) and
    LST (from Landsat) into a single call for vegetation-surface temperature analysis.

    Args:
        polygon: Area of interest as WKT string, GeoJSON dict, or Shapely geometry (EPSG:4326)
        ndvi_start_date: Start date for NDVI calculation (YYYY-MM-DD)
        ndvi_end_date: End date for NDVI calculation (YYYY-MM-DD)
        lst_start_date: Start date for LST calculation (YYYY-MM-DD)
        lst_end_date: End date for LST calculation (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage for both products
        water_vapor: Precipitable water content for LST calculation (g/cm²)

    Returns:
        Dictionary containing:
            - 'ndvi': xarray.DataArray with NDVI values
            - 'lst': numpy array with LST values in Celsius
            - 'lst_metadata': Dict with LST metadata

    Example:
        >>> from shapely.geometry import Polygon
        >>> aoi = Polygon([(2.15, 41.38), (2.16, 41.38), (2.16, 41.39), (2.15, 41.39)])
        >>> results = get_vegetation_surface_indices(aoi)
        >>> print(f"NDVI: {float(results['ndvi'].mean()):.3f}")
        >>> print(f"LST: {np.nanmean(results['lst']):.1f}°C")
    """
    section("Vegetation-Surface Temperature Analysis")

    ndvi = get_ndvi(
        polygon=polygon,
        start_date=ndvi_start_date,
        end_date=ndvi_end_date,
        max_cloud_cover=max_cloud_cover,
    )

    lst, lst_metadata = get_lst(
        polygon=polygon,
        start_date=lst_start_date,
        end_date=lst_end_date,
        max_cloud_cover=max_cloud_cover,
        water_vapor=water_vapor,
    )

    return {
        "ndvi": ndvi,
        "lst": lst,
        "lst_metadata": lst_metadata,
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def search_sentinel2_scenes(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 20,
) -> List[Dict]:
    """
    Search for available Sentinel-2 scenes covering a bounding box.

    Args:
        bbox: Bounding box as (minx, miny, maxx, maxy) in WGS84
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage

    Returns:
        List of scene dictionaries with metadata

    Example:
        >>> bbox = (2.0, 41.3, 2.3, 41.5)
        >>> scenes = search_sentinel2_scenes(bbox, "2024-06-01", "2024-08-31")
        >>> for scene in scenes[:3]:
        >>>     print(f"{scene['id']}: {scene['cloud_cover']}% clouds")
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())

    return [
        {
            "id": item.id,
            "datetime": item.properties["datetime"],
            "cloud_cover": item.properties["eo:cloud_cover"],
            "bbox": item.bbox,
        }
        for item in items
    ]


def search_landsat_scenes(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 20,
) -> List[Dict]:
    """
    Search for available Landsat scenes covering a bounding box.

    Args:
        bbox: Bounding box as (minx, miny, maxx, maxy) in WGS84
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage

    Returns:
        List of scene dictionaries with metadata

    Example:
        >>> bbox = (0.55, 41.58, 0.72, 41.67)
        >>> scenes = search_landsat_scenes(bbox, "2024-07-01", "2024-07-31")
        >>> for scene in scenes:
        >>>     print(f"{scene['id']}: {scene['cloud_cover']}% clouds")
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {"lt": max_cloud_cover},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )
    items = list(search.items())

    return [
        {
            "id": item.id,
            "datetime": item.properties["datetime"],
            "cloud_cover": item.properties["eo:cloud_cover"],
            "platform": item.properties["platform"],
            "bbox": item.bbox,
        }
        for item in items
    ]