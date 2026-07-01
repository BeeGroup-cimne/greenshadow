# -*- coding: utf-8 -*-
"""
LiDAR data access and processing functions for the ICGC (Institut Cartogràfic i Geològic de Catalunya).

This module provides functions to:
- Find and filter LiDAR tiles based on area of interest
- Download LAZ files from ICGC data cloud
- Basic tile management and plotting utilities

For DSM/DEM estimation functions, see dsm_dem_estimation.py
For hillshade and shading functions, see shader.py
For utility functions, see utils.py
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd
import pandas as pd
import contextily as ctx
from typing import List, Union, Dict, Optional, Tuple
from shapely.geometry.base import BaseGeometry
import laspy
import numpy as np

# Import utilities from the new modules
from .utils import (
    _session, _reproject_geom, _to_shapely, _resolve_existing_url,
    _download_many, _load_featurecollection, _filter_tiles
)
from .logging_utils import info, warning



def _read_laz(path):
    """
    Read LAZ file and return Nx4 numpy array (x, y, z, classification).
    Returns None if file cannot be read.
    """

    with laspy.open(path) as reader:
        las = reader.read()

    if las is None:
        return None

    if len(las.points) == 0:
        return np.empty((0, 4))

    # Extract coordinates
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    # Extract classification safely
    if hasattr(las, "classification"):
        classification = np.asarray(las.classification, dtype=np.uint8)
    else:
        # If classification not present, fill zeros
        classification = np.zeros_like(x, dtype=np.uint8)

    return np.column_stack((x, y, z, classification))


def icgc_tiles_from_geojson(
    aoi: Union[str, dict, BaseGeometry],
    aoi_epsg: int = 4326,
    aoi_buffer_in_meters: float = 0.0,
    centroid_only: bool = False,
    *,
    pdf_plot_path: Optional[str] = None,
    show_plot: bool = False,
) -> List[Dict]:
    """
    Find ICGC LiDAR tiles that intersect with an area of interest (AOI).
    
    Args:
        aoi: Area of interest as Shapely geometry, WKT string, or GeoJSON dict/str
        aoi_epsg: EPSG code of the AOI coordinate system  
        aoi_buffer_in_meters: Buffer distance to apply to AOI (in meters)
        centroid_only: If True, return only the tile containing AOI centroid
        pdf_plot_path: Optional path to save plot as PDF
        show_plot: Whether to display the plot
    
    Returns:
        List of tile feature dictionaries that intersect the AOI
    
    Example:
        >>> from shapely.geometry import Point
        >>> aoi = Point(2.154, 41.390).buffer(0.01)  # Barcelona area
        >>> tiles = icgc_tiles_from_geojson(aoi, aoi_epsg=4326)
        >>> print(f"Found {len(tiles)} tiles")
    """
    from .utils import BASE_JSON
    
    # Convert AOI to Shapely geometry
    aoi_geom = _to_shapely(aoi)
    
    # Load tiles from ICGC
    features, tiles_epsg = _load_featurecollection(BASE_JSON)
    
    # Filter tiles
    matching = _filter_tiles(
        features=features,
        aoi=aoi_geom,
        aoi_epsg=aoi_epsg,
        target_epsg=tiles_epsg,
        aoi_buffer_in_meters=aoi_buffer_in_meters,
        centroid_only=centroid_only
    )
    
    # Create plot if requested
    if pdf_plot_path or show_plot:
        _plot_tiles_and_aoi(
            tiles=matching,
            tiles_epsg=tiles_epsg,
            aoi=aoi_geom,
            aoi_epsg=aoi_epsg,
            aoi_buffer_in_meters=aoi_buffer_in_meters,
            pdf_path=pdf_plot_path,
            show=show_plot
        )
    
    return matching


def icgc_download_from_geojson(
    out_dir: str,
    tiles: List[Dict],
    aoi: Union[str, dict, BaseGeometry],
    aoi_epsg: int = 4326,
    aoi_buffer_in_meters: float = 0.0,
    max_workers: int = 8,
) -> List[str]:
    """
    Download LAZ files for given tiles.
    
    Args:
        out_dir: Output directory for downloaded files
        tiles: List of tile dictionaries (from icgc_tiles_from_geojson)
        aoi: Area of interest (used for logging/validation)
        aoi_epsg: EPSG code of AOI
        aoi_buffer_in_meters: Buffer applied to AOI
        centroid_only: Whether only centroid tile was requested
        max_workers: Number of parallel download workers
    
    Returns:
        List of paths to downloaded LAZ files
    
    Example:
        >>> tiles = icgc_tiles_from_geojson(aoi, aoi_epsg=4326)
        >>> paths = icgc_download_from_geojson("./data", tiles, aoi, aoi_epsg=4326)
        >>> print(f"Downloaded {len(paths)} files")
    """
    if not tiles:
        tiles = gs.icgc_tiles_from_geojson(
            aoi=aoi,
            aoi_epsg=aoi_epsg,
            aoi_buffer_in_meters=aoi_buffer_in_meters,
            centroid_only=False,
            pdf_plot_path=f"{wd}/{municipality_name}/tiles.pdf"
        )
    
    urls = []
    s = _session()
    
    for tile in tiles:
        props = tile.get("properties", {})
        id1k = props.get("ID1K", "")
        id10k = props.get("ID10K", "")
        
        if not id1k and not id10k:
            warning(f"Tile missing IDs: {props}")
            continue
        
        url = _resolve_existing_url(s, id1k, id10k)
        if url:
            urls.append(url)
        else:
            warning(f"No accessible URL found for tile {id1k}/{id10k}")
    
    if not urls:
        warning("No downloadable URLs found")
        return []
        
    # Download files
    return _download_many(urls, out_dir, max_workers=max_workers)


def _plot_tiles_and_aoi(
    tiles: List[Dict],
    tiles_epsg: int,
    aoi: BaseGeometry,
    aoi_epsg: int,
    aoi_buffer_in_meters: float,
    pdf_path: Optional[str] = None,
    show: bool = False
):
    """Plot tiles and AOI for visualization."""
    
    if not tiles:
        warning("No tiles to plot")
        return
    
    # Convert tiles to GeoDataFrame
    tiles_gdf = gpd.GeoDataFrame.from_features(tiles, crs=f"EPSG:{tiles_epsg}")
    
    # Reproject AOI to tiles CRS if needed
    if aoi_epsg != tiles_epsg:
        aoi_tiles_crs = _reproject_geom(aoi, f"EPSG:{aoi_epsg}", f"EPSG:{tiles_epsg}")
    else:
        aoi_tiles_crs = aoi
    
    # Apply buffer for visualization
    if aoi_buffer_in_meters > 0:
        aoi_buffered = aoi_tiles_crs.buffer(aoi_buffer_in_meters)
    else:
        aoi_buffered = aoi_tiles_crs
    
    # Create AOI GeoDataFrame
    aoi_gdf = gpd.GeoDataFrame([1], geometry=[aoi_tiles_crs], crs=f"EPSG:{tiles_epsg}")
    aoi_buf_gdf = gpd.GeoDataFrame([1], geometry=[aoi_buffered], crs=f"EPSG:{tiles_epsg}")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot tiles
    tiles_gdf.plot(ax=ax, facecolor='lightblue', edgecolor='blue', alpha=0.7, linewidth=1)
    
    # Plot AOI
    aoi_gdf.plot(ax=ax, facecolor='red', alpha=0.8, label='AOI')
    if aoi_buffer_in_meters > 0:
        aoi_buf_gdf.plot(ax=ax, facecolor='none', edgecolor='red', 
                        linestyle='--', alpha=0.8, label=f'AOI + {aoi_buffer_in_meters}m buffer')
    
    # Add basemap
    try:
        # Convert to Web Mercator for basemap
        tiles_web = tiles_gdf.to_crs(epsg=3857)
        aoi_web = aoi_gdf.to_crs(epsg=3857)
        
        # Get bounds in Web Mercator
        bounds = tiles_web.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        
        ctx.add_basemap(ax, crs=tiles_gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
    except Exception as e:
        warning(f"Could not add basemap: {e}")
    
    # Formatting
    ax.set_title(f'ICGC LiDAR Tiles and AOI\n{len(tiles)} tiles found', fontsize=14, fontweight='bold')
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label='AOI'),
        Patch(facecolor='red', alpha=0.5, label='Tiles')
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    
    # Add tile labels
    for idx, row in tiles_gdf.iterrows():
        centroid = row.geometry.centroid
        tile_id = row.get('id1k', row.get('id10k', f'tile_{idx}'))
        ax.annotate(tile_id, (centroid.x, centroid.y), 
                   ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save PDF if requested
    if pdf_path:
        os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
        fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        info(f"Plot saved to {pdf_path}")
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def list_available_tiles(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    epsg: int = 4326
) -> pd.DataFrame:
    """
    List all available ICGC LiDAR tiles, optionally filtered by bounding box.
    
    Args:
        bbox: Optional bounding box (minx, miny, maxx, maxy) to filter tiles
        epsg: EPSG code of the bounding box coordinates
    
    Returns:
        DataFrame with tile information
    
    Example:
        >>> # List all tiles
        >>> all_tiles = list_available_tiles()
        >>> print(f"Total tiles: {len(all_tiles)}")
        >>> 
        >>> # Filter by Barcelona area
        >>> bbox = (2.0, 41.3, 2.3, 41.5)  # minx, miny, maxx, maxy
        >>> bcn_tiles = list_available_tiles(bbox, epsg=4326)
        >>> print(f"Barcelona area tiles: {len(bcn_tiles)}")
    """
    from shapely.geometry import box
    from .utils import BASE_JSON
    
    info("Loading ICGC tile index...")
    features, tiles_epsg = _load_featurecollection(BASE_JSON)
    
    # Convert features to GeoDataFrame
    tiles_gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{tiles_epsg}")
    
    # Filter by bounding box if provided
    if bbox is not None:
        bbox_geom = box(*bbox)
        filtered_tiles = _filter_tiles(
            features=features,
            aoi=bbox_geom,
            aoi_epsg=epsg,
            target_epsg=tiles_epsg,
            aoi_buffer_in_meters=0.0,
            centroid_only=False
        )
        tiles_gdf = gpd.GeoDataFrame.from_features(filtered_tiles, crs=f"EPSG:{tiles_epsg}")
    
    # Convert to regular DataFrame for easier viewing
    result_df = pd.DataFrame(tiles_gdf.drop(columns='geometry'))
    
    # Add geometry bounds as separate columns
    bounds = tiles_gdf.bounds
    result_df['minx'] = bounds['minx']
    result_df['miny'] = bounds['miny'] 
    result_df['maxx'] = bounds['maxx']
    result_df['maxy'] = bounds['maxy']
    result_df['tile_crs'] = f'EPSG:{tiles_epsg}'
    
    info(f"Found {len(result_df)} tiles")
    
    return result_df


def get_tile_info(tile_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific tile.
    
    Args:
        tile_id: Tile identifier (id1k or id10k)
    
    Returns:
        Dictionary with tile information, or None if not found
    
    Example:
        >>> info = get_tile_info("some_tile_id")
        >>> if info:
        >>>     print(f"Tile area: {info['geometry'].area} sq meters")
    """
    from .utils import BASE_JSON
    
    features, tiles_epsg = _load_featurecollection(BASE_JSON)
    
    for feature in features:
        props = feature.get("properties", {})
        if props.get("id1k") == tile_id or props.get("id10k") == tile_id:
            return {
                'properties': props,
                'geometry': feature.get("geometry"),
                'crs': f'EPSG:{tiles_epsg}',
                'bounds': gpd.GeoSeries([feature.get("geometry")]).bounds.iloc[0].to_dict() if feature.get("geometry") else None
            }
    
    return None
