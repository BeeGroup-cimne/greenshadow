# -*- coding: utf-8 -*-
import fnmatch
import math
from typing import List, Dict, Optional, Tuple, Any, Sequence
import pandas as pd
from rasterio.enums import Resampling
from rasterio.warp import reproject
from osgeo import gdal
import datetime as dt
from .utils import _parse_iso8601_duration, _date_range_local_to_utc, has_cuda, rasterize_polygon_mask
from .logging_utils import info
from numba import njit, prange, cuda
import scipy.ndimage as ndi
import sys
import os
import rasterio
import multiprocessing as mp
import concurrent.futures
from tqdm import tqdm
from shapely.ops import transform
import pyproj
from concurrent.futures import ThreadPoolExecutor
import warnings
import pytz
import re
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from scipy.ndimage import binary_dilation, zoom
from skimage.transform import resize
import imageio
import geopandas as gpd
from rasterio.features import geometry_mask, rasterize
import cv2

def rasterize_aoi_mask(aoi, reference_path: str) -> np.ndarray:
    """
    Rasterize a shapely AOI to a boolean mask aligned with reference raster.
    True  = inside AOI
    False = outside AOI
    """

    with rasterio.open(reference_path) as src:
        shape = (src.height, src.width)
        transform = src.transform

    mask = rasterize(
        [(aoi, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    return mask.astype(bool)


@njit(parallel=True, fastmath=True)
def _shadow_scan(z_rot, nodata_rot, dz_per_px):
    H, W = z_rot.shape
    lit = np.zeros((H, W), dtype=np.uint8)

    for i in prange(H):
        horizon = -1e30
        for j in range(W):
            if nodata_rot[i, j]:
                lit[i, j] = 1
                continue

            h = z_rot[i, j]
            if h >= horizon:
                lit[i, j] = 1
                horizon = h

            horizon -= dz_per_px

    return lit

def _cast_shadow_mask(
    dsm: np.ndarray,
    pixel_size: float,
    azimuth_deg: float,
    altitude_deg: float,
    *,
    nodata_mask: np.ndarray | None = None,
    z_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:

    if altitude_deg <= 0:
        return np.zeros_like(dsm, dtype=bool), ~nodata_mask

    z = dsm * z_factor

    # Rotate so sun comes from the left
    theta = azimuth_deg + 90.0

    # Rotate DSM (nearest neighbor is sufficient for shadows)
    z_rot = ndi.rotate(
        z,
        theta,
        reshape=False,
        order=0,
        mode="nearest",
        prefilter=False,
    )

    # Rotate nodata mask
    if nodata_mask is not None:
        nodata_rot = ndi.rotate(
            nodata_mask.astype(np.uint8),
            theta,
            reshape=False,
            order=0,
            prefilter=False,
        ) > 0
    else:
        nodata_rot = np.zeros_like(z_rot, dtype=bool)

    valid_rot = ~nodata_rot

    # Vertical change per pixel along ray
    dz_per_px = np.tan(np.deg2rad(altitude_deg)) * pixel_size

    # Numba-accelerated horizon scan
    lit_rot = _shadow_scan(z_rot, nodata_rot, dz_per_px).astype(bool)

    # Rotate results back
    lit = ndi.rotate(
        lit_rot.astype(np.uint8),
        -theta,
        reshape=False,
        order=0,
        prefilter=False,
    ) > 0

    valid = ndi.rotate(
        valid_rot.astype(np.uint8),
        -theta,
        reshape=False,
        order=0,
        prefilter=False,
    ) > 0

    return lit, valid

def _apply_dsm_shadow_to_hillshade(
    hillshade_path: str,
    dsm_path: str,
    out_path: str,
    azimuth: float,
    altitude: float,
    *,
    shadow_value: float = 1.0,
):
    with rasterio.open(dsm_path) as dsrc, rasterio.open(hillshade_path) as hsrc:

        dsm = dsrc.read(1).astype(np.float32)
        hs = hsrc.read(1).astype(np.float32)

        pixel_size = abs(dsrc.transform.a)

        nodata_mask = (
            (dsm == dsrc.nodata)
            | ~np.isfinite(dsm)
        )

        # Fill DSM nodata once (ray continuity)
        if nodata_mask.any():
            _, (iy, ix) = ndi.distance_transform_edt(
                nodata_mask,
                return_indices=True,
            )
            dsm[nodata_mask] = dsm[iy[nodata_mask], ix[nodata_mask]]

        # Cast shadows
        lit, valid = _cast_shadow_mask(
            dsm,
            pixel_size,
            azimuth,
            altitude,
            nodata_mask=nodata_mask,
        )

        # Apply shadows
        hs[~lit & valid] = shadow_value
        hs[~valid] = hsrc.nodata

        profile = hsrc.profile.copy()

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(hs.astype(profile["dtype"]), 1)
            dst.update_tags(**hsrc.tags())

    return out_path

def _generate_hillshade_gdal(
    results_dir: str,
    dsm_path: str,
    azimuth: float,
    altitude: float,
    zFactor: float = 1.0,
    *,
    include_datetime_info: bool = False,
    when: dt.datetime = None,
    compute_edges: bool = False,
    creation_options: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create a GDAL hillshade (local shading only). Returns the raw hillshade path (uint8 0..255).
    """
    os.makedirs(results_dir, exist_ok=True)
    if include_datetime_info:
        out_path = f"{results_dir}/{int(azimuth):03}_{int(altitude):02}_raw_{when.strftime("%Y-%m-%dT%H:%M:%S")}.tif"
    else:
        out_path = f"{results_dir}/{int(azimuth):03}_{int(altitude):02}_raw.tif"
    if os.path.exists(out_path):
        return out_path

    # Optimize creation options for performance
    if creation_options is None:
        creation_options = {
            "TILED": "YES",
            "COMPRESS": "LZW",
            "BIGTIFF": "IF_SAFER"
        }

    # Prefer Options object; if some builds are picky, we can fall back to list-of-opts
    try:
        opts_kwargs = dict(
            format="GTiff",
            computeEdges=bool(compute_edges),
            alg="Horn",
            zFactor=float(zFactor),
            scale=1.0,
            azimuth=float(azimuth),
            altitude=float(altitude),
            creationOptions=creation_options
        )
        opts = gdal.DEMProcessingOptions(**opts_kwargs)
        gdal.DEMProcessing(out_path, dsm_path, "hillshade", options=opts)
    except Exception:
        opt_list = ["-of", "GTiff", "-alg", "Horn", "-z", str(zFactor), "-s", "1",
                    "-az", str(azimuth), "-alt", str(altitude)]
        if compute_edges:
            opt_list.append("-compute_edges")
        for k, v in creation_options.items():
            opt_list += ["-co", f"{k}={v}"]
        gdal.DEMProcessing(out_path, dsm_path, "hillshade", options=opt_list)

    return out_path


# Public wrapper for the private function
def generate_hillshade_gdal(
    results_dir: str,
    dsm_path: str,
    azimuth: float,
    altitude: float,
    zFactor: float = 1.0,
    *,
    include_datetime_info: bool = False,
    when: dt.datetime = None,
    compute_edges: bool = False,
    creation_options: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create a GDAL hillshade. Returns the path to the generated hillshade file.
    """
    return _generate_hillshade_gdal(
        results_dir=results_dir,
        dsm_path=dsm_path,
        azimuth=azimuth,
        altitude=altitude,
        zFactor=zFactor,
        include_datetime_info=include_datetime_info,
        when=when,
        compute_edges=compute_edges,
        creation_options=creation_options
    )


# def _cast_shadow_mask(
#     dsm: np.ndarray,
#     pixel_size: float,
#     azimuth_deg: float,
#     altitude_deg: float,
#     *,
#     nodata_mask: np.ndarray | None = None,
#     z_factor: float = 1.0,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Returns:
#       lit: True where illuminated
#       valid: True where inside original footprint (not padding from rotation)
#     """
#
#     if altitude_deg <= 0:
#         return np.zeros_like(dsm, dtype=bool), ~nodata_mask
#
#     z = dsm * z_factor
#
#     # Rotate so sun comes from the left
#     theta = azimuth_deg + 90
#
#     # Rotate DSM
#     z_rot = ndi.rotate(z, theta, reshape=False, order=1, mode="nearest")
#
#     # Rotate nodata mask (this tracks what is real vs padding)
#     if nodata_mask is not None:
#         nodata_rot = ndi.rotate(nodata_mask.astype(np.uint8), theta,
#                                 reshape=False, order=0) > 0
#         valid_rot = ~nodata_rot
#     else:
#         nodata_rot = np.zeros_like(z_rot, dtype=bool)
#         valid_rot = np.ones_like(z_rot, dtype=bool)
#
#     H, W = z_rot.shape
#     lit_rot = np.zeros_like(z_rot, dtype=bool)
#
#     # Vertical change per pixel along ray
#     dz_per_px = np.tan(np.deg2rad(altitude_deg)) * pixel_size
#
#     for i in range(H):
#         horizon = -np.inf
#         for j in range(W):
#             h = z_rot[i, j]
#
#             if nodata_rot[i, j]:
#                 lit_rot[i, j] = True
#                 continue
#
#             if h >= horizon:
#                 lit_rot[i, j] = True
#                 horizon = h
#             else:
#                 lit_rot[i, j] = False
#
#             horizon -= dz_per_px
#
#     # Rotate lit mask back
#     lit = ndi.rotate(lit_rot.astype(np.uint8), -theta,
#                      reshape=False, order=0) > 0
#
#     # Rotate validity mask back (CRUCIAL PART)
#     valid = ndi.rotate(valid_rot.astype(np.uint8), -theta,
#                        reshape=False, order=0) > 0
#
#     return lit, valid
#
#
# def _apply_dsm_shadow_to_hillshade(
#     hillshade_path: str,
#     dsm_path: str,
#     out_path: str,
#     azimuth: float,
#     altitude: float,
#     *,
#     shadow_value: float = 1.0,
# ):
#     with rasterio.open(dsm_path) as dsrc, rasterio.open(hillshade_path) as hsrc:
#         dsm = dsrc.read(1).astype(np.float32)
#         hs = hsrc.read(1).astype(np.float32)
#
#         pixel_size = abs(dsrc.transform.a)
#         nodata_mask = (dsm == dsrc.nodata) | ~np.isfinite(dsm)
#
#         # Fill DSM nodata so rays are continuous
#         if nodata_mask.any():
#             _, (iy, ix) = ndi.distance_transform_edt(nodata_mask, return_indices=True)
#             dsm[nodata_mask] = dsm[iy[nodata_mask], ix[nodata_mask]]
#
#         # Compute cast shadows + validity footprint
#         lit, valid = _cast_shadow_mask(
#             dsm,
#             pixel_size,
#             azimuth,
#             altitude,
#             nodata_mask=nodata_mask
#         )
#
#         # Apply shadows only where valid
#         hs[~lit & valid] = shadow_value
#
#         # Outside original footprint → preserve nodata
#         hs[~valid] = hsrc.nodata
#
#         # Copy full profile + metadata
#         profile = hsrc.profile.copy()
#
#         with rasterio.open(out_path, "w", **profile) as dst:
#             dst.write(hs.astype(profile["dtype"]), 1)
#
#             # Inherit all metadata (CRUCIAL PART)
#             dst.update_tags(**hsrc.tags())


def _apply_height_shadow_to_hillshade(
    hillshade_dir: str,
    dem_path: str,
    dsm_path: str,
    out_dir: str,
    *,
    height_m: float = 1.5,            # DEM + height threshold
    pattern: str = "*.tif",           # hillshade filename pattern
    recursive: bool = True,
    overwrite: bool = False,
    mask_value: int | float = 1,      # value to write where DSM > DEM+height
) -> list[str]:
    """
    For each hillshade GeoTIFF in `hillshade_dir`, produce an output raster where:
      - if DSM > DEM + height_m   → output = mask_value (e.g., 1)
      - else                      → output = hillshade pixel value

    Notes:
      • This is a per-pixel vertical test (no ray-tracing).
      • DSM/DEM are reprojected to each hillshade grid. If all hillshades share the grid,
        reprojection is cached and reused for speed.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect hillshade files
    root = Path(hillshade_dir)
    files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if not files:
        return []

    # Open DEM & DSM once
    dem_ds = rasterio.open(dem_path)
    dsm_ds = rasterio.open(dsm_path)

    # Cache reprojected DEM/DSM by (height, width, transform, crs)
    reproj_cache: Dict[Tuple[int, int, Tuple[float, ...], str], Tuple[np.ndarray, np.ndarray]] = {}

    out_paths: list[str] = []

    try:
        for hf in tqdm(files, desc="Applying height-based shadows"):
            with rasterio.open(hf) as hs:
                hs_band = hs.read(1)  # keep dtype as is (often uint8)
                hs_mask = (hs.read_masks(1) != 0)  # True where valid
                H, W = hs.height, hs.width
                key = (H, W, tuple(hs.transform)[:6], str(hs.crs) if hs.crs else "")

                if key not in reproj_cache or overwrite:
                    # Reproject DEM→hillshade grid
                    dem_r = np.full((H, W), np.nan, dtype=np.float32)
                    reproject(
                        source=rasterio.band(dem_ds, 1),
                        destination=dem_r,
                        src_transform=dem_ds.transform,
                        src_crs=dem_ds.crs,
                        src_nodata=dem_ds.nodata,
                        dst_transform=hs.transform,
                        dst_crs=hs.crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                    # Reproject DSM→hillshade grid
                    dsm_r = np.full((H, W), np.nan, dtype=np.float32)
                    reproject(
                        source=rasterio.band(dsm_ds, 1),
                        destination=dsm_r,
                        src_transform=dsm_ds.transform,
                        src_crs=dsm_ds.crs,
                        src_nodata=dsm_ds.nodata,
                        dst_transform=hs.transform,
                        dst_crs=hs.crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                    reproj_cache[key] = (dem_r, dsm_r)
                else:
                    dem_r, dsm_r = reproj_cache[key]

                # Compute mask: DSM > DEM + height
                dem_plus = dem_r + float(height_m)
                valid_cmp = np.isfinite(dem_plus) & np.isfinite(dsm_r) & hs_mask
                mask = np.zeros((H, W), dtype=bool)
                mask[valid_cmp] = dsm_r[valid_cmp] > dem_plus[valid_cmp]

                # Compose output (where mask True → mask_value else hillshade)
                # Preserve dtype of hillshade, but write nodata wherever hillshade is nodata.
                out = hs_band.astype(np.float32)
                out[mask] = float(mask_value)

                # Respect hillshade nodata
                nodata = hs.nodata
                if nodata is None:
                    # If no nodata defined, keep input zeros if masked out by hs_mask==0
                    # or set an explicit nodata
                    nodata = 0 if np.issubdtype(hs_band.dtype, np.integer) else np.nan
                out[~hs_mask] = nodata

                # Cast back to original dtype if possible
                if np.issubdtype(hs_band.dtype, np.integer):
                    # clip to dtype range
                    info = np.iinfo(hs_band.dtype)
                    out = np.clip(np.round(out), info.min, info.max).astype(hs_band.dtype)
                else:
                    out = out.astype("float32")

                # Write result next to input
                rel = hf.relative_to(root)
                out_path = Path(out_dir) / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)

                prof = hs.profile.copy()
                prof.update(
                    driver="GTiff",
                    count=1,
                    dtype=out.dtype.name if hasattr(out.dtype, "name") else str(out.dtype),
                    compress="deflate",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    interleave="band",
                    nodata=nodata,
                )
                with rasterio.open(out_path, "w", **prof) as dst:
                    dst.write(out, 1)

                out_paths.append(str(out_path))
    finally:
        dem_ds.close()
        dsm_ds.close()

    return out_paths

def generate_multiple_hillshades_gdal(
    raster_dir: str,
    dsm_path: str,
    azimuth_altitude: list[tuple[float, float]],
    *,
    related_datetimes=None,
    zFactor: float = 1.0,
    compute_edges: bool = False,
    creation_options=None,
    include_datetime_info: bool = False,
    qgis_temporal: bool = False,
    max_workers: int = 2,
    overwrite: bool = False,
    aoi=None,
    aoi_crs: str = "EPSG:4326",
    nodata_value: float = -9999.0,
) -> list[str]:
    """
    Generate hillshades normalized to [0,1], AOI-masked.
    """

    os.makedirs(raster_dir, exist_ok=True)

    valid_positions = []
    for i, (az, alt) in enumerate(azimuth_altitude):
        if alt > 0:
            when = related_datetimes[i] if related_datetimes else None
            valid_positions.append((i, az, alt, when))

    if not valid_positions:
        return []

    max_workers = min(max_workers, len(valid_positions), mp.cpu_count())

    def process_single(args):
        i, az, alt, when = args

        base = f"{int(az):03}_{int(alt):02}"
        if when and qgis_temporal:
            base += "_" + (when.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(when, "strftime") else str(when))
        out_path = os.path.join(raster_dir, f"{base}.tif")

        if os.path.exists(out_path) and not overwrite:
            return out_path

        try:
            hs_raw = _generate_hillshade_gdal(
                results_dir=raster_dir,
                when=when,
                dsm_path=dsm_path,
                azimuth=az,
                altitude=alt,
                zFactor=zFactor,
                compute_edges=compute_edges,
                creation_options=creation_options,
                include_datetime_info=include_datetime_info,
            )

            final_path = _apply_dsm_shadow_to_hillshade(
                hillshade_path=hs_raw,
                dsm_path=dsm_path,
                out_path=hs_raw.replace("_raw", ""),
                azimuth=az,
                altitude=alt,
            )

            with rasterio.open(final_path) as src:
                hs = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                raster_crs = src.crs

            # Normalize 0–255 → 0–1
            hs /= 255.0
            np.clip(hs, 0.0, 1.0, out=hs)

            if aoi is not None:
                transformer = pyproj.Transformer.from_crs(
                    aoi_crs,
                    raster_crs,
                    always_xy=True,
                ).transform
                aoi_mask = rasterize_aoi_mask(transform(transformer, aoi), final_path)
                hs[~aoi_mask] = nodata_value

            profile.update(
                dtype="float32",
                nodata=nodata_value,
                compress="deflate",
                predictor=3,
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(hs, 1)

            if hs_raw != final_path and os.path.exists(hs_raw):
                os.remove(hs_raw)

            return out_path

        except Exception as e:
            print(f"Error az={az} alt={alt}: {e}")
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm(ex.map(process_single, valid_positions),
                      total=len(valid_positions),
                      desc="Generating hillshades"):
            if r:
                results.append(r)

    return results


def _solar_position_az_el(
    times_utc: pd.DatetimeIndex,
    lat: float,
    lon: float,
    *,
    use_pysolar: bool = True,
    use_sunpy: bool = False,
) -> pd.DataFrame:
    """
    Calculate solar azimuth and elevation for given times and location.
    
    Args:
        times_utc: UTC timestamps
        lat: Latitude in degrees
        lon: Longitude in degrees
        use_pysolar: Whether to try using pysolar library
        use_sunpy: Whether to try using sunpy library
    
    Returns:
        DataFrame with 'azimuth' and 'elevation' columns
    """
    results = []
    
    # Try to use pysolar if available
    if use_pysolar:
        try:
            import pysolar
            for t in times_utc:
                az = pysolar.solar.get_azimuth(lat, lon, t)
                el = pysolar.solar.get_altitude(lat, lon, t)
                # Pysolar already uses meteorological convention: north = 0°, east = 90°, south = 180°, west = 270°
                # No conversion needed
                results.append({'azimuth': az, 'elevation': el})
            
            return pd.DataFrame(results, index=times_utc)
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"Pysolar calculation failed: {e}")
    
    # Try to use sunpy if available
    if use_sunpy:
        try:
            from sunpy.coordinates import get_sun
            from astropy.coordinates import EarthLocation
            from astropy.time import Time
            import astropy.units as u
            
            location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)
            astropy_times = Time(times_utc)
            
            sun_coords = get_sun(astropy_times)
            sun_altaz = sun_coords.transform_to(
                astropy.coordinates.AltAz(location=location, obstime=astropy_times)
            )
            
            for i, t in enumerate(times_utc):
                az = float(sun_altaz[i].az.deg)
                el = float(sun_altaz[i].alt.deg)
                results.append({'azimuth': az, 'elevation': el})
            
            return pd.DataFrame(results, index=times_utc)
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"SunPy calculation failed: {e}")
    
    # Fallback: simplified solar position calculation
    warnings.warn("Using simplified solar position calculation. Install pysolar or sunpy for better accuracy.")
    
    for t in times_utc:
        # Simplified solar position calculation
        # This is not accurate - just a placeholder
        hour_of_year = t.timetuple().tm_yday * 24 + t.hour
        
        # Very rough approximation
        declination = 23.45 * math.sin(math.radians(360 * (284 + t.timetuple().tm_yday) / 365))
        hour_angle = 15 * (t.hour - 12)  # degrees from solar noon
        
        # Solar elevation (very rough)
        lat_rad = math.radians(lat)
        decl_rad = math.radians(declination)
        hour_rad = math.radians(hour_angle)
        
        elevation = math.degrees(math.asin(
            math.sin(decl_rad) * math.sin(lat_rad) + 
            math.cos(decl_rad) * math.cos(lat_rad) * math.cos(hour_rad)
        ))
        
        # Solar azimuth (very rough)
        azimuth = 180 + hour_angle  # Simplified
        azimuth = azimuth % 360
        
        results.append({'azimuth': azimuth, 'elevation': elevation})
    
    return pd.DataFrame(results, index=times_utc)


def _floor_to_step(x: np.ndarray, step: float, low: float, high: float) -> np.ndarray:
    """Floor values to discrete steps within bounds."""
    x_clamped = np.clip(x, low, high)
    return low + step * np.floor((x_clamped - low) / step)


def generate_azimuth_altitude_pairs_by_date(
    date: dt.date,
    lat: float,            # Madrid by default
    lon: float,
    *,
    tz_name: str = "Europe/Madrid",
    granularity: str = "PT1H",       # ISO 8601 duration
    solar_degrees_discretization: float = 1.0,
    only_daylight: bool = True,
    weather_df: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate azimuth/altitude pairs for solar positions on a given date.
    
    Args:
        date: Date to generate positions for
        lat: Latitude in degrees
        lon: Longitude in degrees  
        tz_name: Timezone name (e.g., "Europe/Madrid")
        granularity: Time step as ISO 8601 duration string (e.g., "PT15M", "PT1H")
        solar_degrees_discretization: Round positions to this many degrees
        only_daylight: Only include positions above horizon
        weather_df: Optional weather Pandas dataframe
    
    Returns:
        Dictionary with 'df' containing DataFrame of positions and metadata
    """
    # Parse granularity
    granularity_delta = _parse_iso8601_duration(granularity)
    
    # Generate UTC time range
    times_utc = _date_range_local_to_utc(date, date, tz_name, granularity_delta, only_daylight, lat, lon)
    
    if len(times_utc) == 0:
        return {
            'df': pd.DataFrame(columns=['datetime_local', 'datetime_utc', 'azimuth', 'elevation', 'azimuth_altitude']),
            'azimuth_altitude_pairs': [],
            'metadata': {'date': date, 'lat': lat, 'lon': lon, 'tz_name': tz_name}
        }
    
    # Calculate solar positions
    solar_df = _solar_position_az_el(times_utc, lat, lon)
    
    # Convert back to local time for reference
    tz = pytz.timezone(tz_name)
    times_local = [t.astimezone(tz) for t in times_utc]
    
    # Apply discretization
    if solar_degrees_discretization > 0:
        azimuth = _floor_to_step(solar_df['azimuth'].values, 
                                solar_degrees_discretization, 0, 360)
        elevation = _floor_to_step(solar_df['elevation'].values,
                                  solar_degrees_discretization, -90, 90)
    else:
        azimuth = solar_df['azimuth'].values
        elevation = solar_df['elevation'].values
    
    # Filter daylight hours if requested
    if only_daylight:
        daylight_mask = elevation > 0
        times_utc_filtered = [t for i, t in enumerate(times_utc) if daylight_mask[i]]
        times_local_filtered = [t for i, t in enumerate(times_local) if daylight_mask[i]]
        azimuth = azimuth[daylight_mask]
        elevation = elevation[daylight_mask]
    else:
        times_utc_filtered = list(times_utc)
        times_local_filtered = times_local
    
    # Create combined tuples
    azimuth_altitude = list(zip(azimuth, elevation))
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'datetime_local': times_local_filtered,
        'datetime_utc': times_utc_filtered,
        'azimuth': azimuth,
        'elevation': elevation,
        'azimuth_altitude': azimuth_altitude
    })
    
    # Integrate weather data if provided
    if weather_df is not None:
        try:
            # Convert weather_df to pandas if it's a polars DataFrame
            if hasattr(weather_df, 'to_pandas'):
                weather_pd = weather_df.to_pandas()
            else:
                weather_pd = weather_df
            
            # Ensure time columns are datetime
            result_df['datetime_local'] = pd.to_datetime(result_df['datetime_local'])
            weather_pd['time'] = pd.to_datetime(weather_pd['time'])
            
            # Convert weather UTC time to local timezone for proper joining
            tz = pytz.timezone(tz_name)
            weather_pd['time_local'] = weather_pd['time'].dt.tz_convert(tz)
            
            # Set local time as index for resampling/interpolation
            weather_indexed = weather_pd.set_index('time_local').sort_index()
            
            # Select weather columns to join (exclude time and spatial columns)
            weather_cols = [col for col in weather_indexed.columns 
                          if col not in ['time', 'latitude', 'longitude', 'weatherStation', 'sunPosition', 
                                       'sunAzimuthDiscrete', 'sunElevationDiscrete']]
            weather_subset = weather_indexed[weather_cols]
            
            # Resample weather data to match our granularity and interpolate
            granularity_pandas = granularity.replace('PT', '').replace('M', 'min').replace('H', 'h')
            weather_resampled = weather_subset.resample(granularity_pandas).interpolate(method='linear')
            
            # Create a temporary DataFrame with local times for joining
            temp_df = result_df[['datetime_local']].copy()
            temp_df = temp_df.set_index('datetime_local')
            
            # Left join with weather data using local time
            joined = temp_df.join(weather_resampled, how='left')
            
            # Add weather columns to result DataFrame
            for col in joined.columns:
                result_df[col] = joined[col].values
                
        except Exception as e:
            print(f"Warning: Failed to integrate weather data: {e}")
            # Continue without weather data
    
    return {
        'df': result_df,
        'azimuth_altitude_pairs': azimuth_altitude,
        'metadata': {
            'date': date,
            'lat': lat, 
            'lon': lon,
            'tz_name': tz_name,
            'granularity': granularity,
            'discretization': solar_degrees_discretization
        }
    }


def generate_azimuth_altitude_pairs_by_year(
    year: int = dt.date.today().year,
    *,
    lat: float = 40.4168,
    lon: float = -3.7038,
    tz_name: str = "Europe/Madrid",
    granularity: str = "PT1H",
    solar_degrees_discretization: float = 1.0,
    only_daylight: bool = True,
    weather_df: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate azimuth/altitude pairs for solar positions across a full year.

    Includes optional weather integration (same semantics as daily function).
    """

    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)

    # --------------------------------------------------
    # Parse granularity
    # --------------------------------------------------
    granularity_delta = _parse_iso8601_duration(granularity)

    # --------------------------------------------------
    # Generate UTC timestamps
    # --------------------------------------------------
    times_utc = _date_range_local_to_utc(
        start_date,
        end_date,
        tz_name,
        granularity_delta,
        only_daylight,
        lat,
        lon,
    )

    if len(times_utc) == 0:
        return {
            "df": pd.DataFrame(
                columns=[
                    "datetime_local",
                    "datetime_utc",
                    "azimuth",
                    "elevation",
                    "azimuth_altitude",
                ]
            ),
            "azimuth_altitude_pairs": [],
            "metadata": {
                "year": year,
                "lat": lat,
                "lon": lon,
                "tz_name": tz_name,
            },
        }

    # --------------------------------------------------
    # Solar positions
    # --------------------------------------------------
    solar_df = _solar_position_az_el(times_utc, lat, lon)

    # --------------------------------------------------
    # Local timestamps
    # --------------------------------------------------
    tz = pytz.timezone(tz_name)
    times_local = [t.astimezone(tz) for t in times_utc]

    # --------------------------------------------------
    # Discretization
    # --------------------------------------------------
    if solar_degrees_discretization > 0:
        azimuth = _floor_to_step(
            solar_df["azimuth"].values,
            solar_degrees_discretization,
            0,
            360,
        )
        elevation = _floor_to_step(
            solar_df["elevation"].values,
            solar_degrees_discretization,
            -90,
            90,
        )
    else:
        azimuth = solar_df["azimuth"].values
        elevation = solar_df["elevation"].values

    # --------------------------------------------------
    # Daylight filter
    # --------------------------------------------------
    if only_daylight:
        mask = elevation > 0
        azimuth = azimuth[mask]
        elevation = elevation[mask]
        times_utc = [t for i, t in enumerate(times_utc) if mask[i]]
        times_local = [t for i, t in enumerate(times_local) if mask[i]]

    azimuth_altitude = list(zip(azimuth, elevation))

    # --------------------------------------------------
    # Result DataFrame
    # --------------------------------------------------
    result_df = pd.DataFrame(
        {
            "datetime_local": times_local,
            "datetime_utc": times_utc,
            "azimuth": azimuth,
            "elevation": elevation,
            "azimuth_altitude": azimuth_altitude,
        }
    )

    # ==================================================
    # WEATHER INTEGRATION (IDENTICAL TO DAILY FUNCTION)
    # ==================================================
    if weather_df is not None:
        try:
            if hasattr(weather_df, "to_pandas"):
                weather_pd = weather_df.to_pandas()
            else:
                weather_pd = weather_df

            result_df["datetime_local"] = pd.to_datetime(result_df["datetime_local"])
            weather_pd["time"] = pd.to_datetime(weather_pd["time"])

            tz = pytz.timezone(tz_name)
            weather_pd["time_local"] = weather_pd["time"].dt.tz_convert(tz)

            weather_indexed = weather_pd.set_index("time_local").sort_index()

            weather_cols = [
                c
                for c in weather_indexed.columns
                if c
                not in [
                    "time",
                    "latitude",
                    "longitude",
                    "weatherStation",
                    "sunPosition",
                    "sunAzimuthDiscrete",
                    "sunElevationDiscrete",
                ]
            ]

            weather_subset = weather_indexed[weather_cols]

            granularity_pandas = (
                granularity.replace("PT", "")
                .replace("M", "min")
                .replace("H", "h")
            )

            weather_resampled = weather_subset.resample(
                granularity_pandas
            ).interpolate(method="linear")

            tmp = result_df[["datetime_local"]].set_index("datetime_local")
            joined = tmp.join(weather_resampled, how="left")

            for col in joined.columns:
                result_df[col] = joined[col].values

        except Exception as e:
            print(f"Warning: Failed to integrate weather data: {e}")

    # --------------------------------------------------
    # Unique azimuth-altitude pairs
    # --------------------------------------------------
    unique_pairs = sorted(set(azimuth_altitude))

    return {
        "df": result_df,
        "azimuth_altitude_pairs": unique_pairs,
        "metadata": {
            "year": year,
            "lat": lat,
            "lon": lon,
            "tz_name": tz_name,
            "granularity": granularity,
            "discretization": solar_degrees_discretization,
            "unique_positions": len(unique_pairs),
        },
    }


def delete_matching_files(
    root_dir: str,
    pattern: str,
    *,
    mode: str = "glob",  # "glob" | "regex"
    recursive: bool = True,
    dry_run: bool = True,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Delete files matching a pattern.
    
    Args:
        root_dir: Root directory to search
        pattern: Pattern to match files against
        mode: "glob" for shell-style wildcards, "regex" for regular expressions
        recursive: Search subdirectories recursively
        dry_run: If True, don't actually delete files (just return what would be deleted)
        exclude_patterns: List of patterns to exclude from deletion
    
    Returns:
        Tuple of (deleted_files, error_files)
    """
    
    deleted_files = []
    error_files = []
    exclude_patterns = exclude_patterns or []
    
    if mode == "glob":
        if recursive:
            files_to_check = Path(root_dir).rglob(pattern)
        else:
            files_to_check = Path(root_dir).glob(pattern)
    elif mode == "regex":
        pattern_re = re.compile(pattern)
        files_to_check = []
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if pattern_re.match(file):
                    files_to_check.append(Path(root) / file)
            if not recursive:
                break
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    for file_path in files_to_check:
        try:
            # Check exclude patterns
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if mode == "glob":
                    if fnmatch.fnmatch(file_path.name, exclude_pattern):
                        should_exclude = True
                        break
                elif mode == "regex":
                    if re.match(exclude_pattern, file_path.name):
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
            
            if not dry_run:
                file_path.unlink()
            deleted_files.append(str(file_path))
            
        except Exception as e:
            error_files.append(f"{file_path}: {e}")
    
    return deleted_files, error_files


def apply_height_shadow_to_hillshades(
    hillshade_dir: str,
    dem_path: str,
    dsm_path: str,
    out_dir: str,
    height_m: float = 0.5,
    *,
    pattern: str = "*.tif",
    recursive: bool = True,
    overwrite: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
    aoi=None,
    aoi_crs: str = "EPSG:4326",
    nodata_value: float = -9999.0,
) -> list[str]:
    """
    Apply height-based shadowing using AOI as spatial authority.
    """

    hillshade_dir = Path(hillshade_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(hillshade_dir.rglob(pattern) if recursive else hillshade_dir.glob(pattern))
    files = [f for f in files if f.is_file()]

    if not files:
        return []

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype(np.float32)

    with rasterio.open(dsm_path) as dsm_src:
        dsm = dsm_src.read(1).astype(np.float32)
        raster_crs = dsm_src.crs

    if aoi is not None:
        transformer = pyproj.Transformer.from_crs(
            aoi_crs,
            raster_crs,
            always_xy=True,
        ).transform
        aoi_mask = rasterize_aoi_mask(transform(transformer, aoi), dsm_path)
    else:
        aoi_mask = np.ones_like(dsm, bool)

    obstacle_mask = aoi_mask & ((dsm - dem) > height_m)

    def process(path: Path):
        try:
            out_path = out_dir / path.relative_to(hillshade_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if os.path.exists(out_path) and not overwrite:
                return str(out_path)

            with rasterio.open(path) as src:
                hs = src.read(1).astype(np.float32)
                profile = src.profile.copy()

            if hs.shape != obstacle_mask.shape:
                warnings.warn(f"Shape mismatch: {path}")
                return None

            hs[obstacle_mask] = 0.0
            hs[~aoi_mask] = nodata_value
            np.clip(hs, 0.0, 1.0, out=hs, where=aoi_mask)

            profile.update(
                dtype="float32",
                nodata=nodata_value,
                compress="deflate",
                predictor=3,
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(hs, 1)

            return str(out_path)

        except Exception as e:
            warnings.warn(str(e))
            return None

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            res = list(tqdm(ex.map(process, files), total=len(files)))
    else:
        res = [process(f) for f in tqdm(files)]

    return [r for r in res if r]


def discretize_solar_position(
    azimuth_deg: float,
    altitude_deg: float,
    *,
    azimuth_step: int = 3,
    altitude_step: int = 3,
) -> tuple[int, int]:
    az_bin = int(round(azimuth_deg / azimuth_step) * azimuth_step) % 360
    alt_bin = int(round(altitude_deg / altitude_step) * altitude_step)
    return az_bin, alt_bin


def load_hillshade(
    hillshade_dir: Path,
    azimuth: float,
    altitude: float,
) -> np.ndarray:
    """
    Load hillshade generated by generate_multiple_hillshades_gdal.

    Output:
      float32 array with values clipped to [0, 1]
    """

    az_i = int(azimuth)
    alt_i = int(altitude)
    prefix = f"{az_i:03d}_{alt_i:02d}"

    candidates = sorted(
        p for p in hillshade_dir.glob(f"{prefix}*.tif")
        if not p.name.endswith("_raw.tif")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No valid hillshade for az={az_i}, alt={alt_i} in {hillshade_dir}"
        )

    fn = candidates[0]

    if fn.stat().st_size == 0:
        raise ValueError(f"Corrupt hillshade (0 bytes): {fn}")

    with rasterio.open(fn) as ds:
        hs = ds.read(1).astype(np.float32)
        nodata = ds.nodata

    if nodata is not None:
        hs = np.where(hs == nodata, 0.0, hs)

    return hs

def compute_surface_solar_energy(
    solar_df,
    *,
    hillshade_dir,
    svf_path,
    out_path,
    mode: str = "total",   # "total" | "monthly" | "fourier_decomposition"
    aoi=None,
    aoi_crs: str = "EPSG:4326",
    nodata_value: float = -9999.0,
    overwrite: bool = False,
) -> None:
    """
    Compute surface solar energy with different aggregation strategies.

    Modes
    -----
    total:
        Bands:
          1 : Direct annual energy (Wh)
          2 : Diffuse annual energy (Wh)

    monthly:
        Bands:
          1–12  : Direct monthly (Jan–Dec)
          13–24 : Diffuse monthly (Jan–Dec)
          25    : Direct annual
          26    : Diffuse annual

    fourier_decomposition:
        Bands:
          1 : A0   (direct isotropic)
          2 : A1c  (direct cos(az))
          3 : A1s  (direct sin(az))
          4 : A2   (direct sin(alt))
          5 : D0   (diffuse isotropic)
          6 : D1   (diffuse sin(alt))

        These coefficients allow exact POA reconstruction.
    """

    import numpy as np
    import rasterio
    import pyproj
    from pathlib import Path
    from tqdm import tqdm
    from shapely.ops import transform as shp_transform

    if mode not in {"total", "monthly", "fourier_decomposition"}:
        raise ValueError("mode must be 'total', 'monthly', or 'fourier_decomposition'")
    out_path = out_path.replace(".tif", f"_{mode}.tif")

    if os.path.exists(out_path) and not overwrite:
        return out_path

    hillshade_dir = Path(hillshade_dir)
    svf_path = Path(svf_path)
    out_path = Path(out_path)

    # ==========================================================
    # SOLAR DATA PREPARATION
    # ==========================================================
    df = solar_df.copy()

    df["az_bin"] = df.azimuth.astype(int)
    df["alt_bin"] = df.elevation.astype(int)

    df["dt_hours"] = (
        df.datetime_utc.diff()
        .dt.total_seconds()
        .shift(-1)
        .fillna(0.0) / 3600.0
    )

    df = df[df.elevation > 0]

    if mode == "monthly":
        df["month"] = df.datetime_utc.dt.month
        month_groups = dict(tuple(df.groupby("month")))
    else:
        month_groups = {1: df}

    # ==========================================================
    # GRID & SVF
    # ==========================================================
    with rasterio.open(svf_path) as src:
        profile = src.profile.copy()
        H, W = src.height, src.width
        transform = src.transform
        raster_crs = src.crs
        svf = src.read(1).astype(np.float32)

    pixel_area = abs(transform.a * transform.e)

    # ==========================================================
    # AOI MASK
    # ==========================================================
    if aoi is not None:
        transformer = pyproj.Transformer.from_crs(
            aoi_crs, raster_crs, always_xy=True
        ).transform
        aoi_proj = shp_transform(transformer, aoi)
        aoi_mask = rasterize_aoi_mask(aoi_proj, svf_path)
    else:
        aoi_mask = np.ones((H, W), dtype=bool)

    # ==========================================================
    # OUTPUT PROFILE
    # ==========================================================
    if mode == "total":
        n_bands = 2
    elif mode == "monthly":
        n_bands = 26
    else:  # fourier
        n_bands = 6

    profile.update(
        dtype="float32",
        nodata=nodata_value,
        count=n_bands,
        compress="zstd",
        zstd_level=9,
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile):
        pass

    # ==========================================================
    # ACCUMULATORS
    # ==========================================================
    if mode == "fourier_decomposition":
        A0  = np.zeros((H, W), np.float32)
        A1c = np.zeros((H, W), np.float32)
        A1s = np.zeros((H, W), np.float32)
        A2  = np.zeros((H, W), np.float32)
        D0  = np.zeros((H, W), np.float32)
        D1  = np.zeros((H, W), np.float32)
    else:
        direct_annual  = np.zeros((H, W), np.float32)
        diffuse_annual = np.zeros((H, W), np.float32)

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    for month_key, df_month in month_groups.items():
        if df_month.empty:
            continue

        if mode == "monthly":
            direct_month  = np.zeros((H, W), np.float32)
            diffuse_month = np.zeros((H, W), np.float32)

        hill_groups = df_month.groupby(["az_bin", "alt_bin"])

        for (az, alt), g in tqdm(
            hill_groups,
            desc=f"Solar energy ({mode})",
            unit="group",
            dynamic_ncols=True,
        ):
            dni_E = np.sum(g.DNI.values * g.dt_hours.values).astype(np.float32)
            dhi_E = np.sum(g.DHI.values * g.dt_hours.values).astype(np.float32)

            if dni_E == 0 and dhi_E == 0:
                continue

            hill = load_hillshade(
                hillshade_dir,
                azimuth=az,
                altitude=alt,
            ).astype(np.float32)

            if mode == "fourier_decomposition":
                θ = np.deg2rad(az)
                φ = np.deg2rad(alt)

                _accumulate_fourier(
                    A0, A1c, A1s, A2,
                    D0, D1,
                    hill, svf,
                    dni_E, dhi_E,
                    θ, φ,
                    aoi_mask,
                    pixel_area,
                )
            else:
                _accumulate_energy_split(
                    direct_month if mode == "monthly" else direct_annual,
                    diffuse_month if mode == "monthly" else diffuse_annual,
                    svf,
                    hill,
                    dni_E,
                    dhi_E,
                    aoi_mask,
                    pixel_area,
                )

        if mode == "monthly":
            direct_annual  += direct_month
            diffuse_annual += diffuse_month

            with rasterio.open(out_path, "r+") as dst:
                direct_month[~aoi_mask]  = nodata_value
                diffuse_month[~aoi_mask] = nodata_value
                dst.write(direct_month,  month_key)
                dst.write(diffuse_month, month_key + 12)

    # ==========================================================
    # FINAL WRITE
    # ==========================================================
    with rasterio.open(out_path, "r+") as dst:
        if mode == "fourier_decomposition":
            for arr, band in zip(
                (A0, A1c, A1s, A2, D0, D1),
                range(1, 7),
            ):
                arr[~aoi_mask] = nodata_value
                dst.write(arr, band)

            dst.set_band_description(1, "A0 direct")
            dst.set_band_description(2, "A1c cos(az)")
            dst.set_band_description(3, "A1s sin(az)")
            dst.set_band_description(4, "A2 sin(alt)")
            dst.set_band_description(5, "D0 diffuse")
            dst.set_band_description(6, "D1 sin(alt)")

        else:
            direct_annual[~aoi_mask]  = nodata_value
            diffuse_annual[~aoi_mask] = nodata_value

            if mode == "monthly":
                dst.write(direct_annual, 25)
                dst.write(diffuse_annual, 26)
                dst.set_band_description(1, "Direct - January")
                dst.set_band_description(2, "Direct - February")
                dst.set_band_description(3, "Direct - March")
                dst.set_band_description(4, "Direct - April")
                dst.set_band_description(5, "Direct - May")
                dst.set_band_description(6, "Direct - June")
                dst.set_band_description(7, "Direct - July")
                dst.set_band_description(8, "Direct - August")
                dst.set_band_description(9, "Direct - September")
                dst.set_band_description(10, "Direct - October")
                dst.set_band_description(11, "Direct - November")
                dst.set_band_description(12, "Direct - December")
                dst.set_band_description(13, "Diffuse - January")
                dst.set_band_description(14, "Diffuse - February")
                dst.set_band_description(15, "Diffuse - March")
                dst.set_band_description(16, "Diffuse - April")
                dst.set_band_description(17, "Diffuse - May")
                dst.set_band_description(18, "Diffuse - June")
                dst.set_band_description(19, "Diffuse - July")
                dst.set_band_description(20, "Diffuse - August")
                dst.set_band_description(21, "Diffuse - September")
                dst.set_band_description(22, "Diffuse - October")
                dst.set_band_description(23, "Diffuse - November")
                dst.set_band_description(24, "Diffuse - December")
                dst.set_band_description(25, "Direct - Annual")
                dst.set_band_description(26, "Diffuse - Annual")
            else:
                dst.write(direct_annual, 1)
                dst.write(diffuse_annual, 2)
                dst.set_band_description(1, "Direct")
                dst.set_band_description(2, "Diffuse")

@njit(parallel=True, fastmath=True)
def _accumulate_energy_split(
    E_dir, E_dif,
    svf, hill,
    dni_E, dhi_E,
    valid_mask,
    pixel_area,
):
    H, W = E_dir.shape
    for i in prange(H):
        for j in range(W):
            if not valid_mask[i, j]:
                continue
            E_dir[i, j] += dni_E * hill[i, j] * pixel_area
            E_dif[i, j] += dhi_E * svf[i, j] * pixel_area


@njit(parallel=True, fastmath=True)
def _accumulate_fourier(
    A0, A1c, A1s, A2,
    D0, D1,
    hill, svf,
    dni_E, dhi_E,
    theta, phi,
    valid_mask,
    pixel_area,
):
    cθ = np.cos(theta)
    sθ = np.sin(theta)
    sφ = np.sin(phi)

    H, W = A0.shape
    for i in prange(H):
        for j in range(W):
            if not valid_mask[i, j]:
                continue

            E_dir = dni_E * hill[i, j] * pixel_area
            E_dif = dhi_E * svf[i, j] * pixel_area

            A0[i, j]  += E_dir
            A1c[i, j] += E_dir * cθ
            A1s[i, j] += E_dir * sθ
            A2[i, j]  += E_dir * sφ

            D0[i, j]  += E_dif
            D1[i, j]  += E_dif * sφ

def calculate_sky_view_factor(
    dsm_path: str,
    out_path: str,
    search_radius: float = 100.0,
    angular_resolution: float = 5.0,
    nodata_value: float = -9999.0,
    *,
    cuda: bool | None = None,
    hybrid: bool = False,
    subsample_factor: int = 2,
    building_buffer_m: float = 40.0,
    height_threshold: float = 3.0,
    class_path: str | None = None,
    building_classes: Sequence[int] | None = None,
    vegetation_classes: Sequence[int] | None = None,
) -> str:

    if cuda is None:
        cuda = has_cuda()

    if cuda and hybrid:
        return _calculate_svf_cuda_hybrid(
            dsm_path=dsm_path,
            out_path=out_path,
            search_radius=search_radius,
            angular_resolution=angular_resolution,
            subsample_factor=subsample_factor,
            building_buffer_m=building_buffer_m,
            height_threshold=height_threshold,
            class_path=class_path,
            building_classes=building_classes,
            vegetation_classes=vegetation_classes,
            nodata_value=nodata_value,
        )

    if cuda:
        return _calculate_svf_cuda(
            dsm_path=dsm_path,
            out_path=out_path,
            search_radius=search_radius,
            angular_resolution=angular_resolution,
            nodata_value=nodata_value,
        )

    return _calculate_svf_cpu(
        dsm_path=dsm_path,
        out_path=out_path,
        search_radius=search_radius,
        angular_resolution=angular_resolution,
        nodata_value=nodata_value,
    )


def _calculate_svf_cpu(
    dsm_path: str,
    out_path: str,
    search_radius: float,
    angular_resolution: float,
    max_height_angle: float,
    nodata_value: float,
    creation_options: Optional[Dict[str, str]],
    row_block_size: int,
) -> str:
    """
    Existing CPU/Numba SVF (row-based kernel).
    """

    info("Calculating Sky View Factor (CPU / Numba)")

    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        pixel_size = abs(src.transform[0])

    height, width = dsm.shape

    if nodata is not None:
        valid_mask = (dsm != nodata) & np.isfinite(dsm)
    else:
        valid_mask = np.isfinite(dsm)

    svf = np.full(dsm.shape, nodata_value, dtype=np.float32)

    azimuth_angles = np.arange(0.0, 360.0, angular_resolution, dtype=np.float32)
    azimuth_rad = np.deg2rad(azimuth_angles)

    search_radius_px = int(search_radius / pixel_size)
    step_px = max(1, int(search_radius_px * 0.02))
    max_height_angle_rad = np.deg2rad(max_height_angle)

    n_blocks = (height + row_block_size - 1) // row_block_size

    with tqdm(total=n_blocks, desc="SVF CPU", unit="block") as pbar:
        for i in range(n_blocks):
            r0 = i * row_block_size
            r1 = min(r0 + row_block_size, height)

            _svf_kernel_rows(
                dsm,
                valid_mask,
                svf,
                azimuth_rad,
                pixel_size,
                search_radius_px,
                step_px,
                max_height_angle_rad,
                r0,
                r1,
            )
            pbar.update(1)

    profile.update(dtype="float32", count=1, nodata=nodata_value)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(svf, 1)

    return out_path


def build_svf_importance_mask(
    class_raster: np.ndarray,
    *,
    building_classes: Sequence[int],
    vegetation_classes: Sequence[int],
    buffer_px: int = 5,
) -> np.ndarray:
    """
    True where full-resolution SVF is required.
    """
    from scipy.ndimage import binary_dilation

    importance = np.isin(class_raster, list(building_classes) + list(vegetation_classes))

    if buffer_px > 0:
        importance = binary_dilation(importance, iterations=buffer_px)

    return importance

def downsample_dsm(dsm: np.ndarray, factor: int) -> np.ndarray:
    h, w = dsm.shape
    h2 = h // factor
    w2 = w // factor

    dsm_crop = dsm[:h2 * factor, :w2 * factor]
    return dsm_crop.reshape(
        h2, factor, w2, factor
    ).mean(axis=(1, 3))


def upsample_svf(
    svf_coarse: np.ndarray,
    factor: int,
    out_shape: tuple[int, int],
) -> np.ndarray:
    svf_up = zoom(svf_coarse, zoom=factor, order=1)
    return svf_up[:out_shape[0], :out_shape[1]]


def _fit_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Crop or pad array to match target shape.
    Pads with edge values if needed.
    """
    h, w = shape
    ah, aw = arr.shape

    # Crop if larger
    arr = arr[:h, :w]

    # Pad if smaller
    if arr.shape[0] < h or arr.shape[1] < w:
        pad_h = h - arr.shape[0]
        pad_w = w - arr.shape[1]

        arr = np.pad(
            arr,
            ((0, pad_h), (0, pad_w)),
            mode="edge",
        )

    return arr


@cuda.jit
def normalize_svf_kernel(svf, valid_mask, n_az):
    i, j = cuda.grid(2)
    rows, cols = svf.shape

    if i >= rows or j >= cols:
        return

    if not valid_mask[i, j]:
        return

    svf[i, j] = svf[i, j] / n_az

@cuda.jit
def svf_horizon_kernel(
    dsm,
    valid_mask,
    svf_acc,
    dx,
    dy,
    pixel_size,
    max_steps,
):
    i, j = cuda.grid(2)
    rows, cols = dsm.shape

    if i >= rows or j >= cols:
        return
    if not valid_mask[i, j]:
        return

    z0 = dsm[i, j]
    max_angle = 0.0

    for step in range(1, max_steps + 1):
        ii = int(i + step * dy)
        jj = int(j + step * dx)

        if ii < 0 or ii >= rows or jj < 0 or jj >= cols:
            break

        dz = dsm[ii, jj] - z0
        ang = math.atan2(dz, step * pixel_size)

        if ang > max_angle:
            max_angle = ang

    svf_acc[i, j] += math.cos(max_angle) ** 2


def _calculate_svf_cuda_hybrid(
    dsm_path: str,
    out_path: str,
    *,
    search_radius: float,
    angular_resolution: float,
    subsample_factor: int,
    building_buffer_m: float,
    height_threshold: float,
    class_path: str | None,
    building_classes: Sequence[int] | None,
    vegetation_classes: Sequence[int] | None,
    nodata_value: float,
) -> str:

    info("Calculating Sky View Factor (CUDA, hybrid – TRUE)")

    # --- Read DSM ---
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        pixel_size = abs(src.transform[0])

    H, W = dsm.shape

    # --- Valid mask ---
    valid = np.isfinite(dsm)
    if nodata is not None:
        valid &= (dsm != nodata)

    # --- Buffer in pixels ---
    building_buffer_px = max(1, int(building_buffer_m / pixel_size))

    # ==========================================================
    # IMPORTANCE MASK
    # ==========================================================
    if class_path is not None:
        info("Using class raster for SVF importance mask")

        with rasterio.open(class_path) as src:
            class_raster = src.read(1)

        if building_classes is None:
            raise ValueError("building_classes must be provided with class_path")

        if vegetation_classes is None:
            vegetation_classes = set()

        importance_mask = build_svf_importance_mask(
            class_raster,
            building_classes=building_classes,
            vegetation_classes=vegetation_classes,
            buffer_px=building_buffer_px,
        )

        importance_mask &= valid

    else:
        info("Using height-based SVF importance mask")

        ground = np.percentile(dsm[valid], 10)
        importance_mask = valid & ((dsm - ground) > height_threshold)
        importance_mask = binary_dilation(
            importance_mask,
            iterations=building_buffer_px,
        )

    frac = 100.0 * importance_mask.mean()
    info(f"Hybrid SVF: {frac:.1f}% full-resolution pixels")

    # ==========================================================
    # COARSE SVF (FULL AREA, CHEAP)
    # ==========================================================
    factor = subsample_factor
    dsm_coarse = downsample_dsm(dsm, factor)

    coarse_dsm_path = out_path + ".coarse_dsm.tif"
    coarse_svf_path = out_path + ".coarse.tif"

    coarse_profile = profile.copy()
    coarse_profile.update(
        height=dsm_coarse.shape[0],
        width=dsm_coarse.shape[1],
        transform=profile["transform"] * rasterio.Affine.scale(factor, factor),
    )

    with rasterio.open(coarse_dsm_path, "w", **coarse_profile) as dst:
        dst.write(dsm_coarse, 1)

    _calculate_svf_cuda(
        dsm_path=coarse_dsm_path,
        out_path=coarse_svf_path,
        search_radius=search_radius,
        angular_resolution=angular_resolution,
        nodata_value=nodata_value,
    )

    with rasterio.open(coarse_svf_path) as ds:
        svf_coarse = ds.read(1)

    os.remove(coarse_dsm_path)
    os.remove(coarse_svf_path)

    svf_coarse_up = zoom(svf_coarse, zoom=factor, order=1)
    svf_coarse_up = _fit_to_shape(svf_coarse_up, (H, W))

    # ==========================================================
    # FULL SVF ONLY WHERE NEEDED (MASKED CUDA)
    # ==========================================================
    masked_valid = valid & importance_mask

    azimuths = np.deg2rad(
        np.arange(0.0, 360.0, angular_resolution, dtype=np.float32)
    )
    n_az = len(azimuths)

    max_steps = int(search_radius / pixel_size)

    d_dsm = cuda.to_device(dsm)
    d_valid = cuda.to_device(masked_valid)
    d_svf = cuda.device_array_like(dsm)
    d_svf[:] = 0.0

    threads = (16, 16)
    blocks = (
        (H + threads[0] - 1) // threads[0],
        (W + threads[1] - 1) // threads[1],
    )

    with tqdm(
        azimuths,
        desc="SVF CUDA azimuths (masked)",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as pbar:
        for az in pbar:
            dx = math.cos(az)
            dy = -math.sin(az)

            svf_horizon_kernel[blocks, threads](
                d_dsm,
                d_valid,
                d_svf,
                dx,
                dy,
                pixel_size,
                max_steps,
            )
            cuda.synchronize()

    normalize_svf_kernel[blocks, threads](
        d_svf,
        d_valid,
        n_az,
    )
    cuda.synchronize()

    svf_full = d_svf.copy_to_host()

    # ==========================================================
    # MERGE
    # ==========================================================
    svf = np.where(
        importance_mask,
        svf_full,
        svf_coarse_up,
    )

    svf = np.where(valid, np.clip(svf, 0.0, 1.0), nodata_value)

    # --- Write output ---
    profile.update(dtype="float32", nodata=nodata_value)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(svf, 1)
        dst.update_tags(
            method="cuda_hybrid_true",
            subsample_factor=subsample_factor,
            building_buffer_m=building_buffer_m,
        )

    info("Hybrid SVF computation completed successfully")
    info(f"Output written to {out_path}")

    return out_path


def _calculate_svf_cuda(
    dsm_path: str,
    out_path: str,
    search_radius: float,
    angular_resolution: float,
    nodata_value: float,
) -> str:
    """
    CUDA-based physically correct 2D angular Sky View Factor (SVF).

    - One CUDA kernel per azimuth
    - Progress logging visible (tqdm + cuda.synchronize)
    - Correct normalization on GPU
    """

    info("Calculating Sky View Factor (CUDA, 2D angular)")

    # --- Read DSM ---
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        pixel_size = abs(src.transform[0])

    rows, cols = dsm.shape
    info(f"DSM size: {cols} × {rows}, pixel size: {pixel_size:.2f} m")

    # --- Valid mask ---
    if nodata is not None:
        valid_mask = (dsm != nodata) & np.isfinite(dsm)
    else:
        valid_mask = np.isfinite(dsm)

    # --- Geometry ---
    azimuths = np.deg2rad(
        np.arange(0.0, 360.0, angular_resolution, dtype=np.float32)
    )
    n_az = len(azimuths)

    max_steps = int(search_radius / pixel_size)
    if max_steps < 2:
        raise ValueError("search_radius too small for DSM resolution")

    info(f"Azimuth directions: {n_az}")
    info(f"Ray steps: {max_steps}")

    # --- GPU memory ---
    d_dsm = cuda.to_device(dsm)
    d_valid = cuda.to_device(valid_mask)
    d_svf = cuda.device_array_like(dsm)
    d_svf[:] = 0.0

    threads = (16, 16)
    blocks = (
        (rows + threads[0] - 1) // threads[0],
        (cols + threads[1] - 1) // threads[1],
    )

    # --- CUDA azimuth loop with visible progress ---
    with tqdm(
        azimuths,
        desc="SVF CUDA azimuths",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as pbar:
        for az in pbar:
            dx = math.cos(az)
            dy = -math.sin(az)

            svf_horizon_kernel[blocks, threads](
                d_dsm,
                d_valid,
                d_svf,
                dx,
                dy,
                pixel_size,
                max_steps,
            )

            # 🔴 REQUIRED: force sync so tqdm updates
            cuda.synchronize()

    # --- Normalize on GPU ---
    normalize_svf_kernel[blocks, threads](
        d_svf,
        d_valid,
        n_az,
    )
    cuda.synchronize()

    # --- Copy back ---
    svf = d_svf.copy_to_host()
    svf = np.where(valid_mask, np.clip(svf, 0.0, 1.0), nodata_value)

    # --- Write output ---
    profile.update(
        dtype="float32",
        count=1,
        nodata=nodata_value,
        driver="GTiff",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(svf, 1)
        dst.update_tags(
            units="dimensionless",
            description="Sky View Factor",
            method="cuda_2d_angular_horizon",
            angular_resolution=f"{angular_resolution}deg",
            search_radius=f"{search_radius}m",
        )

    info("SVF CUDA computation completed successfully")
    info(f"Output written to {out_path}")

    return out_path


def pad_to_macroblock(img, block=16):
    h, w = img.shape[:2]
    pad_h = (-h) % block
    pad_w = (-w) % block

    if pad_h == 0 and pad_w == 0:
        return img

    return np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="edge",
    )


def generate_daily_shadow_mp4(
    solar_df,
    *,
    hillshade_dir,
    out_mp4,
    fps: int = 24,
    duration_seconds: int = 20,
    zoom: float = 4.0,
    center_xy: tuple[int, int] | None = None,
    frame_size: tuple[int, int] = (1920, 1088),
    overwrite: bool = False,
    timestamp_fmt: str = "%Y-%m-%d %H:%M UTC",
    timestamp_position: tuple[int, int] = (30, 60),
    timestamp_scale: float = 1.0,
    timestamp_thickness: int = 2,
    timestamp_padding: int = 10,
    timestamp_box_alpha: float = 0.45,
    solar_line_scale: float | None = None,
):
    """
    Generate an MP4 animation of daily terrain shadows with:
      - bounded LRU hillshade cache
      - OpenCV resize
      - timestamp overlay
      - solar altitude / azimuth overlay (1 decimal)
    """

    from functools import lru_cache
    import gc

    hillshade_dir = Path(hillshade_dir)
    out_mp4 = Path(out_mp4)

    if out_mp4.exists() and not overwrite:
        return

    # ----------------------------------------------------------
    # Prepare solar data
    # ----------------------------------------------------------
    df = solar_df.copy()
    df = df[df.elevation > 0]
    if df.empty:
        raise ValueError("No sun-above-horizon data available")

    df["az_bin"] = df.azimuth.astype(int)
    df["alt_bin"] = df.elevation.astype(int)
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    times = df.datetime_utc.astype("int64").to_numpy() / 1e9
    az = df.azimuth.to_numpy(dtype=np.float32)
    alt = df.elevation.to_numpy(dtype=np.float32)
    az_bin = df.az_bin.to_numpy(dtype=np.int32)
    alt_bin = df.alt_bin.to_numpy(dtype=np.int32)

    # ----------------------------------------------------------
    # LRU Hillshade Cache (bounded)
    # ----------------------------------------------------------
    @lru_cache(maxsize=8)
    def get_hillshade_cached(a: int, h: int) -> np.ndarray:
        hs = load_hillshade(
            hillshade_dir,
            azimuth=a,
            altitude=h,
        )
        return hs.astype(np.float32)

    # ----------------------------------------------------------
    # Geometry reference
    # ----------------------------------------------------------
    ref = get_hillshade_cached(int(az_bin[0]), int(alt_bin[0]))
    H, W = ref.shape

    if center_xy is None:
        cx, cy = W // 2, H // 2
    else:
        cx, cy = center_xy

    # ----------------------------------------------------------
    # Crop window (aspect-ratio safe)
    # ----------------------------------------------------------
    target_ar = frame_size[0] / frame_size[1]

    base_w = int(W / zoom)
    base_h = int(H / zoom)

    if base_w / base_h > target_ar:
        win_h = base_h
        win_w = int(base_h * target_ar)
    else:
        win_w = base_w
        win_h = int(base_w / target_ar)

    win_w = min(win_w, W)
    win_h = min(win_h, H)

    x0 = max(0, cx - win_w // 2)
    y0 = max(0, cy - win_h // 2)
    x1 = min(W, x0 + win_w)
    y1 = min(H, y0 + win_h)

    if x1 - x0 < win_w:
        x0 = max(0, x1 - win_w)
    if y1 - y0 < win_h:
        y0 = max(0, y1 - win_h)

    # ----------------------------------------------------------
    # Temporal sampling
    # ----------------------------------------------------------
    total_frames = fps * duration_seconds
    t_video = np.linspace(times.min(), times.max(), total_frames)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if solar_line_scale is None:
        solar_line_scale = timestamp_scale * 0.85

    # ----------------------------------------------------------
    # Video writer
    # ----------------------------------------------------------
    with imageio.get_writer(
        out_mp4,
        fps=fps,
        codec="libx264",
        macro_block_size=16,
    ) as writer:

        idx = 0

        for t in tqdm(t_video, desc="Generating shadow MP4"):

            while idx < len(times) - 2 and times[idx + 1] < t:
                idx += 1

            t0, t1 = times[idx], times[idx + 1]
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)

            # Interpolate solar geometry
            alt_i = (1 - w) * alt[idx] + w * alt[idx + 1]
            az_i = (1 - w) * az[idx] + w * az[idx + 1]

            # Load only 2 hillshades (LRU keeps max 8)
            hs0 = get_hillshade_cached(int(az_bin[idx]), int(alt_bin[idx]))[y0:y1, x0:x1]
            hs1 = get_hillshade_cached(int(az_bin[idx + 1]), int(alt_bin[idx + 1]))[y0:y1, x0:x1]

            # In-place blend (no extra float64 temp)
            frame = hs0.copy()
            frame *= (1.0 - w)
            frame += hs1 * w

            np.clip(frame, 0.0, 1.0, out=frame)
            frame = (frame * 255.0).astype(np.uint8)

            # Resize + grayscale → RGB
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            # --------------------------------------------------
            # Overlay text
            # --------------------------------------------------
            ts = np.datetime64(int(t), "s").astype("datetime64[s]").astype(object)
            line1 = ts.strftime(timestamp_fmt)
            line2 = f"Alt: {alt_i:.1f} deg   Az: {az_i:.1f} deg"

            (w1, h1), b1 = cv2.getTextSize(
                line1, font, timestamp_scale, timestamp_thickness
            )
            (w2, h2), b2 = cv2.getTextSize(
                line2, font, solar_line_scale, timestamp_thickness
            )

            x, y = timestamp_position
            line_gap = 6

            box_x0 = x - timestamp_padding
            box_y0 = y - h1 - timestamp_padding
            box_x1 = x + max(w1, w2) + timestamp_padding
            box_y1 = (
                y
                + b1
                + h2
                + line_gap
                + b2
                + timestamp_padding
            )

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (box_x0, box_y0),
                (box_x1, box_y1),
                (0, 0, 0),
                -1,
            )

            frame = cv2.addWeighted(
                overlay, timestamp_box_alpha, frame, 1 - timestamp_box_alpha, 0
            )

            cv2.putText(
                frame,
                line1,
                (x + 2, y + 2),
                font,
                timestamp_scale,
                (0, 0, 0),
                timestamp_thickness + 1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                line2,
                (x + 2, y + h2 + line_gap + 2),
                font,
                solar_line_scale,
                (0, 0, 0),
                timestamp_thickness + 1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                line1,
                (x, y),
                font,
                timestamp_scale,
                (255, 255, 255),
                timestamp_thickness,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                line2,
                (x, y + h2 + line_gap),
                font,
                solar_line_scale,
                (255, 255, 255),
                timestamp_thickness,
                cv2.LINE_AA,
            )

            writer.append_data(frame)

            # Explicit cleanup (extra safety)
            del hs0, hs1, frame
            gc.collect()

    print(f"Shadow MP4 written to: {out_mp4}")


def resize_with_aspect_and_padding(
    img: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    pad_value: int = 0,
) -> np.ndarray:
    """
    Resize img preserving aspect ratio and pad to (out_h, out_w).
    img: (H, W) uint8
    """
    h, w = img.shape
    src_ratio = w / h
    dst_ratio = out_w / out_h

    if src_ratio > dst_ratio:
        # limited by width
        new_w = out_w
        new_h = int(round(out_w / src_ratio))
    else:
        # limited by height
        new_h = out_h
        new_w = int(round(out_h * src_ratio))

    resized = resize(
        img,
        (new_h, new_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(img.dtype)

    canvas = np.full((out_h, out_w), pad_value, dtype=img.dtype)

    y0 = (out_h - new_h) // 2
    x0 = (out_w - new_w) // 2

    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def pv_potential_from_fourier_raster(
    fourier_raster_path,
    *,
    out_raster_path,
    polygons_gdf,
    tilt,
    azimuth,
    latitude_deg,
    technology,
    coverage_ratio,
    albedo=0.2,
    nodata=-9999.0,
):
    import rasterio
    import numpy as np

    # --- Technology parameters ---
    TECH = {
        "mono_Si": dict(eta=0.20, k_dni=0.95, k_dhi=0.70, PR=0.80),
        "poly_Si": dict(eta=0.18, k_dni=0.93, k_dhi=0.68, PR=0.78),
        "CdTe":    dict(eta=0.17, k_dni=0.90, k_dhi=0.75, PR=0.82),
    }
    tech = TECH[technology]

    phi_bar = np.deg2rad(latitude_deg)

    with rasterio.open(fourier_raster_path) as src:
        profile = src.profile.copy()
        A0  = src.read(1)
        A1c = src.read(2)
        A1s = src.read(3)
        A2  = src.read(4)
        D0  = src.read(5)
        D1  = src.read(6)

    H, W = A0.shape

    # --- Polygon mask ---
    poly_mask = rasterize_polygon_mask(polygons_gdf, fourier_raster_path)

    # --- Tilt ---
    if np.isscalar(tilt):
        tilt_arr = np.full((H, W), np.deg2rad(tilt), dtype=np.float32)
    else:
        with rasterio.open(tilt) as src:
            tilt_arr = np.deg2rad(src.read(1))

    # --- Azimuth ---
    if np.isscalar(azimuth):
        az_arr = np.full((H, W), np.deg2rad(azimuth), dtype=np.float32)
    else:
        with rasterio.open(azimuth) as src:
            az_arr = np.deg2rad(src.read(1))

    out = np.full((H, W), nodata, dtype=np.float32)

    pv_from_fourier_kernel_masked(
        A0, A1c, A1s, A2,
        D0, D1,
        tilt_arr, az_arr,
        phi_bar,
        tech["k_dni"], tech["k_dhi"],
        tech["eta"], tech["PR"],
        coverage_ratio,
        albedo,
        poly_mask,
        nodata,
        out,
    )

    profile.update(count=1, nodata=nodata, dtype="float32")

    with rasterio.open(out_raster_path, "w", **profile) as dst:
        dst.write(out, 1)

    return out_raster_path

@njit(parallel=True, fastmath=True)
def pv_from_fourier_kernel_masked(
    A0, A1c, A1s, A2,
    D0, D1,
    tilt, azimuth,
    phi_bar,
    k_dni, k_dhi,
    eta_pv, PR,
    coverage,
    albedo,
    mask,
    nodata,
    out,
):
    H, W = A0.shape
    sin_phi = np.sin(phi_bar)
    cos_phi = np.cos(phi_bar)

    for i in prange(H):
        for j in range(W):
            if not mask[i, j]:
                out[i, j] = nodata
                continue

            beta = tilt[i, j]
            gamma = azimuth[i, j]

            cosb = np.cos(beta)
            sinb = np.sin(beta)

            # --- Direct ---
            E_dir = k_dni * (
                A0[i, j]
                + A1c[i, j] * np.cos(gamma)
                + A1s[i, j] * np.sin(gamma)
            ) * (sin_phi * cosb + cos_phi * sinb)

            # --- Diffuse ---
            E_dif = k_dhi * (
                D0[i, j] * (1.0 + cosb) * 0.5
                + D1[i, j] * sinb
            )

            # --- Ground reflected ---
            E_ref = (A0[i, j] + D0[i, j]) * albedo * (1.0 - cosb) * 0.5

            out[i, j] = (
                (E_dir + E_dif + E_ref)
                * eta_pv * PR * coverage
            )


@njit(parallel=True)
def zonal_sum(values, zones, out):
    H, W = values.shape
    for i in prange(H):
        for j in range(W):
            zid = zones[i, j]
            if zid >= 0:
                out[zid] += values[i, j]


def aggregate_pv_by_polygons(
    pv_raster: str,
    polygons_gdf: gpd.GeoDataFrame,
    id_field: str,
    nodata: float = -9999.0,
):
    """
    Aggregate PV raster values over Polygon or MultiPolygon geometries.
    One output row per unique id_field (e.g. plant_id).
    """

    results = []

    with rasterio.open(pv_raster) as src:
        data = src.read(1)
        transform = src.transform

        for plant_id, group in polygons_gdf.groupby(id_field):
            # Union all geometries (Polygon + MultiPolygon safe)
            geom = group.geometry.unary_union

            mask = geometry_mask(
                [geom],
                out_shape=data.shape,
                transform=transform,
                invert=True,
            )

            values = data[mask]
            values = values[values != nodata]

            if values.size == 0:
                results.append({
                    id_field: plant_id,
                    "pv_kwh": 0.0,
                    "area_m2": 0.0,
                })
                continue

            pixel_area = abs(transform.a * transform.e)
            area = values.size * pixel_area

            results.append({
                id_field: plant_id,
                "pv_kwh": values.sum() / 1000.0,
                "area_m2": area,
                "mean_pv_kwh_m2": values.mean() / 1000.0,
            })

    return gpd.GeoDataFrame(results)



VEGETATION_SHADOWING = np.array(
    [
        # DECIDUOUS = 0
        [
            [0.10, 0.10, 0.15, 0.25, 0.35, 0.45, 0.50, 0.50, 0.40, 0.25, 0.15, 0.10],
            [0.20, 0.20, 0.30, 0.50, 0.65, 0.75, 0.80, 0.80, 0.70, 0.50, 0.30, 0.20],
            [0.30, 0.30, 0.40, 0.65, 0.80, 0.88, 0.92, 0.92, 0.85, 0.65, 0.45, 0.30],
        ],

        # EVERGREEN = 1
        [
            [0.30, 0.30, 0.32, 0.35, 0.38, 0.42, 0.45, 0.45, 0.42, 0.38, 0.32, 0.30],
            [0.55, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.75, 0.70, 0.65, 0.58, 0.55],
            [0.75, 0.75, 0.78, 0.80, 0.85, 0.88, 0.90, 0.90, 0.88, 0.85, 0.80, 0.75],
        ],
    ],
    dtype=np.float32,
)


ICGC_DSM_CLASS_TO_HEIGHT_INDEX = np.array(
    [-1, -1, -1, 0, 1, 2], dtype=np.int8
)

@njit(inline="always")
def get_vegetation_shadowing_from_dsm(
    leaf_type: int,   # 0=deciduous, 1=evergreen
    dsm_class: int,   # 3, 4, 5
    month: int,       # 1–12
) -> float:
    height_idx = DSM_CLASS_TO_HEIGHT_INDEX[dsm_class]
    if height_idx < 0:
        return 0.0  # no vegetation → no attenuation
    return VEGETATION_SHADOWING[leaf_type, height_idx, month - 1]

