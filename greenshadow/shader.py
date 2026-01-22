# -*- coding: utf-8 -*-
import os
import fnmatch
import math
from tqdm import tqdm
import warnings
from typing import List, Union, Dict, Optional, Tuple, Any
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pathlib import Path
from scipy.ndimage import interpolation as nvi
from osgeo import gdal
import datetime as dt
import concurrent.futures
import multiprocessing as mp
from .utils import _parse_iso8601_duration, _date_range_local_to_utc
import numpy as np
import rasterio
import scipy.ndimage as ndi
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def _shadow_scan(z_rot, nodata_rot, dz_per_px):
    H, W = z_rot.shape
    lit = np.zeros((H, W), dtype=np.uint8)

    for i in nb.prange(H):
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
    azimuth_altitude: List[Tuple[float, float]],
    *,
    related_datetimes: Optional[List[Union[dt.datetime, str]]] = None,
    zFactor: float = 1.0,
    compute_edges: bool = False,
    creation_options: Optional[Dict[str, str]] = None,
    include_datetime_info: bool = True,
    qgis_temporal: bool = True,
    max_workers: int = 2,  # Default to 2 workers to avoid I/O contention
    overwrite: bool = False
) -> List[str]:
    """
    Generate multiple hillshades with cast shadows from azimuth/altitude pairs.
    
    Args:
        raster_dir: Output directory for hillshades
        dsm_path: Path to DSM raster
        azimuth_altitude: List of (azimuth, altitude) tuples
        related_datetimes: Optional list of datetime objects corresponding to each az/alt pair
        shadow_floor: Minimum shadow value (0.0 = full shadow, 1.0 = no shadow effect)
        zFactor: Vertical exaggeration factor
        compute_edges: Compute hillshade at edges
        creation_options: GDAL creation options
        include_datetime_info: Include datetime metadata in output files
        qgis_temporal: Rename files with ISO timestamps for QGIS temporal
        max_workers: Number of parallel workers
        overwrite: Whether to overwrite existing files
        max_shadow_distance: Maximum shadow distance in meters for shadow casting
        strip_height: Height of processing strips in pixels for memory efficiency
    
    Returns:
        List of paths to generated hillshade files
    """
    
    os.makedirs(raster_dir, exist_ok=True)
    
    if related_datetimes and len(related_datetimes) != len(azimuth_altitude):
        raise ValueError("related_datetimes must have same length as azimuth_altitude")
    
    # Filter out below-horizon positions first
    valid_positions = []
    for i, (azimuth, altitude) in enumerate(azimuth_altitude):
        if altitude > 0:
            when = related_datetimes[i] if related_datetimes else None
            valid_positions.append((i, azimuth, altitude, when))
    
    if not valid_positions:
        return []
    
    # Optimize number of workers
    max_workers = min(max_workers, len(valid_positions), mp.cpu_count())
    
    def process_single_hillshade(args):
        # args=valid_positions[6]
        """Process a single hillshade (for parallel execution)."""
        i, azimuth, altitude, when = args
        
        # Check if final file already exists (skip if not overwriting)
        if not overwrite:
            test_final_path = f"{raster_dir}/{int(azimuth):03}_{int(altitude):02}.tif"
            if when and qgis_temporal:
                iso_str = when.strftime('%Y%m%dT%H%M%S') if isinstance(when, dt.datetime) else str(when)
                test_final_path = f"{raster_dir}/{int(azimuth):03}_{int(altitude):02}_{iso_str}.tif"
            if os.path.exists(test_final_path):
                return test_final_path
        
        try:
            # Generate raw hillshade
            hs_raw = _generate_hillshade_gdal(
                results_dir=raster_dir,
                include_datetime_info=include_datetime_info,
                when=when,
                dsm_path=dsm_path,
                azimuth=azimuth,
                altitude=altitude,
                zFactor=zFactor,
                compute_edges=compute_edges,
                creation_options=creation_options
            )
            
            # Apply shadow mask
            final_path = _apply_dsm_shadow_to_hillshade(
                hillshade_path=hs_raw,
                dsm_path=dsm_path,
                out_path=hs_raw.replace("_raw",""),
                azimuth=azimuth,
                altitude=altitude
            )
            
            # Clean up raw file if different from final
            if hs_raw != final_path and os.path.exists(hs_raw):
                try:
                    os.remove(hs_raw)
                except Exception:
                    pass
                    
            return final_path
        except Exception as e:
            print(f"Error processing hillshade for az={azimuth}, alt={altitude}: {e}")
            return None
    
    final_paths = []
    
    if max_workers == 1:
        # Single-threaded execution
        for args in tqdm(valid_positions, desc="Generating hillshades"):
            result = process_single_hillshade(args)
            if result:
                final_paths.append(result)
    else:
        # Multi-threaded execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_single_hillshade, args): args 
                             for args in valid_positions}
            
            # Collect results with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                             total=len(future_to_args), desc="Generating hillshades"):
                try:
                    result = future.result()
                    if result:
                        final_paths.append(result)
                except Exception as e:
                    args = future_to_args[future]
                    print(f"Error in future for args {args}: {e}")
    
    return final_paths


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
    import pytz
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
            import pytz
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
            granularity_pandas = granularity.replace('PT', '').replace('M', 'min').replace('H', 'H')
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
) -> Dict[str, Any]:
    """
    Generate azimuth/altitude pairs for solar positions across a full year.
    
    Similar to generate_azimuth_altitude_pairs_by_date but for entire year.
    """
    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)
    
    # Parse granularity
    granularity_delta = _parse_iso8601_duration(granularity)
    
    # Generate UTC time range
    times_utc = _date_range_local_to_utc(start_date, end_date, tz_name, granularity_delta, only_daylight, lat, lon)
    
    if len(times_utc) == 0:
        return {
            'azimuth_altitude_pairs': [],
            'metadata': {'year': year, 'lat': lat, 'lon': lon, 'tz_name': tz_name}
        }
    
    # Calculate solar positions
    solar_df = _solar_position_az_el(times_utc, lat, lon)
    
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
        azimuth = azimuth[daylight_mask]
        elevation = elevation[daylight_mask]
    
    # Create combined tuples and remove duplicates
    azimuth_altitude = list(set(zip(azimuth, elevation)))
    azimuth_altitude.sort()  # Sort for consistency
    
    return {
        'azimuth_altitude_pairs': azimuth_altitude,
        'metadata': {
            'year': year,
            'lat': lat,
            'lon': lon, 
            'tz_name': tz_name,
            'granularity': granularity,
            'discretization': solar_degrees_discretization,
            'unique_positions': len(azimuth_altitude)
        }
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
    import re
    from pathlib import Path
    
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
    mask_value: float = 1,
    parallel: bool = False,
    max_workers: int = 4,
) -> List[str]:
    """
    Apply height-based shadow masking to existing hillshades.
    
    This function takes existing hillshade files and applies additional shadowing
    based on the height difference between DSM and DEM at a specified height.
    
    Args:
        hillshade_dir: Directory containing input hillshades
        dem_path: Path to DEM raster
        dsm_path: Path to DSM raster  
        out_dir: Output directory for modified hillshades
        height_m: Height above ground for shadow calculation
        pattern: File pattern to match hillshades
        recursive: Search subdirectories
        overwrite: Overwrite existing output files
        mask_value: Value to set for shadowed areas
        parallel: Use parallel processing
        max_workers: Number of parallel workers
    
    Returns:
        List of output file paths
    """
    import glob
    from pathlib import Path
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Find hillshade files
    if recursive:
        hillshade_files = list(Path(hillshade_dir).rglob(pattern))
    else:
        hillshade_files = list(Path(hillshade_dir).glob(pattern))
    
    hillshade_files = [str(f) for f in hillshade_files if f.is_file()]
    
    if not hillshade_files:
        warnings.warn(f"No hillshade files found matching {pattern} in {hillshade_dir}")
        return []
    
    # Read DEM and DSM
    try:
        with rasterio.open(dem_path) as dem_src:
            dem = dem_src.read(1).astype(np.float32)
            dem_profile = dem_src.profile
            dem_transform = dem_src.transform
            dem_crs = dem_src.crs
        
        with rasterio.open(dsm_path) as dsm_src:
            dsm = dsm_src.read(1).astype(np.float32)
            dsm_profile = dsm_src.profile
    except Exception as e:
        raise RuntimeError(f"Error reading DEM/DSM: {e}")
    
    # Check that DEM and DSM are compatible
    if dem.shape != dsm.shape:
        raise ValueError(f"DEM shape {dem.shape} != DSM shape {dsm.shape}")
    
    # Calculate height difference mask
    height_diff = dsm - dem
    obstacle_mask = height_diff > height_m
    
    output_files = []
    
    def process_hillshade(hs_path: str) -> str:
        """Process a single hillshade file."""
        try:
            # Determine output path
            rel_path = os.path.relpath(hs_path, hillshade_dir)
            out_path = os.path.join(out_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            if os.path.exists(out_path) and not overwrite:
                return out_path
            
            # Read hillshade
            with rasterio.open(hs_path) as hs_src:
                hs = hs_src.read(1).astype(np.float32)
                hs_profile = hs_src.profile.copy()
            
            # Check compatibility
            if hs.shape != obstacle_mask.shape:
                warnings.warn(f"Hillshade {hs_path} shape {hs.shape} != mask shape {obstacle_mask.shape}")
                return ""
            
            # Apply height-based shadow
            hs_modified = hs.copy()
            hs_modified[obstacle_mask] = mask_value
            
            # Write output
            with rasterio.open(out_path, 'w', **hs_profile) as dst:
                dst.write(hs_modified.astype(hs_profile['dtype']), 1)
            
            return out_path
            
        except Exception as e:
            warnings.warn(f"Error processing {hs_path}: {e}")
            return ""
    
    # Process files
    if parallel and max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_hillshade, hillshade_files),
                total=len(hillshade_files),
                desc="Applying height shadows"
            ))
    else:
        results = []
        for hs_path in tqdm(hillshade_files, desc="Applying height shadows"):
            results.append(process_hillshade(hs_path))
    
    # Filter successful results
    output_files = [r for r in results if r]
    
    return output_files


def average_hillshades(
    hillshade_dir: str,
    out_path: str,
    *,
    pattern: str = "*.tif",
    recursive: bool = True,
    exclude_patterns: Optional[List[str]] = None,
    aggregation: str = "mean",  # "mean" | "median" | "max" | "min"
    dtype: str = "float32",
    nodata: Optional[float] = None,
    chunk_size: int = 10,  # Process files in chunks to manage memory
) -> str:
    """
    Average multiple hillshade rasters into a single output.
    
    Args:
        hillshade_dir: Directory containing hillshade files
        out_path: Path for output averaged hillshade
        pattern: File pattern to match
        recursive: Search recursively
        exclude_patterns: Patterns to exclude
        aggregation: Aggregation method ("mean", "median", "max", "min")
        dtype: Output data type
        nodata: NoData value for output
        chunk_size: Number of files to process at once (memory management)
    
    Returns:
        Path to output file
    """
    from pathlib import Path
    
    # Find hillshade files
    if recursive:
        hillshade_files = list(Path(hillshade_dir).rglob(pattern))
    else:
        hillshade_files = list(Path(hillshade_dir).glob(pattern))
    
    hillshade_files = [str(f) for f in hillshade_files if f.is_file()]
    
    # Apply exclude patterns
    if exclude_patterns:
        filtered_files = []
        for f in hillshade_files:
            fname = os.path.basename(f)
            should_exclude = any(fnmatch.fnmatch(fname, pat) for pat in exclude_patterns)
            if not should_exclude:
                filtered_files.append(f)
        hillshade_files = filtered_files
    
    if not hillshade_files:
        raise ValueError(f"No hillshade files found in {hillshade_dir}")
    
    print(f"Found {len(hillshade_files)} hillshade files to average")
    
    # Read first file to get dimensions and profile
    with rasterio.open(hillshade_files[0]) as src:
        profile = src.profile.copy()
        shape = src.shape
        first_data = src.read(1).astype(np.float64)
    
    # Update profile
    profile.update(dtype=dtype, nodata=nodata, compress='deflate', tiled=True,
                   blockxsize=256, blockysize=256)
    
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    # Initialize accumulator
    if aggregation == "mean":
        accumulator = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.int32)
    elif aggregation == "median":
        all_data = []
    elif aggregation == "max":
        accumulator = np.full(shape, -np.inf, dtype=np.float64)
    elif aggregation == "min":
        accumulator = np.full(shape, np.inf, dtype=np.float64)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Process files in chunks
    for i in tqdm(range(0, len(hillshade_files), chunk_size), desc=f"Averaging with {aggregation}"):
        chunk_files = hillshade_files[i:i+chunk_size]
        
        if aggregation == "median":
            # For median, collect all data first
            chunk_data = []
            for hs_path in chunk_files:
                try:
                    with rasterio.open(hs_path) as src:
                        data = src.read(1).astype(np.float64)
                        if data.shape != shape:
                            warnings.warn(f"Shape mismatch in {hs_path}: {data.shape} != {shape}")
                            continue
                        chunk_data.append(data)
                except Exception as e:
                    warnings.warn(f"Error reading {hs_path}: {e}")
                    continue
            
            if chunk_data:
                all_data.extend(chunk_data)
        
        else:
            # For other methods, process incrementally
            for hs_path in chunk_files:
                try:
                    with rasterio.open(hs_path) as src:
                        data = src.read(1).astype(np.float64)
                        
                        if data.shape != shape:
                            warnings.warn(f"Shape mismatch in {hs_path}: {data.shape} != {shape}")
                            continue
                        
                        # Handle nodata
                        valid_mask = np.isfinite(data)
                        if src.nodata is not None:
                            valid_mask &= (data != src.nodata)
                        
                        if aggregation == "mean":
                            accumulator[valid_mask] += data[valid_mask]
                            count[valid_mask] += 1
                        elif aggregation == "max":
                            accumulator = np.maximum(accumulator, data)
                        elif aggregation == "min":
                            accumulator = np.minimum(accumulator, data)
                            
                except Exception as e:
                    warnings.warn(f"Error processing {hs_path}: {e}")
                    continue
    
    # Finalize result
    if aggregation == "mean":
        result = np.divide(accumulator, count, out=np.zeros_like(accumulator), where=(count > 0))
        if nodata is not None:
            result[count == 0] = nodata
    elif aggregation == "median":
        if all_data:
            print(f"Computing median from {len(all_data)} arrays...")
            all_array = np.stack(all_data, axis=0)
            result = np.median(all_array, axis=0)
        else:
            result = np.full(shape, nodata if nodata is not None else 0, dtype=np.float64)
    elif aggregation in ["max", "min"]:
        result = accumulator
        if aggregation == "max":
            result[result == -np.inf] = nodata if nodata is not None else 0
        else:  # min
            result[result == np.inf] = nodata if nodata is not None else 0
    
    # Convert to output dtype
    if dtype == "uint8":
        result = np.clip(result, 0, 255).astype(np.uint8)
    elif dtype == "float32":
        result = result.astype(np.float32)
    else:
        result = result.astype(dtype)
    
    # Write output
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(result, 1)
    
    print(f"Averaged hillshade saved to {out_path}")
    return out_path


def calculate_sky_view_factor(
        dsm_path: str,
        out_path: str,
        search_radius: float = 100.0,
        angular_resolution: float = 5.0,
        max_height_angle: float = 90.0,
        nodata_fill: float = 1.0,
        creation_options: Optional[Dict[str, str]] = None
) -> str:
    """
    Calculate Sky View Factor (SVF) from a Digital Surface Model.

    Sky View Factor measures the fraction of the sky hemisphere that is visible 
    from each point, ranging from 0 (completely obstructed) to 1 (completely open).

    Parameters:
    -----------
    dsm_path : str
        Path to input Digital Surface Model (DSM) raster
    out_path : str
        Path for output Sky View Factor raster
    search_radius : float, default 100.0
        Maximum radius (in meters) to search for obstructing features
    angular_resolution : float, default 5.0
        Angular resolution (in degrees) for horizon calculation
        Lower values = more accurate but slower
    max_height_angle : float, default 90.0
        Maximum height angle (in degrees) to consider for SVF calculation
    nodata_fill : float, default 1.0
        Value to assign to nodata areas (1.0 = completely open sky)
    creation_options : Optional[Dict[str, str]]
        GDAL creation options for output raster

    Returns:
    --------
    str : Path to the created Sky View Factor raster

    Notes:
    ------
    - SVF values range from 0 to 1
    - Urban canyons have low SVF (~0.1-0.4)
    - Open areas have high SVF (~0.8-1.0)
    - Useful for urban climate, solar potential, and thermal comfort analysis
    """

    print(f"Calculating Sky View Factor from {dsm_path}", flush=True)
    print(f"Search radius: {search_radius}m, Angular resolution: {angular_resolution}°", flush=True)

    # Read DSM
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1, masked=True)
        transform = src.transform
        profile = src.profile.copy()
        nodata = src.nodata

        # Get pixel size (assuming square pixels)
        pixel_size = abs(transform[0])
        height, width = dsm.shape

    print(f"DSM dimensions: {width}x{height}, pixel size: {pixel_size:.2f}m", flush=True)

    # Calculate search radius in pixels
    search_radius_pixels = int(search_radius / pixel_size)

    # Generate azimuth angles for horizon calculation
    azimuth_angles = np.arange(0, 360, angular_resolution)
    n_azimuths = len(azimuth_angles)

    print(f"Computing SVF for {n_azimuths} azimuth directions...", flush=True)

    # Initialize Sky View Factor array
    svf = np.full_like(dsm.data, nodata_fill, dtype=np.float32)

    # Handle masked/nodata areas
    if hasattr(dsm, 'mask'):
        valid_mask = ~dsm.mask
        dsm_data = dsm.data
    else:
        if nodata is not None:
            valid_mask = (dsm != nodata) & np.isfinite(dsm)
        else:
            valid_mask = np.isfinite(dsm)
        dsm_data = dsm

    # Create coordinate grids
    rows, cols = np.mgrid[0:height, 0:width]

    # Process each pixel that has valid data
    valid_rows, valid_cols = np.where(valid_mask)
    total_pixels = len(valid_rows)

    print(f"Processing {total_pixels} valid pixels...", flush=True)

    # Improved processing strategy for better interpolation
    print("Creating optimized processing grid...", flush=True)

    # Use moderate downsampling with better spatial distribution
    if total_pixels > 10000:  # Only downsample for large datasets
        step_size = max(2, min(4, int(np.sqrt(total_pixels / 5000))))  # More conservative
        # Create more evenly distributed sample points
        process_rows = valid_rows[::step_size]
        process_cols = valid_cols[::step_size]
        print(f"Using moderate downsampling with step size {step_size}", flush=True)
    else:
        # For small datasets, process all pixels for best quality
        process_rows = valid_rows
        process_cols = valid_cols
        step_size = 1
        print("Processing all pixels (small dataset)", flush=True)

    print(f"Processing {len(process_rows)} sample points (from {total_pixels} total)", flush=True)

    # Pre-calculate direction vectors for all azimuths
    azimuth_rad = np.deg2rad(azimuth_angles)
    dx_all = np.sin(azimuth_rad) / pixel_size
    dy_all = np.cos(azimuth_rad) / pixel_size

    # Fully vectorized processing using batch approach
    # Use larger batches for better performance and clearer progress reporting
    if len(process_rows) > 10000:
        batch_size = min(2000, len(process_rows) // 20)  # Aim for ~20 batches for large datasets
    else:
        batch_size = min(1000, len(process_rows) // 5)  # Aim for ~5 batches for smaller datasets

    batch_size = max(100, batch_size)  # Minimum batch size
    n_batches = (len(process_rows) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"Processing in {n_batches} batches of {batch_size} pixels..."):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(process_rows))

        batch_rows = process_rows[start_idx:end_idx]
        batch_cols = process_cols[start_idx:end_idx]
        batch_size_actual = len(batch_rows)

        # Initialize batch results
        batch_max_angles = np.zeros((batch_size_actual, n_azimuths))
        batch_heights = dsm_data[batch_rows, batch_cols]

        # Vectorized ray casting with adaptive step size for speed
        step_increment = max(1, search_radius_pixels // 25)  # Use fewer samples along rays

        for step in range(step_increment, search_radius_pixels + 1, step_increment):
            # Calculate sample positions for all pixels and all azimuths
            # Shape: (batch_size, n_azimuths)
            sample_rows_all = batch_rows[:, np.newaxis] + dy_all[np.newaxis, :] * step
            sample_cols_all = batch_cols[:, np.newaxis] + dx_all[np.newaxis, :] * step

            # Check bounds for all samples
            valid_samples_mask = ((sample_rows_all >= 0) & (sample_rows_all < height - 1) &
                                  (sample_cols_all >= 0) & (sample_cols_all < width - 1))

            if not np.any(valid_samples_mask):
                break

            # Sample heights using bilinear interpolation for all valid positions
            valid_positions = np.where(valid_samples_mask)
            if len(valid_positions[0]) > 0:
                try:
                    # Efficient batch interpolation
                    sample_heights = nvi.map_coordinates(
                        dsm_data,
                        np.array([sample_rows_all[valid_samples_mask], sample_cols_all[valid_samples_mask]]),
                        order=1,
                        mode='constant',
                        cval=0,
                        prefilter=False
                    )

                    # Calculate elevation angles
                    distance = step * pixel_size
                    pixel_heights = batch_heights[valid_positions[0]]
                    height_diffs = sample_heights - pixel_heights
                    elevation_angles = np.rad2deg(np.arctan(height_diffs / distance))

                    # Update maximum angles efficiently
                    current_max = batch_max_angles[valid_positions]
                    batch_max_angles[valid_positions] = np.maximum(current_max, elevation_angles)

                except:
                    continue

        # Calculate SVF for entire batch
        batch_max_angles = np.clip(batch_max_angles, 0, max_height_angle)
        batch_max_angles_rad = np.deg2rad(batch_max_angles)
        batch_svf_values = np.mean(np.cos(batch_max_angles_rad), axis=1)

        # Store results
        svf[batch_rows, batch_cols] = batch_svf_values

    # High-quality interpolation to fill missing pixels
    computed_mask = svf != nodata_fill

    if step_size > 1 and np.any(computed_mask):
        print("Interpolating to full resolution using bilinear interpolation...", flush=True)

        # Use scipy's griddata for high-quality interpolation
        from scipy.interpolate import griddata

        missing_mask = (valid_mask & (svf == nodata_fill))

        if np.any(missing_mask):
            # Get coordinates and values of computed points
            computed_rows, computed_cols = np.where(computed_mask)
            computed_values = svf[computed_mask]

            # Get coordinates of missing points
            missing_rows, missing_cols = np.where(missing_mask)

            if len(computed_rows) > 3:  # Need at least 3 points for interpolation
                print(f"Interpolating {len(missing_rows)} missing pixels from {len(computed_rows)} computed points",
                      flush=True)

                # Prepare point coordinates
                points = np.column_stack([computed_rows, computed_cols])
                values = computed_values
                xi = np.column_stack([missing_rows, missing_cols])

                # Use linear interpolation (bilinear)
                try:
                    interpolated_values = griddata(
                        points, values, xi,
                        method='linear',
                        fill_value=nodata_fill
                    )

                    # For points that couldn't be interpolated linearly, use nearest neighbor
                    still_missing = interpolated_values == nodata_fill
                    if np.any(still_missing):
                        nearest_values = griddata(
                            points, values, xi[still_missing],
                            method='nearest'
                        )
                        interpolated_values[still_missing] = nearest_values

                    # Assign interpolated values
                    svf[missing_rows, missing_cols] = interpolated_values

                except Exception as e:
                    print(f"Interpolation failed, using nearest neighbor fallback: {e}", flush=True)
                    # Fallback to simple nearest neighbor
                    from scipy.spatial.distance import cdist
                    distances = cdist(xi, points)
                    nearest_indices = np.argmin(distances, axis=1)
                    svf[missing_rows, missing_cols] = values[nearest_indices]

    print("SVF calculation complete. Saving results...", flush=True)

    # Prepare output profile - simple configuration to avoid TIFF issues
    profile.update(
        dtype='float32',
        count=1,
        nodata=None,  # SVF doesn't use nodata, uses fill value instead
        driver='GTiff'
    )

    # Create output directory if needed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write SVF raster
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(svf, 1)

        # Add metadata
        dst.update_tags(
            1,
            units='dimensionless',
            description='Sky View Factor',
            valid_range='0.0 to 1.0',
            method='horizon_based',
            search_radius=f'{search_radius}m',
            angular_resolution=f'{angular_resolution}°'
        )

    print(f"Sky View Factor saved to: {out_path}", flush=True)

    # Print statistics
    valid_svf = svf[valid_mask]
    print(f"SVF Statistics:", flush=True)
    print(f"  Mean: {np.mean(valid_svf):.3f}", flush=True)
    print(f"  Min:  {np.min(valid_svf):.3f}", flush=True)
    print(f"  Max:  {np.max(valid_svf):.3f}", flush=True)
    print(f"  Std:  {np.std(valid_svf):.3f}", flush=True)

    return out_path