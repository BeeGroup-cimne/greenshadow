# -*- coding: utf-8 -*-
import re
import os
import sys
import json
import hashlib
import fnmatch
import subprocess
import shutil
import requests
import math
import tempfile
import psutil
import time
from typing import List, Union, Dict, Optional, Tuple, Any, Sequence
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely import wkt as shapely_wkt
from shapely.validation import explain_validity, make_valid
from shapely.ops import transform
from collections import Counter, defaultdict
import pyproj
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from zipfile import ZipFile, BadZipFile
import tarfile
from rasterio.enums import MergeAlg, Resampling
from scipy.ndimage import distance_transform_edt
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from rasterio.windows import Window
try:
    from .logging_utils import info, warning, error, tqdm
except ImportError:
    from greenshadow.logging_utils import info, warning, error, tqdm

# Constants
BASE_JSON = "https://datacloud.icgc.cat/datacloud/lidar-territorial/json/lidar-territorial-tall.json"
BASE_DATA = "https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/"
UA = "Mozilla/5.0 (compatible; icgc-lidar/1.1)"


# ---------------- HTTP session w/ retries ----------------
def _session() -> requests.Session:
    """Create HTTP session with retry configuration."""
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


# ---------------- Geometry helpers ----------------

def _reproject_geom(geom, src_crs="EPSG:25831", dst_crs="EPSG:4326"):
    """Reproject a Shapely geometry from src_crs to dst_crs."""
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


def _to_shapely(geom: Union[str, dict, BaseGeometry]) -> BaseGeometry:
    """Convert various geometry formats to Shapely geometry."""
    if isinstance(geom, BaseGeometry):
        return geom
    if isinstance(geom, str):
        s = geom.strip()
        if s.startswith("{") or s.startswith("["):
            return shape(json.loads(s))
        return shapely_wkt.loads(s)
    if isinstance(geom, dict):
        return shape(geom)
    raise ValueError("AOI must be shapely geometry, WKT string, or GeoJSON (dict/str)")


def _extract_epsg(crs_obj: dict) -> int:
    """Extract EPSG code from CRS object."""
    try:
        name = crs_obj.get("properties", {}).get("name", "")
        m = re.search(r"EPSG:(\d+)", name)
        return int(m.group(1)) if m else 25831
    except Exception:
        return 25831


def shapely_ops_transform(geom: BaseGeometry, func):
    """Apply transformation function to Shapely geometry."""
    return transform(func, geom)


# ---------------- File and URL utilities ----------------

def _safe_name(s: str) -> str:
    """Create safe filename from string."""
    return re.sub(r'[^\w\-_.]', '_', s)


def _candidate_urls(id1k: str, id10k: str) -> List[str]:
    """Generate candidate URLs for LAZ files."""
    return [
        f"{BASE_DATA}full10km{id10k}/lidar-territorial-v3r1-full1km{id1k}-2021-2023.laz",
    ]


def _resolve_existing_url(s: requests.Session, id1k: str, id10k: str) -> Optional[str]:
    """Find first existing URL from candidates."""
    for url in _candidate_urls(id1k, id10k):
        try:
            resp = s.head(url, timeout=10)
            if resp.status_code == 200:
                return url
        except Exception:
            continue
    return None


def _download_one(s: requests.Session, url: str, out_dir: str) -> str:
    """Download single file to directory."""
    final_path, temp_path = _target_paths_for_url(url, out_dir)
    
    if os.path.exists(final_path):
        return final_path
        
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        with s.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(temp_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
        
        # Atomic move
        shutil.move(temp_path, final_path)
        return final_path
        
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise RuntimeError(f"Download failed for {url}: {e}")


def _target_paths_for_url(url: str, out_dir: str) -> tuple[str, str]:
    """Generate target and temporary paths for download."""
    fname = url.split("/")[-1]
    if not fname or fname == url:
        fname = hashlib.md5(url.encode()).hexdigest()[:12] + ".laz"
    
    fname_safe = _safe_name(fname)
    final_path = os.path.join(out_dir, fname_safe)
    temp_path = final_path + ".tmp"
    return final_path, temp_path


def _download_many(urls: List[str], out_dir: str, max_workers: int = 8) -> List[str]:
    """Download multiple files with progress tracking."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not urls:
        return []
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Check existing files
    existing = []
    to_download = []
    
    for url in urls:
        final_path, _ = _target_paths_for_url(url, out_dir)
        if os.path.exists(final_path):
            existing.append(final_path)
        else:
            to_download.append(url)
    
    if not to_download:
        info(f"All {len(existing)} files already exist")
        return existing
    
    info(f"Downloading {len(to_download)} files (+ {len(existing)} existing)")
    
    downloaded = []
    errors = []
    s = _session()
    
    def download_url(url):
        try:
            return _download_one(s, url, out_dir)
        except Exception as e:
            return f"ERROR: {url}: {e}"
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_url, url): url for url in to_download}
        
        for future in tqdm(as_completed(futures), total=len(to_download), desc="Downloading"):
            result = future.result()
            if isinstance(result, str) and result.startswith("ERROR:"):
                errors.append(result)
            else:
                downloaded.append(result)
    
    if errors:
        error(f"Download errors ({len(errors)}):")
        for err in errors[:5]:  # Show first 5 errors
            error(f"  {err}")
        if len(errors) > 5:
            error(f"  ... and {len(errors)-5} more")
    
    all_paths = existing + downloaded
    info(f"Total files available: {len(all_paths)}")
    return all_paths


# ---------------- Load & normalize tiles ----------------
def _load_featurecollection(url: str) -> Tuple[List[Dict], int]:
    """Load GeoJSON FeatureCollection from URL."""
    s = _session()
    r = s.get(url, timeout=60)
    r.raise_for_status()
    fc = r.json()
    if fc.get("type") != "FeatureCollection":
        raise RuntimeError("JSON is not a FeatureCollection")
    epsg = _extract_epsg(fc.get("crs", {}))
    return fc.get("features", []), epsg


def _filter_tiles(
    features: List[Dict],
    aoi: BaseGeometry,
    aoi_epsg: int,
    target_epsg: int,
    aoi_buffer_in_meters: float = 0.0,
    centroid_only: bool = False
) -> List[Dict]:
    """Filter tiles based on AOI intersection."""
    
    # Reproject AOI to target CRS if needed
    if aoi_epsg != target_epsg:
        aoi_target = _reproject_geom(aoi, f"EPSG:{aoi_epsg}", f"EPSG:{target_epsg}")
    else:
        aoi_target = aoi
    
    # Apply buffer
    if aoi_buffer_in_meters > 0:
        aoi_target = aoi_target.buffer(aoi_buffer_in_meters)
    
    if centroid_only:
        # Pick tile containing AOI centroid
        centroid = aoi_target.centroid
        for feat in features:
            geom_dict = feat.get("geometry")
            if not geom_dict:
                continue
            try:
                tile_geom = shape(geom_dict)
                if tile_geom.contains(centroid):
                    return [feat]
            except Exception:
                continue
        return []
    else:
        # Filter by intersection
        matching = []
        for feat in features:
            geom_dict = feat.get("geometry")
            if not geom_dict:
                continue
            try:
                tile_geom = shape(geom_dict)
                if tile_geom.intersects(aoi_target):
                    matching.append(feat)
            except Exception:
                continue
        return matching


# ---------------- GDAL utilities ----------------

# Configure GDAL to use exceptions for better error handling.
# Calling UseExceptions() explicitly also silences the GDAL >= 3.7 FutureWarning
# ("Neither gdal.UseExceptions() nor gdal.DontUseExceptions() has been explicitly
# called"), which is emitted on the first GDAL call of the process otherwise.
try:
    from osgeo import gdal
except ImportError:
    gdal = None


def _init_gdal():
    """Initialize GDAL with exception handling."""
    if gdal is None:
        return False
    try:
        gdal.UseExceptions()
        return True
    except Exception:
        return False

# Try to initialize GDAL - store result for later use
_gdal_initialized = _init_gdal()

def _ensure_dir(p):
    """Ensure directory exists."""
    os.makedirs(p, exist_ok=True)


def _gdal_exists():
    """Check if GDAL is available."""
    try:
        from osgeo import gdal
        return True
    except ImportError:
        return False


def _fill_nodata(tile_path: str, nodata: float, max_dist_px: int = 10) -> bool:
    """Fill nodata values in raster using GDAL."""
    if not _gdal_exists():
        return False
    
    try:
        from osgeo import gdal
        ds = gdal.Open(tile_path, gdal.GA_Update)
        if ds is None:
            return False
        
        band = ds.GetRasterBand(1)
        result = gdal.FillNodata(band, None, max_dist_px, 0)
        
        band = None
        ds = None
        
        return result == 0  # GDAL returns 0 on success
    except Exception:
        return False

def _build_raster_profile(
    width: int,
    height: int,
    transform: Affine,
    crs,
    nodata: float,
    dtype,
    compress = "lzw",
    block_size: int = 1024,
):
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": block_size,
        "blockysize": block_size,
    }

def _build_profile_from_reference(
    data,
    reference_profile
):
    profile = reference_profile.copy()
    profile.update(
        dtype=data.dtype,
        count=1 if data.ndim == 2 else data.shape[0],
        height=data.shape[-2],
        width=data.shape[-1]
    )
    return profile

def _write_raster_in_blocks(dst, arr, block_size=1024):
    height, width = arr.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):

            h = min(block_size, height - y)
            w = min(block_size, width - x)

            window = Window(x, y, w, h)

            dst.write(
                arr[y:y+h, x:x+w],
                1,
                window=window
            )

def _save_raster(
    data,
    reference_profile,
    output_tif: str
):
    """
    Save an in-memory raster (NumPy array) to GeoTIFF.

    Parameters
    ----------
    data : np.ndarray
        Raster data. Shape (H, W) or (bands, H, W).
    reference_profile : dict
        Rasterio profile / metadata dictionary.
    output_tif : str
        Output GeoTIFF path.
    """
    profile = _build_profile_from_reference(data, reference_profile)

    if data.ndim == 2:
        profile.update(count=1, dtype=data.dtype)
        bands = [data]
    elif data.ndim == 3:
        profile.update(count=data.shape[0], dtype=data.dtype)
        bands = data
    else:
        raise ValueError("Data must be 2D or 3D numpy array")

    with rasterio.open(output_tif, "w", **profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band, i)


# ---------------- PDAL pipeline utilities ----------------

def _pdal_run_pipeline(pipeline_json: str, prefer_python=True):
    """Run PDAL pipeline from JSON string."""
    import json as json_lib
    
    # Try Python PDAL first if preferred
    if prefer_python:
        try:
            import pdal
            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()
            return pipeline
        except ImportError:
            pass
        except Exception as e:
            warning(f"Python PDAL failed: {e}")
    
    # Fall back to command-line PDAL
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(pipeline_json)
            pipeline_file = f.name
        
        try:
            subprocess.run(['pdal', 'pipeline', pipeline_file], 
                         check=True, capture_output=True, text=True)
        finally:
            os.unlink(pipeline_file)
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"PDAL command failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("PDAL not found. Install PDAL or python-pdal.")


def _infer_bounds_from_headers(laz_paths: Sequence[str]) -> Tuple[float, float, float, float]:
    """Infer bounds from LAZ file headers."""
    import laspy
    
    bounds_list = []
    for p in laz_paths:
        try:
            with laspy.open(p) as lf:
                hdr = lf.header
                bounds_list.append((hdr.x_min, hdr.y_min, hdr.x_max, hdr.y_max))
        except Exception:
            continue
    
    if not bounds_list:
        raise ValueError("Could not read bounds from any LAZ file")
    
    minx = min(b[0] for b in bounds_list)
    miny = min(b[1] for b in bounds_list)
    maxx = max(b[2] for b in bounds_list)
    maxy = max(b[3] for b in bounds_list)
    
    return minx, miny, maxx, maxy


# ---------------- File pattern matching ----------------

def find_files_by_pattern(
    root_dir: str,
    pattern: str,
    *,
    mode: str = "glob",  # "glob" | "regex"
    recursive: bool = True,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Find files matching a pattern."""
    import re
    from pathlib import Path
    
    exclude_patterns = exclude_patterns or []
    found_files = []
    
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
        
        if not should_exclude:
            found_files.append(str(file_path))
    
    return found_files


# ---------------- Coordinate and projection utilities ----------------

def degrees_to_utm_zone(longitude: float) -> int:
    """Convert longitude to UTM zone number."""
    return int((longitude + 180) / 6) + 1


def utm_epsg_from_lonlat(longitude: float, latitude: float) -> int:
    """Get UTM EPSG code from longitude/latitude."""
    zone = degrees_to_utm_zone(longitude)
    if latitude >= 0:
        return 32600 + zone  # Northern hemisphere
    else:
        return 32700 + zone  # Southern hemisphere


def reproject_bounds(
    bounds: Tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str
) -> Tuple[float, float, float, float]:
    """Reproject bounding box to different CRS."""
    from shapely.geometry import box
    
    bbox = box(*bounds)
    bbox_reproj = _reproject_geom(bbox, src_crs, dst_crs)
    return bbox_reproj.bounds


# ---------------- Validation utilities ----------------

def validate_geometry(geom: BaseGeometry) -> Tuple[bool, Optional[str]]:
    """Validate Shapely geometry and return error message if invalid."""
    if geom.is_valid:
        return True, None
    else:
        return False, explain_validity(geom)


def fix_geometry(geom: BaseGeometry) -> BaseGeometry:
    """Attempt to fix invalid geometry."""
    if geom.is_valid:
        return geom
    
    try:
        # Try buffer(0) - often fixes self-intersections
        fixed = geom.buffer(0)
        if fixed.is_valid:
            return fixed
    except Exception:
        pass
    
    try:
        # Try simplify
        simplified = geom.simplify(1e-6)
        if simplified.is_valid:
            return simplified
    except Exception:
        pass
    
    # Return original if nothing worked
    return geom


# ---------------- Math utilities ----------------

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


def linear_interpolate(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Linear interpolation between two points."""
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


# ---------------- Configuration and environment ----------------

def hard_print(msg):
    sys.stderr.write(msg)
    sys.stderr.flush()

def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment."""
    import sys
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'working_directory': os.getcwd(),
    }
    
    # Check for optional dependencies
    optional_deps = {}
    for dep in ['gdal', 'pdal', 'laspy', 'pysolar', 'sunpy']:
        try:
            __import__(dep)
            optional_deps[dep] = True
        except ImportError:
            optional_deps[dep] = False
    
    info['optional_dependencies'] = optional_deps
    
    return info


def create_output_directory(base_path: str, create_subdirs: bool = True) -> str:
    """Create output directory structure."""
    os.makedirs(base_path, exist_ok=True)
    
    if create_subdirs:
        subdirs = ['dsm', 'dem', 'hillshades', 'temp', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    
    return base_path


# Time functions ----


def _parse_iso8601_duration(s: str) -> dt.timedelta:
    """Parse ISO 8601 duration string to timedelta."""
    import re

    if not s.startswith('P'):
        raise ValueError(f"Invalid ISO 8601 duration: {s}")

    # Simple regex for common patterns like PT15M, PT1H, P1D
    pattern = r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?'
    match = re.match(pattern, s)

    if not match:
        raise ValueError(f"Cannot parse ISO 8601 duration: {s}")

    days, hours, minutes, seconds = match.groups()

    total_seconds = 0
    if days:
        total_seconds += int(days) * 86400
    if hours:
        total_seconds += int(hours) * 3600
    if minutes:
        total_seconds += int(minutes) * 60
    if seconds:
        total_seconds += float(seconds)

    return dt.timedelta(seconds=total_seconds)


def _iso8601_from_timedelta(delta: dt.timedelta) -> str:
    """Convert timedelta to ISO 8601 duration string."""
    total_seconds = int(delta.total_seconds())

    if total_seconds >= 86400:
        days = total_seconds // 86400
        remainder = total_seconds % 86400
        if remainder == 0:
            return f"P{days}D"
        else:
            hours = remainder // 3600
            minutes = (remainder % 3600) // 60
            seconds = remainder % 60
            time_part = ""
            if hours:
                time_part += f"{hours}H"
            if minutes:
                time_part += f"{minutes}M"
            if seconds:
                time_part += f"{seconds}S"
            return f"P{days}DT{time_part}" if time_part else f"P{days}D"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        time_part = ""
        if hours:
            time_part += f"{hours}H"
        if minutes:
            time_part += f"{minutes}M"
        if seconds:
            time_part += f"{seconds}S"

        return f"PT{time_part}" if time_part else "PT0S"


def _detect_granularity(
        times: Union[pd.DatetimeIndex, List[dt.datetime]],
        *,
        tolerance_factor: float = 1.1,
        min_samples: int = 3
) -> Optional[dt.timedelta]:
    """
    Detect the granularity (time step) from a series of timestamps.

    Args:
        times: Series of datetime objects
        tolerance_factor: How much variation to allow in detected granularity
        min_samples: Minimum number of samples needed for detection

    Returns:
        Detected granularity as timedelta, or None if cannot detect
    """
    if len(times) < min_samples:
        return None

    if isinstance(times, list):
        times = sorted(times)
    else:
        times = times.sort_values()

    # Compute differences
    diffs = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]

    if not diffs:
        return None

    # Find most common difference
    diff_counts = Counter(diffs)
    most_common_diff = diff_counts.most_common(1)[0][0]

    # Check if this difference is consistent
    consistent_count = sum(1 for d in diffs if abs(d - most_common_diff) <= most_common_diff * (tolerance_factor - 1))

    if consistent_count >= len(diffs) * 0.8:  # 80% consistency threshold
        return dt.timedelta(seconds=most_common_diff)

    return None


# def add_granularity(base: dt.datetime, granularity: dt.timedelta, n: int = 1) -> dt.datetime:
#     """Add granularity to a datetime n times."""
#     return base + granularity * n

def delete_all_files_in_directory(directory: str, delete_subdirectories: bool = False):
    """
    Deletes all files inside the specified directory.

    Parameters:
    - directory (str): The path to the directory where files will be deleted.
    - delete_subdirectories (bool): If True, also deletes all subdirectories within the directory.
                                    If False, subdirectories are left intact.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Remove the file or symbolic link
                elif os.path.isdir(file_path) and delete_subdirectories:
                    shutil.rmtree(file_path)  # Remove the directory and its contents if flag is set
            except Exception as e:
                warning(f'Failed to delete {file_path}. Reason: {e}')


def _date_range_local_to_utc(
        date_start: dt.date,
        date_end: dt.date,
        tz_name: str,
        granularity: dt.timedelta,
        only_daylight: bool = True,
        lat: float = None,
        lon: float = None,
) -> pd.DatetimeIndex:
    """Generate datetime range in local timezone and convert to UTC.
    
    Args:
        date_start: Start date
        date_end: End date
        tz_name: Timezone name (e.g., "Europe/Madrid")
        granularity: Time step as timedelta
        only_daylight: If True, filter to only include times when sun is above horizon
        lat: Latitude in degrees (required for solar elevation filtering)
        lon: Longitude in degrees (required for solar elevation filtering)
        
    Returns:
        DatetimeIndex of UTC times, filtered for daylight if requested
    """
    import pytz

    tz = pytz.timezone(tz_name)

    # Start at midnight local time
    dt_start_naive = dt.datetime.combine(date_start, dt.time.min)
    dt_end_naive = dt.datetime.combine(date_end, dt.time.max)

    # Localize to timezone
    dt_start_local = tz.localize(dt_start_naive)
    dt_end_local = tz.localize(dt_end_naive)

    # Generate range in local time
    current = dt_start_local
    times_local = []

    while current <= dt_end_local:
        times_local.append(current)
        current += granularity

    # Convert to UTC
    times_utc = [t.astimezone(pytz.UTC) for t in times_local]
    
    if only_daylight and lat is not None and lon is not None:
        # Filter based on actual solar elevation > 0
        import pvlib
        
        filtered_times = []
        for t_utc in times_utc:
            # Calculate solar position for this time and location
            solar_pos = pvlib.solarposition.get_solarposition(
                time=pd.DatetimeIndex([t_utc]), 
                latitude=lat, 
                longitude=lon
            )
            elevation = solar_pos['elevation'].iloc[0]
            
            # Only include times when sun is above horizon
            if elevation > 0:
                filtered_times.append(t_utc)
        
        times_utc = filtered_times
    elif only_daylight:
        # Fallback to simple hour filter if lat/lon not provided
        filtered_times = []
        for t_local, t_utc in zip(times_local, times_utc):
            hour = t_local.hour
            if 6 <= hour <= 18:  # Simple daylight filter
                filtered_times.append(t_utc)
        times_utc = filtered_times

    return pd.DatetimeIndex(times_utc)


def _write_geotiff_datetime_meta(
        path: str,
        when: Optional[dt.datetime] = None,
        end_when: Optional[dt.datetime] = None,
        azimuth: Optional[float] = None,
        altitude: Optional[float] = None
):
    """Write datetime and solar metadata to GeoTIFF."""
    if not any([when, end_when, azimuth, altitude]):
        return

    try:
        ds = gdal.Open(path, gdal.GA_Update)
        if ds is None:
            return

        if when:
            ds.SetMetadataItem("DATETIME_START", when.isoformat())
        if end_when:
            ds.SetMetadataItem("DATETIME_END", end_when.isoformat())
        if azimuth is not None:
            ds.SetMetadataItem("SOLAR_AZIMUTH", str(azimuth))
        if altitude is not None:
            ds.SetMetadataItem("SOLAR_ALTITUDE", str(altitude))

        ds = None  # Close dataset
    except Exception as e:
        warnings.warn(f"Could not write metadata to {path}: {e}")


def _extract_dt_from_name(path: str) -> Optional[dt.datetime]:
    """Extract datetime from filename if it contains ISO timestamp."""
    import re
    fname = os.path.basename(path)
    # Look for ISO datetime pattern
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)', fname)
    if match:
        try:
            return dt.datetime.fromisoformat(match.group(1).replace('Z', '+00:00'))
        except ValueError:
            pass
    return None


def _maybe_rename_with_iso(tif_path: str, when: dt.datetime, do_rename: bool) -> str:
    """Rename file with ISO timestamp if requested."""
    if not do_rename or not when:
        return tif_path

    dirname = os.path.dirname(tif_path)
    basename = os.path.basename(tif_path)
    name, ext = os.path.splitext(basename)

    # Remove existing timestamp if present
    import re
    name_clean = re.sub(r'_\d{4}-\d{2}-\d{2}T[\d:.-]+(?:Z|[+-]\d{2}:\d{2})?', '', name)

    # Add new timestamp
    iso_str = when.strftime("%Y-%m-%dT%H:%M:%S")
    new_name = f"{name_clean}_{iso_str}{ext}"
    new_path = os.path.join(dirname, new_name)

    try:
        if tif_path != new_path:
            os.rename(tif_path, new_path)
        return new_path
    except Exception as e:
        warnings.warn(f"Could not rename {tif_path}: {e}")
        return tif_path


def _parse_dt(v):
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v
    return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))


## CUDA

def has_cuda() -> bool:
    """
    Returns True if a CUDA-capable GPU is available and usable by Numba.
    """
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False

def cuda_info() -> dict:
    """
    Fully robust CUDA diagnostic for Numba.
    """
    info = {
        "available": False,
        "device_name": None,
        "compute_capability": None,
        "total_memory_GB": None,
    }

    try:
        from numba import cuda

        if not cuda.is_available():
            return info

        cuda.select_device(0)
        dev = cuda.get_current_device()

        # Try context memory info first (most reliable)
        try:
            ctx = cuda.current_context()
            _, total = ctx.get_memory_info()
            total_mem_gb = round(total / 1024**3, 2)
        except Exception:
            total_mem_gb = None

        info.update(
            available=True,
            device_name=dev.name.decode() if isinstance(dev.name, bytes) else dev.name,
            compute_capability=dev.compute_capability,
            total_memory_GB=total_mem_gb,
        )

    except Exception as e:
        info["error"] = str(e)

    return info


## Plotting functions

def plot_raster(
    arr: np.ndarray,
    *,
    nodata_value=None,
    base_height: float = 3.0,
    title: str | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
) -> None:
    """
    Plot a 2D raster with:
      - black = min value
      - white = max value
      - NoData = transparent
      - automatic figsize based on array aspect ratio

    Parameters
    ----------
    arr : np.ndarray
        2D raster array (Y, X).
    nodata_value : optional
        Value to treat as NoData (masked). If None, NaN/Inf are masked.
    base_height : float
        Figure height in inches; width is computed automatically.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")

    ny, nx = arr.shape
    aspect = nx / ny
    figsize = (base_height * aspect, base_height)

    # Mask NoData
    if nodata_value is not None:
        data = np.ma.masked_equal(arr, nodata_value)
    else:
        data = np.ma.masked_invalid(arr)

    cmap = plt.cm.gray.copy()
    cmap.set_bad(color=(0, 0, 0, 0))  # transparent NoData

    plt.figure(figsize=figsize)
    plt.imshow(
        data,
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
    )

    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()
    
def force_raster_value_based_on_geometries(
    *,
    raster_path: Optional[str] = None,
    new_raster_path: Optional[str] = None,
    raster_data: Optional[np.ndarray] = None,
    transform=None,
    profile: Optional[dict] = None,
    geoms=None,
    target_value=None,
    protected_values: Optional[Sequence[int]] = None,
    mode: str = "set",
    all_touched: bool = True,
    return_array: bool = False,
    return_mask: bool = False,
):
    """
    Deterministic behavior:

    - If new_raster_path → ALWAYS writes file
    - If raster_data OR return_array → returns array
    - If return_mask:
        - returns mask
        - if returning array → returns (array, mask)
        - if NOT returning array → returns mask ONLY (early exit)
    """

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    if raster_data is None and raster_path is None:
        raise ValueError("Provide either raster_path or raster_data")

    if raster_data is not None and transform is None:
        raise ValueError("transform must be provided with raster_data")

    if geoms is None:
        raise ValueError("geoms must be provided")

    if protected_values is not None:
        protected_values = set(int(v) for v in protected_values)

    # ------------------------------------------------------------
    # Load raster
    # ------------------------------------------------------------
    if raster_data is not None:
        data = raster_data
        shape = data.shape
        nodata = profile.get("nodata") if profile else None

    else:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata
            transform = src.transform
            shape = data.shape

    # ------------------------------------------------------------
    # Rasterize geometries → mask
    # ------------------------------------------------------------
    geom_mask = rasterize(
        ((g, 1) for g in geoms if g is not None and not g.is_empty),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)

    # ------------------------------------------------------------
    # MASK-ONLY MODE (early exit)
    # ------------------------------------------------------------
    if return_mask and not return_array and new_raster_path is None:
        return geom_mask

    # ------------------------------------------------------------
    # Prepare overwrite mask
    # ------------------------------------------------------------
    if protected_values is None:
        overwrite_mask = geom_mask
    else:
        protected_arr = np.fromiter(protected_values, dtype=data.dtype)
        overwrite_mask = geom_mask & ~np.isin(data, protected_arr)

    if nodata is not None:
        overwrite_mask &= data != nodata

    # ------------------------------------------------------------
    # Apply operation
    # ------------------------------------------------------------
    out = data.copy()

    if np.any(overwrite_mask):

        if mode == "set":
            if target_value is None:
                raise ValueError("target_value cannot be None when mode='set'")
            out[overwrite_mask] = target_value

        elif mode == "add":
            out[overwrite_mask] += target_value

        elif mode == "multiply":
            out[overwrite_mask] *= target_value

        elif mode == "min":
            out[overwrite_mask] = np.minimum(out[overwrite_mask], target_value)

        elif mode == "max":
            out[overwrite_mask] = np.maximum(out[overwrite_mask], target_value)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    else:
        info("No pixels intersect geometries → raster unchanged")

    # ------------------------------------------------------------
    # WRITE FILE (always if requested)
    # ------------------------------------------------------------
    if new_raster_path is not None:
        with rasterio.open(new_raster_path, "w", **profile) as dst:
            dst.write(out, 1)

    # ------------------------------------------------------------
    # RETURN LOGIC
    # ------------------------------------------------------------
    if return_array:
        if return_mask:
            return out, geom_mask
        return out

    if return_mask:
        return geom_mask

    return new_raster_path


def _filter_by_class_priority_grid(
    pts: np.ndarray,
    lidar_class_priority,
    window_diameter: float
):
    """
    Grid-based class-priority filtering for LiDAR points.

    Parameters
    ----------
    pts : np.ndarray
        Shape (N, 4): [x, y, z, classification]
    lidar_class_priority : iterable
        Ordered list of classes (first = highest priority)
    window_diameter : float
        Neighborhood diameter in meters

    Returns
    -------
    np.ndarray
        Filtered points
    """

    if pts is None or len(pts) == 0:
        return pts

    # ------------------------------------------------------------
    # Priority mapping
    # ------------------------------------------------------------
    priority_map = {cls: i for i, cls in enumerate(lidar_class_priority)}
    pt_priority = np.array(
        [priority_map.get(c, np.inf) for c in pts[:, 3]],
        dtype=np.float32
    )

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # ------------------------------------------------------------
    # Grid definition (voxel size = radius)
    # ------------------------------------------------------------
    radius = window_diameter / 2.0
    cell_size = radius

    inv_cell = 1.0 / cell_size

    # Compute integer grid coordinates
    ix = np.floor(x * inv_cell).astype(np.int32)
    iy = np.floor(y * inv_cell).astype(np.int32)

    # ------------------------------------------------------------
    # Build spatial hash (grid)
    # ------------------------------------------------------------
    grid = defaultdict(list)

    for i in range(len(pts)):
        grid[(ix[i], iy[i])].append(i)

    # ------------------------------------------------------------
    # Neighbor offsets (3x3 neighborhood)
    # ------------------------------------------------------------
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1), ( 0, 0), ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    radius_sq = radius * radius

    keep_mask = np.ones(len(pts), dtype=bool)

    # ------------------------------------------------------------
    # Main filtering loop
    # ------------------------------------------------------------
    for i in range(len(pts)):

        if not keep_mask[i]:
            continue

        pi = pt_priority[i]
        xi, yi, zi = x[i], y[i], z[i]
        ci_x, ci_y = ix[i], iy[i]

        remove = False

        # Check neighboring cells
        for dx, dy in neighbor_offsets:
            cell = (ci_x + dx, ci_y + dy)

            if cell not in grid:
                continue

            for j in grid[cell]:

                if i == j:
                    continue

                # ------------------------------------------------
                # Priority condition (strictly higher priority)
                # ------------------------------------------------
                if pt_priority[j] >= pi:
                    continue

                # ------------------------------------------------
                # Height condition (neighbor must be above)
                # ------------------------------------------------
                if z[j] < zi:
                    continue

                # ------------------------------------------------
                # Distance check (within radius)
                # ------------------------------------------------
                dx_ = x[j] - xi
                dy_ = y[j] - yi

                if dx_ * dx_ + dy_ * dy_ > radius_sq:
                    continue

                # Found dominant neighbor → remove point
                remove = True
                break

            if remove:
                break

        if remove:
            keep_mask[i] = False

    return pts[keep_mask]


def draw_geometries_ui(
    *,
    title=(
        "Draw geometries\n"
        "LMB: add vertex | RMB: close ring | ENTER: commit feature | ESC: finish"
    ),
):
    """
    Draw Polygons and MultiPolygons interactively (PyCharm-safe).

    Returns
    -------
    geometries : list[Polygon | MultiPolygon]
    """

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)

    current_ring = []
    current_rings = []
    features = []

    ring_line, = ax.plot([], [], "-o", color="red")

    def redraw():
        if current_ring:
            xs, ys = zip(*current_ring)
            ring_line.set_data(xs, ys)
        else:
            ring_line.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal current_ring, current_rings

        if event.inaxes != ax:
            return

        # LEFT CLICK → add vertex
        if event.button == 1:
            current_ring.append((event.xdata, event.ydata))
            redraw()

        # RIGHT CLICK → close ring
        elif event.button == 3 and len(current_ring) >= 3:
            ring = Polygon(current_ring)
            current_rings.append(ring)

            xs, ys = ring.exterior.xy
            ax.plot(xs, ys, color="blue")

            current_ring = []
            redraw()

    def on_key(event):
        nonlocal current_ring, current_rings, features

        # ENTER → commit feature
        if event.key == "enter" and current_rings:
            if len(current_rings) == 1:
                features.append(current_rings[0])
            else:
                features.append(MultiPolygon(current_rings))

            current_rings = []
            redraw()

        # ESC → finish
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    return features


def rasterize_polygon_mask(polygons_gdf, reference_raster):
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        shape = (src.height, src.width)

    mask = rasterize(
        ((geom, 1) for geom in polygons_gdf.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    return mask.astype(bool)

def _ensure_same_crs(gdf_left: gpd.GeoDataFrame,
                     gdf_right: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if gdf_left.crs != gdf_right.crs:
        gdf_right = gdf_right.to_crs(gdf_left.crs)
    return gdf_left, gdf_right

def _fetch_html(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def _download_file(dir, url, file=None, file_format=None):
    if file is None:
        if file_format is None:
            file_format = url.split('/')[-1].split('.')[-1]
        file = f"tmp_{hashlib.md5(url.encode()).hexdigest()}.{file_format}"
    
    file_path = os.path.join(dir, file)
    if not os.path.exists(file_path):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(file_path, 'wb') as archivo:
                archivo.write(response.content)
            info(f"Downloaded: {file}")
        except requests.RequestException as e:
            error(f"Failed to download {url}: {e}")
            raise

def _unzip_directory(zip_directory, unzip_directory):
    for file in os.listdir(zip_directory):
        if file.endswith(".zip"):
            zip_path = os.path.join(zip_directory, file)
            try:
                with ZipFile(zip_path, 'r') as zip_file:
                    zip_file.extractall(unzip_directory)
                info(f"Extracted: {file}")
            except BadZipFile:
                warning(f"Bad zip file, removing: {file}")
                os.remove(zip_path)

def _delete_all_files_in_directory(directory: str, delete_subdirectories: bool = False):
    """
    Deletes all files inside the specified directory.

    Parameters:
    - directory (str): The path to the directory where files will be deleted.
    - delete_subdirectories (bool): If True, also deletes all subdirectories within the directory.
                                    If False, subdirectories are left intact.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Remove the file or symbolic link
                elif os.path.isdir(file_path) and delete_subdirectories:
                    shutil.rmtree(file_path)  # Remove the directory and its contents if flag is set
            except Exception as e:
                warning(f'Failed to delete {file_path}. Reason: {e}')

def _validate_and_fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validates and fixes invalid geometries in a GeoDataFrame.

    Parameters:
    - gdf (gpd.GeoDataFrame): The GeoDataFrame with geometries to validate and fix.

    Returns:
    - gpd.GeoDataFrame: The GeoDataFrame with valid geometries.
    """
    # Fix invalid geometries by applying buffer(0) or using make_valid
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
    return gdf


def _untar_directory(tar_directory, untar_directory, files_to_extract):

    # Create the extraction directory if it doesn't exist, or remove all the files
    if not os.path.exists(untar_directory):
        os.makedirs(untar_directory)
    else:
        shutil.rmtree(untar_directory)
        os.makedirs(untar_directory)

    for file in os.listdir(tar_directory):

        try:
            # Determine the mode based on file extension
            if tar_directory.endswith(".tar.gz") or tar_directory.endswith(".tgz"):
                mode = 'r:gz'
            elif tar_directory.endswith(".tar.bz2") or tar_directory.endswith(".tbz"):
                mode = 'r:bz2'
            else:
                mode = 'r'

            # Open the tar file
            with tarfile.open(f"{tar_directory}{file}", mode) as tar:

                # Initialize a counter to create unique filenames
                counter = 0

                # Find and extract members that match the pattern
                for member in tar.getmembers():

                    newfile = file.replace(file.split(".")[-1], member.name.split(".")[-1])

                    if fnmatch.fnmatch(member.name, files_to_extract):

                        # Define the full path for the extracted file
                        extract_full_path = os.path.join(untar_directory, f"{'' if counter == 0 else str(counter)}{newfile}")

                        # Extract the file content to a temporary location
                        extracted_file = tar.extractfile(member)
                        with open(extract_full_path, 'wb') as out_file:
                            out_file.write(extracted_file.read())

                        counter += 1

        except Exception:
            os.remove(f"{tar_directory}{file}")

def plot_points_and_footprint(points3d, polygon):

    import numpy as np
    import plotly.graph_objects as go

    points3d = np.asarray(points3d)
    x, y, z = points3d.T

    coords = np.array(polygon.exterior.coords)
    coords = coords[:, :2]

    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    xs = coords[:, 0]
    ys = coords[:, 1]

    # Lift contour well above roof
    zs = np.full(xs.shape, z.max() + 3.0)

    fig = go.Figure()

    # Plot points smaller
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=2),
        name="Points"
    ))

    # Plot contour
    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="red", width=20),
        name="Footprint"
    ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        title="Points + Elevated Footprint"
    )

    fig.show()

    return fig


def plot_plane_debug_pyvista(points3d, polygon, tilt, azimuth):
    """
    Visualize:
    - 3D point cloud
    - Polygon footprint
    - Plane reconstructed from tilt/azimuth
    - Normal vector
    - Coordinate axes
    """

    import numpy as np
    import pyvista as pv

    points3d = np.asarray(points3d)

    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("points3d must be (N,3)")

    if polygon is None or polygon.is_empty:
        info("Polygon invalid.")
        return

    # -----------------------------------------
    # Convert tilt/azimuth → normal (downslope)
    # -----------------------------------------

    tilt_rad = np.radians(tilt)
    az_rad = np.radians(azimuth)

    # downslope direction
    nx = np.sin(tilt_rad) * np.sin(az_rad)
    ny = np.sin(tilt_rad) * np.cos(az_rad)
    nz = np.cos(tilt_rad)

    normal = np.array([nx, ny, nz])
    normal /= np.linalg.norm(normal)

    # -----------------------------------------
    # Anchor plane at centroid
    # -----------------------------------------

    centroid = points3d.mean(axis=0)
    x0, y0, z0 = centroid

    def plane_z(x, y):
        return z0 - (normal[0]*(x - x0) + normal[1]*(y - y0)) / normal[2]

    # -----------------------------------------
    # Create plane grid over polygon bbox
    # -----------------------------------------

    minx, miny, maxx, maxy = polygon.bounds

    resolution = 50
    xx = np.linspace(minx, maxx, resolution)
    yy = np.linspace(miny, maxy, resolution)
    xx, yy = np.meshgrid(xx, yy)

    zz = plane_z(xx, yy)

    grid_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    plane_surface = pv.StructuredGrid(
        xx,
        yy,
        zz
    )

    # -----------------------------------------
    # Create polygon polyline
    # -----------------------------------------

    coords = np.array(polygon.exterior.coords)
    coords = coords[:, :2]

    z_lift = points3d[:,2].max() + 0.2
    footprint_pts = np.column_stack([
        coords[:,0],
        coords[:,1],
        np.full(len(coords), z_lift)
    ])

    n = len(footprint_pts)
    lines = np.hstack([[n], np.arange(n)])
    footprint = pv.PolyData(footprint_pts)
    footprint.lines = lines

    # -----------------------------------------
    # Create point cloud
    # -----------------------------------------

    cloud = pv.PolyData(points3d)

    # -----------------------------------------
    # Create normal vector arrow
    # -----------------------------------------

    arrow = pv.Arrow(
        start=centroid,
        direction=normal,
        scale=2.0
    )

    # -----------------------------------------
    # Plot
    # -----------------------------------------

    plotter = pv.Plotter()
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=4)
    plotter.add_mesh(plane_surface, color="lightblue", opacity=0.6)
    plotter.add_mesh(footprint, color="red", line_width=4)
    plotter.add_mesh(arrow, color="green")

    plotter.add_axes()
    plotter.show_grid()

    plotter.show()



def analyze_polygon_vs_points(points3d, polygon):
    """
    Analyze scale and overlap between point cloud and polygon footprint.
    """

    import numpy as np

    points3d = np.asarray(points3d)

    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("points3d must be (N,3)")

    if polygon is None or polygon.is_empty:
        info("Polygon is None or empty.")
        return

    # -------------------------------
    # Point cloud extent
    # -------------------------------

    x_min, x_max = points3d[:, 0].min(), points3d[:, 0].max()
    y_min, y_max = points3d[:, 1].min(), points3d[:, 1].max()

    pc_width = x_max - x_min
    pc_height = y_max - y_min
    pc_area_bbox = pc_width * pc_height

    # -------------------------------
    # Polygon extent
    # -------------------------------

    p_minx, p_miny, p_maxx, p_maxy = polygon.bounds
    poly_width = p_maxx - p_minx
    poly_height = p_maxy - p_miny
    poly_bbox_area = poly_width * poly_height
    poly_area = polygon.area

    # -------------------------------
    # Overlap check
    # -------------------------------

    overlap_x = not (p_maxx < x_min or p_minx > x_max)
    overlap_y = not (p_maxy < y_min or p_miny > y_max)
    overlaps = overlap_x and overlap_y

    # -------------------------------
    # Relative size ratio
    # -------------------------------

    if pc_area_bbox > 0:
        ratio = poly_bbox_area / pc_area_bbox
    else:
        ratio = 0

    # -------------------------------
    # Report
    # -------------------------------

    info("----- Point Cloud Extent -----")
    info(f"X range: {x_min:.3f} → {x_max:.3f} ({pc_width:.3f} m)")
    info(f"Y range: {y_min:.3f} → {y_max:.3f} ({pc_height:.3f} m)")
    info(f"Bounding box area: {pc_area_bbox:.3f} m²")

    info("\n----- Polygon Extent -----")
    info(f"X range: {p_minx:.3f} → {p_maxx:.3f} ({poly_width:.3f} m)")
    info(f"Y range: {p_miny:.3f} → {p_maxy:.3f} ({poly_height:.3f} m)")
    info(f"Polygon area: {poly_area:.3f} m²")
    info(f"Polygon bbox area: {poly_bbox_area:.3f} m²")

    info("\n----- Overlap Check -----")
    info(f"Overlaps point cloud XY: {overlaps}")

    info("\n----- Relative Size -----")
    info(f"Polygon bbox / Point cloud bbox ratio: {ratio:.6f}")

    if ratio < 0.01:
        warning("Polygon is very small relative to scene → will look invisible in 3D.")

    if not overlaps:
        warning("Polygon does NOT overlap point cloud in XY.")


def _write_array_to_tiff(
    array: np.ndarray,
    out_path: str,
    transform,
    crs,
    nodata=None,
    dtype=None,
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 256,
):
    """
    Write a 2D NumPy array to a GeoTIFF file.

    Parameters
    ----------
    array : np.ndarray
        2D array (height, width)
    out_path : str
        Output GeoTIFF path
    transform : rasterio.Affine
        Affine transform
    crs : rasterio CRS or EPSG int
        Coordinate reference system
    nodata : optional
        Nodata value
    dtype : optional
        Force output dtype (default: array.dtype)
    compress : str
        Compression method (default: 'deflate')
    tiled : bool
        Whether to tile the raster
    blocksize : int
        Block size for tiling
    """

    if array.ndim != 2:
        raise ValueError("Only 2D arrays supported")

    if dtype is None:
        dtype = array.dtype

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "tiled": tiled,
        "blockxsize": blocksize if tiled else None,
        "blockysize": blocksize if tiled else None,
        "nodata": nodata,
    }

    # Remove None values from profile
    profile = {k: v for k, v in profile.items() if v is not None}

    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

    return out_path

class RAMLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, label: str):
        mem = psutil.virtual_memory()

        with open(self.log_file, "a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{label} | "
                f"used={mem.used / 1e9:.2f}GB | "
                f"avail={mem.available / 1e9:.2f}GB | "
                f"percent={mem.percent}%\n"
            )