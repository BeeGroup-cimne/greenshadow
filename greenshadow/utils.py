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
from typing import List, Union, Dict, Optional, Tuple, Any, Sequence
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely import wkt as shapely_wkt
from shapely.validation import explain_validity
from shapely.ops import transform
from collections import Counter
import pyproj
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
import numpy as np
import datetime as dt
import pandas as pd

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
    from tqdm import tqdm
    
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
        print(f"All {len(existing)} files already exist")
        return existing
    
    print(f"Downloading {len(to_download)} files (+ {len(existing)} existing)")
    
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
        print(f"Download errors ({len(errors)}):")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors)-5} more")
    
    all_paths = existing + downloaded
    print(f"Total files available: {len(all_paths)}")
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

# Configure GDAL to use exceptions for better error handling
def _init_gdal():
    """Initialize GDAL with exception handling."""
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

def _nearest_fill(mask: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Fill masked values in arr by nearest valid neighbor (fast EDT indices)."""
    _, (iy, ix) = ndi.distance_transform_edt(mask, return_indices=True)
    out = arr.copy()
    out[mask] = arr[iy[mask], ix[mask]]
    return out

def _build_raster_profile(
    width: int,
    height: int,
    transform: Affine,
    crs,
    nodata: float,
    dtype
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
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
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

# ---------------- Array utilities ----------------

def _nearest_fill(mask: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Fill masked values in arr by nearest valid neighbor."""
    _, (iy, ix) = distance_transform_edt(mask, return_indices=True)
    out = arr.copy()
    out[mask] = arr[iy[mask], ix[mask]]
    return out


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
            print(f"Python PDAL failed: {e}")
    
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
    sys.stdout.write(msg)
    sys.stdout.flush()

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
