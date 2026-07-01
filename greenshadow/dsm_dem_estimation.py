# STRtree moved between modules across shapely versions
try:
    from shapely.strtree import STRtree  # shapely 1.x and 2.x
except Exception:  # pragma: no cover
    try:
        from shapely import STRtree       # shapely 2.x alt path
    except Exception:
        STRtree = None

# -*- coding: utf-8 -*-
import os
import warnings
from typing import Union, Optional, Tuple, Any, Sequence
import geopandas as gpd
import rasterio
from tqdm import tqdm
import json
import shapely.geometry as sg
from rasterio.features import rasterize
from pathlib import Path
import hashlib
from rasterio.transform import Affine
from shapely.geometry import mapping
from shapely.vectorized import contains as contains_xy
from rasterio.crs import CRS
from pyproj import Transformer
from shapely.ops import transform as shp_transform
from scipy.interpolate import griddata
from scipy.ndimage import (
    distance_transform_edt, convolve, label, binary_dilation,
    uniform_filter, gaussian_filter, median_filter, grey_opening,
    grey_closing, sobel
)
from skimage.restoration import inpaint, denoise_bilateral
import numpy as np
import laspy
from numba import njit, prange
try:
    from .logging_utils import error
    from .utils import _build_raster_profile, force_raster_value_based_on_geometries, RAMLogger, _write_raster_in_blocks, _filter_by_class_priority_grid
except ImportError:
    from greenshadow.logging_utils import error
    from greenshadow.utils import _build_raster_profile, force_raster_value_based_on_geometries, RAMLogger, _write_raster_in_blocks, _filter_by_class_priority_grid
from scipy.spatial import cKDTree
import psutil
from numba import set_num_threads, get_num_threads
import gc
import math
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock
from rasterio.windows import Window
from rasterio.transform import from_origin
from shapely.geometry import box as shp_box

# Set number of threads when Numba parallel execution
set_num_threads(int(np.ceil(get_num_threads()/3)))

def _hybrid_dsm_fill(arr, nodata, building_mask):

    filled = arr.copy()

    mask = (arr == nodata) & building_mask

    if not np.any(mask):
        return arr

    labeled, n = label(mask)
    counts = np.bincount(labeled.ravel())

    for hid in range(1, n + 1):

        hole = labeled == hid
        size = counts[hid]

        # ------------------------------
        # SMALL HOLES
        # ------------------------------
        if size <= 4:

            med = median_filter(filled, size=3)
            filled[hole] = med[hole]

        # ------------------------------
        # MEDIUM HOLES
        # ------------------------------
        elif size <= 200:

            tmp = _harmonic_fill(filled, nodata, hole)
            filled[hole] = tmp[hole]

        # ------------------------------
        # LARGE HOLES
        # ------------------------------
        else:

            boundary = binary_dilation(hole) & (~hole)

            by, bx = np.where(boundary)
            bz = filled[boundary]

            valid = (bz != nodata) & np.isfinite(bz)

            if np.sum(valid) < 3:
                continue

            bx = bx[valid]
            by = by[valid]
            bz = bz[valid]

            hy, hx = np.where(hole)

            # distance weighted interpolation
            dx = hx[:, None] - bx
            dy = hy[:, None] - by

            dist = np.sqrt(dx*dx + dy*dy) + 1e-6

            w = 1 / dist

            z = np.sum(w * bz, axis=1) / np.sum(w, axis=1)

            filled[hy, hx] = z

    return filled

def _dw_plane_fill(arr, nodata, mask, k=8, max_boundary_points=500):

    filled = arr.copy()

    labeled, n = label(mask)

    for hid in range(1, n + 1):

        hole = labeled == hid

        # ---------------------------------------------
        # Find boundary pixels
        # ---------------------------------------------
        boundary = binary_dilation(hole) & (~hole)

        by, bx = np.where(boundary)
        bz = arr[boundary]

        valid = (bz != nodata) & np.isfinite(bz)

        if np.sum(valid) < 3:
            continue

        bx = bx[valid]
        by = by[valid]
        bz = bz[valid]

        # ---------------------------------------------
        # Subsample boundary for speed
        # ---------------------------------------------
        if bx.size > max_boundary_points:
            idx = np.random.choice(bx.size, max_boundary_points, replace=False)
            bx, by, bz = bx[idx], by[idx], bz[idx]

        # ---------------------------------------------
        # Build KD-tree from boundary samples
        # ---------------------------------------------
        tree = cKDTree(np.column_stack((bx, by)))

        # ---------------------------------------------
        # Hole pixels
        # ---------------------------------------------
        hy, hx = np.where(hole)

        # Query nearest boundary points
        k_eff = min(k, bx.size)
        dist, idx = tree.query(np.column_stack((hx, hy)), k=k_eff)

        # Ensure 2D arrays
        if k_eff == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        # ---------------------------------------------
        # Distance weighted interpolation
        # ---------------------------------------------
        weights = 1.0 / (dist + 1e-6)

        z_pred = np.sum(weights * bz[idx], axis=1) / np.sum(weights, axis=1)

        filled[hy, hx] = z_pred

    return filled


def _plane_fill_roofs_fast(arr, nodata, building_mask):

    filled = arr.copy()

    # holes only inside buildings
    mask = (arr == nodata) & building_mask

    if not np.any(mask):
        return arr

    labeled, n = label(building_mask)

    for bid in range(1, n + 1):

        roof = labeled == bid

        # valid pixels of this roof
        valid = roof & (arr != nodata)

        if np.sum(valid) < 3:
            continue

        ry, rx = np.where(valid)
        rz = arr[valid]

        # plane fit
        A = np.c_[rx, ry, np.ones_like(rx)]
        coef, *_ = np.linalg.lstsq(A, rz, rcond=None)

        a, b, c = coef

        # pixels to fill
        hole = roof & mask
        hy, hx = np.where(hole)

        if hy.size == 0:
            continue

        z = a * hx + b * hy + c

        filled[hy, hx] = z

    return filled

def _nearest_fill(
    mask: np.ndarray,
    arr: np.ndarray,
    aoi_mask: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
    *,
    dist_dtype=np.int32,
) -> np.ndarray:
    """
    In-place capable nearest-neighbor gap filling using EDT indices.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where True indicates gaps to fill
    arr : np.ndarray
        Input array
    aoi_mask : Optional[np.ndarray]
        AOI restriction mask
    out : Optional[np.ndarray]
        Output buffer (in-place if provided)
    dist_dtype : dtype
        dtype for EDT indices (int32 is usually enough and saves memory)

    Returns
    -------
    np.ndarray
        Filled array (same as `out` if provided)
    """

    # --------------------------------------------------
    # Effective mask
    # --------------------------------------------------
    if aoi_mask is not None:
        effective_mask = mask & aoi_mask
        valid_data_mask = (~mask) & aoi_mask
    else:
        effective_mask = mask
        valid_data_mask = ~mask

    if not np.any(effective_mask):
        # No fill needed
        return arr if out is None else out

    if not np.any(valid_data_mask):
        # No valid data to interpolate from
        if out is None:
            return arr.copy()
        else:
            out[:] = arr
            return out

    # --------------------------------------------------
    # Prepare output
    # --------------------------------------------------
    if out is None:
        out = arr.copy()
    else:
        # preserve original values
        if out is not arr:
            out[:] = arr

    # --------------------------------------------------
    # Distance transform (CRITICAL MEMORY POINT)
    # --------------------------------------------------
    # NOTE: ~valid_data_mask → distance to nearest valid pixel
    _, (iy, ix) = distance_transform_edt(
        ~valid_data_mask,
        return_indices=True
    )

    # Optional: cast to reduce memory footprint
    if dist_dtype is not None:
        iy = iy.astype(dist_dtype, copy=False)
        ix = ix.astype(dist_dtype, copy=False)

    # --------------------------------------------------
    # In-place fill
    # --------------------------------------------------
    out[effective_mask] = arr[iy[effective_mask], ix[effective_mask]]

    return out

def _plane_fill_roof(arr, nodata, mask):

    import numpy as np
    from scipy.ndimage import binary_dilation, label

    filled = arr.copy()

    labeled, n = label(mask)

    for hid in range(1, n + 1):

        hole = labeled == hid

        # boundary pixels
        boundary = binary_dilation(hole) & (~hole)

        by, bx = np.where(boundary)
        bz = arr[boundary]

        valid = (bz != nodata) & np.isfinite(bz)

        if np.sum(valid) < 3:
            continue

        bx = bx[valid]
        by = by[valid]
        bz = bz[valid]

        # -------------------------------------
        # Fit plane z = ax + by + c
        # -------------------------------------
        A = np.c_[bx, by, np.ones_like(bx)]
        coef, *_ = np.linalg.lstsq(A, bz, rcond=None)

        a, b, c = coef

        # -------------------------------------
        # Fill hole pixels
        # -------------------------------------
        hy, hx = np.where(hole)

        z_pred = a * hx + b * hy + c

        filled[hy, hx] = z_pred

    return filled

def _nearest_fill_chunked(
    mask: np.ndarray,
    arr: np.ndarray,
    max_ram_gb: float = 8.0,
    aoi_mask: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Memory-aware nearest-neighbor gap filling with in-place support.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where True indicates gaps to fill
    arr : np.ndarray
        Input array
    max_ram_gb : float
        Maximum RAM budget
    aoi_mask : Optional[np.ndarray]
        AOI restriction mask
    out : Optional[np.ndarray]
        Output buffer (in-place if provided)

    Returns
    -------
    np.ndarray
        Filled array (same as `out` if provided)
    """

    # --------------------------------------------------
    # Prepare output buffer
    # --------------------------------------------------
    if out is None:
        out = arr.copy()
    else:
        if out is not arr:
            out[:] = arr

    height, width = arr.shape

    # --------------------------------------------------
    # Estimate memory (EDT dominates)
    # --------------------------------------------------
    arr_size_gb = arr.nbytes / (1024**3)
    mask_size_gb = mask.nbytes / (1024**3)

    edt_memory_factor = 5.0
    required_memory_gb = (arr_size_gb + mask_size_gb) * edt_memory_factor

    # --------------------------------------------------
    # FULL processing (no chunking)
    # --------------------------------------------------
    if required_memory_gb <= max_ram_gb * 0.5:
        _nearest_fill(
            mask,
            arr,
            aoi_mask=aoi_mask,
            out=out,   # 🔥 in-place
        )
        return out

    # --------------------------------------------------
    # CHUNKED processing
    # --------------------------------------------------
    target_chunk_memory_gb = max_ram_gb * 0.6
    pixels_per_gb = (height * width) / required_memory_gb
    target_chunk_pixels = int(pixels_per_gb * target_chunk_memory_gb)

    chunk_side = int(np.sqrt(target_chunk_pixels))
    chunk_side = max(64, min(chunk_side, min(height, width)))

    overlap = 32
    step = chunk_side - overlap

    print(f"[nearest_fill_chunked] chunk={chunk_side} overlap={overlap}")

    # --------------------------------------------------
    # Iterate chunks
    # --------------------------------------------------
    for y_start in range(0, height, step):
        for x_start in range(0, width, step):

            y_end = min(y_start + chunk_side, height)
            x_end = min(x_start + chunk_side, width)

            # Views (no copies)
            chunk_mask = mask[y_start:y_end, x_start:x_end]
            chunk_arr = out[y_start:y_end, x_start:x_end]  # 🔥 operate directly on output

            chunk_aoi_mask = None
            if aoi_mask is not None:
                chunk_aoi_mask = aoi_mask[y_start:y_end, x_start:x_end]

            # Effective mask
            if chunk_aoi_mask is not None:
                effective_mask = chunk_mask & chunk_aoi_mask
            else:
                effective_mask = chunk_mask

            if not np.any(effective_mask):
                continue

            # --------------------------------------------------
            # IN-PLACE fill on the chunk view
            # --------------------------------------------------
            _nearest_fill(
                effective_mask,
                chunk_arr,
                aoi_mask=chunk_aoi_mask,
                out=chunk_arr,   # 🔥 true in-place
            )

            # --------------------------------------------------
            # Overlap trimming (avoid edge artifacts)
            # --------------------------------------------------
            uy0 = overlap // 2 if y_start > 0 else 0
            ux0 = overlap // 2 if x_start > 0 else 0

            uy1 = (y_end - y_start) - (overlap // 2 if y_end < height else 0)
            ux1 = (x_end - x_start) - (overlap // 2 if x_end < width else 0)

            # NOTE:
            # We already wrote in-place, so we only need to restore edges
            # from original arr where overlap should not dominate

            if uy0 > 0:
                chunk_arr[:uy0, :] = arr[y_start:y_start + uy0, x_start:x_end]
            if ux0 > 0:
                chunk_arr[:, :ux0] = arr[y_start:y_end, x_start:x_start + ux0]

            if uy1 < (y_end - y_start):
                chunk_arr[uy1:, :] = arr[y_start + uy1:y_end, x_start:x_end]
            if ux1 < (x_end - x_start):
                chunk_arr[:, ux1:] = arr[y_start:y_end, x_start + ux1:x_end]

    return out

def _infer_bounds_from_lidar(paths: Sequence[str], classification: Sequence[int]) -> Tuple[float, float, float, float]:
    """Infer bounds from LiDAR files by reading headers."""
    bounds_list = []
    for p in paths:
        try:
            with laspy.open(p) as lf:
                hdr = lf.header
                if hasattr(hdr, 'x_min'):
                    bounds_list.append((hdr.x_min, hdr.y_min, hdr.x_max, hdr.y_max))
                else:
                    # Fallback: read some points to estimate bounds
                    pts = lf.read().points[:1000]  # Sample first 1000 points
                    if len(pts) > 0:
                        x, y = pts['X'], pts['Y']
                        bounds_list.append((x.min(), y.min(), x.max(), y.max()))
        except Exception as e:
            warnings.warn(f"Could not read bounds from {p}: {e}")
            continue
    
    if not bounds_list:
        raise ValueError("Could not determine bounds from any LiDAR file")
    
    # Compute union of all bounds
    minx = min(b[0] for b in bounds_list)
    miny = min(b[1] for b in bounds_list)
    maxx = max(b[2] for b in bounds_list)
    maxy = max(b[3] for b in bounds_list)
    
    return minx, miny, maxx, maxy


def _aligned_grid(minx: float, miny: float, maxx: float, maxy: float, res: float):
    """Create aligned grid parameters for rasterization."""
    # Snap bounds to grid
    minx_snap = np.floor(minx / res) * res
    miny_snap = np.floor(miny / res) * res
    maxx_snap = np.ceil(maxx / res) * res
    maxy_snap = np.ceil(maxy / res) * res
    
    # Calculate grid dimensions
    width = int(np.round((maxx_snap - minx_snap) / res))
    height = int(np.round((maxy_snap - miny_snap) / res))
    
    # Create affine transform
    transform = Affine.translation(minx_snap, maxy_snap) * Affine.scale(res, -res)
    
    return width, height, transform

def _count_invalid_values(arr, name="array"):

    n_nan = np.isnan(arr).sum()
    n_posinf = np.isposinf(arr).sum()
    n_neginf = np.isneginf(arr).sum()

    total = arr.size

    print(
        f"{name}: "
        f"NaN={n_nan} "
        f"+Inf={n_posinf} "
        f"-Inf={n_neginf} "
        f"({(n_nan+n_posinf+n_neginf)/total:.6%} invalid)"
    )

def _rasterize_agg(
    laz_path: Optional[str] = None,
    laz_points: Optional[np.ndarray] = None,
    width: int = None,
    height: int = None,
    transform: Affine = None,
    classification: Sequence[int] = (0,),
    unclassified_classes: Sequence[int] = (1,),
    vegetation_classes: Sequence[int] = (3, 4, 5),  # By default correspond to ICGC classes
    building_classes: Sequence[int] = (6,),  # By default correspond to ICGC classes
    terrain_classes: Sequence[int] = (2, 8),  # By default correspond to ICGC classes
    lidar_class_priority: Optional[Sequence[int]] = None,
    lidar_class_window_diameter: Optional[float] = 0.5,
    min_class_fraction: float = 0.10,
    aggregation: str = "max",
    nodata: float = -9999.0,
    aoi: Optional[Any] = None,
    aoi_buffer_in_meters: float = 0.0,
    return_classification: bool = False,
    filter_raster: Optional[rasterio.io.DatasetReader] = None,
    filter_raster_band: int = 1,
    filter_points_from_class: Optional[tuple] = None,
    filter_mode: str = "exclude",
    cache_path: Optional[str] = None,
    CLASS_NODATA: int = 255,
    epsg: int = 25831,
    reclassify_unclassified: bool = True,
    drop_unclassified_after_reclassify: bool = True,
    unclassified_global_fraction_threshold: float = 0.20,
    min_unclassified_pixel_fraction: float = 0.10,
    reclassify_neighborhood_radius: float = 1.5,
    reclassify_hag_min: float = 1.5,
    reclassify_hag_terrain_max: float = 0.3,
    reclassify_planar_threshold: float = 0.5,
    reclassify_rough_threshold: float = 0.2,
    reclassify_target_building: int = 6,
    reclassify_target_vegetation: int = 4,
    reclassify_target_terrain: int = 2,
):
    """
    Rasterize LiDAR points with aggregation, AOI filtering and raster-based filtering.
    """

    if cache_path is not None:
        def _cache_hash(params):
            s = json.dumps(params, sort_keys=True, default=str)
            return hashlib.sha1(s.encode()).hexdigest()[:16]
        
        params = dict(
            width=width,
            height=height,
            transform=tuple(transform),
            classification=classification,
            vegetation_classes=vegetation_classes,
            building_classes=building_classes,
            terrain_classes=terrain_classes,
            min_class_fraction=min_class_fraction,
            aggregation=aggregation,
            nodata=nodata,
            aoi_bounds=aoi.bounds if aoi is not None else None,
            aoi_buffer_in_meters=aoi_buffer_in_meters,
            filter_points_from_class=filter_points_from_class,
            filter_mode=filter_mode,
            epsg=25831,
        )
        param_hash = _cache_hash(params)
        base = os.path.splitext(cache_path)[0]
        cache_path = f"{base}_{param_hash}.tif"

    profile = _build_raster_profile(
        width=width,
        height=height,
        transform=transform,
        crs=CRS.from_epsg(epsg),
        nodata=nodata,
        dtype=np.float32,
        compress="lzw"
    )

    # ------------------------------------------------------------
    # Cache loading
    # ------------------------------------------------------------
    if cache_path is not None and os.path.exists(cache_path):

        with rasterio.open(cache_path) as src:
            raster = src.read(1)

            if return_classification and src.count >= 2:
                class_raster = src.read(2).astype(np.uint8)
            else:
                class_raster = None

            profile = src.profile

        if return_classification:
            return raster, class_raster, profile
        else:
            return raster, profile

    # ------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------
    if laz_path is None and laz_points is None:
        raise ValueError("Either laz_path or laz_points must be provided.")

    if laz_path is not None and laz_points is not None:
        raise ValueError("Provide only one of laz_path or laz_points.")

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    if laz_path is not None:

        import laspy

        with laspy.open(laz_path) as lf:
            las = lf.read()

        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
        class_vals = np.asarray(las.classification, dtype=np.uint8)

    else:
        # laz_points expected shape: Nx3 or Nx4
        x = laz_points[:, 0].astype(np.float32)
        y = laz_points[:, 1].astype(np.float32)
        z = laz_points[:, 2].astype(np.float32)

        if laz_points.shape[1] >= 4:
            class_vals = laz_points[:, 3].astype(np.uint8)
        else:
            class_vals = None

    # ------------------------------------------------------------
    # Classification filter
    # ------------------------------------------------------------
    if classification and class_vals is not None:
        cls_mask = np.isin(class_vals, classification)
        if not np.any(cls_mask):
            raster = np.full((height, width), nodata, np.float32)
            profile = _build_raster_profile(width, height, transform, None, nodata, raster.dtype)
            if return_classification:
                return raster, np.full((height, width), 1, np.uint8), profile
            return raster, profile

        x, y, z = x[cls_mask], y[cls_mask], z[cls_mask]
        class_vals = class_vals[cls_mask]

    # ------------------------------------------------------------
    # AOI filtering
    # ------------------------------------------------------------
    if aoi is not None and x.size > 0:
        try:
            geom = aoi.buffer(aoi_buffer_in_meters) if aoi_buffer_in_meters > 0 else aoi
            minx, miny, maxx, maxy = geom.bounds

            bbox = (
                    (x >= minx) & (x <= maxx) &
                    (y >= miny) & (y <= maxy)
            )

            if np.any(bbox):
                inside = contains_xy(geom, x[bbox], y[bbox])
                keep = np.zeros_like(x, bool)
                keep[np.where(bbox)[0]] = inside
                x, y, z = x[keep], y[keep], z[keep]
                if class_vals is not None:
                    class_vals = class_vals[keep]
            else:
                x = y = z = np.empty(0)
                if class_vals is not None:
                    class_vals = np.empty(0)

        except Exception as e:
            warnings.warn(f"AOI filtering failed: {e}")

    # ------------------------------------------------------------
    # Raster-based filtering
    # ------------------------------------------------------------
    if filter_raster is not None and filter_points_from_class and x.size:
        rows, cols = rasterio.transform.rowcol(
            filter_raster.transform, x, y, op=np.floor
        )
        # Convert to integers for array indexing
        rows = rows.astype(int)
        cols = cols.astype(int)
        
        valid = (
            (rows >= 0) & (rows < filter_raster.height) &
            (cols >= 0) & (cols < filter_raster.width)
        )

        in_range = np.zeros_like(x, bool)
        if np.any(valid):
            band = filter_raster.read(filter_raster_band)
            in_range[valid] = np.isin(
                band[rows[valid], cols[valid]],
                filter_points_from_class
            )

        keep = ~in_range if filter_mode == "exclude" else in_range
        x, y, z, class_vals = x[keep], y[keep], z[keep], class_vals[keep]

    # ------------------------------------------------------------
    # Geometric reclassification of unclassified points,
    # then drop whatever stayed unclassified.
    # ------------------------------------------------------------
    if (
            reclassify_unclassified
            and class_vals is not None
            and x.size > 0
    ):
        from reclassify_unclassified import reclassify_unclassified_points

        class_vals, reclass_info = reclassify_unclassified_points(
            x, y, z, class_vals,
            unclassified_classes=unclassified_classes,
            terrain_classes=terrain_classes,
            target_building_class=reclassify_target_building,
            target_vegetation_class=reclassify_target_vegetation,
            target_terrain_class=reclassify_target_terrain,
            transform=transform,
            width=width,
            height=height,
            unclassified_global_fraction_threshold=unclassified_global_fraction_threshold,
            min_unclassified_pixel_fraction=min_unclassified_pixel_fraction,
            neighborhood_radius=reclassify_neighborhood_radius,
            hag_min=reclassify_hag_min,
            hag_terrain_max=reclassify_hag_terrain_max,
            planar_threshold=reclassify_planar_threshold,
            rough_threshold=reclassify_rough_threshold,
        )

        if reclass_info["ran"]:
            import logging
            logging.info(
                "Reclassified tile (frac=%.2f): +%d building, +%d vegetation, +%d terrain, %d kept unclassified",
                reclass_info["global_unclassified_fraction"],
                reclass_info["n_reclassified_to_building"],
                reclass_info["n_reclassified_to_vegetation"],
                reclass_info["n_reclassified_to_terrain"],
                reclass_info["n_left_unclassified"],
            )

    # ------------------------------------------------------------
    # Drop any points still in unclassified_classes.
    # Runs whether or not reclassification triggered, so tiles below
    # the global fraction gate also get cleaned.
    # ------------------------------------------------------------
    if (
            drop_unclassified_after_reclassify
            and class_vals is not None
            and x.size > 0
    ):
        keep = ~np.isin(class_vals, np.asarray(unclassified_classes, dtype=np.uint8))

        n_before = x.size
        x = x[keep]
        y = y[keep]
        z = z[keep]
        class_vals = class_vals[keep]
        n_after = x.size

        if n_after < n_before:
            import logging
            logging.info(
                "Dropped %d remaining unclassified points (%.2f%% of tile).",
                n_before - n_after,
                100.0 * (n_before - n_after) / n_before,
            )

        # Bail out cleanly if nothing survived
        if x.size == 0:
            raster = np.full((height, width), nodata, np.float32)
            profile = _build_raster_profile(
                width, height, transform, None, nodata, raster.dtype
            )
            if return_classification:
                return raster, np.full((height, width), CLASS_NODATA, np.uint8), profile
            return raster, profile

    # ------------------------------------------------------------
    # NOTE on cache_path:
    # Add to the `params` dict at the top of the function so the hash
    # picks up the new behavior:
    #
    #     reclassify_unclassified=reclassify_unclassified,
    #     drop_unclassified_after_reclassify=drop_unclassified_after_reclassify,
    #     unclassified_global_fraction_threshold=unclassified_global_fraction_threshold,
    #     min_unclassified_pixel_fraction=min_unclassified_pixel_fraction,
    #     reclassify_neighborhood_radius=reclassify_neighborhood_radius,
    #     reclassify_hag_min=reclassify_hag_min,
    #     reclassify_hag_terrain_max=reclassify_hag_terrain_max,
    #     reclassify_planar_threshold=reclassify_planar_threshold,
    #     reclassify_rough_threshold=reclassify_rough_threshold,
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Priority-based filtering (remove ground under roofs, etc.)
    # ------------------------------------------------------------
    if (
            lidar_class_priority is not None and
            lidar_class_window_diameter is not None and
            class_vals is not None and
            x.size > 0
    ):
        pts = np.column_stack((x, y, z, class_vals))

        pts = _filter_by_class_priority_grid(
            pts,
            lidar_class_priority,
            lidar_class_window_diameter
        )

        if pts is None or len(pts) == 0:
            raster = np.full((height, width), nodata, np.float32)
            profile = _build_raster_profile(width, height, transform, None, nodata, raster.dtype)

            if return_classification:
                return raster, np.full((height, width), CLASS_NODATA, np.uint8), profile

            return raster, profile

        # unpack back
        x = pts[:, 0].astype(np.float32)
        y = pts[:, 1].astype(np.float32)
        z = pts[:, 2].astype(np.float32)
        class_vals = pts[:, 3].astype(np.uint8)

    # ------------------------------------------------------------
    # Pixel mapping  ✅ FIXED
    # ------------------------------------------------------------
    inv = ~transform
    px, py = inv * (x, y)

    px = np.floor(px).astype(np.int64)
    py = np.floor(py).astype(np.int64)

    valid = (
            (px >= 0) & (px < width) &
            (py >= 0) & (py < height)
    )

    if not np.any(valid):
        raster = np.full((height, width), nodata, np.float32)
        profile = _build_raster_profile(width, height, transform, None, nodata, raster.dtype)
        if return_classification:
            return raster, np.full((height, width), 1, np.uint8), profile
        return raster, profile

    px, py = px[valid], py[valid]
    z = z[valid]
    if class_vals is not None:
        class_vals = class_vals[valid]

    # ------------------------------------------------------------
    # Initialise rasters
    # ------------------------------------------------------------
    raster = np.full((height, width), nodata, np.float32)
    class_raster = (
        np.full((height, width), 1, np.uint8)
        if return_classification else None
    )

    # ------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------
    if aggregation == "max":
        # --- height ---
        np.maximum.at(raster, (py, px), z)

    elif aggregation == "smart_for_roofs":
        flat_idx = py * width + px
        order = np.argsort(flat_idx)
        flat_idx = flat_idx[order]
        z_sorted = z[order]
        unique_idx, start_idx = np.unique(flat_idx, return_index=True)
        rows = unique_idx // width
        cols = unique_idx % width
        n = len(unique_idx)
        med_vals = np.empty(n, dtype=np.float32)

        for i in range(n):
            start = start_idx[i]
            end = start_idx[i + 1] if i + 1 < n else len(z_sorted)
            v = z_sorted[start:end]
            if v.size >= 3:
                med = np.median(v)
                mad = np.median(np.abs(v - med))
                if mad > 0:
                    v = v[np.abs(v - med) < 3 * mad]
            med_vals[i] = np.median(v)
        raster[rows, cols] = med_vals
        raster[~np.isfinite(raster)] = nodata

    elif aggregation == "min":
        # --- height ---
        np.minimum.at(raster, (py, px), z)


    elif aggregation == "mean":
        # --- height ---
        count = np.zeros((height, width), np.uint32)
        sum_r = np.zeros((height, width), np.float64)

        np.add.at(sum_r, (py, px), z)
        np.add.at(count, (py, px), 1)

        mask = count > 0
        raster[mask] = (sum_r[mask] / count[mask]).astype(np.float32)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation} ('mean', 'max', 'smart_for_roofs', 'min' are available)")

    ######
    # Classification aggregation
    ######

    if return_classification and class_vals is not None:

        valid = (
                (px >= 0) & (px < width) &
                (py >= 0) & (py < height)
        )

        px = px[valid]
        py = py[valid]
        class_vals = class_vals[valid]

        if px.size == 0:
            class_raster = np.full((height, width), CLASS_NODATA, dtype=np.uint8)

        else:
            flat_idx = py * width + px

            order = np.argsort(flat_idx, kind="mergesort")
            flat_idx = flat_idx[order].astype(np.int64)
            class_vals = class_vals[order].astype(np.uint8)

            unique_pix, start_idx = np.unique(flat_idx, return_index=True)

            vegetation_arr = np.array(list(vegetation_classes), dtype=np.uint8)
            building_arr = np.array(list(building_classes), dtype=np.uint8)

            class_raster_flat = _classify_pixels_numba(
                flat_idx,
                class_vals,
                unique_pix,
                start_idx,
                width,
                height,
                vegetation_arr,
                building_arr,
                min_class_fraction,
                CLASS_NODATA
            )

            class_raster = class_raster_flat.reshape(height, width)

    # ------------------------------------------------------------
    # Cache saving
    # ------------------------------------------------------------
    if cache_path is not None:

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        write_profile = profile.copy()

        if return_classification and class_raster is not None:
            write_profile.update(count=2, dtype="float32")
        else:
            write_profile.update(count=1)

        with rasterio.open(cache_path, "w", **write_profile) as dst:
            dst.write(raster.astype(np.float32), 1)

            if return_classification and class_raster is not None:
                dst.write(class_raster.astype(np.float32), 2)

    return (raster, class_raster, profile) if return_classification else (raster, profile)


# ----------------------------------------------------------------------
# Closed-form eigenvalues of a 3x3 real symmetric matrix.
# Smith (1961). Returns sorted ascending: l0 <= l1 <= l2.
# ----------------------------------------------------------------------
@njit(cache=True, inline="always")
def _eigvals_sym3(a, b, c, d, e, f):
    """
    Eigenvalues of [[a, b, c],
                    [b, d, e],
                    [c, e, f]]
    """
    # Trace / 3
    q = (a + d + f) / 3.0

    # Deviatoric matrix B = A - q*I
    a1 = a - q
    d1 = d - q
    f1 = f - q

    # p2 = sum of squares of B / 6
    p2 = (a1 * a1 + d1 * d1 + f1 * f1
          + 2.0 * (b * b + c * c + e * e)) / 6.0

    if p2 <= 1e-20:
        # Already (nearly) diagonal & isotropic
        return q, q, q

    p = np.sqrt(p2)

    # detB / 2 = r
    detB = (a1 * (d1 * f1 - e * e)
            - b * (b * f1 - e * c)
            + c * (b * e - d1 * c))
    r = detB / 2.0 / (p ** 3)

    # Numerical clamp
    if r <= -1.0:
        phi = np.pi / 3.0
    elif r >= 1.0:
        phi = 0.0
    else:
        phi = np.arccos(r) / 3.0

    # Eigenvalues, sorted descending then we return ascending
    eig2 = q + 2.0 * p * np.cos(phi)
    eig0 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    eig1 = 3.0 * q - eig0 - eig2  # trace conservation

    # Sort ascending
    if eig0 > eig1:
        eig0, eig1 = eig1, eig0
    if eig1 > eig2:
        eig1, eig2 = eig2, eig1
    if eig0 > eig1:
        eig0, eig1 = eig1, eig0

    return eig0, eig1, eig2


# ----------------------------------------------------------------------
# Build a 2D bin index: returns (cell_start, cell_count, sorted_idx, nx, ny)
# such that all points in cell (ix, iy) are sorted_idx[start : start+count].
# ----------------------------------------------------------------------
@njit(cache=True)
def _build_grid_index(x, y, x0, y0, cell, nx, ny):
    n = x.shape[0]
    cell_id = np.empty(n, dtype=np.int64)
    for i in range(n):
        ix = int((x[i] - x0) / cell)
        iy = int((y[i] - y0) / cell)
        if ix < 0:
            ix = 0
        elif ix >= nx:
            ix = nx - 1
        if iy < 0:
            iy = 0
        elif iy >= ny:
            iy = ny - 1
        cell_id[i] = iy * nx + ix

    sorted_idx = np.argsort(cell_id, kind="mergesort")
    cell_id_sorted = cell_id[sorted_idx]

    ncells = nx * ny
    cell_start = np.zeros(ncells + 1, dtype=np.int64)
    for i in range(n):
        cell_start[cell_id_sorted[i] + 1] += 1
    for c in range(ncells):
        cell_start[c + 1] += cell_start[c]

    return cell_start, sorted_idx


# ----------------------------------------------------------------------
# Per-pixel ground elevation (min z over terrain points), plus a fallback
# to nearest non-empty pixel for sparse areas. Pixel size = cell.
# ----------------------------------------------------------------------
@njit(cache=True)
def _ground_grid(xt, yt, zt, x0, y0, cell, nx, ny):
    """Min-z grid over terrain points; NaN where no terrain point fell."""
    g = np.full((ny, nx), np.nan, dtype=np.float32)
    for i in range(xt.shape[0]):
        ix = int((xt[i] - x0) / cell)
        iy = int((yt[i] - y0) / cell)
        if 0 <= ix < nx and 0 <= iy < ny:
            v = g[iy, ix]
            zi = zt[i]
            if np.isnan(v) or zi < v:
                g[iy, ix] = zi
    return g


@njit(cache=True, inline="always")
def _ground_lookup(g, ix, iy, nx, ny, search_radius_cells):
    """Return ground at (ix, iy) or nearest non-NaN within radius; NaN if none."""
    v = g[iy, ix]
    if not np.isnan(v):
        return v
    # Expanding ring search
    for r in range(1, search_radius_cells + 1):
        best = np.nan
        for dy in range(-r, r + 1):
            jy = iy + dy
            if jy < 0 or jy >= ny:
                continue
            # Only the ring boundary
            if dy == -r or dy == r:
                dx_range_start = -r
                dx_range_end = r + 1
            else:
                dx_range_start = -r
                dx_range_end = r + 1
                # but only edges
            for dx in range(dx_range_start, dx_range_end):
                if dy != -r and dy != r and dx != -r and dx != r:
                    continue
                jx = ix + dx
                if jx < 0 or jx >= nx:
                    continue
                w = g[jy, jx]
                if not np.isnan(w):
                    if np.isnan(best) or w < best:
                        best = w
        if not np.isnan(best):
            return best
    return np.nan


# ----------------------------------------------------------------------
# Main reclassification kernel.
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _reclassify_kernel(
        x, y, z, class_vals,
        is_unclassified,  # bool array, same length
        cell_start, sorted_idx,  # all-points grid
        x0, y0, cell, nx, ny,
        ground_grid,  # (ny, nx)
        pixel_x0, pixel_y0, pixel_size, pix_nx, pix_ny,  # output-raster grid for pixel-fraction gate
        pixel_unclass_count,  # (pix_ny, pix_nx) int
        pixel_total_count,  # (pix_ny, pix_nx) int
        min_pixel_fraction,
        hag_min,
        hag_terrain_max,
        planar_threshold,
        rough_threshold,
        min_neighbors,
        target_building, target_vegetation, target_terrain,
        ground_search_radius_cells,
):
    n = x.shape[0]
    out = class_vals.copy()

    for i in prange(n):
        if not is_unclassified[i]:
            continue

        # Pixel-fraction gate (uses output raster grid, not the neighborhood grid)
        ipx = int((x[i] - pixel_x0) / pixel_size)
        ipy = int((y[i] - pixel_y0) / pixel_size)
        if 0 <= ipx < pix_nx and 0 <= ipy < pix_ny:
            tot = pixel_total_count[ipy, ipx]
            if tot > 0:
                frac = pixel_unclass_count[ipy, ipx] / tot
                if frac < min_pixel_fraction:
                    continue
            else:
                continue
        else:
            continue

        # Ground lookup
        ix = int((x[i] - x0) / cell)
        iy = int((y[i] - y0) / cell)
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            continue
        g = _ground_lookup(ground_grid, ix, iy, nx, ny, ground_search_radius_cells)
        if np.isnan(g):
            continue
        hag = z[i] - g
        # Don't bail out yet — terrain candidates live at low HAG.
        # The HAG-vs-planarity decision happens after the covariance is computed.

        # Gather neighbors from the 3x3 cell block
        # First pass: count
        cnt = 0
        for dy in range(-1, 2):
            jy = iy + dy
            if jy < 0 or jy >= ny:
                continue
            for dx in range(-1, 2):
                jx = ix + dx
                if jx < 0 or jx >= nx:
                    continue
                cell_idx = jy * nx + jx
                cnt += cell_start[cell_idx + 1] - cell_start[cell_idx]

        if cnt < min_neighbors:
            continue

        # Second pass: accumulate mean and covariance (3x3 symmetric)
        # Using running sums
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for dy in range(-1, 2):
            jy = iy + dy
            if jy < 0 or jy >= ny:
                continue
            for dx in range(-1, 2):
                jx = ix + dx
                if jx < 0 or jx >= nx:
                    continue
                cell_idx = jy * nx + jx
                cs = cell_start[cell_idx]
                ce = cell_start[cell_idx + 1]
                for k in range(cs, ce):
                    pi = sorted_idx[k]
                    sx += x[pi]
                    sy += y[pi]
                    sz += z[pi]

        inv = 1.0 / cnt
        mx = sx * inv
        my = sy * inv
        mz = sz * inv

        cxx = 0.0
        cyy = 0.0
        czz = 0.0
        cxy = 0.0
        cxz = 0.0
        cyz = 0.0
        for dy in range(-1, 2):
            jy = iy + dy
            if jy < 0 or jy >= ny:
                continue
            for dx in range(-1, 2):
                jx = ix + dx
                if jx < 0 or jx >= nx:
                    continue
                cell_idx = jy * nx + jx
                cs = cell_start[cell_idx]
                ce = cell_start[cell_idx + 1]
                for k in range(cs, ce):
                    pi = sorted_idx[k]
                    dxp = x[pi] - mx
                    dyp = y[pi] - my
                    dzp = z[pi] - mz
                    cxx += dxp * dxp
                    cyy += dyp * dyp
                    czz += dzp * dzp
                    cxy += dxp * dyp
                    cxz += dxp * dzp
                    cyz += dyp * dzp

        cxx *= inv;
        cyy *= inv;
        czz *= inv
        cxy *= inv;
        cxz *= inv;
        cyz *= inv

        l0, l1, l2 = _eigvals_sym3(cxx, cxy, cxz, cyy, cyz, czz)
        if l2 <= 1e-20:
            continue
        planarity = (l1 - l0) / l2

        if hag <= hag_terrain_max:
            # Low and planar -> terrain. Low and rough -> probably noise / low veg, leave as 0.
            if planarity >= planar_threshold:
                out[i] = target_terrain
        elif hag >= hag_min:
            if planarity >= planar_threshold:
                out[i] = target_building
            elif planarity <= rough_threshold:
                out[i] = target_vegetation
            # else: ambiguous, leave as 0
        # else: hag in (hag_terrain_max, hag_min) — ambiguous low/mid, leave as 0

    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def reclassify_unclassified_points(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        class_vals: np.ndarray,
        *,
        unclassified_classes: Sequence[int] = (1,),
        terrain_classes: Sequence[int] = (2, 8),
        target_building_class: int = 6,
        target_vegetation_class: int = 4,
        target_terrain_class: int = 2,
        transform=None,  # rasterio Affine for the output raster (pixel grid)
        width: int = None,
        height: int = None,
        unclassified_global_fraction_threshold: float = 0.20,
        min_unclassified_pixel_fraction: float = 0.10,
        neighborhood_radius: float = 1.5,  # meters; controls grid cell size
        hag_min: float = 1.5,  # meters above ground for building/vegetation
        hag_terrain_max: float = 0.3,  # meters; up to this height, planar -> terrain
        planar_threshold: float = 0.5,
        rough_threshold: float = 0.2,
        min_neighbors: int = 8,
        ground_search_radius_cells: int = 3,
) -> Tuple[np.ndarray, dict]:
    """
    Returns
    -------
    new_class_vals : np.ndarray (uint8)
        Updated classifications.
    info : dict
        Diagnostic info: ran, frac, n_to_building, n_to_vegetation, n_kept_unclassified.
    """
    info = {
        "ran": False,
        "global_unclassified_fraction": 0.0,
        "n_reclassified_to_building": 0,
        "n_reclassified_to_vegetation": 0,
        "n_reclassified_to_terrain": 0,
        "n_left_unclassified": 0,
    }

    if class_vals is None or x.size == 0:
        return class_vals, info

    is_unclassified = np.isin(class_vals, np.asarray(unclassified_classes, dtype=np.uint8))
    n_unclass = int(is_unclassified.sum())
    frac = n_unclass / x.size
    info["global_unclassified_fraction"] = float(frac)

    if frac < unclassified_global_fraction_threshold:
        return class_vals, info  # gate 1 fails

    # Build neighborhood grid over ALL points
    cell = float(neighborhood_radius)
    x0 = float(x.min()) - cell
    y0 = float(y.min()) - cell
    x1 = float(x.max()) + cell
    y1 = float(y.max()) + cell
    nx = max(1, int(np.ceil((x1 - x0) / cell)))
    ny = max(1, int(np.ceil((y1 - y0) / cell)))

    cell_start, sorted_idx = _build_grid_index(
        x.astype(np.float64), y.astype(np.float64),
        x0, y0, cell, nx, ny,
    )

    # Ground grid from terrain points
    is_terrain = np.isin(class_vals, np.asarray(terrain_classes, dtype=np.uint8))
    if is_terrain.sum() == 0:
        # No terrain points -> can't compute HAG; bail out gracefully
        return class_vals, info
    ground_grid = _ground_grid(
        x[is_terrain].astype(np.float64),
        y[is_terrain].astype(np.float64),
        z[is_terrain].astype(np.float32),
        x0, y0, cell, nx, ny,
    )

    # Pixel-fraction gate counts (on the OUTPUT raster grid)
    if transform is None or width is None or height is None:
        raise ValueError("transform, width, height are required for the pixel-fraction gate.")

    inv_t = ~transform
    px = np.floor(np.asarray((x, y)).T @ np.array([[inv_t.a, inv_t.d], [inv_t.b, inv_t.e]])
                  + np.array([inv_t.c, inv_t.f]))
    # The above is a vectorized inverse-transform; equivalent to ~transform * (x, y)
    pix_col = px[:, 0].astype(np.int64)
    pix_row = px[:, 1].astype(np.int64)

    pix_total = np.zeros((height, width), dtype=np.int64)
    pix_unclass = np.zeros((height, width), dtype=np.int64)
    valid_pix = (pix_col >= 0) & (pix_col < width) & (pix_row >= 0) & (pix_row < height)
    np.add.at(pix_total, (pix_row[valid_pix], pix_col[valid_pix]), 1)
    np.add.at(
        pix_unclass,
        (pix_row[valid_pix & is_unclassified], pix_col[valid_pix & is_unclassified]),
        1,
    )

    # Output raster origin & cell size (assumes north-up affine: transform.e < 0)
    pixel_size_x = transform.a
    pixel_size_y = -transform.e  # positive
    # We need a single pixel size; assert square-ish
    pixel_size = float(pixel_size_x)
    # Map (x, y) -> integer pixel using floor((x - pixel_x0)/pixel_size); for y
    # we mirror because rows grow downward.
    # Easier: convert output pixel grid into world coords for the kernel.
    # The kernel uses (pixel_x0, pixel_y0, pixel_size) where pixel_y0 is the
    # SOUTH edge of the raster, and rows count upward. We rebuild that:
    raster_west = transform.c
    raster_north = transform.f
    raster_south = raster_north - pixel_size_y * height
    pixel_x0 = raster_west
    pixel_y0 = raster_south
    pix_nx = width
    pix_ny = height

    # The pixel_unclass / pixel_total in the kernel are indexed as (iy, ix) with
    # iy growing northward, but we built them with rasterio rows (growing south).
    # Flip vertically so the kernel sees iy=0 at the south.
    pix_unclass_kernel = pix_unclass[::-1, :].copy()
    pix_total_kernel = pix_total[::-1, :].copy()

    new_class = _reclassify_kernel(
        x.astype(np.float64), y.astype(np.float64), z.astype(np.float32),
        class_vals.astype(np.uint8),
        is_unclassified,
        cell_start, sorted_idx,
        x0, y0, cell, nx, ny,
        ground_grid,
        pixel_x0, pixel_y0, pixel_size, pix_nx, pix_ny,
        pix_unclass_kernel, pix_total_kernel,
        float(min_unclassified_pixel_fraction),
        float(hag_min),
        float(hag_terrain_max),
        float(planar_threshold),
        float(rough_threshold),
        int(min_neighbors),
        int(target_building_class),
        int(target_vegetation_class),
        int(target_terrain_class),
        int(ground_search_radius_cells),
    )

    info["ran"] = True
    diff = new_class != class_vals
    info["n_reclassified_to_building"] = int(((new_class == target_building_class) & diff).sum())
    info["n_reclassified_to_vegetation"] = int(((new_class == target_vegetation_class) & diff).sum())
    info["n_reclassified_to_terrain"] = int(((new_class == target_terrain_class) & diff).sum())
    info["n_left_unclassified"] = int((is_unclassified & ~diff).sum())

    return new_class, info


@njit(parallel=True)
def _classify_pixels_numba(
    flat_idx,
    class_vals,
    unique_pix,
    start_idx,
    width,
    height,
    vegetation_classes,
    building_classes,
    min_class_fraction,
    CLASS_NODATA
):
    n_pixels = width * height
    out = np.full(n_pixels, CLASS_NODATA, np.uint8)

    MAX_CLASSES = 16

    for i in prange(len(unique_pix)):

        start = start_idx[i]
        end = start_idx[i + 1] if i + 1 < len(start_idx) else len(flat_idx)

        pix = unique_pix[i]

        # -------------------------------------
        # local counting (thread-local)
        # -------------------------------------
        uniq = np.empty(MAX_CLASSES, dtype=np.uint8)
        counts = np.zeros(MAX_CLASSES, dtype=np.int32)
        n_uniq = 0

        for j in range(start, end):
            c = class_vals[j]

            found = False
            for k in range(n_uniq):
                if uniq[k] == c:
                    counts[k] += 1
                    found = True
                    break

            if not found:
                if n_uniq < MAX_CLASSES:
                    uniq[n_uniq] = c
                    counts[n_uniq] = 1
                    n_uniq += 1

        # -------------------------------------
        # total
        # -------------------------------------
        total = 0
        for k in range(n_uniq):
            total += counts[k]

        if total == 0:
            continue

        # -------------------------------------
        # majority
        # -------------------------------------
        max_idx = 0
        for k in range(1, n_uniq):
            if counts[k] > counts[max_idx]:
                max_idx = k

        assigned = uniq[max_idx]

        # -------------------------------------
        # vegetation override
        # -------------------------------------
        veg_votes = 0
        veg_best_idx = -1

        for k in range(n_uniq):
            c = uniq[k]

            for vc in vegetation_classes:
                if c == vc:
                    veg_votes += counts[k]
                    if veg_best_idx == -1 or counts[k] > counts[veg_best_idx]:
                        veg_best_idx = k

        if veg_best_idx != -1:
            if veg_votes / total >= min_class_fraction:
                assigned = uniq[veg_best_idx]

        else:
            # ---------------------------------
            # building override
            # ---------------------------------
            bld_votes = 0
            bld_best_idx = -1

            for k in range(n_uniq):
                c = uniq[k]

                for bc in building_classes:
                    if c == bc:
                        bld_votes += counts[k]
                        if bld_best_idx == -1 or counts[k] > counts[bld_best_idx]:
                            bld_best_idx = k

            if bld_best_idx != -1:
                if bld_votes / total >= min_class_fraction:
                    assigned = uniq[bld_best_idx]

        out[pix] = assigned

    return out

def _median_per_pixel(px, py, z, width):
    flat = py * width + px

    order = np.argsort(flat, kind="mergesort")
    flat = flat[order]
    z = z[order]

    unique, idx_start = np.unique(flat, return_index=True)

    rows = unique // width
    cols = unique % width

    out = np.empty(len(unique), dtype=np.float32)

    for i in range(len(unique)):
        start = idx_start[i]
        end = idx_start[i + 1] if i + 1 < len(unique) else len(z)

        v = z[start:end]

        if v.size >= 3:
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            if mad > 0:
                v = v[np.abs(v - med) < 3 * mad]

        out[i] = np.median(v)

    return rows, cols, out

def _remove_local_outliers(raster, nodata, resolution):
    threshold = max(1.2, abs(resolution) * 4)

    valid = raster != nodata

    tmp = raster.copy()
    tmp[~valid] = np.nan

    # neighbourhood median
    med = median_filter(tmp, size=3, mode="nearest")

    # deviation from neighbourhood
    diff = np.abs(tmp - med)

    # outliers (both high and low)
    outliers = (diff > threshold) & valid

    raster[outliers] = med[outliers]

    raster[raster>9000] = nodata

    return raster


def _ridge_preserving_filter(raster, nodata, resolution):
    threshold = max(1.2, abs(resolution) * 4)

    valid = raster != nodata

    tmp = raster.copy()

    # replace nodata with local median reference
    med_global = np.median(tmp[valid]) if np.any(valid) else 0
    tmp[~valid] = med_global

    med_x = median_filter(tmp, size=(1, 3), mode="nearest")
    med_y = median_filter(tmp, size=(3, 1), mode="nearest")

    diff_x = np.abs(tmp - med_x)
    diff_y = np.abs(tmp - med_y)

    replace = (diff_x > threshold) & (diff_y > threshold) & valid

    raster[replace] = np.minimum(med_x, med_y)[replace]

    raster[~np.isfinite(raster)] = nodata

    return raster

def _create_aoi_mask(
    aoi_geometry: Any,
    width: int, 
    height: int, 
    transform: Any,
    buffer_meters: float = 0.0
) -> np.ndarray:
    """
    Create a boolean mask from AOI geometry.
    
    Parameters:
    -----------
    aoi_geometry : geometry
        Shapely geometry or similar object with bounds and contains methods
    width, height : int
        Raster dimensions
    transform : Affine
        Raster transform
    buffer_meters : float
        Additional buffer in meters
        
    Returns:
    --------
    np.ndarray : boolean mask where True = inside AOI
    """
    
    # Buffer the geometry if requested
    if buffer_meters > 0:
        aoi_buffered = aoi_geometry.buffer(buffer_meters)
    else:
        aoi_buffered = aoi_geometry
    
    # Create mask using rasterio.features.rasterize
    # This is much faster than checking each pixel individually
    try:
        mask = rasterize(
            [aoi_buffered],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=1,
            dtype='uint8'
        ).astype(bool)
        
        return mask
        
    except Exception as e:
        print(f"Warning: Could not create optimized AOI mask, falling back to pixel-by-pixel method: {e}")
        
        # Fallback: pixel-by-pixel method (slower but more reliable)
        mask = np.zeros((height, width), dtype=bool)
        
        # Convert pixel coordinates to world coordinates
        rows, cols = np.mgrid[0:height, 0:width]
        xs, ys = transform * (cols, rows)
        
        # Create points and test containment in chunks to manage memory
        chunk_size = 100000  # Process in chunks to avoid memory issues
        total_pixels = height * width
        
        for i in range(0, total_pixels, chunk_size):
            end_i = min(i + chunk_size, total_pixels)
            flat_idx = np.arange(i, end_i)
            row_idx = flat_idx // width
            col_idx = flat_idx % width
            
            chunk_xs = xs[row_idx, col_idx]
            chunk_ys = ys[row_idx, col_idx]
            
            # Test containment for this chunk
            for j, (x, y) in enumerate(zip(chunk_xs, chunk_ys)):
                if aoi_buffered.contains(sg.Point(x, y)):
                    r, c = row_idx[j], col_idx[j]
                    mask[r, c] = True
                    
        return mask

@njit(parallel=True, fastmath=True)
def _compute_slope_aspect(
    dem,
    slope,
    aspect,
    nodata,
    pixel_size,
):
    H, W = dem.shape

    for i in prange(1, H - 1):
        for j in range(1, W - 1):

            zc = dem[i, j]
            if zc == nodata:
                slope[i, j] = nodata
                aspect[i, j] = nodata
                continue

            zn = dem[i - 1, j]
            zs = dem[i + 1, j]
            zw = dem[i, j - 1]
            ze = dem[i, j + 1]

            if (
                zn == nodata or zs == nodata or
                zw == nodata or ze == nodata
            ):
                slope[i, j] = nodata
                aspect[i, j] = nodata
                continue

            dzdx = (ze - zw) / (2.0 * pixel_size)
            dzdy = (zn - zs) / (2.0 * pixel_size)

            # slope
            slope_val = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
            slope[i, j] = slope_val

            # aspect (0=N, clockwise)
            asp = np.arctan2(dzdx, dzdy)
            if asp < 0:
                asp += 2.0 * np.pi

            aspect[i, j] = asp




def _interpolation_support_mask(
    arr: np.ndarray,
    nodata: float,
    max_pix_dist: int
) -> np.ndarray:
    """
    Determine where interpolation (not extrapolation) is possible.

    A nodata pixel is valid for interpolation if there are valid pixels
    in opposite directions within max_pix_dist.
    """

    valid = arr != nodata

    # -----------------------------
    # Directional kernels
    # -----------------------------

    k_left = np.zeros((1, 2 * max_pix_dist + 1))
    k_left[0, :max_pix_dist] = 1

    k_right = np.zeros((1, 2 * max_pix_dist + 1))
    k_right[0, max_pix_dist + 1:] = 1

    k_up = np.zeros((2 * max_pix_dist + 1, 1))
    k_up[:max_pix_dist, 0] = 1

    k_down = np.zeros((2 * max_pix_dist + 1, 1))
    k_down[max_pix_dist + 1:, 0] = 1

    # Diagonals
    k_diag1_a = np.eye(2 * max_pix_dist + 1)
    k_diag1_a[max_pix_dist:, :] = 0

    k_diag1_b = np.eye(2 * max_pix_dist + 1)
    k_diag1_b[:max_pix_dist + 1, :] = 0

    k_diag2_a = np.fliplr(np.eye(2 * max_pix_dist + 1))
    k_diag2_a[max_pix_dist:, :] = 0

    k_diag2_b = np.fliplr(np.eye(2 * max_pix_dist + 1))
    k_diag2_b[:max_pix_dist + 1, :] = 0

    # -----------------------------
    # Convolutions
    # -----------------------------

    left  = convolve(valid.astype(np.uint8), k_left,  mode="constant") > 0
    right = convolve(valid.astype(np.uint8), k_right, mode="constant") > 0

    up    = convolve(valid.astype(np.uint8), k_up,    mode="constant") > 0
    down  = convolve(valid.astype(np.uint8), k_down,  mode="constant") > 0

    diag1a = convolve(valid.astype(np.uint8), k_diag1_a, mode="constant") > 0
    diag1b = convolve(valid.astype(np.uint8), k_diag1_b, mode="constant") > 0

    diag2a = convolve(valid.astype(np.uint8), k_diag2_a, mode="constant") > 0
    diag2b = convolve(valid.astype(np.uint8), k_diag2_b, mode="constant") > 0

    # -----------------------------
    # Opposite direction support
    # -----------------------------

    horiz  = left & right
    vert   = up & down
    diag1  = diag1a & diag1b
    diag2  = diag2a & diag2b

    support = horiz | vert | diag1 | diag2

    return support

def _roof_median_propagation(arr, nodata, building_mask, iterations=6):

    filled = arr.copy()

    hole_mask = (arr == nodata) & building_mask

    for _ in range(iterations):

        if not np.any(hole_mask):
            break

        med = median_filter(filled, size=3)

        update = hole_mask & (med != nodata)

        filled[update] = med[update]

        hole_mask = (filled == nodata) & building_mask

    return filled

def _smart_roof_fill(arr, nodata, mask_to_fill, aoi_mask=None):

    filled = arr.copy()

    # -----------------------------------------
    # restrict mask to AOI
    # -----------------------------------------
    if aoi_mask is not None:
        mask_to_fill = mask_to_fill & aoi_mask

    if not np.any(mask_to_fill):
        return arr

    # -----------------------------------------
    # distance inside holes
    # -----------------------------------------
    dist_pix = distance_transform_edt(mask_to_fill).astype(np.float32)

    contour_mask = mask_to_fill & (dist_pix <= 4)
    interior_mask = mask_to_fill & (dist_pix > 4)

    # -----------------------------------------
    # NEAREST propagation (edge zone)
    # -----------------------------------------
    if np.any(contour_mask):

        valid = (arr != nodata)

        if aoi_mask is not None:
            valid &= aoi_mask

        dist, inds = distance_transform_edt(
            ~valid,
            return_indices=True
        )

        nearest_vals = arr[inds[0], inds[1]]

        filled[contour_mask] = nearest_vals[contour_mask]

    # -----------------------------------------
    # BIHARMONIC interpolation (interior)
    # -----------------------------------------
    if np.any(interior_mask):

        arr_tmp = filled.copy()

        valid = (arr_tmp != nodata)

        if aoi_mask is not None:
            valid &= aoi_mask

        if not np.any(valid):
            return filled

        init_val = np.median(arr_tmp[valid])

        arr_tmp[mask_to_fill] = init_val

        biharm = inpaint.inpaint_biharmonic(arr_tmp, mask_to_fill)

        filled[interior_mask] = biharm[interior_mask]

    return filled


def _compute_roi(mask, pad=20):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    ymin = max(0, ymin - pad)
    ymax = min(mask.shape[0], ymax + pad + 1)
    xmin = max(0, xmin - pad)
    xmax = min(mask.shape[1], xmax + pad + 1)

    return np.s_[ymin:ymax, xmin:xmax]

def _smart_roof_fill_safe(
    arr: np.ndarray,
    nodata: float,
    mask_to_fill: np.ndarray,
    *,
    aoi_mask: np.ndarray | None = None,
    max_ram_gb: float = 8.0,
    contour_px: int = 4,
    roi_pad_px: int = 20,
    biharmonic_max_pixels: int = 50_000,
    logger = None
) -> np.ndarray:
    """
    Memory-safe roof gap filling.

    Strategy:
    ---------
    1. Restrict to ROI (massive memory reduction)
    2. Split into contour vs interior using local EDT
    3. Contour → chunked nearest fill
    4. Interior:
        - small → biharmonic
        - large → nearest (fallback)
    """

    # -----------------------------------------
    # Restrict mask to AOI
    # -----------------------------------------
    if aoi_mask is not None:
        mask_to_fill = mask_to_fill & aoi_mask

    if not np.any(mask_to_fill):
        return arr

    # -----------------------------------------
    # Compute ROI
    # -----------------------------------------
    if logger:
        logger.log("smart_for_roofs filling / compute roi")
    roi = _compute_roi(mask_to_fill, pad=roi_pad_px)
    if roi is None:
        return arr

    arr_roi = arr[roi]
    mask_roi = mask_to_fill[roi]

    if aoi_mask is not None:
        aoi_roi = aoi_mask[roi]
    else:
        aoi_roi = None

    # -----------------------------------------
    # Distance transform ONLY in ROI
    # -----------------------------------------
    if logger:
        logger.log("smart_for_roofs filling / distance transform in roi")
    dist_pix = distance_transform_edt(mask_roi)

    contour_mask = mask_roi & (dist_pix <= contour_px)
    interior_mask = mask_roi & (dist_pix > contour_px)

    filled_roi = arr_roi.copy()

    # -----------------------------------------
    # NEAREST propagation (contour)
    # -----------------------------------------
    if logger:
        logger.log("smart_for_roofs filling / nearest propagation")
    if np.any(contour_mask):

        valid = (arr_roi != nodata)
        if aoi_roi is not None:
            valid &= aoi_roi

        # fill only contour region
        filled_tmp = _nearest_fill_chunked(
            mask=~valid,   # gaps
            arr=arr_roi,
            aoi_mask=aoi_roi,
            max_ram_gb=max_ram_gb
        )

        filled_roi[contour_mask] = filled_tmp[contour_mask]

    # -----------------------------------------
    # INTERIOR interpolation
    # -----------------------------------------
    if logger:
        logger.log("smart_for_roofs filling / interior propagation")
    if np.any(interior_mask):

        interior_pixels = int(np.sum(interior_mask))

        valid = (filled_roi != nodata)
        if aoi_roi is not None:
            valid &= aoi_roi

        if not np.any(valid):
            arr_out = arr.copy()
            arr_out[roi] = filled_roi
            return arr_out

        # -------------------------------------
        # SMALL regions → biharmonic
        # -------------------------------------
        if logger:
            logger.log("smart_for_roofs filling / interior propagation / small regions")
        if interior_pixels <= biharmonic_max_pixels:

            arr_tmp = filled_roi.copy()

            init_val = np.median(arr_tmp[valid])
            arr_tmp[mask_roi] = init_val

            try:
                biharm = inpaint.inpaint_biharmonic(arr_tmp, mask_roi)
                filled_roi[interior_mask] = biharm[interior_mask]
            except Exception:
                # fallback if solver fails
                filled_tmp = _nearest_fill_chunked(
                    mask=mask_roi,
                    arr=filled_roi,
                    aoi_mask=aoi_roi,
                    max_ram_gb=max_ram_gb
                )
                filled_roi[interior_mask] = filled_tmp[interior_mask]

        # -------------------------------------
        # LARGE regions → fallback (critical)
        # -------------------------------------
        else:
            if logger:
                logger.log("smart_for_roofs filling / interior propagation / large regions")
            filled_tmp = _nearest_fill_chunked(
                mask=mask_roi,
                arr=filled_roi,
                aoi_mask=aoi_roi,
                max_ram_gb=max_ram_gb
            )
            filled_roi[interior_mask] = filled_tmp[interior_mask]

    # -----------------------------------------
    # Write back to full raster
    # -----------------------------------------
    result = arr.copy()
    result[roi] = filled_roi

    return result

def _fast_dem_fill(arr, nodata=-9999.0):

    mask = arr == nodata

    if not np.any(mask):
        return arr

    # distance transform returns nearest valid pixel indices
    dist, inds = distance_transform_edt(
        mask,
        return_indices=True
    )

    filled = arr.copy()
    filled[mask] = arr[inds[0][mask], inds[1][mask]]

    return filled

def _fill_gaps(
    arr: np.ndarray,
    nodata: float,
    method: str,
    resolution: float,
    max_ram_gb: float,
    dist_threshold: float = 1,
    aoi_mask: Optional[np.ndarray] = None,
    logger = None,
) -> np.ndarray:

    mask = arr == nodata

    if not np.any(mask):
        return arr

    # -----------------------------------------
    # Ensure interpolation support (no extrapolation)
    # -----------------------------------------
    if logger:
        logger.log("support_mask")
    support_mask = _interpolation_support_mask(
        arr,
        nodata,
        max_pix_dist=int(np.ceil(dist_threshold / resolution))
    )

    mask_to_fill = mask & support_mask

    if aoi_mask is not None:
        mask_to_fill &= aoi_mask

    if not np.any(mask_to_fill):
        return arr


    # =================================================
    # SMART METHOD
    # =================================================
    if method == "smart_for_roofs":
        if logger:
            logger.log("fill gaps using smart_for_roofs method")
        return _smart_roof_fill_safe(
            arr,
            nodata,
            mask_to_fill,
            aoi_mask=aoi_mask,
            logger = logger
        )

    elif method == "smart_general":

        filled = arr.copy()
        # ---------------------------------------------
        # Identify holes
        # ---------------------------------------------
        labeled, n = label(mask)
        counts = np.bincount(labeled.ravel())

        # thresholds (can tune)
        SMALL = 3
        LARGE = 200

        small_ids = np.where((counts > 0) & (counts <= SMALL))[0]
        medium_ids = np.where((counts > SMALL) & (counts <= LARGE))[0]
        large_ids = np.where(counts > LARGE)[0]

        # remove background
        small_ids = small_ids[small_ids != 0]
        medium_ids = medium_ids[medium_ids != 0]
        large_ids = large_ids[large_ids != 0]

        # ---------------------------------------------
        # FAST LUT masks
        # ---------------------------------------------
        lut_small = np.zeros(len(counts), dtype=bool)
        lut_medium = np.zeros(len(counts), dtype=bool)
        lut_large = np.zeros(len(counts), dtype=bool)

        lut_small[small_ids] = True
        lut_medium[medium_ids] = True
        lut_large[large_ids] = True

        small_mask = lut_small[labeled]
        medium_mask = lut_medium[labeled]
        large_mask = lut_large[labeled]

        if aoi_mask is not None:
            small_mask &= aoi_mask
            medium_mask &= aoi_mask
            large_mask &= aoi_mask

        # ---------------------------------------------
        # Small holes → biharmonic
        # ---------------------------------------------
        if np.any(small_mask):
            tmp = _bilinear_fill(arr, nodata, small_mask)
            filled[small_mask] = tmp[small_mask]

        # ---------------------------------------------
        # Medium holes → nearest
        # ---------------------------------------------
        if np.any(medium_mask):
            tmp = _nearest_fill_chunked(mask_to_fill, arr, max_ram_gb, aoi_mask)
            filled[medium_mask] = tmp[medium_mask]

        # ---------------------------------------------
        # Large holes → minimum contour value per hole
        # ---------------------------------------------

        if np.any(large_mask):

            large_labels = np.unique(labeled[large_mask])
            large_labels = large_labels[large_labels != 0]

            for hid in large_labels:

                hole_mask = labeled == hid

                boundary = binary_dilation(hole_mask) & (~hole_mask)

                boundary_vals = arr[boundary]

                # Remove nodata and invalid values
                boundary_vals = boundary_vals[
                    (boundary_vals != nodata) &
                    np.isfinite(boundary_vals)
                    ]

                if boundary_vals.size == 0:
                    continue

                fill_val = boundary_vals.min()

                filled[hole_mask] = fill_val

        # ---------------------------------------------
        # Prevent holes lower than contour
        # ---------------------------------------------
        boundary = binary_dilation(mask) & (~mask)

        boundary_vals = arr[boundary]
        boundary_vals = boundary_vals[
            (boundary_vals != nodata) & np.isfinite(boundary_vals)
            ]

        if boundary_vals.size > 0:
            min_contour = boundary_vals.min()
            filled[mask] = np.maximum(filled[mask], min_contour)

        return filled

    # =================================================
    # CLASSIC METHODS
    # =================================================
    elif method == "dem":
        return _fast_dem_fill(arr, nodata)

    elif method == "dw_plane":
        return _dw_plane_fill(arr, nodata, mask_to_fill)
        
    elif method == "median":
        return _median_fill(arr, nodata, mask_to_fill)

    elif method == "nearest":
        return _nearest_fill_chunked(mask_to_fill, arr, max_ram_gb, aoi_mask)

    elif method == "bilinear":
        return _bilinear_fill(arr, nodata, mask_to_fill)

    elif method == "idw":
        return _idw_fill(arr, nodata, mask_to_fill)

    elif method == "harmonic":
        return _harmonic_fill(arr, nodata, mask_to_fill)

    elif method == "biharmonic":
        return _biharmonic_fill(arr, nodata, mask_to_fill)

    else:
        raise ValueError(f"Unknown fill_gaps method: {method}")

def _median_fill(arr, nodata, mask, window_size=3):
    """
    Fast median fill ignoring nodata values.
    """

    import numpy as np
    from scipy.ndimage import median_filter

    filled = arr.copy()

    # Replace nodata with NaN so they don't influence the median
    arr_nan = arr.astype(float)
    arr_nan[arr_nan == nodata] = np.nan

    # Compute median filter
    med = median_filter(arr_nan, size=window_size, mode="nearest")

    # Only update masked pixels
    valid = mask & np.isfinite(med)
    filled[valid] = med[valid]

    return filled

def _harmonic_fill(arr, nodata, aoi_mask=None):
    from scipy.ndimage import laplace
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    filled = arr.copy()
    mask = arr == nodata

    if aoi_mask is not None:
        mask &= aoi_mask

    if not np.any(mask):
        return arr

    h, w = arr.shape
    idx_map = -np.ones((h, w), int)

    unknown_idx = np.where(mask)
    for i, (y, x) in enumerate(zip(*unknown_idx)):
        idx_map[y, x] = i

    N = len(unknown_idx[0])
    A = lil_matrix((N, N))
    b = np.zeros(N)

    for i, (y, x) in enumerate(zip(*unknown_idx)):
        A[i, i] = 4
        neighbors = [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]

        for ny, nx in neighbors:
            if 0 <= ny < h and 0 <= nx < w:
                if mask[ny, nx]:
                    j = idx_map[ny, nx]
                    A[i, j] = -1
                else:
                    b[i] += arr[ny, nx]

    solution = spsolve(A.tocsr(), b)

    filled[mask] = solution

    return filled

def _idw_fill(arr, nodata, aoi_mask=None, power=2):

    from scipy.spatial import cKDTree

    filled = arr.copy()
    mask = arr == nodata

    if aoi_mask is not None:
        mask &= aoi_mask

    if not np.any(mask):
        return arr

    h, w = arr.shape
    yy, xx = np.indices((h, w))

    valid = ~mask

    pts_valid = np.column_stack((xx[valid], yy[valid]))
    vals_valid = arr[valid]

    pts_missing = np.column_stack((xx[mask], yy[mask]))

    tree = cKDTree(pts_valid)
    dist, idx = tree.query(pts_missing, k=8)

    weights = 1.0 / (dist + 1e-6) ** power
    weights /= weights.sum(axis=1, keepdims=True)

    filled_vals = np.sum(vals_valid[idx] * weights, axis=1)

    filled[mask] = filled_vals

    return filled

def _bilinear_fill(arr, nodata, aoi_mask=None):
    """
    Fast bilinear-like interpolation using convolution.
    Much faster than scipy.griddata for raster gaps.
    """

    import numpy as np
    from scipy.ndimage import convolve

    filled = arr.copy()

    # -----------------------------------------
    # Mask of pixels to fill
    # -----------------------------------------
    mask = arr == nodata

    if aoi_mask is not None:
        mask &= aoi_mask

    if not np.any(mask):
        return arr

    # -----------------------------------------
    # Valid mask
    # -----------------------------------------
    valid = arr != nodata

    # -----------------------------------------
    # Bilinear kernel
    # -----------------------------------------
    kernel = np.array([
        [0.25, 0.5, 0.25],
        [0.5 , 1.0, 0.5 ],
        [0.25, 0.5, 0.25]
    ], dtype=np.float32)

    # -----------------------------------------
    # Convolution numerator
    # -----------------------------------------
    weighted_sum = convolve(
        np.where(valid, arr, 0.0),
        kernel,
        mode="nearest"
    )

    # -----------------------------------------
    # Convolution denominator
    # -----------------------------------------
    weight_sum = convolve(
        valid.astype(np.float32),
        kernel,
        mode="nearest"
    )

    # Avoid division by zero
    interp = weighted_sum / np.maximum(weight_sum, 1e-6)

    # -----------------------------------------
    # Fill only allowed pixels
    # -----------------------------------------
    filled[mask] = interp[mask]

    return filled

def _biharmonic_fill(arr, nodata, mask):

    if not np.any(mask):
        return arr

    arr_copy = arr.astype(np.float64).copy()

    valid = (~mask) & np.isfinite(arr_copy)

    if not np.any(valid):
        return arr

    fill_val = np.mean(arr_copy[valid])

    arr_copy[mask] = fill_val

    filled = inpaint.inpaint_biharmonic(arr_copy, mask)

    return filled.astype(arr.dtype)

def build_digital_model_from_lidar_pure(
        # ---- optional outputs ----
        laz_paths=None,
        out_path=None,
        return_arrays=False,
        *,
        laz_points=None,
        out_classification_path=None,
        out_slope_path=None,
        out_aspect_path=None,
        cache_rasterization=True,

        # ---- grid ----
        resolution=1.0,
        bounds=None,
        epsg=25831,

        # ---- LiDAR aggregation ----
        classification=(2, 8,),  # By default correspond to terrain ICGC classes
        vegetation_classes: Sequence[int] = (3, 4, 5),  # By default correspond to ICGC classes
        building_classes: Sequence[int] = (6, 17, 76, 77),  # By default correspond to ICGC classes
        terrain_classes: Sequence[int] = (2, 8),  # By default correspond to ICGC classes
        min_class_fraction: float = 0.0,
        aggregation="mean",
        lidar_class_priority=None,
        lidar_class_window_diameter=0.5,

        # Could be "min", "max", "mean". It is the aggregation used in the rasterisation of the LIDAR points
        nodata=-9999.0,

        # ---- memory ----
        max_ram_gb=None,

        # ---- AOI ----
        aoi=None,
        aoi_epsg=None,
        aoi_buffer_in_meters=0.0,

        # ---- raster-based filtering ----
        filter_raster_path=None,
        filter_raster_band=1,
        filter_points_from_class=None,
        filter_mode="exclude",

        # ---- isolated pixels ----
        isolated_points_exclusion_ratio=None,
        isolated_window_size=3,

        # ---- Digital Model smoothing ----
        smooth_classes=None,
        smooth_mode=None,
        smooth_unary=False,
        smooth_window_size=None,
        smooth_iterations=1,

        # ---- slope/aspect smoothing ----
        slope_smooth_window=None,
        slope_smooth_iterations=1,
        aspect_smooth_window=None,
        aspect_smooth_iterations=1,

        # ---- gap filling ----
        fill_gaps="nearest",

        # ---- classification correction ----
        classification_correction_buildings_gdf=None,
        classification_correction_buildings_mode="enforce",
        classification_correction_building_class_value=6,
        classification_correction_building_buffer_m=1.0,
        classification_correction_avoid_vegetation_classes: bool = True,

        block_size=1024  # or match GeoTIFF block size
):
    """
    DSM/DTM builder with dual-pass classification preservation.

    CRITICAL INVARIANT (restored):
    - Classification votes are counted ONLY where elevation contributes.
    """
    logger = RAMLogger("/tmp/ram_debug.log")

    CLASS_NODATA = 255  # must not be in `classification`

    if laz_paths is None and laz_points is None:
        raise ValueError("Either laz_paths or laz_points must be provided.")

    if laz_paths is not None and laz_points is not None:
        raise ValueError("Provide only one of laz_paths or laz_points.")

    # ------------------------------------------------------------
    # AOI reprojection
    # ------------------------------------------------------------
    aoi_transformed = None
    if aoi is not None:
        if aoi_epsg is None:
            raise ValueError("aoi_epsg must be provided when aoi is given")
        if aoi_epsg != epsg:
            transformer = Transformer.from_crs(
                f"EPSG:{aoi_epsg}", f"EPSG:{epsg}", always_xy=True
            )
            aoi_transformed = shp_transform(transformer.transform, aoi)
        else:
            aoi_transformed = aoi

    # ------------------------------------------------------------
    # Bounds / grid
    # ------------------------------------------------------------
    if bounds is None:
        if laz_points is not None:
            x = laz_points[:, 0]
            y = laz_points[:, 1]
            bounds = (x.min(), y.min(), x.max(), y.max())
        else:
            bounds = _infer_bounds_from_lidar(laz_paths, classification)

    minx, miny, maxx, maxy = bounds
    width, height, transform = _aligned_grid(
        minx, miny, maxx, maxy, resolution
    )
    profile = _build_raster_profile(
        width=width,
        height=height,
        transform=transform,
        crs=CRS.from_epsg(epsg),
        nodata=nodata,
        dtype=np.float32,
        compress="lzw"
    )

    # ------------------------------------------------------------
    # AOI mask
    # ------------------------------------------------------------
    aoi_mask = None
    if aoi_transformed is not None:
        aoi_mask = _create_aoi_mask(
            aoi_transformed,
            width,
            height,
            transform,
            aoi_buffer_in_meters,
        )

    # ------------------------------------------------------------
    # Digital Model initialization
    # ------------------------------------------------------------
    if aggregation == "mean":
        dm = np.zeros((height, width), dtype=np.float64)
        count = np.zeros((height, width), dtype=np.uint32)
    elif aggregation == "max":
        dm = np.full((height, width), -np.inf, dtype=np.float32)
        count = None
    elif aggregation == "smart_for_roofs":
        dm = np.full((height, width), nodata, dtype=np.float32)
        count = None
    elif aggregation == "min":
        dm = np.full((height, width), np.inf, dtype=np.float32)
        count = None
    else:
        raise ValueError(f"Unknown aggregation: {aggregation} ('mean', 'max', 'smart_for_roofs', 'min' are available)")
    # ------------------------------------------------------------
    # CLASS COUNTERS
    # ------------------------------------------------------------
    class_counts = {
        int(cls): np.zeros((height, width), dtype=np.uint32)
        for cls in classification
    }

    # ------------------------------------------------------------
    # Filter raster
    # ------------------------------------------------------------
    filter_raster = (
        rasterio.open(filter_raster_path)
        if filter_raster_path else None
    )

    # ------------------------------------------------------------
    # PROCESS LiDAR FILES
    # ------------------------------------------------------------
    if cache_rasterization:
        def _cache_name_from_path(path):
            return hashlib.sha1(str(path).encode()).hexdigest()[:16] + ".tif"
        out_path_p = Path(out_path)
        cache_hash = hashlib.sha1(str(out_path_p).encode()).hexdigest()[:12]
        cache_dir = out_path_p.parent / f"{str(out_path_p.name).split(".")[0]}_{cache_hash}"
        os.makedirs(cache_dir, exist_ok=True)
    logger.log("rasterisation")
    if laz_points is not None:
        if cache_rasterization:
            cache_file = cache_dir / "points_cache.tif"
        raster, class_raster, _ = _rasterize_agg(
            laz_path=None,
            laz_points=laz_points,
            width=width,
            height=height,
            transform=transform,
            classification=classification,
            vegetation_classes=vegetation_classes,
            building_classes=building_classes,
            terrain_classes=terrain_classes,
            min_class_fraction=min_class_fraction,
            aggregation=aggregation,
            nodata=nodata,
            aoi=aoi_transformed,
            aoi_buffer_in_meters=aoi_buffer_in_meters,
            return_classification=True,
            filter_raster=filter_raster,
            filter_raster_band=filter_raster_band,
            filter_points_from_class=filter_points_from_class,
            filter_mode=filter_mode,
            lidar_class_priority=lidar_class_priority,
            lidar_class_window_diameter=0.5,
            cache_path=str(cache_file) if cache_rasterization else None,
            CLASS_NODATA=CLASS_NODATA,
            epsg=epsg
        )
        elev_valid = raster != nodata
        del laz_points

        # ---- Elevation aggregation ----
        if aggregation == "mean":
            dm[elev_valid] += raster[elev_valid]
            count[elev_valid] += 1
        elif aggregation == "max":
            dm[elev_valid] = np.maximum(dm[elev_valid], raster[elev_valid])
        elif aggregation == "min":
            dm[elev_valid] = np.minimum(dm[elev_valid], raster[elev_valid])
        elif aggregation == "smart_for_roofs":
            mask = elev_valid & (dm == nodata)
            dm[mask] = raster[mask]
            overlap = elev_valid & (dm != nodata)
            dm[overlap] = np.maximum(dm[overlap], raster[overlap])
        # ---- Classification density (SUPPORT-ALIGNED) ----
        for cls in class_counts:
            mask = (class_raster == cls) & elev_valid
            class_counts[cls][mask] += 1

    else:

        for laz_path in (tqdm(laz_paths,desc="Reading and rasterising the LAZ files...") if len(laz_paths)>12 else laz_paths):
            if cache_rasterization:
                cache_file = cache_dir / _cache_name_from_path(laz_path)
            raster, class_raster, _ = _rasterize_agg(
                laz_path=laz_path,
                laz_points=None,
                width=width,
                height=height,
                transform=transform,
                classification=classification,
                vegetation_classes=vegetation_classes,
                building_classes=building_classes,
                terrain_classes=terrain_classes,
                min_class_fraction=min_class_fraction,
                aggregation=aggregation,
                nodata=nodata,
                aoi=aoi_transformed,
                aoi_buffer_in_meters=aoi_buffer_in_meters,
                return_classification=True,
                filter_raster=filter_raster,
                filter_raster_band=filter_raster_band,
                filter_points_from_class=filter_points_from_class,
                filter_mode=filter_mode,
                lidar_class_priority=lidar_class_priority,
                lidar_class_window_diameter=0.5,
                cache_path=str(cache_file) if cache_rasterization else None,
                CLASS_NODATA=CLASS_NODATA,
                epsg=epsg
            )

            elev_valid = raster != nodata

            if aggregation == "mean":
                dm[elev_valid] += raster[elev_valid]
                count[elev_valid] += 1
            elif aggregation == "max":
                dm[elev_valid] = np.maximum(dm[elev_valid], raster[elev_valid])
            elif aggregation == "min":
                dm[elev_valid] = np.minimum(dm[elev_valid], raster[elev_valid])
            elif aggregation == "smart_for_roofs":
                dm[elev_valid] = raster[elev_valid]

            for cls in class_counts:
                mask = (class_raster == cls) & elev_valid
                class_counts[cls][mask] += 1
    del raster
    del class_raster
    gc.collect()

    # ------------------------------------------------------------
    # FINALIZE Digital Model
    # ------------------------------------------------------------
    if aggregation == "mean":
        valid = count > 0
        dm[valid] /= count[valid]
        dm[~valid] = nodata
        dm = dm.astype(np.float32)
    elif aggregation == "max":
        dm[dm == -np.inf] = nodata
    elif aggregation == "min":
        dm[dm == np.inf] = nodata
    elif aggregation == "smart_for_roofs":
        dm[~np.isfinite(dm)] = nodata
    # ------------------------------------------------------------
    # ISOLATED PIXELS
    # ------------------------------------------------------------
    if isolated_points_exclusion_ratio is not None:
        dm = _remove_isolated_pixels(
            dm, nodata,
            ratio=isolated_points_exclusion_ratio,
            window_size=isolated_window_size,
            aoi_mask=aoi_mask,
        )

    # ------------------------------------------------------------
    # FINALIZE CLASSIFICATION (MAJORITY VOTE)
    # ------------------------------------------------------------
    class_dm = np.full((height, width), CLASS_NODATA, dtype=np.uint8)

    max_count = np.zeros((height, width), dtype=np.uint32)

    for c, counts in class_counts.items():
        mask = counts > max_count

        class_dm[mask] = c
        max_count[mask] = counts[mask]
    del class_counts
    gc.collect()

    # ------------------------------------------------------------
    # AOI MASK
    # ------------------------------------------------------------
    if aoi_mask is not None:
        dm[~aoi_mask] = nodata
        class_dm[~aoi_mask] = CLASS_NODATA

    # ------------------------------------------------------------
    # GAP FILL (ELEVATION)
    # ------------------------------------------------------------
    logger.log("start filling")
    if fill_gaps is not None:
        if max_ram_gb is None:
            mem = psutil.virtual_memory()
            max_ram_gb = mem.available / (1024 ** 3) * 0.6
        logger.log(f"max_ram_gb={max_ram_gb}")
        dm = _fill_gaps(
            dm,
            nodata,
            method=fill_gaps,
            resolution=resolution,
            dist_threshold=resolution * 10,
            max_ram_gb=max_ram_gb,
            aoi_mask=aoi_mask,
            logger=logger
        )
    gc.collect()

    # ------------------------------------------------------------
    # GAP FILL (CLASSIFICATION)
    # ------------------------------------------------------------
    logger.log("start filling class")
    class_mask = (class_dm == CLASS_NODATA)
    if class_mask.any():
        class_dm = _nearest_fill_chunked(
            class_mask,
            class_dm.astype(np.float32),
            max_ram_gb,
            aoi_mask=aoi_mask,
        ).astype(np.uint8)
    del class_mask
    gc.collect()

    # ------------------------------------------------------------
    # BUILDING CLASS CORRECTION (in-memory, no I/O)
    # ------------------------------------------------------------
    if classification_correction_buildings_gdf is not None:

        buildings = classification_correction_buildings_gdf.copy()

        # -----------------------------------------
        # CRS alignment
        # -----------------------------------------
        if buildings.crs and buildings.crs.to_epsg() != epsg:
            buildings = buildings.to_crs(epsg=epsg)

        # -----------------------------------------
        # Buffer
        # -----------------------------------------
        if classification_correction_building_buffer_m > 0:
            buildings["geometry"] = buildings.geometry.buffer(
                classification_correction_building_buffer_m
            )

        if len(buildings):

            target = classification_correction_building_class_value
            unknown = 1  # consider making configurable

            protected = (
                vegetation_classes
                if classification_correction_avoid_vegetation_classes
                else None
            )

            # -----------------------------------------
            # 🔥 Compute mask ONCE (critical optimization)
            # -----------------------------------------
            building_mask = force_raster_value_based_on_geometries(
                raster_data=class_dm,
                transform=transform,
                profile={"nodata": CLASS_NODATA},
                geoms=buildings.geometry,
                return_mask=True,
            )

            # -----------------------------------------
            # MODE: enforce (only inside buildings)
            # -----------------------------------------
            if classification_correction_buildings_mode == "enforce":

                if protected is not None:
                    veg_mask = np.isin(class_dm, protected)
                    apply_mask = building_mask & (~veg_mask)
                else:
                    apply_mask = building_mask

                class_dm[apply_mask] = target

            # -----------------------------------------
            # MODE: strict (inside + remove outside)
            # -----------------------------------------
            elif classification_correction_buildings_mode == "strict":

                if protected is not None:
                    veg_mask = np.isin(class_dm, protected)
                    inside_mask = building_mask & (~veg_mask)
                else:
                    inside_mask = building_mask

                # 1️⃣ assign inside buildings
                class_dm[inside_mask] = target

                # 2️⃣ remove building class outside
                outside_mask = (~building_mask) & (class_dm == target)
                class_dm[outside_mask] = unknown

            else:
                raise ValueError(
                    f"Unknown classification_correction_buildings_mode: "
                    f"{classification_correction_buildings_mode}"
                )

        gc.collect()

    # ------------------------------------------------------------
    # Digital Model SMOOTHING (SEMANTIC + PER-OPERATOR)
    # ------------------------------------------------------------
    logger.log("start smoothing")
    if smooth_classes is not None and smooth_window_size is not None:

        smooth_classes = np.asarray(smooth_classes, dtype=np.uint8)

        # --------------------------------------------------------
        # Determine semantic groups
        # --------------------------------------------------------
        if smooth_unary:
            class_groups = {
                "unary": smooth_classes
            }
        else:
            terrain_vals = np.intersect1d(smooth_classes, terrain_classes)
            vegetation_vals = np.intersect1d(smooth_classes, vegetation_classes)
            building_vals = np.intersect1d(smooth_classes, building_classes)

            class_groups = {}
            if terrain_vals.size:
                class_groups["terrain"] = terrain_vals
            if vegetation_vals.size:
                class_groups["vegetation"] = vegetation_vals
            if building_vals.size:
                class_groups["buildings"] = building_vals

        if class_groups:

            dm_smooth = dm.copy()
            tmp = np.empty_like(dm, dtype=np.float32)
            del dm

            for _ in range(smooth_iterations):

                tmp[:] = dm_smooth

                for group_name, cls_vals in class_groups.items():

                    # ------------------------------------------------
                    # Select smoothing mode for this group
                    # ------------------------------------------------
                    if isinstance(smooth_mode, dict):
                        mode = smooth_mode.get(group_name, None)
                    else:
                        mode = smooth_mode

                    if mode is None:
                        continue  # no smoothing for this group

                    group_mask = np.isin(class_dm, cls_vals)

                    if aoi_mask is not None:
                        group_mask &= aoi_mask

                    valid = group_mask & (dm_smooth != nodata)
                    if not np.any(valid):
                        continue

                    elev = np.where(valid, dm_smooth, 0.0).astype(np.float32)
                    w = valid.astype(np.float32)

                    # ------------------------------------------------
                    # Apply selected smoothing operator
                    # ------------------------------------------------
                    if mode == "box":

                        elev_f = uniform_filter(
                            elev,
                            size=smooth_window_size,
                            mode="constant",
                            cval=0.0,
                        )
                        w_f = uniform_filter(
                            w,
                            size=smooth_window_size,
                            mode="constant",
                            cval=0.0,
                        )

                    elif mode == "gaussian":

                        sigma = smooth_window_size / 3.0

                        elev_f = gaussian_filter(
                            elev,
                            sigma=sigma,
                            mode="constant",
                            cval=0.0,
                        )
                        w_f = gaussian_filter(
                            w,
                            sigma=sigma,
                            mode="constant",
                            cval=0.0,
                        )

                    elif mode == "median":

                        elev_f = median_filter(
                            elev,
                            size=smooth_window_size,
                            mode="constant",
                            cval=0.0,
                        )
                        w_f = uniform_filter(
                            w,
                            size=smooth_window_size,
                            mode="constant",
                            cval=0.0,
                        )

                    elif mode == "opening":

                        elev_f = grey_opening(
                            elev,
                            size=smooth_window_size,
                        )
                        w_f = uniform_filter(
                            w,
                            size=smooth_window_size,
                            mode="constant",
                            cval=0.0,
                        )

                    elif mode == "closing":

                        elev_f = grey_closing(
                            elev,
                            size=smooth_window_size,
                        )
                        w_f = uniform_filter(
                            w,
                            size=smooth_window_size,
                            mode="constant",
                            cval=0.0,
                        )

                    elif mode == "bilateral":

                        # Normalize to avoid scaling issues
                        elev_norm = elev.copy()

                        # Bilateral requires finite values
                        elev_norm[~valid] = 0.0

                        sigma_spatial = smooth_window_size / 2.0
                        sigma_color = resolution * 2  # default, can be parameterized later

                        elev_f = denoise_bilateral(
                            elev_norm,
                            sigma_color=sigma_color,
                            sigma_spatial=sigma_spatial,
                            channel_axis=None,
                        ).astype(np.float32)

                        # Weight normalization same as others
                        w_f = uniform_filter(
                            w,
                            size=smooth_window_size,
                            mode="constant",
                            cval=0.0,
                        )

                    else:
                        raise ValueError(
                            f"Unknown smooth_mode '{mode}' for group '{group_name}'"
                        )

                    # ------------------------------------------------
                    # Normalize and write back
                    # ------------------------------------------------
                    smoothed = np.where(w_f > 0, elev_f / w_f, dm_smooth)

                    tmp[valid] = smoothed[valid]

                dm_smooth, tmp = tmp, dm_smooth

            dm = dm_smooth

    # ------------------------------------------------------------
    # SLOPE / ASPECT AND WRITE RASTERS
    # ------------------------------------------------------------

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if out_slope_path or out_aspect_path:
        slope = np.full_like(dm, nodata, dtype=np.float32)
        aspect = np.full_like(dm, nodata, dtype=np.float32)

        _compute_slope_aspect(dm, slope, aspect, nodata, abs(transform.a))

        if smooth_classes is not None:
            cls = np.asarray(smooth_classes, dtype=np.uint8)

            if slope_smooth_window:
                tmp = slope.copy()
                for _ in range(slope_smooth_iterations):
                    _smooth_slope_by_class(
                        slope, class_dm, cls,
                        nodata, slope_smooth_window, tmp
                    )
                    slope, tmp = tmp, slope

            if aspect_smooth_window:
                tmp = aspect.copy()
                for _ in range(aspect_smooth_iterations):
                    _smooth_aspect_by_class(
                        aspect, class_dm, cls,
                        nodata, aspect_smooth_window, tmp
                    )
                    aspect, tmp = tmp, aspect

        if out_slope_path:
            with rasterio.open(out_slope_path, "w", **profile) as dst:
                _write_raster_in_blocks(dst, slope, block_size)
        if not return_arrays:
            del slope
            gc.collect()

        if out_aspect_path:
            with rasterio.open(out_aspect_path, "w", **profile) as dst:
                _write_raster_in_blocks(dst, aspect, block_size)
        if not return_arrays:
            del aspect
            gc.collect()

    with rasterio.open(out_path, "w", **profile) as dst:
        _write_raster_in_blocks(dst,dm, block_size)
    if not return_arrays:
        del dm
        gc.collect()

    if out_classification_path:
        cp = profile.copy()
        cp.update(dtype="uint8", nodata=CLASS_NODATA)
        with rasterio.open(out_classification_path, "w", **cp) as dst:
            _write_raster_in_blocks(dst, class_dm, block_size)
        if not return_arrays:
            del class_dm
            gc.collect()

    if filter_raster:
        filter_raster.close()

    if return_arrays:
        return dm, class_dm, slope if 'slope' in locals() else None, \
            aspect if 'aspect' in locals() else None, transform
    else:
        return out_path


# =============================================================================
# HELPERS
# =============================================================================

def _laz_bbox(laz_path):
    """Read (minx, miny, maxx, maxy) from a LAZ header without loading points."""
    with laspy.open(str(laz_path)) as f:
        h = f.header
        return (float(h.mins[0]), float(h.mins[1]),
                float(h.maxs[0]), float(h.maxs[1]))


def _snap_to_grid(value, resolution, origin=0.0, mode="floor"):
    """Snap a coordinate to a grid aligned on `origin` with step `resolution`."""
    k = (value - origin) / resolution
    if mode == "floor":
        k = math.floor(k)
    elif mode == "ceil":
        k = math.ceil(k)
    else:
        k = round(k)
    return origin + k * resolution


def _aligned_grid_from_bounds(minx, miny, maxx, maxy, resolution):
    """Build an axis-aligned grid snapped to multiples of `resolution`."""
    minx_s = _snap_to_grid(minx, resolution, mode="floor")
    miny_s = _snap_to_grid(miny, resolution, mode="floor")
    maxx_s = _snap_to_grid(maxx, resolution, mode="ceil")
    maxy_s = _snap_to_grid(maxy, resolution, mode="ceil")
    width  = int(round((maxx_s - minx_s) / resolution))
    height = int(round((maxy_s - miny_s) / resolution))
    transform = from_origin(minx_s, maxy_s, resolution, resolution)
    return width, height, transform, (minx_s, miny_s, maxx_s, maxy_s)


def _compute_tile_buffer_pixels(
    *,
    smooth_window_size,
    smooth_iterations,
    slope_smooth_window,
    slope_smooth_iterations,
    aspect_smooth_window,
    aspect_smooth_iterations,
    fill_gaps,
    resolution,
    user_override=None,
):
    """
    Compute the halo (in pixels) required so that per-tile operations
    (smoothing, slope/aspect smoothing, gap-fill) produce seamless
    results when their core (unbuffered) region is written back.

    Logic:
      - Each smoothing pass with window W can contaminate W//2 pixels
        from its boundary. N iterations compound to (W//2) * N.
      - Slope/aspect use a 3x3 stencil, so +1 px if computed.
      - Gap filling fills up to dist_threshold = resolution*10 → 10 px.
      - Total buffer is the sum of the maximum chain that could
        propagate influence inward (we are conservative and just sum
        the worst-case per stage).
    """
    if user_override is not None:
        return int(user_override)

    halo = 0

    # Elevation smoothing
    if smooth_window_size:
        halo += (int(smooth_window_size) // 2) * max(1, int(smooth_iterations or 1))

    # Slope/aspect operators (3x3 central differences) + their smoothing
    halo_slope = 1
    if slope_smooth_window:
        halo_slope += (int(slope_smooth_window) // 2) * max(1, int(slope_smooth_iterations or 1))
    halo_aspect = 1
    if aspect_smooth_window:
        halo_aspect += (int(aspect_smooth_window) // 2) * max(1, int(aspect_smooth_iterations or 1))

    halo += max(halo_slope, halo_aspect)

    # Gap fill (matches _fill_gaps dist_threshold = resolution * 10)
    # Be generous: the fill can pull from further when interpolating,
    # and we also want to absorb any minor operator asymmetry.
    if fill_gaps is not None:
        halo += 16

    # Safety margin (absorbs rounding + operator asymmetry)
    halo += 8

    return int(halo)


# =============================================================================
# TILE PLAN
# =============================================================================

def _build_tile_plan(
    laz_paths,
    resolution,
    buffer_m,
    aoi_transformed=None,
    aoi_buffer_in_meters=0.0,
):
    """
    One tile per LAZ file. For each file, compute:
        - core bounds (the file's own bbox, snapped to grid)
        - buffered bounds (core + halo in meters, snapped to grid)
        - list of NEIGHBOR LAZ files whose bbox intersects the buffered bbox
    Neighbors are needed so the halo contains real LiDAR points, avoiding
    seam artifacts from gap-fill at tile edges.
    """
    # 1) Read all bboxes once
    all_bboxes = []
    kept_paths = []
    for p in laz_paths:
        try:
            bb = _laz_bbox(p)
        except Exception as e:
            logging.warning(f"Skipping {p}: cannot read header ({e})")
            continue
        all_bboxes.append(bb)
        kept_paths.append(str(p))

    # 2) Spatial index on file bboxes for fast neighbor lookup
    geoms = [shp_box(*bb) for bb in all_bboxes]
    tree = STRtree(geoms) if STRtree is not None else None

    def _candidate_indices(buf_box):
        """Return candidate tile indices that *may* intersect buf_box."""
        if tree is None:
            # Brute-force fallback (fine for a few hundred tiles)
            return range(len(geoms))
        hits = tree.query(buf_box)
        # shapely 2.x -> numpy array of ints; 1.x -> list of geometries
        out = []
        for h in hits:
            if isinstance(h, (int, np.integer)):
                out.append(int(h))
            else:
                # 1.x: find index by identity
                for j, g in enumerate(geoms):
                    if g is h:
                        out.append(j)
                        break
        return out

    tiles = []
    for i, (p, bb) in enumerate(zip(kept_paths, all_bboxes)):
        # AOI filtering
        if aoi_transformed is not None:
            tg = geoms[i]
            if aoi_buffer_in_meters:
                tg = tg.buffer(aoi_buffer_in_meters)
            if not tg.intersects(aoi_transformed):
                continue

        # Buffered bbox for neighbor search
        minx, miny, maxx, maxy = bb
        buf_box = shp_box(
            minx - buffer_m, miny - buffer_m,
            maxx + buffer_m, maxy + buffer_m,
        )
        neighbors = []
        for j in _candidate_indices(buf_box):
            if j == i:
                continue
            if geoms[j].intersects(buf_box):
                neighbors.append(kept_paths[j])

        tiles.append({
            "path": p,
            "bbox": bb,
            "neighbors": neighbors,
        })
    return tiles


def _global_bounds_from_tiles(tiles, resolution, aoi_transformed=None):
    """Union of tile bboxes, optionally clipped to AOI bounds."""
    if not tiles:
        raise ValueError("No LAZ tiles to process.")

    xs_min = min(t["bbox"][0] for t in tiles)
    ys_min = min(t["bbox"][1] for t in tiles)
    xs_max = max(t["bbox"][2] for t in tiles)
    ys_max = max(t["bbox"][3] for t in tiles)

    if aoi_transformed is not None:
        ax0, ay0, ax1, ay1 = aoi_transformed.bounds
        xs_min = max(xs_min, ax0)
        ys_min = max(ys_min, ay0)
        xs_max = min(xs_max, ax1)
        ys_max = min(ys_max, ay1)

    return _aligned_grid_from_bounds(xs_min, ys_min, xs_max, ys_max, resolution)


# =============================================================================
# PER-TILE WORKER
# =============================================================================

def _process_single_tile(tile_kwargs):
    """
    Run build_digital_model_from_lidar_pure on a single LAZ tile and
    return the in-memory arrays + the window describing where the
    tile's CORE (unbuffered) region lives inside the global raster.

    The worker:
      - computes the tile's buffered bounds
      - loads THIS tile + NEIGHBOR tiles (to populate the halo with real data)
      - calls the existing pure function with that bbox as `bounds`
      - clips the result back to the core region
      - returns arrays + destination window in the global raster
    """
    # Silence noisy internals (tqdm from pure fn + numpy warnings)
    import contextlib, io, os as _os
    _os.environ["TQDM_DISABLE"] = "1"

    # Local imports inside the worker so it is picklable across processes
    from .dsm_dem_estimation import build_digital_model_from_lidar_pure  # noqa

    params      = tile_kwargs["params"]
    tile        = tile_kwargs["tile"]
    buffer_m    = tile_kwargs["buffer_m"]
    global_tf   = tile_kwargs["global_transform"]
    global_w    = tile_kwargs["global_width"]
    global_h    = tile_kwargs["global_height"]
    resolution  = params["resolution"]
    epsg        = params["epsg"]
    nodata      = params["nodata"]
    want_slope  = tile_kwargs["want_slope"]
    want_aspect = tile_kwargs["want_aspect"]
    want_class  = tile_kwargs["want_class"]
    tmp_dir     = tile_kwargs["tmp_dir"]
    buildings_gdf_pickle = tile_kwargs.get("buildings_gdf_pickle")

    cminx, cminy, cmaxx, cmaxy = tile["bbox"]

    # ---------- CORE region snapped to the GLOBAL grid ----------
    gx0 = global_tf.c                       # global minx
    gy1 = global_tf.f                       # global maxy
    res = resolution

    core_col_off = int(math.floor((cminx - gx0) / res))
    core_row_off = int(math.floor((gy1 - cmaxy) / res))
    core_col_end = int(math.ceil((cmaxx - gx0) / res))
    core_row_end = int(math.ceil((gy1 - cminy) / res))

    core_col_off = max(0, core_col_off)
    core_row_off = max(0, core_row_off)
    core_col_end = min(global_w, core_col_end)
    core_row_end = min(global_h, core_row_end)

    core_w = core_col_end - core_col_off
    core_h = core_row_end - core_row_off
    if core_w <= 0 or core_h <= 0:
        return None

    # ---------- BUFFERED region (what we actually process) ----------
    buf_col_off = max(0, core_col_off - buffer_m)
    buf_row_off = max(0, core_row_off - buffer_m)
    buf_col_end = min(global_w, core_col_end + buffer_m)
    buf_row_end = min(global_h, core_row_end + buffer_m)

    buf_minx = gx0 + buf_col_off * res
    buf_maxx = gx0 + buf_col_end * res
    buf_maxy = gy1 - buf_row_off * res
    buf_miny = gy1 - buf_row_end * res

    # ---------- Load buildings GDF subset (if any) ----------
    buildings_gdf = None
    if buildings_gdf_pickle is not None:
        import pickle
        with open(buildings_gdf_pickle, "rb") as fh:
            full_gdf, _ = pickle.load(fh)
        idx = list(full_gdf.sindex.intersection((buf_minx, buf_miny, buf_maxx, buf_maxy)))
        if idx:
            buildings_gdf = full_gdf.iloc[idx].copy()

    # ---------- Output paths for this tile (temp) ----------
    tile_id  = hashlib.sha1(tile["path"].encode()).hexdigest()[:12]
    tile_dir = Path(tmp_dir) / tile_id
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_out = tile_dir / "elev.tif"

    # ---------- LAZ files to feed the pure function ----------
    # Self + neighbors so the halo contains REAL points, not synthetic fill.
    # The pure function already clips to `bounds`, so extra area is free-of-cost.
    laz_inputs = [tile["path"]] + list(tile.get("neighbors", []))

    # ---------- Build kwargs for the pure function ----------
    pure_kwargs = dict(params)
    pure_kwargs.update(dict(
        laz_paths=laz_inputs,
        laz_points=None,
        out_path=str(tile_out),
        return_arrays=True,
        bounds=(buf_minx, buf_miny, buf_maxx, buf_maxy),
        out_slope_path=str(tile_dir / "slope.tif") if want_slope else None,
        out_aspect_path=str(tile_dir / "aspect.tif") if want_aspect else None,
        out_classification_path=str(tile_dir / "class.tif") if want_class else None,
        classification_correction_buildings_gdf=buildings_gdf,
        cache_rasterization=False,
    ))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(divide="ignore", invalid="ignore"):
            dm, class_dm, slope, aspect, _tf = build_digital_model_from_lidar_pure(**pure_kwargs)

    # ---------- Clip buffered arrays back to CORE region ----------
    core_off_r = core_row_off - buf_row_off
    core_off_c = core_col_off - buf_col_off

    def _core_crop(arr):
        if arr is None:
            return None
        return np.ascontiguousarray(
            arr[core_off_r:core_off_r + core_h,
                core_off_c:core_off_c + core_w]
        )

    result = {
        "window": Window(core_col_off, core_row_off, core_w, core_h),
        "dm":      _core_crop(dm),
        "class_dm": _core_crop(class_dm) if want_class else None,
        "slope":   _core_crop(slope) if want_slope else None,
        "aspect":  _core_crop(aspect) if want_aspect else None,
        "tile_dir": str(tile_dir),
    }

    del dm, class_dm, slope, aspect
    gc.collect()
    return result


# =============================================================================
# MAIN ENTRY
# =============================================================================

def build_digital_model_from_lidar_tiled(
        laz_paths,
        out_path,
        *,
        # Same signature as the pure function for everything else:
        resolution=1.0,
        bounds=None,
        epsg=25831,
        classification=(2, 8,),
        vegetation_classes=(3, 4, 5),
        building_classes=(6, 17, 76, 77),
        terrain_classes=(2, 8),
        min_class_fraction=0.0,
        aggregation="mean",
        lidar_class_priority=None,
        lidar_class_window_diameter=0.5,
        nodata=-9999.0,
        max_ram_gb=None,
        aoi=None,
        aoi_epsg=None,
        aoi_buffer_in_meters=0.0,
        filter_raster_path=None,
        filter_raster_band=1,
        filter_points_from_class=None,
        filter_mode="exclude",
        isolated_points_exclusion_ratio=None,
        isolated_window_size=3,
        smooth_classes=None,
        smooth_mode=None,
        smooth_unary=False,
        smooth_window_size=None,
        smooth_iterations=1,
        slope_smooth_window=None,
        slope_smooth_iterations=1,
        aspect_smooth_window=None,
        aspect_smooth_iterations=1,
        fill_gaps="nearest",
        classification_correction_buildings_gdf=None,
        classification_correction_buildings_mode="enforce",
        classification_correction_building_class_value=6,
        classification_correction_building_buffer_m=1.0,
        classification_correction_avoid_vegetation_classes=True,
        block_size=1024,

        # ---- NEW: tiling control ----
        out_classification_path=None,
        out_slope_path=None,
        out_aspect_path=None,
        tile_buffer_pixels=None,   # None = auto from smoothing/slope/gap-fill
        n_workers=1,
        tmp_dir=None,
        keep_tile_files=False,
):
    """
    Tiled wrapper around build_digital_model_from_lidar_pure.

    One tile per LAZ file. For each tile we process a buffered region
    (halo) so smoothing / slope / gap-fill produce seam-free results,
    then we write ONLY the core (unbuffered) region into pre-allocated
    output GeoTIFFs via windowed writes.

    Parameters mirror the pure function; extras:
      - tile_buffer_pixels: halo in pixels (auto if None)
      - n_workers: 1 = sequential, >1 = ProcessPoolExecutor
      - tmp_dir: where per-tile intermediates live (auto if None)
      - keep_tile_files: keep per-tile rasters after merging
    """
    # ------------------------------------------------------------
    # Silence numpy divide/invalid warnings emitted inside the
    # pure function's smoothing code. They are harmless (np.where
    # evaluates both branches) but spam the console over 175+ tiles.
    # ------------------------------------------------------------
    _warn_ctx = warnings.catch_warnings()
    _warn_ctx.__enter__()
    warnings.simplefilter("ignore")
    _np_err = np.seterr(divide="ignore", invalid="ignore")

    # ------------------------------------------------------------
    # AOI reprojection
    # ------------------------------------------------------------
    aoi_transformed = None
    if aoi is not None:
        if aoi_epsg is None:
            raise ValueError("aoi_epsg must be provided when aoi is given")
        if aoi_epsg != epsg:
            tr = Transformer.from_crs(f"EPSG:{aoi_epsg}", f"EPSG:{epsg}", always_xy=True)
            aoi_transformed = shp_transform(tr.transform, aoi)
        else:
            aoi_transformed = aoi

    # ------------------------------------------------------------
    # Halo (tile buffer) — computed FIRST, used by tile planner
    # ------------------------------------------------------------
    buffer_px = _compute_tile_buffer_pixels(
        smooth_window_size=smooth_window_size,
        smooth_iterations=smooth_iterations,
        slope_smooth_window=slope_smooth_window,
        slope_smooth_iterations=slope_smooth_iterations,
        aspect_smooth_window=aspect_smooth_window,
        aspect_smooth_iterations=aspect_smooth_iterations,
        fill_gaps=fill_gaps,
        resolution=resolution,
        user_override=tile_buffer_pixels,
    )
    buffer_m = buffer_px * resolution
    logging.info(f"[tiled] per-tile buffer = {buffer_px} px ({buffer_m} m)")

    # ------------------------------------------------------------
    # Tile plan (one tile per LAZ file, with neighbor awareness)
    # ------------------------------------------------------------
    tiles = _build_tile_plan(
        laz_paths,
        resolution=resolution,
        buffer_m=buffer_m,
        aoi_transformed=aoi_transformed,
        aoi_buffer_in_meters=aoi_buffer_in_meters,
    )
    if not tiles:
        raise ValueError("No LAZ tiles intersect the AOI.")
    avg_neighbors = sum(len(t["neighbors"]) for t in tiles) / max(1, len(tiles))
    logging.info(f"[tiled] {len(tiles)} tiles, avg {avg_neighbors:.1f} neighbors/tile")

    # ------------------------------------------------------------
    # Global grid (covering all tile bboxes, optionally clipped by AOI)
    # ------------------------------------------------------------
    if bounds is not None:
        gw, gh, gtf, _ = _aligned_grid_from_bounds(*bounds, resolution)
    else:
        gw, gh, gtf, _ = _global_bounds_from_tiles(tiles, resolution, aoi_transformed)

    logging.info(f"[tiled] global raster: {gw} x {gh} px @ {resolution} m")

    # ------------------------------------------------------------
    # Output GeoTIFFs (pre-allocated, tiled, windowed writes)
    # ------------------------------------------------------------
    want_slope  = out_slope_path  is not None
    want_aspect = out_aspect_path is not None
    want_class  = out_classification_path is not None

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    elev_profile = {
        "driver":    "GTiff",
        "height":    gh,
        "width":     gw,
        "count":     1,
        "dtype":     "float32",
        "crs":       CRS.from_epsg(epsg),
        "transform": gtf,
        "nodata":    nodata,
        "tiled":     True,
        "blockxsize": block_size,
        "blockysize": block_size,
        "compress":  "lzw",
        "BIGTIFF":   "IF_SAFER",
    }
    class_profile = dict(elev_profile, dtype="uint8", nodata=255)

    # Open output datasets (kept open for the whole run)
    dst_elev   = rasterio.open(out_path, "w", **elev_profile)
    # Fill with nodata explicitly (GTiff init is platform-dependent)
    dst_elev.write(np.full((gh, gw), nodata, dtype="float32"), 1)

    dst_slope  = None
    dst_aspect = None
    dst_class  = None
    if want_slope:
        dst_slope = rasterio.open(out_slope_path, "w", **elev_profile)
        dst_slope.write(np.full((gh, gw), nodata, dtype="float32"), 1)
    if want_aspect:
        dst_aspect = rasterio.open(out_aspect_path, "w", **elev_profile)
        dst_aspect.write(np.full((gh, gw), nodata, dtype="float32"), 1)
    if want_class:
        dst_class = rasterio.open(out_classification_path, "w", **class_profile)
        dst_class.write(np.full((gh, gw), 255, dtype="uint8"), 1)

    # ------------------------------------------------------------
    # Buildings GDF: pickle once, workers load-and-filter by bbox
    # ------------------------------------------------------------
    buildings_pickle_path = None
    if classification_correction_buildings_gdf is not None and len(classification_correction_buildings_gdf):
        import pickle, tempfile
        gdf = classification_correction_buildings_gdf
        if gdf.crs and gdf.crs.to_epsg() != epsg:
            gdf = gdf.to_crs(epsg=epsg)
        # Trigger sindex build now so every worker reuses it
        _ = gdf.sindex
        tmp_root = tmp_dir or tempfile.mkdtemp(prefix="dm_tiles_")
        os.makedirs(tmp_root, exist_ok=True)
        buildings_pickle_path = os.path.join(tmp_root, "buildings.pkl")
        with open(buildings_pickle_path, "wb") as fh:
            pickle.dump((gdf, None), fh)
    else:
        if tmp_dir is None:
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix="dm_tiles_")
        else:
            os.makedirs(tmp_dir, exist_ok=True)
        tmp_root = tmp_dir

    # ------------------------------------------------------------
    # Pure-function kwargs carried to each worker (shared across tiles)
    # ------------------------------------------------------------
    shared_params = dict(
        resolution=resolution,
        epsg=epsg,
        classification=classification,
        vegetation_classes=vegetation_classes,
        building_classes=building_classes,
        terrain_classes=terrain_classes,
        min_class_fraction=min_class_fraction,
        aggregation=aggregation,
        lidar_class_priority=lidar_class_priority,
        lidar_class_window_diameter=lidar_class_window_diameter,
        nodata=nodata,
        max_ram_gb=max_ram_gb,
        aoi=aoi,
        aoi_epsg=aoi_epsg,
        aoi_buffer_in_meters=aoi_buffer_in_meters,
        filter_raster_path=filter_raster_path,
        filter_raster_band=filter_raster_band,
        filter_points_from_class=filter_points_from_class,
        filter_mode=filter_mode,
        isolated_points_exclusion_ratio=isolated_points_exclusion_ratio,
        isolated_window_size=isolated_window_size,
        smooth_classes=smooth_classes,
        smooth_mode=smooth_mode,
        smooth_unary=smooth_unary,
        smooth_window_size=smooth_window_size,
        smooth_iterations=smooth_iterations,
        slope_smooth_window=slope_smooth_window,
        slope_smooth_iterations=slope_smooth_iterations,
        aspect_smooth_window=aspect_smooth_window,
        aspect_smooth_iterations=aspect_smooth_iterations,
        fill_gaps=fill_gaps,
        classification_correction_buildings_mode=classification_correction_buildings_mode,
        classification_correction_building_class_value=classification_correction_building_class_value,
        classification_correction_building_buffer_m=classification_correction_building_buffer_m,
        classification_correction_avoid_vegetation_classes=classification_correction_avoid_vegetation_classes,
        block_size=block_size,
    )

    tile_tasks = [
        {
            "tile":   t,
            "params": shared_params,
            "buffer_m": buffer_px,
            "global_transform": gtf,
            "global_width":  gw,
            "global_height": gh,
            "want_slope":  want_slope,
            "want_aspect": want_aspect,
            "want_class":  want_class,
            "tmp_dir":     tmp_root,
            "buildings_gdf_pickle": buildings_pickle_path,
        }
        for t in tiles
    ]

    # ------------------------------------------------------------
    # Run tiles (serial writer, parallel compute)
    # ------------------------------------------------------------
    write_lock = Lock()

    def _write_result(res):
        if res is None:
            return
        win = res["window"]
        with write_lock:
            dst_elev.write(res["dm"].astype("float32"), 1, window=win)
            if want_slope and res["slope"] is not None:
                dst_slope.write(res["slope"].astype("float32"), 1, window=win)
            if want_aspect and res["aspect"] is not None:
                dst_aspect.write(res["aspect"].astype("float32"), 1, window=win)
            if want_class and res["class_dm"] is not None:
                dst_class.write(res["class_dm"].astype("uint8"), 1, window=win)

        if not keep_tile_files:
            import shutil
            shutil.rmtree(res["tile_dir"], ignore_errors=True)

    try:
        if n_workers is None or n_workers <= 1:
            for task in tqdm(tile_tasks, desc="Tiles"):
                res = _process_single_tile(task)
                _write_result(res)
                gc.collect()
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_process_single_tile, t): t for t in tile_tasks}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Tiles"):
                    try:
                        res = fut.result()
                    except Exception as e:
                        t = futures[fut]
                        logging.exception(f"[tiled] tile failed: {t['tile']['path']}: {e}")
                        continue
                    _write_result(res)
                    gc.collect()
    finally:
        dst_elev.close()
        if dst_slope  is not None: dst_slope.close()
        if dst_aspect is not None: dst_aspect.close()
        if dst_class  is not None: dst_class.close()

        if not keep_tile_files:
            import shutil
            # Remove only auto-created tmp
            if tmp_dir is None and buildings_pickle_path:
                shutil.rmtree(os.path.dirname(buildings_pickle_path), ignore_errors=True)

        # Restore numpy err state / warnings filter
        np.seterr(**_np_err)
        _warn_ctx.__exit__(None, None, None)

    return out_path

@njit(parallel=True, fastmath=True)
def _smooth_slope_by_class(
    slope,
    class_raster,
    classes_to_smooth,
    nodata,
    window,
    out,
):
    H, W = slope.shape
    r = window // 2

    for i in prange(H):
        for j in range(W):

            v = slope[i, j]
            if v == nodata:
                out[i, j] = nodata
                continue

            cls = class_raster[i, j]

            # Only smooth pixels whose OWN class is allowed
            allowed = False
            for c in classes_to_smooth:
                if cls == c:
                    allowed = True
                    break

            if not allowed:
                out[i, j] = v
                continue

            acc = 0.0
            n = 0

            for di in range(-r, r + 1):
                ii = i + di
                if ii < 0 or ii >= H:
                    continue

                for dj in range(-r, r + 1):
                    jj = j + dj
                    if jj < 0 or jj >= W:
                        continue

                    # STRICT: neighbor must ALSO be in smooth_classes
                    ncls = class_raster[ii, jj]
                    ok = False
                    for c in classes_to_smooth:
                        if ncls == c:
                            ok = True
                            break
                    if not ok:
                        continue

                    vv = slope[ii, jj]
                    if vv == nodata:
                        continue

                    acc += vv
                    n += 1

            out[i, j] = acc / n if n > 0 else v


@njit(parallel=True, fastmath=True)
def _smooth_slope_unified(
    slope,
    nodata,
    window,
    out,
):
    """Unified slope smoothing that ignores classification."""
    H, W = slope.shape
    r = window // 2

    for i in prange(H):
        for j in range(W):

            v = slope[i, j]
            if v == nodata:
                out[i, j] = nodata
                continue

            acc = 0.0
            n = 0

            for di in range(-r, r + 1):
                ii = i + di
                if ii < 0 or ii >= H:
                    continue

                for dj in range(-r, r + 1):
                    jj = j + dj
                    if jj < 0 or jj >= W:
                        continue

                    vv = slope[ii, jj]
                    if vv == nodata:
                        continue

                    acc += vv
                    n += 1

            out[i, j] = acc / n if n > 0 else v

@njit(parallel=True, fastmath=True)
def _smooth_aspect_by_class(
    aspect,
    class_raster,
    classes_to_smooth,
    nodata,
    window,
    out,
):
    H, W = aspect.shape
    r = window // 2

    for i in prange(H):
        for j in range(W):

            a = aspect[i, j]
            if a == nodata:
                out[i, j] = nodata
                continue

            cls = class_raster[i, j]

            # Only smooth pixels whose OWN class is allowed
            allowed = False
            for c in classes_to_smooth:
                if cls == c:
                    allowed = True
                    break

            if not allowed:
                out[i, j] = a
                continue

            acc_x = 0.0
            acc_y = 0.0
            n = 0

            for di in range(-r, r + 1):
                ii = i + di
                if ii < 0 or ii >= H:
                    continue

                for dj in range(-r, r + 1):
                    jj = j + dj
                    if jj < 0 or jj >= W:
                        continue

                    # STRICT neighbor check
                    ncls = class_raster[ii, jj]
                    ok = False
                    for c in classes_to_smooth:
                        if ncls == c:
                            ok = True
                            break
                    if not ok:
                        continue

                    aa = aspect[ii, jj]
                    if aa == nodata:
                        continue

                    acc_x += np.sin(aa)
                    acc_y += np.cos(aa)
                    n += 1

            if n > 0:
                out[i, j] = np.arctan2(acc_x / n, acc_y / n)
                if out[i, j] < 0:
                    out[i, j] += 2.0 * np.pi
            else:
                out[i, j] = a


@njit(parallel=True, fastmath=True)
def _smooth_aspect_unified(
    aspect,
    nodata,
    window,
    out,
):
    """Unified aspect smoothing that ignores classification."""
    H, W = aspect.shape
    r = window // 2

    for i in prange(H):
        for j in range(W):

            a = aspect[i, j]
            if a == nodata:
                out[i, j] = nodata
                continue

            acc_x = 0.0
            acc_y = 0.0
            n = 0

            for di in range(-r, r + 1):
                ii = i + di
                if ii < 0 or ii >= H:
                    continue

                for dj in range(-r, r + 1):
                    jj = j + dj
                    if jj < 0 or jj >= W:
                        continue

                    aa = aspect[ii, jj]
                    if aa == nodata:
                        continue

                    acc_x += np.sin(aa)
                    acc_y += np.cos(aa)
                    n += 1

            if n > 0:
                out[i, j] = np.arctan2(acc_x / n, acc_y / n)
                if out[i, j] < 0:
                    out[i, j] += 2.0 * np.pi
            else:
                out[i, j] = a

@njit(parallel=True, fastmath=True) 
def _smooth_unified(
    dem,
    nodata,
    window,
    out,
):
    """Unified DEM smoothing that ignores classification."""
    H, W = dem.shape
    r = window // 2

    for i in prange(H):
        for j in range(W):

            z = dem[i, j]
            if z == nodata:
                out[i, j] = nodata
                continue

            acc = 0.0
            n = 0

            for di in range(-r, r + 1):
                ii = i + di
                if ii < 0 or ii >= H:
                    continue

                for dj in range(-r, r + 1):
                    jj = j + dj
                    if jj < 0 or jj >= W:
                        continue

                    zv = dem[ii, jj]
                    if zv == nodata:
                        continue

                    acc += zv
                    n += 1

            out[i, j] = acc / n if n > 0 else z


@njit(parallel=True, fastmath=True)
def _smooth_by_class(
    dem,
    class_raster,
    classes_to_smooth,
    nodata,
    window,
    out,
):
    H, W = dem.shape
    r = window // 2

    for i in prange(H):
        for j in range(W):

            z = dem[i, j]
            if z == nodata:
                out[i, j] = nodata
                continue

            cls = class_raster[i, j]

            smooth = False
            for c in classes_to_smooth:
                if cls == c:
                    smooth = True
                    break

            if not smooth:
                out[i, j] = z
                continue

            acc = 0.0
            n = 0

            for di in range(-r, r + 1):
                ii = i + di
                if ii < 0 or ii >= H:
                    continue

                for dj in range(-r, r + 1):
                    jj = j + dj
                    if jj < 0 or jj >= W:
                        continue

                    if class_raster[ii, jj] != cls:
                        continue

                    zv = dem[ii, jj]
                    if zv == nodata:
                        continue

                    acc += zv
                    n += 1

            out[i, j] = acc / n if n > 0 else z


def _remove_isolated_pixels(
    raster: np.ndarray,
    nodata: float,
    *,
    ratio: float = 0.25,
    window_size: int = 3,
    aoi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Remove isolated valid pixels based on neighborhood support ratio.

    Parameters
    ----------
    raster : np.ndarray
        DEM raster
    nodata : float
        nodata value
    ratio : float
        Minimum fraction of valid neighbors required to keep a pixel
    window_size : int
        Neighborhood size (must be odd, e.g. 3 or 5)
    aoi_mask : Optional[np.ndarray]
        If provided, only pixels inside AOI are considered

    Returns
    -------
    np.ndarray
        Filtered raster
    """

    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1]")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    out = raster.copy()

    valid = raster != nodata
    if aoi_mask is not None:
        valid = valid & aoi_mask

    if not np.any(valid):
        return out

    # Binary mask of valid pixels
    valid_int = valid.astype(np.uint8)

    # Convolution kernel (exclude center pixel)
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    kernel[window_size // 2, window_size // 2] = 0

    # Count valid neighbors
    neighbor_count = convolve(
        valid_int,
        kernel,
        mode="constant",
        cval=0,
    )

    max_neighbors = kernel.sum()
    neighbor_ratio = neighbor_count / max_neighbors

    # Identify isolated pixels
    isolated = valid & (neighbor_ratio < ratio)

    # Remove them
    out[isolated] = nodata

    return out


def _filter_lidar_points_by_raster(
    xs: np.ndarray,
    ys: np.ndarray,
    raster: rasterio.io.DatasetReader,
    band: int,
    value_range: tuple,
    mode: str = "exclude",
) -> np.ndarray:
    """
    Return boolean mask of points to KEEP based on raster value range.
    """

    if mode not in {"exclude", "include"}:
        raise ValueError("filter_mode must be 'exclude' or 'include'")

    # ✅ VECTORISED row/col computation (no scalar casting)
    rows, cols = rasterio.transform.rowcol(
        filter_raster.transform, xs, ys, op=np.floor
    )

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    valid = (
        (rows >= 0) & (rows < raster.height) &
        (cols >= 0) & (cols < raster.width)
    )

    keep = np.ones(xs.shape[0], dtype=bool)

    if not np.any(valid):
        return keep

    # Read band once
    band_data = raster.read(band)

    lo, hi = value_range
    rv = rows[valid].astype(np.intp, copy=False)
    cv = cols[valid].astype(np.intp, copy=False)

    in_range[valid] = (
            (band[rv, cv] >= lo) &
            (band[rv, cv] <= hi)
    )

    if mode == "exclude":
        keep[valid] = ~in_range
    else:  # include
        keep[valid] = in_range

    return keep

def remove_objects_from_dsm(
    dsm_path: str,
    objects: Union[gpd.GeoDataFrame, str],
    out_path: str,
    *,
    buffer_m: float = 1.0,
    nodata_value: float = -9999.0,
    save_mask_debug: Optional[str] = None,
) -> str:
    """
    Remove objects (e.g., buildings) from DSM by masking them as NoData.
    
    This is a simplified version that only masks out object areas without interpolation,
    leaving NoData values for other processing steps to handle.
    
    Parameters
    ----------
    dsm_path : str
        Path to input DSM raster
    objects : Union[gpd.GeoDataFrame, str]
        GeoDataFrame or path to shapefile containing object geometries to remove
    out_path : str
        Path for output masked DSM
    buffer_m : float, default 1.0
        Buffer distance in meters to apply around objects
    nodata_value : float, default -9999.0
        NoData value to assign to masked areas
    save_mask_debug : Optional[str], default None
        Optional path to save debug mask showing masked areas
        
    Returns
    -------
    str
        Path to output masked DSM
    """
    import time
    start_time = time.time()
    
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    # Read objects
    print("Reading objects...")
    if isinstance(objects, str):
        objects_gdf = gpd.read_file(objects)
    else:
        objects_gdf = objects.copy()
        
    if objects_gdf.empty:
        print("No objects to remove, copying original DSM")
        import shutil
        shutil.copy2(dsm_path, out_path)
        return out_path
    
    print(f"Processing {len(objects_gdf)} objects")
    
    # Read DSM metadata first
    print("Reading DSM metadata...")
    with rasterio.open(dsm_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = src.shape
    
    # Ensure objects are in same CRS as DSM
    if objects_gdf.crs is None:
        raise ValueError("Objects GeoDataFrame has no CRS.")
    if crs and objects_gdf.crs != crs:
        print("Converting CRS...")
        objects_gdf = objects_gdf.to_crs(crs)
    
    # Apply buffer if specified - optimize by doing all at once
    print(f"Applying {buffer_m}m buffer...")
    if buffer_m > 0:
        # Use unary_union to merge overlapping geometries first, then buffer
        from shapely.ops import unary_union
        if len(objects_gdf) > 100:  # Only merge if many objects
            print("Merging overlapping geometries for performance...")
            merged_geom = unary_union(objects_gdf.geometry.tolist())
            if hasattr(merged_geom, 'geoms'):
                shapes = [mapping(geom.buffer(buffer_m)) for geom in merged_geom.geoms]
            else:
                shapes = [mapping(merged_geom.buffer(buffer_m))]
        else:
            shapes = [mapping(geom.buffer(buffer_m)) for geom in objects_gdf.geometry if geom and not geom.is_empty]
    else:
        shapes = [mapping(geom) for geom in objects_gdf.geometry if geom and not geom.is_empty]
    
    if shapes:
        print("Creating mask...")
        from rasterio.features import rasterize
        
        mask = rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype='uint8',
            all_touched=True  # This can be faster for small objects
        )
        
        print("Reading and processing DSM data...")
        # Read DSM data only when needed
        with rasterio.open(dsm_path) as src:
            dsm_data = src.read(1)
        
        # Apply mask to DSM (set object areas to NoData)
        dsm_masked = dsm_data.astype(np.float32)
        dsm_masked[mask == 1] = nodata_value
        
        # Update profile
        profile.update(dtype=rasterio.float32, nodata=nodata_value)
        
        print("Saving masked DSM...")
        # Save masked DSM
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(dsm_masked, 1)
            
        # Save debug mask if requested
        if save_mask_debug:
            print("Saving debug mask...")
            mask_profile = profile.copy()
            mask_profile.update(dtype=rasterio.uint8, nodata=0)
            with rasterio.open(save_mask_debug, 'w', **mask_profile) as dst:
                dst.write(mask, 1)
        
        masked_pixels = np.sum(mask)
        total_pixels = mask.size
        print(f"Masked {masked_pixels:,} pixels ({masked_pixels/total_pixels*100:.1f}%) as objects")
        
    else:
        print("No valid object geometries found, copying original DSM")
        import shutil
        shutil.copy2(dsm_path, out_path)
    
    elapsed = time.time() - start_time
    print(f"Object removal completed in {elapsed:.1f} seconds")
    return out_path


def build_dem_from_masked_dsm(
    dsm_path: str,
    out_dem_path: str,
    *,
    interpolation_method: str = "linear",  # "cubic", "linear", "nearest"
    chunk_size: int = 1024,  # Smaller chunks for memory efficiency
    n_jobs: int = -1,
    nodata_out: float = -9999.0,
    save_mask_debug: Optional[str] = None,
) -> str:
    """
    Build DEM from DSM that already has objects masked as NoData.
    
    This is a simplified and faster version specifically for DSMs where objects
    (like buildings) have already been removed and set to NoData values.
    It simply interpolates the NoData areas using surrounding valid elevation data.
    
    Parameters
    ----------
    dsm_path : str
        Path to DSM with objects already masked as NoData
    out_dem_path : str  
        Path for output DEM
    interpolation_method : str, default "linear"
        Interpolation method: "linear", "cubic", "nearest"
    chunk_size : int, default 1024
        Chunk size for processing large rasters
    n_jobs : int, default -1
        Number of parallel jobs (-1 for auto)
    nodata_out : float, default -9999.0
        Output NoData value
    save_mask_debug : Optional[str], default None
        Optional path to save interpolation mask
        
    Returns
    -------
    str
        Path to output DEM
    """
    import time
    start_time = time.time()
    
    os.makedirs(os.path.dirname(os.path.abspath(out_dem_path)), exist_ok=True)
    
    print(f"Processing masked DSM: {dsm_path}")
    with rasterio.open(dsm_path) as ds:
        dsm = ds.read(1).astype(np.float32)
        prof = ds.profile.copy()
        H, W = dsm.shape
        dsm_nodata = ds.nodata
    
    print(f"DSM size: {H} x {W} pixels")
    
    # Create masks
    if dsm_nodata is not None:
        nodata_mask = (dsm == dsm_nodata)
        valid_mask = (dsm != dsm_nodata) & np.isfinite(dsm)
    else:
        nodata_mask = ~np.isfinite(dsm)
        valid_mask = np.isfinite(dsm)
    
    interpolation_pixels = np.sum(nodata_mask)
    valid_pixels = np.sum(valid_mask)
    
    print(f"Found {interpolation_pixels:,} pixels to interpolate ({interpolation_pixels/(H*W)*100:.1f}%)")
    print(f"Using {valid_pixels:,} valid pixels for interpolation")
    
    if interpolation_pixels == 0:
        print("No interpolation needed, copying input")
        import shutil
        shutil.copy2(dsm_path, out_dem_path)
        return out_dem_path
    
    if valid_pixels < 10:
        print("Warning: Too few valid pixels for reliable interpolation")
        dem = np.full_like(dsm, nodata_out)
        dem[valid_mask] = dsm[valid_mask]
    else:
        # Interpolate NoData areas
        print(f"Interpolating using {interpolation_method} method...")
        dem = _interpolate_nodata_fast(dsm, valid_mask, nodata_mask, interpolation_method, chunk_size, n_jobs)
    
    # Save debug mask if requested
    if save_mask_debug:
        print("Saving interpolation mask...")
        mask_prof = prof.copy()
        mask_prof.update(dtype="uint8", nodata=0)
        with rasterio.open(save_mask_debug, 'w', **mask_prof) as dst:
            dst.write(nodata_mask.astype("uint8"), 1)
    
    # Write output
    print(f"Writing DEM to: {out_dem_path}")
    prof.update(dtype="float32", nodata=dsm_nodata if dsm_nodata is not None else nodata_out,
                compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_dem_path, "w", **prof) as dst:
        dst.write(dem, 1)
    
    elapsed = time.time() - start_time
    print(f"DEM generation completed in {elapsed:.1f} seconds!")
    return out_dem_path


def _interpolate_nodata_fast(
    dsm: np.ndarray,
    valid_mask: np.ndarray, 
    nodata_mask: np.ndarray,
    method: str = "linear",
    chunk_size: int = 1024,
    n_jobs: int = -1
) -> np.ndarray:
    """Fast interpolation of NoData areas using surrounding valid pixels."""
    from scipy.interpolate import griddata
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    H, W = dsm.shape
    dem = dsm.copy()
    
    # For small datasets, use simple approach
    total_nodata = np.sum(nodata_mask)
    if total_nodata < 5000 or H*W < chunk_size*chunk_size:
        return _interpolate_nodata_simple(dsm, valid_mask, nodata_mask, method)
    
    # Parallel processing for large datasets
    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), 6)
    
    # Create overlapping chunks
    overlap = 128
    chunks = []
    
    for y_start in range(0, H, chunk_size - overlap):
        for x_start in range(0, W, chunk_size - overlap):
            y_end = min(y_start + chunk_size, H)
            x_end = min(x_start + chunk_size, W)
            
            # Skip chunks with no NoData pixels
            if not np.any(nodata_mask[y_start:y_end, x_start:x_end]):
                continue
                
            chunks.append((y_start, y_end, x_start, x_end))
    
    print(f"Processing {len(chunks)} chunks with {n_jobs} workers...")
    
    def process_chunk(chunk_info):
        y_start, y_end, x_start, x_end = chunk_info
        
        # Add padding for interpolation context
        pad = 64
        y_pad_start = max(0, y_start - pad)
        y_pad_end = min(H, y_end + pad)
        x_pad_start = max(0, x_start - pad)
        x_pad_end = min(W, x_end + pad)
        
        chunk_dsm = dsm[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
        chunk_valid = valid_mask[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
        chunk_nodata = nodata_mask[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
        
        # Interpolate chunk
        chunk_dem = _interpolate_nodata_simple(chunk_dsm, chunk_valid, chunk_nodata, method)
        
        # Extract core area (without padding)
        core_y_start = y_start - y_pad_start
        core_y_end = core_y_start + (y_end - y_start)
        core_x_start = x_start - x_pad_start
        core_x_end = core_x_start + (x_end - x_start)
        
        return (y_start, y_end, x_start, x_end, 
                chunk_dem[core_y_start:core_y_end, core_x_start:core_x_end])
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        
        for future in as_completed(futures):
            y_start, y_end, x_start, x_end, chunk_result = future.result()
            dem[y_start:y_end, x_start:x_end] = chunk_result
    
    return dem


def _interpolate_nodata_simple(dsm: np.ndarray, valid_mask: np.ndarray, nodata_mask: np.ndarray, method: str) -> np.ndarray:
    """Simple interpolation for small areas or single chunks with robust bounds checking."""
    H, W = dsm.shape
    dem = dsm.copy()
    
    if not np.any(nodata_mask) or not np.any(valid_mask):
        return dem
    
    # Get valid data statistics for bounds checking
    valid_values = dsm[valid_mask]
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)
    value_range = max_val - min_val
    
    print(f"Valid elevation range: {min_val:.2f} to {max_val:.2f} meters (range: {value_range:.2f}m)")
    
    # Get valid points for interpolation
    y_valid, x_valid = np.where(valid_mask)
    values_valid = dsm[valid_mask]
    
    # Get points to interpolate
    y_interp, x_interp = np.where(nodata_mask)
    
    if len(values_valid) < 3:
        # Too few points for interpolation, use nearest
        if len(values_valid) > 0:
            dem[nodata_mask] = np.mean(values_valid)
        return dem
    
    # Check distance from NoData pixels to nearest valid pixels
    # This helps avoid extrapolation artifacts
    distance_map = distance_transform_edt(~valid_mask)
    max_interp_distance = np.max(distance_map[nodata_mask])
    
    print(f"Maximum interpolation distance: {max_interp_distance:.1f} pixels")
    
    # For very isolated areas, use conservative nearest neighbor
    if max_interp_distance > 50:  # More than 50 pixels from valid data
        print("Warning: Some areas are very far from valid data, using nearest neighbor interpolation")
        method = 'nearest'
    
    try:
        # Prepare points - IMPORTANT: use (x, y) not (y, x) for griddata!
        if len(values_valid) > 10000:
            # For many points, subsample for performance
            step = max(1, len(values_valid) // 5000)
            x_sub = x_valid[::step]  # Note: x first!
            y_sub = y_valid[::step]  # Then y
            vals_sub = values_valid[::step]
            points = np.column_stack((x_sub, y_sub))  # (x, y) order
            xi = np.column_stack((x_interp, y_interp))  # (x, y) order
        else:
            points = np.column_stack((x_valid, y_valid))  # (x, y) order
            xi = np.column_stack((x_interp, y_interp))  # (x, y) order
            vals_sub = values_valid
        
        print(f"Interpolating {len(xi)} points using {len(points)} reference points with {method} method")
        
        interpolated = griddata(points, vals_sub, xi, method=method, fill_value=np.nan)
        
        # Robust bounds checking - clip to reasonable range
        if method in ['linear', 'cubic']:
            # Allow small overshoot for linear/cubic but clip extreme values
            safety_margin = max(0.1 * value_range, 1.0)  # 10% of range or 1m minimum
            lower_bound = min_val - safety_margin
            upper_bound = max_val + safety_margin
            
            # Identify and report extreme values
            extreme_mask = (interpolated < lower_bound) | (interpolated > upper_bound)
            if np.any(extreme_mask):
                n_extreme = np.sum(extreme_mask)
                extreme_min = np.min(interpolated[extreme_mask]) if np.any(extreme_mask) else 0
                extreme_max = np.max(interpolated[extreme_mask]) if np.any(extreme_mask) else 0
                print(f"Warning: Found {n_extreme} extreme interpolated values (range: {extreme_min:.1f} to {extreme_max:.1f})")
                print("Clipping to valid data range...")
                
                interpolated = np.clip(interpolated, lower_bound, upper_bound)
        
        # Handle any remaining NaN values with nearest neighbor
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            print(f"Filling {np.sum(nan_mask)} NaN values with nearest neighbor")
            nn_interpolated = griddata(points, vals_sub, xi[nan_mask], method='nearest')
            interpolated[nan_mask] = nn_interpolated
        
        # Final safety check
        final_min = np.min(interpolated)
        final_max = np.max(interpolated)
        print(f"Interpolated value range: {final_min:.2f} to {final_max:.2f} meters")
        
        dem[nodata_mask] = interpolated
        
    except Exception as e:
        print(f"Interpolation failed ({e}), using nearest neighbor fallback")
        try:
            points = np.column_stack((x_valid, y_valid))  # (x, y) order
            xi = np.column_stack((x_interp, y_interp))    # (x, y) order
            interpolated = griddata(points, values_valid, xi, method='nearest')
            dem[nodata_mask] = interpolated
            print(f"Nearest neighbor range: {np.min(interpolated):.2f} to {np.max(interpolated):.2f} meters")
        except Exception as e2:
            print(f"Nearest neighbor also failed ({e2}), using mean value")
            mean_val = np.mean(values_valid)
            dem[nodata_mask] = mean_val
            print(f"Using mean value: {mean_val:.2f} meters")
    
    return dem