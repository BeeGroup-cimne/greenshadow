# -*- coding: utf-8 -*-
import os
import warnings
from typing import Iterable, List, Union, Dict, Optional, Tuple, Any, Sequence, Literal
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import Affine
from shapely.geometry import mapping
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, distance_transform_edt, uniform_filter, minimum_filter, convolve, zoom
from scipy.interpolate import SmoothBivariateSpline
import numpy as np
import laspy
from rasterio.crs import CRS
from .utils import _build_raster_profile, _save_raster


def _nearest_fill(mask: np.ndarray, arr: np.ndarray, aoi_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Fill masked values in arr by nearest valid neighbor (fast EDT indices).
    
    Parameters:
    -----------
    mask : np.ndarray
        Boolean mask where True indicates gaps to fill
    arr : np.ndarray
        Array to fill gaps in
    aoi_mask : Optional[np.ndarray]
        Boolean mask where True indicates areas inside AOI. If provided, 
        only gaps inside AOI will be filled.
    """
    from scipy.ndimage import distance_transform_edt
    
    # If AOI mask is provided, only fill gaps within AOI
    if aoi_mask is not None:
        effective_mask = mask & aoi_mask
    else:
        effective_mask = mask
    
    # Only proceed if there are gaps to fill
    if not np.any(effective_mask):
        return arr.copy()
    
    # For finding nearest neighbors, we need to consider the valid data mask
    # Valid data is where we have data AND it's within AOI (if AOI is provided)
    if aoi_mask is not None:
        # Valid data is not a gap and is within AOI
        valid_data_mask = (~mask) & aoi_mask
    else:
        # Valid data is simply not a gap
        valid_data_mask = ~mask
    
    # If we have no valid data to interpolate from, return original array
    if not np.any(valid_data_mask):
        return arr.copy()
    
    # Use distance transform on the inverse of valid data mask to find nearest valid pixels
    _, (iy, ix) = distance_transform_edt(~valid_data_mask, return_indices=True)
    
    out = arr.copy()
    # Only fill the gaps that are within the effective mask (mask AND AOI)
    out[effective_mask] = arr[iy[effective_mask], ix[effective_mask]]
    return out


def _nearest_fill_chunked(mask: np.ndarray, arr: np.ndarray, max_ram_gb: float = 64.0, aoi_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Fill masked values in arr by nearest valid neighbor using chunked processing to manage RAM usage.
    
    Parameters:
    -----------
    mask : np.ndarray
        Boolean mask where True indicates gaps to fill
    arr : np.ndarray
        Array to fill gaps in
    max_ram_gb : float
        Maximum RAM to use in GB
    aoi_mask : Optional[np.ndarray]
        Boolean mask where True indicates areas inside AOI. If provided, 
        gaps outside AOI will not be filled.
    """
    from scipy.ndimage import distance_transform_edt
    
    # Calculate memory requirements
    arr_size_gb = arr.nbytes / (1024**3)
    mask_size_gb = mask.nbytes / (1024**3)
    
    # For distance_transform_edt, we need additional memory for indices arrays (2x the size for iy, ix)
    # Plus temporary arrays during computation (approximately 3x total array size)
    edt_memory_factor = 5.0
    required_memory_gb = (arr_size_gb + mask_size_gb) * edt_memory_factor
    
    if required_memory_gb <= max_ram_gb * 0.8:  # Use 80% of available RAM as safety margin
        # Enough RAM to process the entire array at once
        return _nearest_fill(mask, arr, aoi_mask=aoi_mask)
    
    # Need to process in chunks
    out = arr.copy()
    height, width = arr.shape
    
    # Calculate chunk size based on available RAM
    target_chunk_memory_gb = max_ram_gb * 0.6  # Use 60% of RAM per chunk
    pixels_per_gb = (height * width) / required_memory_gb
    target_chunk_pixels = int(pixels_per_gb * target_chunk_memory_gb)
    
    # Calculate chunk dimensions (prefer square chunks for better locality)
    chunk_side = int(np.sqrt(target_chunk_pixels))
    chunk_side = max(64, min(chunk_side, min(height, width)))  # Minimum 64x64, max array dimension
    
    overlap = 32  # Overlap to avoid edge artifacts
    
    print(f"Processing gap filling in chunks of {chunk_side}x{chunk_side} with {overlap} pixel overlap")
    
    # Process chunks with overlap
    for y_start in range(0, height, chunk_side - overlap):
        for x_start in range(0, width, chunk_side - overlap):
            y_end = min(y_start + chunk_side, height)
            x_end = min(x_start + chunk_side, width)
            
            # Extract chunk
            chunk_mask = mask[y_start:y_end, x_start:x_end]
            chunk_arr = arr[y_start:y_end, x_start:x_end]
            
            # Extract AOI mask chunk if provided
            chunk_aoi_mask = None
            if aoi_mask is not None:
                chunk_aoi_mask = aoi_mask[y_start:y_end, x_start:x_end]
            
            # Skip if no gaps to fill in this chunk (within AOI)
            if chunk_aoi_mask is not None:
                effective_mask = chunk_mask & chunk_aoi_mask
            else:
                effective_mask = chunk_mask
                
            if not np.any(effective_mask):
                continue
            
            # Fill gaps in chunk
            filled_chunk = _nearest_fill(effective_mask, chunk_arr, aoi_mask=chunk_aoi_mask)
            
            # Calculate the region to update (avoid overlaps except for first chunk)
            update_y_start = overlap // 2 if y_start > 0 else 0
            update_x_start = overlap // 2 if x_start > 0 else 0
            update_y_end = chunk_side - overlap // 2 if y_end < height else chunk_side
            update_x_end = chunk_side - overlap // 2 if x_end < width else chunk_side
            
            # Make sure we don't go beyond chunk boundaries
            update_y_end = min(update_y_end, y_end - y_start)
            update_x_end = min(update_x_end, x_end - x_start)
            
            # Update output array
            out[y_start + update_y_start:y_start + update_y_end, 
                x_start + update_x_start:x_start + update_x_end] = \
                filled_chunk[update_y_start:update_y_end, update_x_start:update_x_end]
    
    return out


def _horn_gradients(z: np.ndarray, pixel_size: float):
    """Compute gradients using Horn's method for slope/aspect calculation."""
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32) / (8.0*pixel_size)
    ky = np.array([[ 1,2,1],[ 0,0,0],[-1,-2,-1]], np.float32) / (8.0*pixel_size)
    dzdx = convolve(z, kx, mode="nearest")
    dzdy = convolve(z, ky, mode="nearest")
    return dzdx, dzdy


def _asc_y_eval_fix(xs: np.ndarray, ys_desc: np.ndarray, spline: SmoothBivariateSpline) -> np.ndarray:
    """Fix for spline evaluation with descending Y coordinates."""
    ys_inc = np.sort(ys_desc)
    M = spline(xs, ys_inc, grid=True)
    return M


def fit_spline_adaptive(xs, ys, dem_up, *, px: float, step_px_init: int,
                       step_px_min: int = 5, s: float = 0.0, kx: int = 3, ky: int = 3):
    """Fit spline adaptively by progressively densifying control points."""
    
    def try_fit(step_px):
        if step_px < step_px_min:
            return None, f"step_px={step_px} < min={step_px_min}"
        idx_x = np.arange(0, len(xs), step_px)
        idx_y = np.arange(0, len(ys), step_px)
        try:
            x_ctrl = xs[idx_x]; y_ctrl = ys[idx_y]
            z_ctrl = dem_up[np.ix_(idx_y, idx_x)]
            spline = SmoothBivariateSpline(
                np.tile(x_ctrl, len(y_ctrl)), 
                np.repeat(y_ctrl, len(x_ctrl)), 
                z_ctrl.ravel(), s=s, kx=kx, ky=ky
            )
            return spline, None
        except Exception as e:
            return None, str(e)

    step_px = step_px_init
    while step_px >= step_px_min:
        spline, err = try_fit(step_px)
        if spline is not None:
            return spline
        warnings.warn(f"Spline fit failed at step_px={step_px}: {err}")
        step_px //= 2

    raise RuntimeError(f"All spline attempts failed down to step_px={step_px}")


def _downscale_dem_spline_to_grid(
    coarse_dem_path: str, H: int, W: int, T: Affine, crs,
    *, ctrl_step_m: float = 60.0, s: float = 0.0, kx: int = 3, ky: int = 3,
    resampling: Resampling = Resampling.bilinear
) -> np.ndarray:
    """Downscale a coarse DEM to a finer grid using spline interpolation."""
    
    arr = np.full((H, W), np.nan, np.float32)
    with rasterio.open(coarse_dem_path) as src:
        reproject(rasterio.band(src, 1), arr, src_transform=src.transform, 
                 src_crs=src.crs, src_nodata=src.nodata, dst_transform=T,
                 dst_crs=crs, dst_nodata=np.nan, resampling=resampling)

    arr = _nearest_fill(~np.isfinite(arr), arr)
    px_m = abs(float(T.a))
    step_px_init = max(1, int(ctrl_step_m / px_m))
    
    if step_px_init >= min(H, W) // 3:
        return arr
    
    xs = np.arange(W, dtype=np.float64) * px_m + T.c + 0.5 * px_m
    ys = np.arange(H, dtype=np.float64) * (-px_m) + T.f - 0.5 * px_m
    
    try:
        spline = fit_spline_adaptive(xs, ys, arr, px=px_m, 
                                   step_px_init=step_px_init, s=s, kx=kx, ky=ky)
        return _asc_y_eval_fix(xs, ys, spline)
    except Exception as e:
        warnings.warn(f"Spline downscaling failed: {e}. Using bilinear.")
        return arr


def _projected_harmonic_multiscale(
    dsm: np.ndarray,
    anchors_mask: np.ndarray | None,
    anchor_vals: np.ndarray | None,
    *,
    levels: int = 3,
    relax_iters: int = 50,
    omega: float = 1.8,
) -> np.ndarray:
    """
    Projected harmonic smoothing under obstacle u <= dsm with optional anchors.
    Robust to odd-sized grids: pad once to even at the top, use shape-safe
    restriction/prolongation, crop back to original size at the end.
    """
    
    # Normalize numeric params
    if isinstance(levels, (tuple, list, np.ndarray)):
        if len(levels) == 0:
            raise ValueError("levels cannot be empty")
        levels = int(levels[0])
    else:
        levels = int(levels)
    relax_iters = int(relax_iters)
    omega = float(omega)
    if levels < 1:
        levels = 1

    # Coerce inputs & record original shape
    dsm = np.asarray(dsm, dtype=np.float32)
    H0, W0 = dsm.shape

    # Pad ONCE so top level is even x even
    pad_bottom = H0 % 2
    pad_right  = W0 % 2
    if pad_bottom or pad_right:
        pad = ((0, pad_bottom), (0, pad_right))
        dsm_p = np.pad(dsm, pad, mode="edge")
        if anchors_mask is not None:
            anchors_mask = np.pad(np.asarray(anchors_mask, dtype=bool), pad, mode="edge")
        if anchor_vals is not None:
            anchor_vals = np.pad(np.asarray(anchor_vals, dtype=np.float32), pad, mode="edge")
    else:
        dsm_p = dsm
        if anchors_mask is not None:
            anchors_mask = np.asarray(anchors_mask, dtype=bool)
        if anchor_vals is not None:
            anchor_vals = np.asarray(anchor_vals, dtype=np.float32)

    Hp, Wp = dsm_p.shape
    if anchors_mask is not None:
        assert anchors_mask.shape == (Hp, Wp), f"anchors_mask {anchors_mask.shape} != {(Hp, Wp)}"
    if anchor_vals is not None:
        assert anchor_vals.shape == (Hp, Wp), f"anchor_vals  {anchor_vals.shape} != {(Hp, Wp)}"

    # Initial guess
    u0 = dsm_p.copy()

    # Helper functions for restriction/prolongation
    def _restrict_num(a: np.ndarray) -> np.ndarray:
        """2x2 numeric restriction with aligned slices (works for odd H/W)."""
        H, W = a.shape
        Hs, Ws = H - (H % 2), W - (W % 2)
        a00 = a[0:Hs:2, 0:Ws:2]
        a10 = a[1:Hs:2, 0:Ws:2]
        a01 = a[0:Hs:2, 1:Ws:2]
        a11 = a[1:Hs:2, 1:Ws:2]
        return 0.25 * (a00 + a10 + a01 + a11)

    def _restrict_mask(m: np.ndarray) -> np.ndarray:
        """2x2 boolean restriction as OR over the 2x2 block."""
        m = m.astype(bool, copy=False)
        H, W = m.shape
        Hs, Ws = H - (H % 2), W - (W % 2)
        m00 = m[0:Hs:2, 0:Ws:2]
        m10 = m[1:Hs:2, 0:Ws:2]
        m01 = m[0:Hs:2, 1:Ws:2]
        m11 = m[1:Hs:2, 1:Ws:2]
        return m00 | m10 | m01 | m11

    def _prolong(a: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Prolongate by ~2� with nearest-neighbor."""
        Af = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
        th, tw = target_shape

        # Pad bottom if we're short in rows
        dh = th - Af.shape[0]
        if dh > 0:
            last_row = Af[-1:, :]
            Af = np.vstack([Af, np.repeat(last_row, dh, axis=0)])

        # Pad right if we're short in cols
        dw = tw - Af.shape[1]
        if dw > 0:
            last_col = Af[:, -1:]
            Af = np.hstack([Af, np.repeat(last_col, dw, axis=1)])

        return Af[:th, :tw]

    def _apply_anchors_inplace(u_: np.ndarray, m_: np.ndarray | None, v_: np.ndarray | None) -> None:
        """Shape-safe in-place write: intersect shapes first to avoid mismatches."""
        if m_ is None or v_ is None:
            return
        h = min(u_.shape[0], m_.shape[0], v_.shape[0])
        w = min(u_.shape[1], m_.shape[1], v_.shape[1])
        if h <= 0 or w <= 0:
            return
        uh = u_[:h, :w]
        mh = m_[:h, :w]
        vh = v_[:h, :w]
        uh[mh] = vh[mh]

    def _apply_obstacle_inplace(u_: np.ndarray, dsm_: np.ndarray) -> None:
        """Shape-safe obstacle constraint: u[i,j] = min(u[i,j], dsm[i,j])"""
        h = min(u_.shape[0], dsm_.shape[0])
        w = min(u_.shape[1], dsm_.shape[1])
        if h <= 0 or w <= 0:
            return
        uh = u_[:h, :w]
        dh = dsm_[:h, :w]
        np.minimum(uh, dh, out=uh)

    def _relax_inplace(u_: np.ndarray, iters: int, omega_: float) -> None:
        """SOR relaxation for Laplace's equation �u = 0."""
        h, w = u_.shape
        for _ in range(iters):
            # Proper interior update without wrap-around boundaries
            u_new = u_.copy()
            if h > 2 and w > 2:
                u_new[1:-1, 1:-1] = 0.25 * (
                    u_[0:-2, 1:-1] + u_[2:, 1:-1] +  # up, down
                    u_[1:-1, 0:-2] + u_[1:-1, 2:]    # left, right
                )
                # Apply relaxation only to interior points
                u_[1:-1, 1:-1] = (1.0 - omega_) * u_[1:-1, 1:-1] + omega_ * u_new[1:-1, 1:-1]

    # Build multigrid hierarchy
    us = [u0]; dsms = [dsm_p]; masks = [anchors_mask]; vals = [anchor_vals]
    
    for lv in range(1, levels):
        us.append(_restrict_num(us[-1]))
        dsms.append(_restrict_num(dsms[-1]))
        masks.append(_restrict_mask(masks[-1]) if masks[-1] is not None else None)
        vals.append(_restrict_num(vals[-1]) if vals[-1] is not None else None)

    # Solve from coarse to fine
    for lv in reversed(range(levels)):
        for _ in range(relax_iters):
            _relax_inplace(us[lv], 1, omega)
            _apply_anchors_inplace(us[lv], masks[lv], vals[lv])
            _apply_obstacle_inplace(us[lv], dsms[lv])
        
        if lv > 0:
            # Prolongate correction to finer level
            correction = _prolong(us[lv], us[lv-1].shape)
            us[lv-1] += correction

    # Final result: crop back to original size
    result = us[0][:H0, :W0]
    return result


def remove_buildings_from_dsm(
    dsm_path: str,
    buildings: Union[gpd.GeoDataFrame, str],
    out_path: str,
    *,
    buildings_buffer_m: float = 0.5,
    all_touched: bool = True,
    fill_method: str = "spline",               # "nearest" | "coarse_dem" | "spline"
    coarse_dem_path: Optional[str] = None,     # required for "coarse_dem"/"spline"
    spline_ctrl_step_m: float = 60.0,
    spline_s: float = 1e2,
    feather_m: float = 3.0,
    nodata_out: float = -9999.0,
    save_mask_debug: Optional[str] = None,
) -> str:
    """Remove buildings from DSM by filling with interpolated values."""
    
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with rasterio.open(dsm_path) as ds:
        dsm = ds.read(1).astype(np.float32)
        prof = ds.profile.copy()
        T = ds.transform
        crs = ds.crs
        H, W = dsm.shape
        dsm_nodata = ds.nodata
        px = float(abs(T.a))

    gdf = buildings if isinstance(buildings, gpd.GeoDataFrame) else gpd.read_file(buildings)
    if gdf.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")
    if crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    if buildings_buffer_m:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(buildings_buffer_m)

    shapes = [mapping(g) for g in gdf.geometry if g and not g.is_empty]
    mask = rasterize(shapes=shapes, out_shape=(H, W), transform=T,
                     fill=0, default_value=1, all_touched=all_touched, dtype="uint8").astype(bool)

    if save_mask_debug:
        dbg = prof.copy()
        dbg.update(dtype="uint8", nodata=0, count=1, compress="deflate", tiled=True,
                  blockxsize=256, blockysize=256)
        with rasterio.open(save_mask_debug, "w", **dbg) as dst:
            dst.write(mask.astype("uint8"), 1)

    if fill_method == "nearest":
        repl = _nearest_fill(mask, dsm)
    elif fill_method == "coarse_dem":
        if not coarse_dem_path:
            raise ValueError("coarse_dem_path required for 'coarse_dem'.")
        repl = np.full_like(dsm, np.nan, np.float32)
        with rasterio.open(coarse_dem_path) as src:
            reproject(rasterio.band(src, 1), repl,
                      src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                      dst_transform=T,  dst_crs=crs,  dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
        repl = _nearest_fill(~np.isfinite(repl), repl)
    elif fill_method == "spline":
        if not coarse_dem_path:
            raise ValueError("coarse_dem_path required for 'spline'.")
        repl = _downscale_dem_spline_to_grid(coarse_dem_path, H, W, T, crs,
                                             ctrl_step_m=spline_ctrl_step_m, s=spline_s, kx=3, ky=3)
    else:
        raise ValueError("fill_method must be 'nearest', 'coarse_dem', or 'spline'.")

    out = dsm.copy()
    if feather_m and feather_m > 0:
        rad_px = max(1, int(round(feather_m / px)))
        d_out = distance_transform_edt(~mask)
        alpha = np.clip(d_out / (rad_px + 1e-6), 0.0, 1.0).astype(np.float32)
        blended = alpha * dsm + (1.0 - alpha) * repl
        out[mask] = blended[mask]
    else:
        out[mask] = repl[mask]

    if dsm_nodata is not None:
        out[~np.isfinite(dsm)] = dsm_nodata

    prof.update(dtype="float32", nodata=dsm_nodata if dsm_nodata is not None else nodata_out,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)
    return out_path


def _write_like(profile: dict, path: str, data: np.ndarray, nodata: float):
    """Write array to raster using the given profile."""
    profile_out = profile.copy()
    profile_out.update(
        dtype=data.dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256
    )
    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(data, 1)


def _roi_from_mask(mask: np.ndarray, pad: int = 2):
    """Extract region of interest from mask with padding."""
    if not np.any(mask):
        return slice(0, 1), slice(0, 1)
    y_indices, x_indices = np.where(mask)
    y_min, y_max = max(0, y_indices.min() - pad), min(mask.shape[0], y_indices.max() + pad + 1)
    x_min, x_max = max(0, x_indices.min() - pad), min(mask.shape[1], x_indices.max() + pad + 1)
    return slice(y_min, y_max), slice(x_min, x_max)


def _harmonic_inpaint_roi(base_roi, dsm_roi, fill_roi, iters=80, omega=0.8):
    """Harmonic inpainting within a region of interest."""
    u = base_roi.copy().astype(np.float32)
    for _ in range(iters):
        u_new = 0.25 * (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
        )
        u = (1.0 - omega) * u + omega * u_new
        u[~fill_roi] = base_roi[~fill_roi]  # Preserve boundary
        np.minimum(u, dsm_roi, out=u)  # Obstacle constraint
    return u


def _bilinear_fill(base, known, fill_mask):
    """Fill gaps using bilinear interpolation."""
    from scipy.ndimage import zoom
    # This is a simplified version - in practice, you'd want proper bilinear interpolation
    return _nearest_fill(fill_mask, base)


def _detect_height_disruptions_fast(dsm: np.ndarray, valid_mask: np.ndarray, threshold_m: float) -> np.ndarray:
    """Fast vectorized detection of pixels with large height differences to neighbors."""
    H, W = dsm.shape
    
    # Use convolution to compute max difference with all 8 neighbors efficiently
    from scipy.ndimage import maximum_filter, minimum_filter
    
    # Only consider valid pixels
    dsm_masked = np.where(valid_mask, dsm, np.nan)
    
    # Get local max and min in 3x3 neighborhood (excluding center)
    max_neighbor = maximum_filter(dsm_masked, size=3, mode='constant', cval=np.nan)
    min_neighbor = minimum_filter(dsm_masked, size=3, mode='constant', cval=np.nan)
    
    # Max difference between center pixel and any neighbor
    center_val = dsm_masked
    max_diff = np.maximum(
        np.abs(center_val - max_neighbor),
        np.abs(center_val - min_neighbor)
    )
    
    # Alternative: use gradient magnitude for even faster detection
    dzdx, dzdy = _horn_gradients(dsm_masked, 1.0)  # Using pixel units
    gradient_mag = np.sqrt(dzdx**2 + dzdy**2)
    
    # Disruption is either large local difference OR large gradient
    disruption_mask = (
        (max_diff > threshold_m) | 
        (gradient_mag > threshold_m * 0.7)  # Gradient threshold slightly lower
    ) & valid_mask
    
    return disruption_mask


def _create_buffered_disruption_areas(
    disruption_mask: np.ndarray, 
    buffer_m: float, 
    px_size: float, 
    min_area_pixels: int
) -> np.ndarray:
    """Create buffered polygons around disruption areas."""
    from scipy.ndimage import binary_dilation, label
    
    # Remove small isolated pixels
    labeled, num_features = label(disruption_mask)
    for i in range(1, num_features + 1):
        region = (labeled == i)
        if np.sum(region) < min_area_pixels:
            disruption_mask[region] = False
    
    # Buffer the disruption areas
    if buffer_m > 0:
        buffer_px = max(1, int(np.round(buffer_m / px_size)))
        structure = np.ones((2*buffer_px+1, 2*buffer_px+1))
        buffered_mask = binary_dilation(disruption_mask, structure=structure)
    else:
        buffered_mask = disruption_mask.copy()
    
    return buffered_mask


def _interpolate_dem_chunked(
    dsm: np.ndarray, 
    valid_mask: np.ndarray, 
    fill_mask: np.ndarray, 
    method: str,
    chunk_size: int = 2048,
    n_jobs: int = -1
) -> np.ndarray:
    """Fast chunked interpolation with parallel processing."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing as mp
    
    H, W = dsm.shape
    dem = dsm.copy()
    
    # Areas to interpolate
    interpolate_mask = fill_mask & valid_mask
    if not np.any(interpolate_mask):
        return dem
    
    # Known points: valid AND NOT in fill mask  
    known_mask = valid_mask & ~fill_mask
    if not np.any(known_mask):
        return _nearest_fill(interpolate_mask, dem)
    
    # For very small areas, use simple approach
    total_interp_pixels = np.sum(interpolate_mask)
    if total_interp_pixels < 1000:
        return _interpolate_simple(dsm, valid_mask, fill_mask, method)
    
    # Determine optimal number of workers
    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    # Create chunks that overlap to ensure smooth interpolation
    overlap = 64  # pixels overlap between chunks
    chunks = []
    
    for y_start in range(0, H, chunk_size - overlap):
        for x_start in range(0, W, chunk_size - overlap):
            y_end = min(y_start + chunk_size, H)
            x_end = min(x_start + chunk_size, W)
            
            # Skip chunks with no interpolation needed
            chunk_interp = interpolate_mask[y_start:y_end, x_start:x_end]
            if not np.any(chunk_interp):
                continue
                
            chunks.append((y_start, y_end, x_start, x_end))
    
    # Process chunks in parallel
    def process_chunk(chunk_info):
        y_start, y_end, x_start, x_end = chunk_info
        
        # Extract chunk with padding for interpolation context
        pad = 32
        y_pad_start = max(0, y_start - pad)
        y_pad_end = min(H, y_end + pad)
        x_pad_start = max(0, x_start - pad)
        x_pad_end = min(W, x_end + pad)
        
        chunk_dsm = dsm[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
        chunk_valid = valid_mask[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
        chunk_fill = fill_mask[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
        
        # Interpolate the padded chunk
        chunk_result = _interpolate_simple(chunk_dsm, chunk_valid, chunk_fill, method)
        
        # Extract only the core region (without padding)
        core_y_start = y_start - y_pad_start
        core_y_end = core_y_start + (y_end - y_start)
        core_x_start = x_start - x_pad_start
        core_x_end = core_x_start + (x_end - x_start)
        
        return (y_start, y_end, x_start, x_end, 
                chunk_result[core_y_start:core_y_end, core_x_start:core_x_end])
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        for future in as_completed(futures):
            try:
                y_start, y_end, x_start, x_end, chunk_result = future.result()
                # Only update interpolation areas
                chunk_mask = interpolate_mask[y_start:y_end, x_start:x_end]
                dem[y_start:y_end, x_start:x_end][chunk_mask] = chunk_result[chunk_mask]
            except Exception as e:
                warnings.warn(f"Chunk interpolation failed: {e}")
                continue
    
    return dem


def _interpolate_simple(dsm: np.ndarray, valid_mask: np.ndarray, fill_mask: np.ndarray, method: str) -> np.ndarray:
    """Simple interpolation for small areas or fallback."""
    from scipy.interpolate import griddata
    
    dem = dsm.copy()
    H, W = dsm.shape
    
    interpolate_mask = fill_mask & valid_mask
    if not np.any(interpolate_mask):
        return dem
    
    known_mask = valid_mask & ~fill_mask
    if not np.any(known_mask):
        return _nearest_fill(interpolate_mask, dem)
    
    # Use fast RBF interpolation for better performance
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    
    # Subsample known points if too many (>10000) for speed
    known_indices = np.where(known_mask)
    if len(known_indices[0]) > 10000:
        # Subsample uniformly
        step = len(known_indices[0]) // 10000
        subsample = slice(None, None, step)
        known_y = known_indices[0][subsample]
        known_x = known_indices[1][subsample]
        known_points = np.column_stack([known_y, known_x])
        known_values = dsm[known_y, known_x]
    else:
        known_points = np.column_stack([known_indices[0], known_indices[1]])
        known_values = dsm[known_mask]
    
    interp_indices = np.where(interpolate_mask)
    interp_points = np.column_stack([interp_indices[0], interp_indices[1]])
    
    try:
        # Use linear for speed, fallback to nearest
        if method in ["cubic", "linear"]:
            interp_values = griddata(known_points, known_values, interp_points, 
                                   method='linear', fill_value=np.nan)
            nan_mask = np.isnan(interp_values)
            if np.any(nan_mask):
                interp_values[nan_mask] = griddata(known_points, known_values, 
                                                 interp_points[nan_mask], method='nearest')
        else:
            interp_values = griddata(known_points, known_values, interp_points, method='nearest')
        
        dem[interpolate_mask] = interp_values
        
    except Exception as e:
        warnings.warn(f"Simple interpolation failed: {e}. Using nearest fill.")
        dem = _nearest_fill(interpolate_mask, dem)
    
    return dem


def build_dem_from_dsm_height_disruption(
    dsm_path: str,
    out_dem_path: str,
    *,
    height_threshold_m: float = 2.0,
    buffer_m: float = 1.0,
    interpolation_method: str = "linear",  # "cubic", "linear", "nearest" (linear is faster)
    min_area_pixels: int = 4,
    chunk_size: int = 2048,  # Process in chunks for memory efficiency
    n_jobs: int = -1,  # Parallel processing (-1 = auto-detect cores)
    nodata_out: float = -9999.0,
    save_mask_debug: Optional[str] = None,
) -> str:
    """Build DEM from DSM by detecting height disruptions and interpolating (FAST VERSION)."""
    
    os.makedirs(os.path.dirname(os.path.abspath(out_dem_path)), exist_ok=True)

    print(f"Processing DSM: {dsm_path}")
    with rasterio.open(dsm_path) as ds:
        dsm = ds.read(1).astype(np.float32)
        prof = ds.profile.copy()
        T = ds.transform
        crs = ds.crs
        H, W = dsm.shape
        dsm_nodata = ds.nodata
        px_size = abs(float(T.a))

    print(f"DSM size: {H} x {W} pixels ({px_size:.2f}m resolution)")

    # Create valid data mask
    valid_mask = np.isfinite(dsm)
    if dsm_nodata is not None:
        valid_mask &= (dsm != dsm_nodata)

    print("Detecting height disruptions...")
    # Use fast vectorized detection
    disruption_mask = _detect_height_disruptions_fast(dsm, valid_mask, height_threshold_m)
    
    print(f"Found {np.sum(disruption_mask)} disrupted pixels")
    
    # Create closed polygons and buffer them
    print("Creating buffered disruption areas...")
    buffered_mask = _create_buffered_disruption_areas(
        disruption_mask, buffer_m, px_size, min_area_pixels
    )
    
    print(f"Buffered to {np.sum(buffered_mask)} pixels for interpolation")
    
    if save_mask_debug:
        dbg = prof.copy()
        dbg.update(dtype="uint8", nodata=0, count=1, compress="deflate", tiled=True,
                  blockxsize=256, blockysize=256)
        with rasterio.open(save_mask_debug, "w", **dbg) as dst:
            dst.write(buffered_mask.astype("uint8"), 1)

    # Create DEM by interpolating over disrupted areas
    print("Interpolating DEM values...")
    dem = _interpolate_dem_chunked(
        dsm, valid_mask, buffered_mask, interpolation_method, chunk_size, n_jobs
    )
    
    # Ensure DEM <= DSM everywhere
    print("Applying height constraint (DEM <= DSM)...")
    dem = np.minimum(dem, dsm)

    # Handle nodata values
    if dsm_nodata is not None:
        dem[~valid_mask] = dsm_nodata

    # Write output
    print(f"Writing DEM to: {out_dem_path}")
    prof.update(dtype="float32", nodata=dsm_nodata if dsm_nodata is not None else nodata_out,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_dem_path, "w", **prof) as dst:
        dst.write(dem, 1)
    
    print("DEM generation completed!")
    return out_dem_path


def build_dem_from_dsm_obstacle_fast(
    dsm_path: str,
    out_dem_path: str,
    *,
    buildings: Union[gpd.GeoDataFrame, str, None] = None,
    buildings_buffer_m: float = 0.5,
    all_touched: bool = True,
    method: str = "multiscale_harmonic",
    levels: int = 3,
    relax_iters: int = 50,
    omega: float = 1.8,
    nodata_out: float = -9999.0,
    save_mask_debug: Optional[str] = None,
) -> str:
    """Build DEM from DSM using fast obstacle-constrained smoothing."""
    
    os.makedirs(os.path.dirname(os.path.abspath(out_dem_path)), exist_ok=True)

    with rasterio.open(dsm_path) as ds:
        dsm = ds.read(1).astype(np.float32)
        prof = ds.profile.copy()
        T = ds.transform
        crs = ds.crs
        H, W = dsm.shape
        dsm_nodata = ds.nodata

    # Handle buildings mask if provided
    if buildings is not None:
        gdf = buildings if isinstance(buildings, gpd.GeoDataFrame) else gpd.read_file(buildings)
        if gdf.crs is None:
            raise ValueError("Buildings GeoDataFrame has no CRS.")
        if crs and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        if buildings_buffer_m:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.buffer(buildings_buffer_m)

        shapes = [mapping(g) for g in gdf.geometry if g and not g.is_empty]
        buildings_mask = rasterize(shapes=shapes, out_shape=(H, W), transform=T,
                                 fill=0, default_value=1, all_touched=all_touched, dtype="uint8").astype(bool)
        
        if save_mask_debug:
            dbg = prof.copy()
            dbg.update(dtype="uint8", nodata=0, count=1, compress="deflate", tiled=True,
                      blockxsize=256, blockysize=256)
            with rasterio.open(save_mask_debug, "w", **dbg) as dst:
                dst.write(buildings_mask.astype("uint8"), 1)
    else:
        buildings_mask = None

    # Apply the smoothing algorithm
    if method == "multiscale_harmonic":
        dem = _projected_harmonic_multiscale(
            dsm=dsm,
            anchors_mask=None,  # Don't anchor building pixels
            anchor_vals=None,
            levels=levels,
            relax_iters=relax_iters,
            omega=omega
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle nodata values
    if dsm_nodata is not None:
        dem[~np.isfinite(dsm)] = dsm_nodata

    # Write output
    prof.update(dtype="float32", nodata=dsm_nodata if dsm_nodata is not None else nodata_out,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_dem_path, "w", **prof) as dst:
        dst.write(dem, 1)
    
    return out_dem_path


def refine_dem_with_buffered_interpolation(
    *,
    dsm_path: str,
    dem_path: str,
    out_path: str,
    tol_m: float = 0.20,
    buffer_m: float = 15.0,
    method: str = "harmonic",
    fill_iters: int = 80,
    fill_omega: float = 0.8,
    nodata_out: float = -9999.0,
) -> str:
    """Refine DEM by interpolating in regions where DSM and DEM differ significantly."""
    
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Read inputs
    with rasterio.open(dsm_path) as ds:
        dsm = ds.read(1).astype(np.float32)
        prof = ds.profile.copy()
        T = ds.transform
        px = abs(float(T.a))

    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float32)

    # Find pixels where DSM >> DEM (likely obstacles)
    diff = dsm - dem
    obstacles_mask = diff > tol_m

    if not np.any(obstacles_mask):
        # No refinement needed, just copy the DEM
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(dem, 1)
        return out_path

    # Buffer the obstacles to get interpolation region
    if buffer_m > 0:
        buffer_px = int(np.ceil(buffer_m / px))
        from scipy.ndimage import binary_dilation
        struct = np.ones((2*buffer_px+1, 2*buffer_px+1))
        fill_mask = binary_dilation(obstacles_mask, structure=struct)
    else:
        fill_mask = obstacles_mask.copy()

    # Create output starting from DEM
    out = dem.copy()

    if method == "harmonic":
        # Use harmonic inpainting in buffered regions
        ry, rx = _roi_from_mask(fill_mask)
        if (ry.stop - ry.start) > 1 and (rx.stop - rx.start) > 1:
            base_roi = dem[ry, rx]
            dsm_roi = dsm[ry, rx]
            fill_roi = fill_mask[ry, rx]
            filled_roi = _harmonic_inpaint_roi(base_roi, dsm_roi, fill_roi, 
                                             iters=fill_iters, omega=fill_omega)
            out[ry, rx] = filled_roi
    elif method == "bilinear":
        # Use bilinear interpolation
        out = _bilinear_fill(dem, ~fill_mask, fill_mask)
    else:
        raise ValueError(f"Unknown refinement method: {method}")

    # Write output
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)
    
    return out_path


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


def _rasterize_agg(
    laz_path: str,
    width: int,
    height: int,
    transform: Affine,
    classification: Sequence[int] = (0,),
    aggregation: str = "max",
    nodata: float = -9999.0,
    aoi: Optional[Any] = None,
    aoi_buffer_in_meters: float = 0.0,
    return_classification: bool = False
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, np.ndarray, dict]:
    """
    Rasterize LiDAR points with specified aggregation method.
    
    Returns:
        If return_classification=False: (raster, profile)
        If return_classification=True: (raster, classification_raster, profile)
    """

    # try:
    with laspy.open(laz_path) as lf:
        las = lf.read()

        # --- For consistency, don't extract CRS here - let main function handle it ---
        # This ensures all profiles use the same CRS logic as the final output
        crs = None

    # Filter by classification
    if len(classification)>0:
        mask = np.isin(las.classification, classification)
        if not np.any(mask):
            raster = np.full((height, width), nodata, dtype=np.float32)
            profile = _build_raster_profile(
                width, height, transform, crs, nodata, raster.dtype
            )
            if return_classification:
                class_raster = np.full((height, width), 1, dtype=np.uint8)  # 1 = no data
                return raster, class_raster, profile
            return raster, profile

        x, y, z = las.x[mask], las.y[mask], las.z[mask]
        class_vals = las.classification[mask]
    else:
        x, y, z = las.x, las.y, las.z
        class_vals = las.classification

    # Apply AOI filter if provided
    if aoi is not None:
        try:
            from shapely.geometry import Point
            import geopandas as gpd

            # Get AOI geometry
            if hasattr(aoi, 'bounds'):
                # AOI is a geometry
                aoi_geom = aoi
            elif hasattr(aoi, 'geometry'):
                # AOI is a GeoDataFrame
                aoi_geom = aoi.geometry.union_all() if hasattr(aoi.geometry, 'union_all') else aoi.geometry.unary_union
            else:
                # Try to convert to GeoDataFrame if it's a path
                if isinstance(aoi, str):
                    aoi_gdf = gpd.read_file(aoi)
                    aoi_geom = aoi_gdf.geometry.union_all() if hasattr(aoi_gdf.geometry, 'union_all') else aoi_gdf.geometry.unary_union
                else:
                    aoi_geom = None

            if aoi_geom is not None:
                # Buffer the AOI if specified
                if aoi_buffer_in_meters > 0:
                    aoi_geom = aoi_geom.buffer(aoi_buffer_in_meters)

                # Optimized AOI filtering using vectorized operations
                # First do a quick bounding box filter
                minx, miny, maxx, maxy = aoi_geom.bounds
                bbox_mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
                
                if np.any(bbox_mask):
                    # Only do expensive containment test on points in bounding box
                    bbox_x, bbox_y = x[bbox_mask], y[bbox_mask]
                    
                    # Use shapely's vectorized contains for better performance
                    from shapely import prepare, vectorized
                    prepare(aoi_geom)  # Optimize geometry for repeated queries
                    
                    # Create points array and test containment
                    points_in_bbox = np.column_stack([bbox_x, bbox_y])
                    contains_in_bbox = vectorized.contains(aoi_geom, points_in_bbox[:, 0], points_in_bbox[:, 1])
                    
                    # Map back to original indices
                    within_mask = np.zeros(len(x), dtype=bool)
                    within_mask[bbox_mask] = contains_in_bbox
                else:
                    within_mask = np.zeros(len(x), dtype=bool)

                if np.any(within_mask):
                    x, y, z = x[within_mask], y[within_mask], z[within_mask]
                    class_vals = class_vals[within_mask]
                    print(f"Average height: {np.mean(z[z > 0])}")
                else:
                    # No points in AOI, return empty raster
                    raster = np.full((height, width), nodata, dtype=np.float32)
                    profile = _build_raster_profile(
                        width, height, transform, crs, nodata, raster.dtype
                    )
                    if return_classification:
                        class_raster = np.full((height, width), 1, dtype=np.uint8)
                        # 1 = unclassified points based on
                        # https://datacloud.ide.cat/especificacions/lidar-territorial-v3r1-20250905.pdf
                        return raster, class_raster, profile
                    return raster, profile

        except Exception as e:
            warnings.warn(f"AOI filtering failed: {e}. Processing all points.")
            pass

    # Convert to pixel coordinates
    inv_transform = ~transform
    px, py = inv_transform * (x, y)
    px_int = np.round(px).astype(int)
    py_int = np.round(py).astype(int)

    # Filter valid pixels
    valid = (
        (px_int >= 0) & (px_int < width) &
        (py_int >= 0) & (py_int < height)
    )

    if not np.any(valid):
        raster = np.full((height, width), nodata, dtype=np.float32)
        profile = _build_raster_profile(
            width, height, transform, crs, nodata, raster.dtype
        )
        if return_classification:
            class_raster = np.full((height, width), 1, dtype=np.uint8)  # 1 = no data
            return raster, class_raster, profile
        return raster, profile

    px_int = px_int[valid]
    py_int = py_int[valid]
    z_valid = np.asarray(z[valid])
    class_valid = class_vals[valid]

    raster = np.full((height, width), nodata, dtype=np.float32)

    # Initialize classification raster if needed
    if return_classification:
        class_raster = np.full((height, width), 1, dtype=np.uint8)

    # Aggregation
    if aggregation == "max":
        np.maximum.at(raster, (py_int, px_int), z_valid)

        # For classification with max: use classification of highest point per pixel
        if return_classification:
            # Create temporary arrays to track max elevation and corresponding classification
            for i in range(len(px_int)):
                py, px, z_val, class_val = py_int[i], px_int[i], z_valid[i], class_valid[i]
                if raster[py, px] == nodata or z_val > raster[py, px]:
                    class_raster[py, px] = class_val

    elif aggregation == "min":
        # More efficient min aggregation - modify raster in place
        # First pass: set initial values
        for i in range(len(px_int)):
            py, px, z_val = py_int[i], px_int[i], z_valid[i]
            if raster[py, px] == nodata:
                raster[py, px] = z_val
            else:
                raster[py, px] = min(raster[py, px], z_val)

        # For classification with min: track minimum during the same loop
        if return_classification:
            # Use a more efficient approach - track min values and classes together
            pixel_min_data = {}
            for i in range(len(px_int)):
                py, px, z_val, class_val = py_int[i], px_int[i], z_valid[i], class_valid[i]
                pixel_key = (py, px)
                if pixel_key not in pixel_min_data or z_val < pixel_min_data[pixel_key][0]:
                    pixel_min_data[pixel_key] = (z_val, class_val)
            
            # Apply classifications
            for (py, px), (_, class_val) in pixel_min_data.items():
                class_raster[py, px] = class_val

    elif aggregation == "mean":
        count = np.zeros((height, width), dtype=np.uint32)
        sum_raster = np.zeros((height, width), dtype=np.float64)
        np.add.at(sum_raster, (py_int, px_int), z_valid)
        np.add.at(count, (py_int, px_int), 1)
        mask = count > 0
        raster[mask] = (sum_raster[mask] / count[mask]).astype(np.float32)

        # For classification with mean: use most common classification per pixel
        if return_classification:
            from collections import defaultdict
            pixel_classifications = defaultdict(list)

            # Collect all classifications per pixel
            for i in range(len(px_int)):
                py, px, class_val = py_int[i], px_int[i], class_valid[i]
                pixel_classifications[(py, px)].append(class_val)

            # Find most common classification per pixel
            for (py, px), class_list in pixel_classifications.items():
                from collections import Counter
                most_common_class = Counter(class_list).most_common(1)[0][0]
                class_raster[py, px] = most_common_class

    elif aggregation == "smart":
        # Fast smart aggregation using multi-resolution outlier detection
        from scipy.ndimage import zoom
        
        # Step 1: Create coarse resolution analysis (5x downsampled)
        scale_factor = 5
        coarse_height = max(1, height // scale_factor)
        coarse_width = max(1, width // scale_factor)
        
        # Downscale pixel coordinates for coarse grid
        coarse_px_int = px_int // scale_factor
        coarse_py_int = py_int // scale_factor
        
        # Clip to coarse grid bounds
        valid_coarse = (coarse_px_int >= 0) & (coarse_px_int < coarse_width) & \
                       (coarse_py_int >= 0) & (coarse_py_int < coarse_height)
        
        if np.any(valid_coarse):
            coarse_px_int = coarse_px_int[valid_coarse]
            coarse_py_int = coarse_py_int[valid_coarse]
            coarse_z_valid = z_valid[valid_coarse]
            
            # Create coarse grids for statistics
            coarse_max = np.full((coarse_height, coarse_width), -np.inf, dtype=np.float32)
            coarse_min = np.full((coarse_height, coarse_width), np.inf, dtype=np.float32)
            coarse_count = np.zeros((coarse_height, coarse_width), dtype=np.int32)
            
            # Aggregate coarse statistics
            np.maximum.at(coarse_max, (coarse_py_int, coarse_px_int), coarse_z_valid)
            np.minimum.at(coarse_min, (coarse_py_int, coarse_px_int), coarse_z_valid)
            np.add.at(coarse_count, (coarse_py_int, coarse_px_int), 1)
            
            # Calculate coarse outlier threshold
            valid_coarse_mask = coarse_count > 0
            coarse_height_diff = np.where(valid_coarse_mask, 
                                        coarse_max - coarse_min, 
                                        0)
            
            # Upsample the outlier thresholds back to fine resolution
            outlier_threshold_map = zoom(coarse_height_diff, scale_factor, order=1)
            coarse_min_upsampled = zoom(np.where(valid_coarse_mask, coarse_min, np.nan), 
                                      scale_factor, order=1)
            
            # Ensure upsampled arrays match target dimensions
            if outlier_threshold_map.shape[0] != height or outlier_threshold_map.shape[1] != width:
                outlier_threshold_map = zoom(coarse_height_diff, 
                                           (height/coarse_height, width/coarse_width), 
                                           order=1)
                coarse_min_upsampled = zoom(np.where(valid_coarse_mask, coarse_min, np.nan),
                                          (height/coarse_height, width/coarse_width),
                                          order=1)
        else:
            # Fallback if no valid coarse points
            outlier_threshold_map = np.full((height, width), 10.0)  # 10m default threshold
            coarse_min_upsampled = np.zeros((height, width))
        
        # Step 2: Apply fine-resolution filtering using coarse-derived thresholds
        from collections import defaultdict
        pixel_data = defaultdict(list)
        
        # Collect all heights per fine pixel
        for i in range(len(px_int)):
            py, px, z_val = py_int[i], px_int[i], z_valid[i]
            pixel_data[(py, px)].append((z_val, class_valid[i] if return_classification else None))
        
        # Process each fine pixel with coarse-informed thresholds
        for (py, px), height_data in pixel_data.items():
            if len(height_data) == 1:
                # Single measurement - use as is
                raster[py, px] = height_data[0][0]
                if return_classification:
                    class_raster[py, px] = height_data[0][1]
            else:
                # Multiple measurements - use fast coarse-informed filtering
                heights = np.array([h[0] for h in height_data])
                
                # Get coarse-derived threshold for this pixel
                local_height_range = outlier_threshold_map[py, px] if not np.isnan(outlier_threshold_map[py, px]) else 10.0
                local_baseline = coarse_min_upsampled[py, px] if not np.isnan(coarse_min_upsampled[py, px]) else np.min(heights)
                
                # Simple and fast outlier detection
                median_height = np.median(heights)
                outlier_threshold = median_height + min(10.0, max(3.0, local_height_range * 0.7))
                
                # Filter outliers
                filtered_heights = heights[heights <= outlier_threshold]
                
                if len(filtered_heights) > 0:
                    raster[py, px] = np.mean(filtered_heights)
                    if return_classification:
                        filtered_indices = np.where(heights <= outlier_threshold)[0]
                        # Use most common class (simple approach)
                        filtered_classes = [height_data[i][1] for i in filtered_indices]
                        from collections import Counter
                        most_common_class = Counter(filtered_classes).most_common(1)[0][0]
                        class_raster[py, px] = most_common_class
                else:
                    # All outliers - use minimum
                    raster[py, px] = np.min(heights)
                    if return_classification:
                        min_idx = np.argmin(heights)
                        class_raster[py, px] = height_data[min_idx][1]

    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    profile = _build_raster_profile(
        width, height, transform, crs, nodata, raster.dtype
    )

    if return_classification:
        return raster, class_raster, profile
    return raster, profile


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
    import shapely.geometry as sg
    from rasterio.features import rasterize
    
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


def _calculate_optimal_batch_size(
    laz_paths: Sequence[str],
    aoi: Optional[Any] = None,
    aoi_buffer_in_meters: float = 0.0,
    max_ram_gb: float = 8.0
) -> int:
    """Calculate optimal batch size based on AOI area and available RAM."""
    try:
        import psutil
        # Get available RAM (use 70% to be conservative)
        available_ram_gb = psutil.virtual_memory().available / (1024**3) * 0.7
        target_ram_gb = min(max_ram_gb, available_ram_gb)
    except ImportError:
        warnings.warn("psutil not available. Using default RAM estimate.")
        target_ram_gb = max_ram_gb
    
    # If no AOI provided, use conservative batch size based on file count
    if aoi is None:
        # Each LAZ file is ~300MB, assume 2-3x RAM usage during processing
        ram_per_file_gb = 0.9  # 900MB RAM per 300MB file (3x multiplier)
        batch_size = max(1, int(target_ram_gb / ram_per_file_gb))
        print(f"No AOI provided. Estimated {ram_per_file_gb:.1f}GB RAM per file. Batch size: {batch_size}")
        return min(batch_size, len(laz_paths))
    
    # Calculate AOI area with buffer
    try:
        from shapely.geometry import box
        import geopandas as gpd
        
        if hasattr(aoi, 'bounds'):
            # AOI is a geometry
            aoi_geom = aoi
        elif hasattr(aoi, 'geometry'):
            # AOI is a GeoDataFrame
            aoi_geom = aoi.geometry.union_all() if hasattr(aoi.geometry, 'union_all') else aoi.geometry.unary_union
        else:
            # Fallback to conservative batch size
            batch_size = max(1, int(target_ram_gb * 10))
            return min(batch_size, len(laz_paths))
        
        # Buffer the AOI
        if aoi_buffer_in_meters > 0:
            aoi_geom = aoi_geom.buffer(aoi_buffer_in_meters)
        
        # Calculate area in square kilometers
        area_km2 = aoi_geom.area / (1000 * 1000)
        
        # For 300MB LAZ files, estimate RAM usage more directly
        # Assume each 300MB file covers roughly equal area
        total_area_all_files = area_km2  # Area covered by AOI
        
        # Estimate how many files actually intersect with the AOI
        # This is a rough estimate - in practice, you'd check file bounds vs AOI
        files_intersecting_aoi = len(laz_paths)  # Conservative: assume all files intersect
        
        # Each file needs ~900MB RAM (300MB file * 3x processing overhead)
        ram_per_file_gb = 0.9
        estimated_total_ram_gb = files_intersecting_aoi * ram_per_file_gb
        
        # Calculate batch size to stay within RAM limits
        batch_size = max(1, int(target_ram_gb / ram_per_file_gb))
        
        print(f"AOI area: {area_km2:.2f} km², Estimated RAM per file: {ram_per_file_gb:.1f}GB")
        print(f"Target RAM: {target_ram_gb:.1f}GB, Calculated batch size: {batch_size}")
        
        return min(batch_size, len(laz_paths))
        
    except Exception as e:
        warnings.warn(f"Could not calculate optimal batch size: {e}. Using conservative value.")
        batch_size = max(1, int(target_ram_gb * 10))
        return min(batch_size, len(laz_paths))


def build_digital_model_from_lidar_pure(
    laz_paths: Sequence[str],
    out_path: str,
    out_classification_path: Optional[str] = None,  # Path to save classification raster
    resolution: float = 1.0,
    bounds: Optional[Tuple[float, float, float, float]] = None,  # (minx,miny,maxx,maxy)
    classification: Sequence[int] = (0,),
    epsg: int = 25831,               # EPSG code of LiDAR files (deprecated, use lidar_epsg)
    fill_gaps: Optional[str] = "nearest",  # None | "nearest"
    aggregation: str = "max",              # "max" | "min" | "mean"
    nodata: float = -9999.0,
    batch_size: Optional[int] = None,      # If None, auto-calculated
    max_ram_gb: float = 64.0,              # Max RAM to use for auto-calculation
    aoi: Optional[Any] = None,             # Area of interest geometry
    aoi_epsg: Optional[int] = None,        # EPSG code of AOI
    aoi_buffer_in_meters: float = 0.0,     # Buffer around AOI
    lidar_epsg: Optional[int] = None,      # EPSG code of LiDAR files (preferred over epsg)
) -> str:
    """Build DSM from LiDAR files using pure Python/NumPy implementation with batching.
    
    Parameters:
    -----------
    lidar_epsg : int, default 25831
        EPSG code of the LiDAR coordinate reference system (e.g., 25831 for UTM 31N)
    aoi_epsg : Optional[int], default None
        EPSG code of the AOI coordinate reference system (e.g., 4326 for WGS84).
        If None, assumes AOI is in same CRS as LiDAR files.
    out_classification_path : Optional[str]
        If provided, saves a classification raster showing the main classification 
        value for each pixel (corresponding to the elevation aggregation method used).
    epsg : Optional[int]
        Deprecated parameter. Use lidar_epsg instead.
    """
    
    if not laz_paths:
        raise ValueError("No LAZ files provided")
    
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if out_classification_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_classification_path)), exist_ok=True)

    # Handle deprecated epsg parameter
    if lidar_epsg is None:
        lidar_epsg = epsg
    
    # Transform AOI to LiDAR CRS if necessary
    aoi_transformed = aoi
    if aoi is not None and aoi_epsg is not None and aoi_epsg != lidar_epsg:
        print(f"Transforming AOI from EPSG:{aoi_epsg} to EPSG:{lidar_epsg}")
        try:
            import geopandas as gpd
            from shapely.geometry import mapping

            # Convert AOI to GeoDataFrame if it's not already
            if hasattr(aoi, 'bounds'):
                # AOI is a geometry
                gdf_aoi = gpd.GeoDataFrame({'geometry': [aoi]}, crs=f'EPSG:{aoi_epsg}')
            elif hasattr(aoi, 'geometry'):
                # AOI is already a GeoDataFrame
                gdf_aoi = aoi.copy()
                if gdf_aoi.crs is None:
                    gdf_aoi.crs = f'EPSG:{aoi_epsg}'
            elif isinstance(aoi, str):
                # AOI is a file path
                gdf_aoi = gpd.read_file(aoi)
                if gdf_aoi.crs is None:
                    gdf_aoi.crs = f'EPSG:{aoi_epsg}'
            else:
                raise ValueError("AOI format not recognized")

            # Transform to LiDAR CRS
            gdf_aoi_transformed = gdf_aoi.to_crs(f'EPSG:{lidar_epsg}')
            aoi_transformed = gdf_aoi_transformed.geometry.union_all() if hasattr(gdf_aoi_transformed.geometry, 'union_all') else gdf_aoi_transformed.geometry.unary_union

            print(f"AOI transformed successfully")

        except Exception as e:
            warnings.warn(f"Failed to transform AOI CRS: {e}. Using AOI as-is.")
            aoi_transformed = aoi
    elif aoi is not None and aoi_epsg is None:
        print(f"Assuming AOI is already in LiDAR CRS (EPSG:{lidar_epsg})")
        aoi_transformed = aoi
    
    # Calculate optimal batch size if not provided
    if batch_size is None:
        batch_size = _calculate_optimal_batch_size(
            laz_paths=laz_paths, aoi=None, max_ram_gb=max_ram_gb
        )
    
    print(f"Processing {len(laz_paths)} LAZ files in batches of {batch_size}")
    
    # Determine bounds
    if bounds is None:
        bounds = _infer_bounds_from_lidar(laz_paths, classification)
    
    minx, miny, maxx, maxy = bounds
    
    # Create aligned grid
    width, height, transform = _aligned_grid(minx, miny, maxx, maxy, resolution)
    
    print(f"Creating DSM: {width}x{height} pixels at {resolution}m resolution")
    print(f"Bounds: {bounds}")
    
    # Initialize output raster
    if aggregation == "max":
        dm = np.full((height, width), -np.inf, dtype=np.float32)
        count = None
    elif aggregation == "min":
        dm = np.full((height, width), np.inf, dtype=np.float32)
        count = None
    else:
        dm = np.zeros((height, width), dtype=np.float64)
        count = np.zeros((height, width), dtype=int)
    
    # Initialize classification raster if requested
    if out_classification_path:
        class_dsm = np.full((height, width), 1, dtype=np.uint8)  # 1 = no data
    
    # Process LAZ files in batches
    total_batches = (len(laz_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(laz_paths))
        batch_files = laz_paths[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)")
        
        # Process each file in the current batch
        for enum,laz_path in enumerate(batch_files):
            print(f"{enum+1} - Processing {os.path.basename(laz_path)}")
            
            # Get raster and optionally classification raster
            if out_classification_path:
                raster, class_raster, profile = _rasterize_agg(
                    laz_path, width, height, transform, classification, 
                    aggregation, nodata,
                    return_classification=True
                )
            else:
                raster, profile = _rasterize_agg(
                    laz_path, width, height, transform, classification, 
                    aggregation, nodata,
                    return_classification=False
                )

            if aggregation == "max":
                valid = raster != nodata
                dm[valid] = np.maximum(dm[valid], raster[valid])
                if out_classification_path:
                    class_dsm[valid] = class_raster[valid]
            elif aggregation == "min":
                valid = raster != nodata
                # For min aggregation, we need to be more careful with initialization
                # Only update pixels that have valid data in this file
                if np.any(valid):
                    # Initialize pixels with inf to the first valid value
                    first_valid_mask = (dm == np.inf) & valid
                    dm[first_valid_mask] = raster[first_valid_mask]
                    
                    # For already initialized pixels, take minimum
                    already_valid_mask = (dm != np.inf) & valid
                    dm[already_valid_mask] = np.minimum(dm[already_valid_mask], raster[already_valid_mask])
                    
                if out_classification_path:
                    class_dsm[valid] = class_raster[valid]
            elif aggregation == "mean":
                valid = raster != nodata
                dm[valid] += raster[valid]
                if count is not None:
                    count[valid] += 1
                if out_classification_path:
                    # For mean aggregation, we need to track all classifications
                    # and determine the most common one at the end
                    # For now, just update with the current file's classification
                    class_dsm[valid] = class_raster[valid]
        
        # Force garbage collection after each batch to free memory
        import gc
        gc.collect()
        
        print(f"Batch {batch_idx + 1}/{total_batches} completed")
    
    # Finalize aggregation
    if aggregation == "max":
        dm[dm == -np.inf] = nodata
    elif aggregation == "min":
        dm[dm == np.inf] = nodata
    elif aggregation == "mean":
        if count is not None:
            valid_pixels = count > 0
            dm[valid_pixels] = dm[valid_pixels] / count[valid_pixels]
            dm[~valid_pixels] = nodata
        dm = dm.astype(np.float32)
    
    # Apply AOI mask before gap filling
    aoi_mask = None
    if aoi_transformed is not None:
        print("Creating AOI mask...")
        aoi_mask = _create_aoi_mask(aoi_transformed, width, height, transform, aoi_buffer_in_meters)
        
        # Set pixels outside AOI to nodata
        outside_aoi = ~aoi_mask
        dm[outside_aoi] = nodata
        print(f"Masked {np.sum(outside_aoi)} pixels outside AOI")
    
    # Fill gaps if requested (but don't fill outside AOI)
    if fill_gaps == "nearest":
        gap_mask = (dm == nodata) | ~np.isfinite(dm)
        
        # If we have an AOI mask, exclude areas outside AOI from gap filling
        if aoi_mask is not None:
            gap_mask = gap_mask & aoi_mask
        
        if np.any(gap_mask):
            print("Filling gaps with nearest neighbor interpolation...")
            dm = _nearest_fill_chunked(gap_mask, dm, max_ram_gb, aoi_mask=aoi_mask)
    
    # Write output - use CRS from first processed file if available, otherwise use lidar_epsg
    final_crs = f'EPSG:{lidar_epsg}'
    
    # Try to get CRS from first successfully processed LAZ file
    for laz_path in laz_paths:
        try:
            with laspy.open(laz_path) as lf:
                las = lf.read()
                try:
                    file_crs = CRS.from_wkt(las.header.parse_crs().to_wkt())
                    if file_crs is not None:
                        final_crs = file_crs
                        print(f"Using CRS from LAZ file: {final_crs}")
                        break
                except Exception:
                    continue
        except Exception:
            continue
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': nodata,
        'width': width,
        'height': height,
        'count': 1,
        'crs': final_crs,
        'transform': transform,
        'compress': 'deflate',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    }
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(dm.astype(np.float32), 1)
    
    print(f"Digital model saved in {out_path}")
    
    # Save classification raster if requested
    if out_classification_path:
        class_profile = profile.copy()
        class_profile.update(dtype='uint8', nodata=1)
        
        with rasterio.open(out_classification_path, 'w', **class_profile) as dst:
            dst.write(class_dsm, 1)
        
        print(f"Classification raster saved in {out_classification_path}")
        
        # Print classification statistics
        unique_classes = np.unique(class_dsm[class_dsm != 1])
        print(f"Classification values found: {unique_classes.tolist()}")
    
    return out_path


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
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
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
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt
    
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