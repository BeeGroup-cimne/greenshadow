# -*- coding: utf-8 -*-
import calendar
import fnmatch
import math
from typing import List, Dict, NamedTuple, Optional, Tuple, Any, Sequence
import pandas as pd
from rasterio.enums import Resampling
from rasterio.warp import reproject
from osgeo import gdal
import datetime as dt
from .utils import _parse_iso8601_duration, _date_range_local_to_utc, has_cuda, rasterize_polygon_mask
from .logging_utils import info, warning, tqdm
from numba import njit, prange, cuda, float32, float64
import scipy.ndimage as ndi
import sys
import os
import rasterio
import multiprocessing as mp
import concurrent.futures
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


@njit(parallel=True, fastmath=True, cache=True)
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

# ======================================================================================
# SEMI-TRANSPARENT VEGETATION
#
# The plain cast-shadow model above treats every DSM pixel as an opaque wall, which
# turns a tree into an extruded column and its shadow into a hard black blob. The
# functions below replace that for vegetation with a participating medium:
#
#   * high vegetation is resolved into individual tree crowns, each a half-ellipsoid
#     sized from the canopy height H = DSM - DEM (see ``detect_tree_crowns``);
#   * low and medium vegetation keeps the DSM column, which is a fair description of
#     a shrub and has no treetop to shape;
#   * the ray-march accumulates the *path length* through that medium instead of
#     returning a lit/unlit bit (see ``cast_canopy_path_lengths``).
#
# Path length is stored per LiDAR height class and is independent of the season, so
# one hillshade serves every month of the year: the month only enters at consumption
# time through the extinction coefficient (``vegetation_extinction`` /
# ``apply_canopy_transmittance``). That is what keeps the (azimuth, altitude)
# hillshade cache usable for a full-year run.
#
# Everything here is opt-in. With ``canopy=None`` the hillshade pipeline follows the
# original opaque path and produces byte-identical single-band output.
# ======================================================================================

# Fraction of the beam stopped by one full vertical crossing of the crown.
# rows = leaf type (deciduous / evergreen), then height class, then month.
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

# ICGC LiDAR class -> index into VEGETATION_SHADOWING (-1 = not vegetation).
ICGC_DSM_CLASS_TO_HEIGHT_INDEX = np.array(
    [-1, -1, -1, 0, 1, 2], dtype=np.int8
)

# Nominal crown thickness each row of VEGETATION_SHADOWING is calibrated against.
# The table gives a *total* attenuation; turning it into an extinction coefficient
# needs the thickness that attenuation refers to. These follow the ICGC class
# definitions (low 0.15-2 m, medium 2-5 m, high > 5 m).
VEGETATION_REFERENCE_THICKNESS_M = np.array([1.0, 2.5, 6.0], dtype=np.float32)

N_VEGETATION_CLASSES = 3

# Bands 2-4 of a canopy hillshade, in the order of ICGC_DSM_CLASS_TO_HEIGHT_INDEX.
CANOPY_BAND_NAMES = (
    "canopy_path_low_m",
    "canopy_path_medium_m",
    "canopy_path_high_m",
)

# Stamped on canopy hillshades so readers cannot mistake one for a plain hillshade
# and silently drop the vegetation attenuation.
CANOPY_RASTER_TAGS = {"GREENSHADOW_CANOPY": "path_lengths_m"}

# How hillshades are written. Their values sit on a few levels - k/255 for the
# hillshade, whole ray steps for the canopy path lengths - which the floating-point
# predictor scrambles into noise. ZSTD on the raw floats writes a full-grid canopy
# hillshade 7x faster than DEFLATE with PREDICTOR=3 (0.8 s against 4.4 s) and
# makes it half the size, which also halves the disk reads that bound
# compute_surface_solar_energies. Lossless either way.
HILLSHADE_COMPRESSION = {"compress": "zstd", "zstd_level": 6, "predictor": 1}


def geotiff_is_complete(path) -> bool:
    """
    Whether a GeoTIFF was written out in full: it opens, and every block has
    been written and lies inside the file.

    rasterio does not raise when GDAL fails to write - on a full disk, say,
    the file is closed with blocks missing or empty - so such a file would
    be taken as finished by every later run. Only the TIFF directory is
    read, not the pixels: ~0.1 s for a canopy hillshade.
    """
    path = str(path)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        ds = gdal.Open(path)
        if ds is None or ds.GetDriver().ShortName != "GTiff":
            return ds is not None
        size = os.path.getsize(path)
        # pixel interleaving keeps all the bands in one set of blocks
        separate = ds.GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE") == "BAND"
        for b in range(1, (ds.RasterCount if separate else 1) + 1):
            band = ds.GetRasterBand(b)
            bx, by = band.GetBlockSize()
            for y in range((ds.RasterYSize + by - 1) // by):
                for x in range((ds.RasterXSize + bx - 1) // bx):
                    offset = band.GetMetadataItem(f"BLOCK_OFFSET_{x}_{y}", "TIFF")
                    nbytes = band.GetMetadataItem(f"BLOCK_SIZE_{x}_{y}", "TIFF")
                    if not offset or not nbytes or int(nbytes) == 0:
                        return False
                    if int(offset) + int(nbytes) > size:
                        return False
        return True
    finally:
        ds = None
        gdal.PopErrorHandler()


def _finish_geotiff(tmp_path, path):
    """Move a GeoTIFF written under ``tmp_path`` to ``path``, if it was written in full."""
    if not geotiff_is_complete(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise OSError(
            f"Writing {path} failed: the file came out incomplete "
            "(disk full, or a write error)"
        )
    os.replace(tmp_path, path)


class CanopyModel(NamedTuple):
    """
    The semi-transparent canopy, on the DSM grid.

    crown_top / crown_base
        Upper and lower envelope of the medium, in metres above the vertical datum.
        Only meaningful where ``class_index >= 0``.
    class_index
        Height class of the medium at each pixel (0 low, 1 medium, 2 high, -1 none).
        Selects which path-length band the ray-march writes into.
    opaque
        The surface that still blocks the beam completely: the DSM everywhere except
        vegetation, dropping to the under-canopy surface beneath a crown.
    render_surface
        DSM with crowns replaced by their own envelope, so the local hillshade draws
        domes instead of LiDAR speckle.
    trees
        Per-tree table (row, col, height_m, crown_radius_m) for inspection.
    """

    crown_top: np.ndarray
    crown_base: np.ndarray
    class_index: np.ndarray
    opaque: np.ndarray
    render_surface: np.ndarray
    trees: dict


class VegetationGrowth(NamedTuple):
    """
    How much the canopy is assumed to have grown between the LiDAR flight and the
    year being simulated. ``years`` is negative when simulating before the flight.

    Height and crown diameter are both scaled the same way::

        H' = min(H * height_factor + height_gain_m, max_height_m)
        D' = D * crown_factor + crown_gain_m
    """

    years: float = 0.0
    height_gain_m: float = 0.0
    crown_gain_m: float = 0.0
    height_factor: float = 1.0
    crown_factor: float = 1.0
    max_height_m: float = 40.0

    @property
    def is_identity(self) -> bool:
        return (
            self.height_gain_m == 0.0
            and self.crown_gain_m == 0.0
            and self.height_factor == 1.0
            and self.crown_factor == 1.0
        )


NO_GROWTH = VegetationGrowth()


def _as_date(value) -> Optional[dt.date]:
    """Accept a date, datetime, ISO string, or bare year."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, (int, np.integer)):
        return dt.date(int(value), 7, 1)      # mid-year, so a bare year is unbiased
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    if isinstance(value, pd.Timestamp):
        return value.date()
    raise TypeError(f"Cannot interpret {value!r} as a date")


def vegetation_growth(
    lidar_date=None,
    simulation_date=None,
    *,
    height_rate_m_per_year: float = 0.25,
    crown_rate_m_per_year: float = 0.20,
    relative_rate_per_year: Optional[float] = None,
    max_height_m: float = 40.0,
) -> VegetationGrowth:
    """
    Build the growth correction between a LiDAR acquisition and a simulated year.

    LiDAR is a snapshot. Simulating 2030 shading off a 2022 flight with the trees
    frozen at their 2022 size systematically under-shades; this ages them forward.

    Two ways to express the rate, and they compose:

    * **additive** (default) - ``height_rate_m_per_year`` metres of height and
      ``crown_rate_m_per_year`` metres of crown *diameter* per year. Reasonable
      for established urban trees, which grow roughly linearly over a decade.
    * **relative** - set ``relative_rate_per_year`` (e.g. 0.02 for 2 %/year) to
      scale multiplicatively instead, which suits young or fast-growing stands.

    Args:
        lidar_date: When the point cloud was flown. For ICGC data see
            ``greenshadow.lidar.icgc_lidar_acquisition_date``.
        simulation_date: The date or year being simulated.
        height_rate_m_per_year: Additive height growth.
        crown_rate_m_per_year: Additive crown *diameter* growth.
        relative_rate_per_year: If given, use multiplicative growth instead of additive.
        max_height_m: Hard cap on grown tree height.

    Returns:
        A ``VegetationGrowth``. With either date missing it is the identity, so
        growth stays off unless it was asked for.

    Example:
        >>> flown = gs.lidar.icgc_lidar_acquisition_date(aoi)      # date(2022, 7, 3)
        >>> growth = gs.shader.vegetation_growth(flown, 2030)
        >>> growth.years, round(growth.height_gain_m, 2)
        (7.5, 1.88)
    """
    d0 = _as_date(lidar_date)
    d1 = _as_date(simulation_date)
    if d0 is None or d1 is None:
        return VegetationGrowth(max_height_m=max_height_m)

    years = (d1 - d0).days / 365.25

    if relative_rate_per_year is not None:
        factor = float((1.0 + relative_rate_per_year) ** years)
        return VegetationGrowth(
            years=years,
            height_factor=factor,
            crown_factor=factor,
            max_height_m=max_height_m,
        )

    return VegetationGrowth(
        years=years,
        height_gain_m=float(height_rate_m_per_year * years),
        crown_gain_m=float(crown_rate_m_per_year * years),
        max_height_m=max_height_m,
    )


def vegetation_extinction(
    month: int,
    *,
    leaf_type: int = 0,
    reference_thickness_m: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Beer-Lambert extinction coefficients [1/m], one per LiDAR height class.

    ``VEGETATION_SHADOWING`` gives the fraction ``s`` stopped by a full vertical
    crossing of ``reference_thickness_m`` metres of crown, so
    ``mu = -ln(1 - s) / thickness``. Every other ray direction then follows from
    its own path length, which is what makes crown edges lighter than crown centres.

    Args:
        month: 1-12.
        leaf_type: 0 deciduous, 1 evergreen.
        reference_thickness_m: Override the nominal crown thickness per class.

    Returns:
        float32 array of 3 coefficients, ordered low / medium / high vegetation.
    """
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be 1-12, got {month}")

    thickness = (
        VEGETATION_REFERENCE_THICKNESS_M
        if reference_thickness_m is None
        else np.asarray(reference_thickness_m, dtype=np.float32)
    )
    s = VEGETATION_SHADOWING[int(leaf_type), :, int(month) - 1]
    return (-np.log(np.clip(1.0 - s, 1e-3, 1.0)) / np.maximum(thickness, 0.1)).astype(
        np.float32
    )


def apply_canopy_transmittance(
    base: np.ndarray,
    path_lengths: np.ndarray,
    month: int,
    *,
    leaf_type: int = 0,
    reference_thickness_m: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Combine an opaque-shadow hillshade with its canopy path-length bands.

    Args:
        base: Band 1 of a canopy hillshade - local shading with opaque shadows only.
        path_lengths: Bands 2-4, shape (3, H, W), metres of canopy crossed per class.
        month: 1-12, selects the leaf-on / leaf-off state.
        leaf_type: 0 deciduous, 1 evergreen.
        reference_thickness_m: Passed through to ``vegetation_extinction``.

    Returns:
        float32 direct-beam fraction in [0, 1].
    """
    mu = vegetation_extinction(
        month, leaf_type=leaf_type, reference_thickness_m=reference_thickness_m
    )
    optical_depth = np.tensordot(mu, path_lengths, axes=(0, 0))
    return (base * np.exp(-optical_depth)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Tree crown detection and rasterisation
# --------------------------------------------------------------------------------------
@njit(cache=True)
def _suppress_overlapping_crowns(order_r, order_c, chm, radius_px, claimed):
    """Greedy non-maximum suppression: the tallest pixel wins and claims its crown."""
    n = order_r.size
    tr = np.empty(n, np.int32)
    tc = np.empty(n, np.int32)
    th = np.empty(n, np.float32)
    k = 0
    H, W = chm.shape

    for idx in range(n):
        r = order_r[idx]
        c = order_c[idx]
        if claimed[r, c]:
            continue

        rad = radius_px[r, c]
        ri = int(rad) + 1
        r0 = max(0, r - ri); r1 = min(H, r + ri + 1)
        c0 = max(0, c - ri); c1 = min(W, c + ri + 1)

        for rr in range(r0, r1):
            dr = rr - r
            for cc in range(c0, c1):
                dc = cc - c
                if dr * dr + dc * dc <= rad * rad:
                    claimed[rr, cc] = True

        tr[k] = r
        tc[k] = c
        th[k] = chm[r, c]
        k += 1

    return tr[:k], tc[:k], th[:k]


def detect_tree_crowns(
    chm: np.ndarray,
    high_veg_mask: np.ndarray,
    pixel_size: float,
    *,
    min_tree_height_m: float = 3.0,
    crown_a: float = 1.00,
    crown_b: float = 0.80,
    smooth_sigma_m: float = 1.50,
    max_crown_radius_m: float = 9.0,
    observed_radius_factor: float = 2.0,
    growth: VegetationGrowth = NO_GROWTH,
):
    """
    Find individual tree tops on the canopy height model and size their crowns.

    Crown diameter follows the usual allometry ``D = a * H^b`` with H = DSM - DEM,
    then it is capped by how much crown the point cloud actually shows around the
    peak, so a single street tree cannot inherit a forest-sized crown.

    Args:
        chm: Canopy height model (DSM - DEM), metres.
        high_veg_mask: True on high-vegetation pixels.
        pixel_size: Raster resolution in metres.
        min_tree_height_m: Peaks shorter than this are not trees.
        crown_a: Allometric coefficient a.
        crown_b: Allometric exponent b.
        smooth_sigma_m: Gaussian smoothing before peak detection. Raise it to merge
            a noisy canopy into fewer, larger crowns.
        max_crown_radius_m: Hard cap on crown radius.
        observed_radius_factor: How far the crown may exceed the inscribed radius of
            its own LiDAR blob. LiDAR under-represents thin crown edges, so > 1.
        growth: Ages the trees forward, see ``vegetation_growth``.

    Returns:
        (rows, cols, height_m, crown_radius_m), one entry per tree.
    """
    sigma_px = max(smooth_sigma_m / pixel_size, 0.3)
    chm_s = ndi.gaussian_filter(
        np.where(high_veg_mask, chm, 0.0).astype(np.float32), sigma_px
    )
    chm_s[~high_veg_mask] = 0.0

    # Detection runs on the canopy as flown, so growth never changes *which* trees
    # exist - only how big the ones found are. Ageing happens at the end.

    # allometric crown radius per candidate pixel, in metres
    radius_m = 0.5 * crown_a * np.power(np.maximum(chm_s, 0.0), crown_b)
    np.clip(radius_m, 0.0, max_crown_radius_m, out=radius_m)

    # how much crown the point cloud actually shows around each pixel
    observed_r_m = ndi.distance_transform_edt(
        high_veg_mask, sampling=pixel_size
    ).astype(np.float32)
    radius_m = np.minimum(radius_m, observed_r_m * observed_radius_factor)
    radius_m = np.maximum(radius_m, 1.0)

    candidates = high_veg_mask & (chm_s >= min_tree_height_m)
    if not candidates.any():
        empty_i = np.empty(0, np.int32)
        empty_f = np.empty(0, np.float32)
        return empty_i, empty_i, empty_f, empty_f

    # a peak must dominate a window scaled to its own crown
    window = int(max(3, 2 * np.median(radius_m[candidates]) / pixel_size // 2 * 2 + 1))
    peaks = candidates & (chm_s >= ndi.maximum_filter(chm_s, size=window))

    rr, cc = np.nonzero(peaks)
    order = np.argsort(-chm_s[rr, cc])
    rr = rr[order].astype(np.int32)
    cc = cc[order].astype(np.int32)

    claimed = np.zeros(chm.shape, dtype=np.bool_)
    tr, tc, th = _suppress_overlapping_crowns(
        rr, cc, chm_s, (radius_m / pixel_size).astype(np.float32), claimed
    )
    trad = radius_m[tr, tc]

    if not growth.is_identity:
        # Applied after the observed-footprint cap on purpose: a tree that has
        # grown for a decade is genuinely wider than the blob the flight saw.
        th = np.clip(
            th * growth.height_factor + growth.height_gain_m,
            0.5, growth.max_height_m,
        ).astype(np.float32)
        trad = np.clip(
            trad * growth.crown_factor + 0.5 * growth.crown_gain_m,
            1.0, max_crown_radius_m,
        ).astype(np.float32)

    return tr, tc, th, trad


@njit(cache=True)
def _paint_crowns(
    tr, tc, th, trad_px, ground, crown_top, crown_base, class_index, crown_ratio
):
    """
    Paint every tree as a half-ellipsoid crown standing on its trunk:
    horizontal semi-axis = allometric crown radius, vertical semi-axis =
    crown_ratio * H / 2. Overlapping crowns take the union (max top, min base).
    """
    H, W = crown_top.shape

    for t in range(tr.size):
        r = tr[t]
        c = tc[t]
        h = th[t]
        rad = trad_px[t]
        if rad < 0.5 or h <= 0.0:
            continue

        depth = crown_ratio * h              # vertical extent of the crown
        z_centre = ground[r, c] + h - 0.5 * depth
        semi_z = 0.5 * depth

        ri = int(rad) + 1
        r0 = max(0, r - ri); r1 = min(H, r + ri + 1)
        c0 = max(0, c - ri); c1 = min(W, c + ri + 1)
        inv_r2 = 1.0 / (rad * rad)

        for rr in range(r0, r1):
            dr = rr - r
            for cc in range(c0, c1):
                dc = cc - c
                q = (dr * dr + dc * dc) * inv_r2
                if q > 1.0:
                    continue

                half = semi_z * np.sqrt(1.0 - q)   # ellipsoid half-thickness here
                top = z_centre + half
                base = z_centre - half

                if class_index[rr, cc] < 0:
                    class_index[rr, cc] = 2        # high vegetation
                    crown_top[rr, cc] = top
                    crown_base[rr, cc] = base
                else:
                    if top > crown_top[rr, cc]:
                        crown_top[rr, cc] = top
                    if base < crown_base[rr, cc]:
                        crown_base[rr, cc] = base


def build_canopy_model(
    dsm: np.ndarray,
    dem: np.ndarray,
    classes: np.ndarray,
    *,
    pixel_size: float,
    under_canopy: Optional[np.ndarray] = None,
    low_veg_classes: Sequence[int] = (3, 4),
    high_veg_classes: Sequence[int] = (5,),
    crown_ratio: float = 0.65,
    growth: VegetationGrowth = NO_GROWTH,
    **crown_kwargs,
) -> CanopyModel:
    """
    Turn DSM + DEM + LiDAR classes into the semi-transparent canopy the ray-march needs.

    High-vegetation pixels are resolved into ellipsoidal tree crowns sized from
    H = DSM - DEM. Low and medium vegetation keeps the DSM column. Any
    high-vegetation pixel no crown disc reached falls back to the column too, so
    the gaps between packed crowns still attenuate.

    Args:
        dsm: Digital surface model, metres.
        dem: Bare-earth model on the same grid, metres.
        classes: LiDAR class raster on the same grid (ICGC codes).
        pixel_size: Resolution in metres.
        under_canopy: Surface the beam sees beneath a crown, typically the
            vegetation-free DSM. Defaults to ``dem``.
        low_veg_classes: Classes kept as a DSM column.
        high_veg_classes: Classes resolved into tree crowns.
        crown_ratio: Crown depth as a fraction of tree height.
        growth: Ages the canopy forward, see ``vegetation_growth``.
        **crown_kwargs: Forwarded to ``detect_tree_crowns``.

    Returns:
        A ``CanopyModel`` on the input grid.
    """
    dsm = np.asarray(dsm, dtype=np.float32)
    dem = np.asarray(dem, dtype=np.float32)
    chm = np.clip(dsm - dem, 0.0, None).astype(np.float32)

    high_mask = np.isin(classes, tuple(high_veg_classes))
    low_mask = np.isin(classes, tuple(low_veg_classes))
    veg_mask = high_mask | low_mask

    under = dem if under_canopy is None else np.asarray(under_canopy, dtype=np.float32)
    opaque = np.where(veg_mask, np.minimum(under, dsm), dsm).astype(np.float32)

    crown_top = np.zeros_like(dsm)
    crown_base = np.zeros_like(dsm)
    class_index = np.full(dsm.shape, -1, dtype=np.int8)

    # ---- high vegetation: ellipsoidal treetops ----
    tr, tc, th, trad = detect_tree_crowns(
        chm, high_mask, pixel_size, growth=growth, **crown_kwargs
    )
    if tr.size:
        _paint_crowns(
            tr, tc, th,
            (trad / pixel_size).astype(np.float32),
            dem, crown_top, crown_base, class_index,
            float(crown_ratio),
        )

    # ---- low/medium vegetation, and any high-veg pixel no crown reached ----
    column = veg_mask & (class_index < 0)
    if column.any():
        top = dsm
        if not growth.is_identity:
            grown = dem + np.clip(
                chm * growth.height_factor + growth.height_gain_m,
                0.0, growth.max_height_m,
            )
            top = np.where(high_mask, np.maximum(grown, dsm), dsm).astype(np.float32)
        base = np.minimum(opaque, top)
        crown_top[column] = top[column]
        crown_base[column] = base[column]
        for cls_code in tuple(low_veg_classes) + tuple(high_veg_classes):
            idx = int(ICGC_DSM_CLASS_TO_HEIGHT_INDEX[cls_code])
            if idx < 0:
                continue
            here = column & (classes == cls_code)
            class_index[here] = idx

    # a crown must have some thickness, otherwise it contributes no optical depth
    crown_top = np.maximum(crown_top, crown_base + 0.3).astype(np.float32)
    crown_top[class_index < 0] = 0.0
    crown_base[class_index < 0] = 0.0

    render_surface = np.where(
        class_index >= 0, np.maximum(crown_top, opaque), dsm
    ).astype(np.float32)

    return CanopyModel(
        crown_top=crown_top,
        crown_base=crown_base,
        class_index=class_index,
        opaque=opaque,
        render_surface=render_surface,
        trees={
            "row": tr,
            "col": tc,
            "height_m": th,
            "crown_radius_m": trad,
        },
    )


def build_canopy_model_from_rasters(
    dsm_path: str,
    dem_path: str,
    class_path: str,
    *,
    under_canopy_path: Optional[str] = None,
    growth: VegetationGrowth = NO_GROWTH,
    **kwargs,
) -> CanopyModel:
    """
    Read the rasters a canopy model needs and build it on the DSM grid.

    DEM, class and under-canopy rasters are reprojected onto the DSM grid, so they
    need not share its extent. Holes are nearest-filled so rays stay continuous.

    Args:
        dsm_path: Digital surface model - defines the output grid.
        dem_path: Bare-earth model.
        class_path: LiDAR class raster (ICGC codes).
        under_canopy_path: Vegetation-free DSM. Falls back to the DEM.
        growth: See ``vegetation_growth``.
        **kwargs: Forwarded to ``build_canopy_model``.

    Returns:
        A ``CanopyModel`` aligned to the DSM.
    """
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        pixel_size = abs(src.transform.a)
        dsm_nodata = src.nodata

    def _on_dsm_grid(path, resampling, dtype, fill):
        with rasterio.open(path) as src:
            out = np.full((profile["height"], profile["width"]), fill, dtype=dtype)
            reproject(
                source=rasterio.band(src, 1),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=profile["transform"],
                dst_crs=profile["crs"],
                dst_nodata=fill,
                resampling=resampling,
            )
        return out

    dem = _on_dsm_grid(dem_path, Resampling.bilinear, np.float32, np.nan)
    classes = _on_dsm_grid(class_path, Resampling.nearest, np.uint8, 255)
    under = (
        _on_dsm_grid(under_canopy_path, Resampling.bilinear, np.float32, np.nan)
        if under_canopy_path
        else None
    )

    invalid = ~np.isfinite(dsm)
    if dsm_nodata is not None:
        invalid |= dsm == dsm_nodata

    def _fill(arr):
        bad = ~np.isfinite(arr)
        if bad.any() and not bad.all():
            _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
            arr[bad] = arr[iy[bad], ix[bad]]
        return arr

    dsm[invalid] = np.nan
    dsm = _fill(dsm)
    dem = _fill(dem)
    if under is not None:
        under = _fill(under)

    return build_canopy_model(
        dsm, dem, classes,
        pixel_size=pixel_size,
        under_canopy=under,
        growth=growth,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# Ray-march through the canopy
# --------------------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _canopy_path_scan(
    z_rot, opaque_rot, ctop_rot, cbase_rot, cidx_rot, nodata_rot,
    dz_per_px, ds, zmax, max_steps,
):
    """
    Sun comes from the left of the rotated frame. For every pixel, march column by
    column towards it, accumulating metres of canopy crossed per height class and
    stopping dead at the first opaque hit.
    """
    H, W = z_rot.shape
    lit = np.ones((H, W), dtype=np.uint8)
    path = np.zeros((3, H, W), dtype=np.float32)

    for i in prange(H):
        for j in range(W):
            if nodata_rot[i, j]:
                continue

            ray = z_rot[i, j]
            jj = j
            for _ in range(max_steps):
                jj -= 1
                if jj < 0:
                    break

                ray += dz_per_px
                if ray > zmax:
                    break
                if nodata_rot[i, jj]:
                    continue

                if opaque_rot[i, jj] >= ray:
                    lit[i, j] = 0
                    # fully dark already: the path lengths would never be used
                    path[0, i, j] = 0.0
                    path[1, i, j] = 0.0
                    path[2, i, j] = 0.0
                    break

                ci = cidx_rot[i, jj]
                if ci >= 0 and cbase_rot[i, jj] <= ray <= ctop_rot[i, jj]:
                    path[ci, i, j] += ds

    return lit, path


def _rotate(arr, theta, order=0, cval=0.0):
    return ndi.rotate(
        arr, theta, reshape=False, order=order, mode="constant",
        cval=cval, prefilter=False,
    )


def cast_canopy_path_lengths(
    canopy: CanopyModel,
    pixel_size: float,
    azimuth_deg: float,
    altitude_deg: float,
    *,
    nodata_mask: Optional[np.ndarray] = None,
    max_steps: int = 20000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ray-march the canopy for one sun position.

    Args:
        canopy: The model from ``build_canopy_model``.
        pixel_size: Resolution in metres.
        azimuth_deg: Solar azimuth, 0 = North, clockwise.
        altitude_deg: Solar elevation in degrees.
        nodata_mask: True where the DSM has no data.
        max_steps: Safety cap on march length. The march normally stops far sooner,
            when the ray clears the tallest obstacle in the scene.

    Returns:
        ``(lit, path_lengths, valid)`` where ``lit`` is the opaque shadow mask,
        ``path_lengths`` has shape (3, H, W) in metres per height class, and
        ``valid`` is the original footprint (rotation padding excluded).
    """
    shape = canopy.opaque.shape
    if nodata_mask is None:
        nodata_mask = np.zeros(shape, dtype=bool)

    if altitude_deg <= 0:
        return (
            np.zeros(shape, dtype=bool),
            np.zeros((N_VEGETATION_CLASSES,) + shape, dtype=np.float32),
            ~nodata_mask,
        )

    theta = azimuth_deg + 90.0
    below_everything = np.float32(-1.0e6)

    z_rot = _rotate(canopy.render_surface, theta, cval=below_everything)
    opaque_rot = _rotate(canopy.opaque, theta, cval=below_everything)
    ctop_rot = _rotate(canopy.crown_top, theta, cval=0.0)
    cbase_rot = _rotate(canopy.crown_base, theta, cval=0.0)
    cidx_rot = _rotate(canopy.class_index, theta, cval=-1)
    nodata_rot = _rotate(nodata_mask.astype(np.uint8), theta, cval=1) > 0

    altitude = np.deg2rad(altitude_deg)
    dz_per_px = np.float32(np.tan(altitude) * pixel_size)
    ds = np.float32(pixel_size / max(np.cos(altitude), 1e-3))

    inside = ~nodata_mask
    zmax = np.float32(
        max(
            float(canopy.opaque[inside].max()) if inside.any() else 0.0,
            float(canopy.crown_top.max()),
        )
    )

    lit_rot, path_rot = _canopy_path_scan(
        z_rot, opaque_rot, ctop_rot, cbase_rot, cidx_rot, nodata_rot,
        dz_per_px, ds, zmax, int(max_steps),
    )

    lit = _rotate(lit_rot, -theta, cval=1) > 0
    path = np.stack(
        [_rotate(path_rot[k], -theta, order=1, cval=0.0) for k in range(path_rot.shape[0])]
    )
    np.clip(path, 0.0, None, out=path)

    valid = _rotate(_rotate(inside.astype(np.uint8), theta, cval=0), -theta, cval=0) > 0
    return lit, path.astype(np.float32), valid


def _apply_dsm_shadow_to_hillshade(
    hillshade_path: str,
    dsm_path: str,
    out_path: str,
    azimuth: float,
    altitude: float,
    *,
    shadow_value: float = 1.0,
    canopy: Optional[CanopyModel] = None,
):
    """
    Add cast shadows to a local hillshade.

    With ``canopy=None`` this is the original opaque model: one band, every DSM
    pixel blocks the beam. With a ``CanopyModel`` the output gains three more
    bands holding the metres of canopy the beam crossed, per LiDAR height class,
    so the season can be applied later without recomputing the geometry.
    """
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

        if canopy is None:
            path_lengths = None
            lit, valid = _cast_shadow_mask(
                dsm,
                pixel_size,
                azimuth,
                altitude,
                nodata_mask=nodata_mask,
            )
        else:
            if canopy.opaque.shape != dsm.shape:
                raise ValueError(
                    f"canopy model shape {canopy.opaque.shape} does not match the "
                    f"DSM grid {dsm.shape}; build it from this DSM"
                )
            lit, path_lengths, valid = cast_canopy_path_lengths(
                canopy,
                pixel_size,
                azimuth,
                altitude,
                nodata_mask=nodata_mask,
            )

        # Apply shadows
        hs[~lit & valid] = shadow_value
        hs[~valid] = hsrc.nodata

        profile = hsrc.profile.copy()

        if path_lengths is None:
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(hs.astype(profile["dtype"]), 1)
                dst.update_tags(**hsrc.tags())
        else:
            # path lengths are metres, so the uint8 hillshade profile will not do
            profile.update(dtype="float32", count=1 + N_VEGETATION_CLASSES)
            path_lengths = path_lengths.astype(np.float32)
            path_lengths[:, ~valid] = 0.0
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(hs.astype(np.float32), 1)
                for band, name in enumerate(CANOPY_BAND_NAMES, start=2):
                    dst.write(path_lengths[band - 2], band)
                    dst.set_band_description(band, name)
                dst.set_band_description(1, "hillshade_opaque")
                dst.update_tags(**hsrc.tags())
                dst.update_tags(**CANOPY_RASTER_TAGS)

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
                if "GREENSHADOW_CANOPY" in hs.tags():
                    raise ValueError(
                        f"{hf.name} carries canopy path-length bands, which this "
                        f"per-pixel height test would silently drop. Regenerate the "
                        f"hillshades with canopy=None to use height shadows."
                    )
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
    canopy: Optional[CanopyModel] = None,
    cuda: bool | None = None,
) -> list[str]:
    """
    Generate hillshades normalized to [0,1], AOI-masked.

    Args:
        cuda: Run on the GPU. ``None`` uses it when CUDA (driver + libNVVM) is
            usable. The GPU path keeps the rasters on the device for every sun
            position and marches each ray straight towards the sun on the DSM
            grid, instead of rotating the grid, so it differs from the CPU path
            by a few nearest-neighbour pixels along shadow edges. ``max_workers``
            then sets the number of writer threads.
        canopy: Optional semi-transparent vegetation model from
            ``build_canopy_model_from_rasters``. When given, trees stop being
            opaque walls: each output gains three bands holding the metres of
            canopy the beam crossed per LiDAR height class, and the local shading
            is computed on the crown envelope so treetops read as domes. The
            season is applied later by ``apply_canopy_transmittance``, so one
            hillshade still serves every month. Leave it ``None`` for the
            original single-band opaque behaviour.
    """

    os.makedirs(raster_dir, exist_ok=True)

    valid_positions = []
    for i, (az, alt) in enumerate(azimuth_altitude):
        if alt > 0:
            when = related_datetimes[i] if related_datetimes else None
            valid_positions.append((i, az, alt, when))

    if not valid_positions:
        return []

    if cuda is None:
        cuda = has_cuda()
    elif cuda and not has_cuda():
        warning(
            "CUDA requested but not usable (no GPU, driver or libNVVM); "
            "falling back to the CPU hillshades."
        )
        cuda = False

    if cuda:
        return _generate_hillshades_cuda(
            raster_dir,
            dsm_path,
            valid_positions,
            zFactor=zFactor,
            compute_edges=compute_edges,
            creation_options=creation_options,
            qgis_temporal=qgis_temporal,
            max_workers=max_workers,
            overwrite=overwrite,
            aoi=aoi,
            aoi_crs=aoi_crs,
            nodata_value=nodata_value,
            canopy=canopy,
        )

    # The local hillshade wants the crown envelope, not the raw LiDAR canopy top.
    shading_dsm_path = dsm_path
    render_tmp = None
    if canopy is not None:
        with rasterio.open(dsm_path) as src:
            render_profile = src.profile.copy()
        if canopy.render_surface.shape != (render_profile["height"], render_profile["width"]):
            raise ValueError(
                "canopy model does not match the DSM grid; build it from this DSM"
            )
        render_profile.update(dtype="float32", count=1, compress="deflate", predictor=3)
        render_tmp = os.path.join(raster_dir, "_canopy_render_surface.tif")
        with rasterio.open(render_tmp, "w", **render_profile) as dst:
            dst.write(canopy.render_surface.astype(np.float32), 1)
        shading_dsm_path = render_tmp

    max_workers = min(max_workers, len(valid_positions), mp.cpu_count())

    def process_single(args):
        i, az, alt, when = args

        base = f"{int(az):03}_{int(alt):02}"
        if when and qgis_temporal:
            base += "_" + (when.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(when, "strftime") else str(when))
        out_path = os.path.join(raster_dir, f"{base}.tif")

        if not overwrite and geotiff_is_complete(out_path):
            return out_path

        try:
            hs_raw = _generate_hillshade_gdal(
                results_dir=raster_dir,
                when=when,
                dsm_path=shading_dsm_path,
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
                canopy=canopy,
            )

            with rasterio.open(final_path) as src:
                hs = src.read(1).astype(np.float32)
                # metres of canopy crossed, left in metres - only band 1 is a ratio
                path_lengths = (
                    src.read(list(range(2, src.count + 1))).astype(np.float32)
                    if src.count > 1 else None
                )
                profile = src.profile.copy()
                raster_crs = src.crs
                tags = src.tags()

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
                if path_lengths is not None:
                    path_lengths[:, ~aoi_mask] = nodata_value

            profile.update(
                dtype="float32",
                nodata=nodata_value,
                **HILLSHADE_COMPRESSION,
            )

            tmp_path = out_path + ".part"
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(hs, 1)
                if path_lengths is not None:
                    dst.set_band_description(1, "hillshade_opaque")
                    for band, name in enumerate(CANOPY_BAND_NAMES, start=2):
                        dst.write(path_lengths[band - 2], band)
                        dst.set_band_description(band, name)
                    dst.update_tags(**tags)
            _finish_geotiff(tmp_path, out_path)

            if hs_raw != final_path and os.path.exists(hs_raw):
                os.remove(hs_raw)

            return out_path

        except Exception as e:
            warning(f"Error az={az} alt={alt}: {e}")
            return None

    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for r in tqdm(ex.map(process_single, valid_positions),
                          total=len(valid_positions),
                          desc="Generating hillshades"):
                if r:
                    results.append(r)
    finally:
        if render_tmp and os.path.exists(render_tmp):
            os.remove(render_tmp)

    return results


# ======================================================================================
# CUDA HILLSHADES
#
# The same rasters as the CPU path above, without its two costs. The CPU path rotates
# the grid so the sun comes from the left, scans the rows and rotates back: twelve
# ``ndi.rotate`` calls per sun position with the canopy. And it reads the DSM, fills
# its holes, writes the raw GDAL hillshade and reads it back for every position.
# Here the DSM, canopy and AOI go to the device once for the whole run, the Horn
# hillshade is computed in place, and each thread marches its own ray straight towards
# the sun on the DSM grid, so nothing is rotated and nothing is lost in the corners
# that a same-shape rotation cuts off.
# ======================================================================================

# Rows per kernel launch. Keeps every launch short on a GPU that also drives a
# display, where one long kernel would trip the driver watchdog.
_HILLSHADE_ROWS_PER_LAUNCH = 256
_HILLSHADE_THREADS = (32, 8)


@cuda.jit(cache=True)
def _horn_hillshade_kernel(
    z, nodata, out, r0,
    inv_ewres, inv_nsres, sin_alt_254, cos_az_254, sin_az_254, square_z,
    compute_edges,
):
    """
    GDAL's Horn hillshade, as ``gdaldem hillshade`` writes it: 0 = nodata, 1..255.
    In float64 on purpose: it is cheap next to the ray-march and keeps the rounding
    identical to GDAL's.
    """
    j, i = cuda.grid(2)
    i += r0
    rows, cols = z.shape
    if i >= rows or j >= cols:
        return

    out[i, j] = 0.0
    if nodata[i, j]:
        return

    zc = z[i, j]
    w = cuda.local.array(9, float64)
    k = 0
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ii = i + di
            jj = j + dj
            if ii < 0 or ii >= rows or jj < 0 or jj >= cols or nodata[ii, jj]:
                # GDAL leaves the pixel nodata, or with -compute_edges
                # stands the centre in for the missing neighbour
                if not compute_edges:
                    return
                w[k] = zc
            else:
                w[k] = z[ii, jj]
            k += 1

    x = ((w[0] + w[3] + w[3] + w[6]) - (w[2] + w[5] + w[5] + w[8])) * inv_ewres
    y = ((w[6] + w[7] + w[7] + w[8]) - (w[0] + w[1] + w[1] + w[2])) * inv_nsres
    cang = (sin_alt_254 - (y * cos_az_254 - x * sin_az_254)) / math.sqrt(
        1.0 + square_z * (x * x + y * y)
    )
    out[i, j] = math.floor((1.0 if cang <= 0.0 else 1.0 + cang) + 0.5)


@cuda.jit(cache=True)
def _opaque_shadow_kernel(
    z, nodata, aoi, hs, r0,
    dx, dy, dz, zmax, max_steps, shadow, c255, nodata_value,
):
    """
    GPU twin of ``_shadow_scan``: shadowed when any DSM pixel along the ray rises
    above it. Turns the raw 0..255 hillshade in ``hs`` into the final 0..1 band.
    """
    j, i = cuda.grid(2)
    i += r0
    rows, cols = z.shape
    if i >= rows or j >= cols:
        return

    if not aoi[i, j]:
        hs[i, j] = nodata_value
        return
    if nodata[i, j]:
        hs[i, j] = 0.0
        return

    fi = float32(i)
    fj = float32(j)
    last_row = float32(rows - 1)
    last_col = float32(cols - 1)
    z0 = z[i, j]
    lit = True
    for s in range(1, max_steps + 1):
        # from the start every step, so float32 rounding does not drift the ray
        fs = float32(s)
        ri = fi + fs * dy
        rj = fj + fs * dx
        if ri < 0 or ri > last_row or rj < 0 or rj > last_col:
            break
        ray = z0 + fs * dz
        if ray > zmax:
            break
        ii = int(ri + float32(0.5))
        jj = int(rj + float32(0.5))
        if nodata[ii, jj]:
            continue
        if z[ii, jj] > ray:
            lit = False
            break

    if lit:
        hs[i, j] = min(max(hs[i, j] / c255, float32(0.0)), float32(1.0))
    else:
        hs[i, j] = shadow


@cuda.jit(cache=True)
def _canopy_shadow_kernel(
    render, opaque, ctop, cbase, cidx, nodata, aoi, hs, path, r0,
    dx, dy, dz, ds, zmax, max_steps, shadow, c255, nodata_value,
):
    """
    GPU twin of ``_canopy_path_scan``: metres of canopy crossed per height class,
    stopping dead at the first opaque hit.
    """
    j, i = cuda.grid(2)
    i += r0
    rows, cols = render.shape
    if i >= rows or j >= cols:
        return

    if not aoi[i, j]:
        hs[i, j] = nodata_value
        path[0, i, j] = nodata_value
        path[1, i, j] = nodata_value
        path[2, i, j] = nodata_value
        return
    if nodata[i, j]:
        hs[i, j] = 0.0
        path[0, i, j] = 0.0
        path[1, i, j] = 0.0
        path[2, i, j] = 0.0
        return

    fi = float32(i)
    fj = float32(j)
    last_row = float32(rows - 1)
    last_col = float32(cols - 1)
    z0 = render[i, j]
    lit = True
    p0 = float32(0.0)
    p1 = float32(0.0)
    p2 = float32(0.0)
    for s in range(1, max_steps + 1):
        fs = float32(s)
        ri = fi + fs * dy
        rj = fj + fs * dx
        if ri < 0 or ri > last_row or rj < 0 or rj > last_col:
            break
        ray = z0 + fs * dz
        if ray > zmax:
            break
        ii = int(ri + float32(0.5))
        jj = int(rj + float32(0.5))
        if nodata[ii, jj]:
            continue
        if opaque[ii, jj] >= ray:
            # fully dark already: the path lengths would never be used
            lit = False
            p0 = float32(0.0)
            p1 = float32(0.0)
            p2 = float32(0.0)
            break
        ci = cidx[ii, jj]
        if ci >= 0 and cbase[ii, jj] <= ray and ray <= ctop[ii, jj]:
            if ci == 0:
                p0 += ds
            elif ci == 1:
                p1 += ds
            else:
                p2 += ds

    path[0, i, j] = p0
    path[1, i, j] = p1
    path[2, i, j] = p2
    if lit:
        hs[i, j] = min(max(hs[i, j] / c255, float32(0.0)), float32(1.0))
    else:
        hs[i, j] = shadow


def _write_hillshade(out_path, hs, path_lengths, profile):
    # via a temporary name, so an interrupted or failed write never leaves a
    # truncated file that the next run would skip as already done
    tmp_path = out_path + ".part"
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(hs, 1)
        if path_lengths is not None:
            dst.set_band_description(1, "hillshade_opaque")
            for band, name in enumerate(CANOPY_BAND_NAMES, start=2):
                dst.write(path_lengths[band - 2], band)
                dst.set_band_description(band, name)
            dst.update_tags(**CANOPY_RASTER_TAGS)
    _finish_geotiff(tmp_path, out_path)
    return out_path


def _generate_hillshades_cuda(
    raster_dir: str,
    dsm_path: str,
    positions: list,
    *,
    zFactor: float,
    compute_edges: bool,
    creation_options: Optional[Dict[str, str]],
    qgis_temporal: bool,
    max_workers: int,
    overwrite: bool,
    aoi,
    aoi_crs: str,
    nodata_value: float,
    canopy: Optional[CanopyModel],
    shadow_value: float = 1.0,
    canopy_max_steps: int = 20000,
) -> list[str]:
    """
    CUDA path of ``generate_multiple_hillshades_gdal``, same file names and bands.
    """
    results = []
    jobs = []
    queued = set()
    for _, az, alt, when in positions:
        base = f"{int(az):03}_{int(alt):02}"
        if when and qgis_temporal:
            base += "_" + (when.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(when, "strftime") else str(when))
        out_path = os.path.join(raster_dir, f"{base}.tif")
        results.append(out_path)
        # positions that round to one file name are computed once
        if out_path in queued or (not overwrite and geotiff_is_complete(out_path)):
            continue
        queued.add(out_path)
        jobs.append((out_path, float(az), float(alt)))

    if not jobs:
        return results

    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        dsm_nodata = src.nodata
        geotransform = src.transform
        raster_crs = src.crs

    shape = dsm.shape
    rows, cols = shape
    pixel_size = abs(geotransform.a)

    nodata_mask = ~np.isfinite(dsm)
    if dsm_nodata is not None:
        nodata_mask |= dsm == dsm_nodata
    inside = ~nodata_mask

    # The local hillshade wants the crown envelope, not the raw LiDAR canopy top.
    if canopy is not None:
        if canopy.opaque.shape != shape:
            raise ValueError(
                f"canopy model shape {canopy.opaque.shape} does not match the "
                f"DSM grid {shape}; build it from this DSM"
            )
        shade = canopy.render_surface.astype(np.float32)
        shade_nodata = ~np.isfinite(shade)
        if dsm_nodata is not None:
            shade_nodata |= shade == dsm_nodata
        zmax = max(
            float(canopy.opaque[inside].max()) if inside.any() else 0.0,
            float(canopy.crown_top.max()),
        )
    else:
        shade = dsm
        shade_nodata = nodata_mask
        zmax = float(dsm[inside].max()) if inside.any() else 0.0

    if aoi is not None:
        transformer = pyproj.Transformer.from_crs(aoi_crs, raster_crs, always_xy=True).transform
        aoi_mask = rasterize_aoi_mask(transform(transformer, aoi), dsm_path)
    else:
        aoi_mask = np.ones(shape, dtype=bool)

    profile = {
        "driver": "GTiff",
        "width": cols,
        "height": rows,
        "crs": raster_crs,
        "transform": geotransform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "IF_SAFER",
        **{k.lower(): v for k, v in (creation_options or {}).items()},
    }
    profile.update(
        dtype="float32",
        count=1 + (N_VEGETATION_CLASSES if canopy is not None else 0),
        nodata=nodata_value,
        num_threads="ALL_CPUS",
        **HILLSHADE_COMPRESSION,
    )

    info(
        f"Hillshades on GPU ({cuda.get_current_device().name}): "
        f"{len(jobs)} sun positions, {cols} × {rows} px"
    )

    d_shade = cuda.to_device(np.ascontiguousarray(shade, dtype=np.float32))
    d_shade_nodata = cuda.to_device(shade_nodata.astype(np.uint8))
    d_nodata = cuda.to_device(nodata_mask.astype(np.uint8))
    d_aoi = cuda.to_device(aoi_mask.astype(np.uint8))
    d_hs = cuda.device_array(shape, dtype=np.float32)
    if canopy is not None:
        d_opaque = cuda.to_device(canopy.opaque.astype(np.float32))
        d_ctop = cuda.to_device(canopy.crown_top.astype(np.float32))
        d_cbase = cuda.to_device(canopy.crown_base.astype(np.float32))
        d_cidx = cuda.to_device(canopy.class_index.astype(np.int8))
        d_path = cuda.device_array((N_VEGETATION_CLASSES,) + shape, dtype=np.float32)
    del shade, shade_nodata, nodata_mask, aoi_mask

    inv_ewres = 1.0 / geotransform.a
    inv_nsres = 1.0 / geotransform.e
    z_scaled = zFactor / 8.0  # Horn's 8 * scale, with gdaldem's scale = 1
    opaque_max_steps = rows + cols
    band_rows = _HILLSHADE_ROWS_PER_LAUNCH
    grid_x = (cols + _HILLSHADE_THREADS[0] - 1) // _HILLSHADE_THREADS[0]

    def launches():
        for r0 in range(0, rows, band_rows):
            n = min(band_rows, rows - r0)
            yield (grid_x, (n + _HILLSHADE_THREADS[1] - 1) // _HILLSHADE_THREADS[1]), r0

    # Writing the deflated GeoTIFF is now the slow part, so it runs in the
    # background while the GPU moves on; bounded so the host keeps at most
    # max_workers finished rasters waiting in memory.
    n_writers = max(1, max_workers)
    pending = set()
    with ThreadPoolExecutor(max_workers=n_writers) as writer:
        for out_path, az, alt in tqdm(jobs, desc="Generating hillshades (CUDA)"):
            az_rad = math.radians(az)
            alt_rad = math.radians(alt)
            cos_alt_z_254 = 254.0 * math.cos(alt_rad) * z_scaled
            for grid, r0 in launches():
                _horn_hillshade_kernel[grid, _HILLSHADE_THREADS](
                    d_shade, d_shade_nodata, d_hs, r0,
                    inv_ewres, inv_nsres,
                    254.0 * math.sin(alt_rad),
                    math.cos(az_rad) * cos_alt_z_254,
                    math.sin(az_rad) * cos_alt_z_254,
                    z_scaled * z_scaled,
                    bool(compute_edges),
                )

            # unit step towards the sun, in (row, col); rows grow southwards
            dx = np.float32(math.sin(az_rad))
            dy = np.float32(-math.cos(az_rad))
            dz = np.float32(math.tan(alt_rad) * pixel_size)
            common = (
                np.float32(shadow_value / 255.0),
                np.float32(255.0),
                np.float32(nodata_value),
            )
            for grid, r0 in launches():
                if canopy is None:
                    _opaque_shadow_kernel[grid, _HILLSHADE_THREADS](
                        d_shade, d_nodata, d_aoi, d_hs, r0,
                        dx, dy, dz, np.float32(zmax), opaque_max_steps, *common,
                    )
                else:
                    _canopy_shadow_kernel[grid, _HILLSHADE_THREADS](
                        d_shade, d_opaque, d_ctop, d_cbase, d_cidx, d_nodata,
                        d_aoi, d_hs, d_path, r0,
                        dx, dy, dz,
                        np.float32(pixel_size / max(math.cos(alt_rad), 1e-3)),
                        np.float32(zmax), int(canopy_max_steps), *common,
                    )

            hs = d_hs.copy_to_host()
            path_lengths = d_path.copy_to_host() if canopy is not None else None

            if len(pending) >= n_writers:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    f.result()
            pending.add(writer.submit(_write_hillshade, out_path, hs, path_lengths, profile))

        for f in concurrent.futures.as_completed(pending):
            f.result()

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


#: Weather columns that describe the record rather than the weather, and are
#: therefore not carried onto the solar table.
_WEATHER_METADATA_COLUMNS = (
    "time", "latitude", "longitude", "weatherStation", "sunPosition",
    "sunAzimuthDiscrete", "sunElevationDiscrete", "sunElevation", "sunAzimuth",
)

#: How GHI is split into its direct and diffuse parts.
IRRADIANCE_DECOMPOSITIONS = ("era5", "dirint", "disc", "erbs", "weather")


def _attach_weather(
    result_df: pd.DataFrame,
    weather_df: Any,
    *,
    lat: float,
    lon: float,
    decomposition: str = "disc",
    time_offset_minutes: float = -30.0,
) -> pd.DataFrame:
    """
    Put the weather on the solar table, with GHI split at the table's own sun.

    ERA5-Land's ``ssrd`` is all-sky: clouds are already in it, and in the GHI
    the weather frame carries. What is not in it is the split into direct and
    diffuse, which is what decides how much a shadow costs and how much a
    tilted panel gains. That split is redone here, at the table's exact sun
    position, rather than taken from the frame:

    * ``time_offset_minutes`` moves each weather record to the middle of the
      period it describes. An ERA5 value stamped T is the mean over the hour
      before T, so -30 is right for ERA5 and ERA5-Land; leaving the stamp where
      it is pins every hour's sunshine half an hour late, which tilts the whole
      year towards the afternoon.
    * ``decomposition`` picks the model; ``docs/irradiance_era5.md`` compares
      them all against PVGIS-SARAH3, which is how ``"disc"`` became the
      default: closest on the annual DNI and on tilted planes. ``"dirint"`` (Perez et al. 1992) reads
      cloudiness from the clearness index **and** from how much it changes hour
      to hour, which is what tells broken cloud from thin haze; ``"disc"`` uses
      the clearness index alone; ``"erbs"`` is the classic diffuse-fraction
      fit; ``"weather"`` keeps whatever DNI and DHI the frame already has.
      ``"era5"`` does not model the split at all: it takes the beam fraction
      ERA5's own clouds produced - ``directHorizontal`` over ``GHI_ERA5``,
      both joined by ``weather.load_ERA5Land_with_direct_radiation`` - and
      applies it to the frame's GHI. Without those columns it falls back to
      ``"dirint"`` with a warning.

    Adds or replaces the weather columns in place - ``GHI``, ``DNI``, ``DHI``
    in W/m2 and ``clearness_index`` - and returns the frame.
    """
    import pvlib

    if decomposition not in IRRADIANCE_DECOMPOSITIONS:
        raise ValueError(f"decomposition must be one of {IRRADIANCE_DECOMPOSITIONS}")

    weather = weather_df.to_pandas() if hasattr(weather_df, "to_pandas") else weather_df.copy()

    stamps = pd.to_datetime(weather["time"])
    stamps = stamps.dt.tz_localize("UTC") if stamps.dt.tz is None else stamps.dt.tz_convert("UTC")
    weather.index = pd.DatetimeIndex(stamps) + pd.Timedelta(minutes=time_offset_minutes)
    weather = weather.sort_index()
    weather = weather[~weather.index.duplicated()]

    columns = [
        column for column in weather.columns
        if column not in _WEATHER_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(weather[column])
    ]

    target = pd.DatetimeIndex(pd.to_datetime(result_df["datetime_utc"], utc=True))
    union = weather.index.union(target)
    values = (
        weather[columns].reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(target)
    )

    for column in columns:
        result_df[column] = values[column].to_numpy()

    if "GHI" not in values:
        return result_df

    ghi = values["GHI"].clip(lower=0.0).fillna(0.0)

    position = pvlib.solarposition.get_solarposition(target, lat, lon)
    zenith = position["apparent_zenith"]
    extra = pvlib.irradiance.get_extra_radiation(target)

    result_df["clearness_index"] = pvlib.irradiance.clearness_index(
        ghi, zenith, extra).to_numpy()

    if decomposition == "weather":
        return result_df

    pressure = values["surfacePressure"] if "surfacePressure" in values else 101325.0

    if decomposition == "era5" and not {"directHorizontal", "GHI_ERA5"} <= set(values):
        warnings.warn("irradiance_decomposition='era5' needs the directHorizontal and "
                      "GHI_ERA5 columns of load_ERA5Land_with_direct_radiation; "
                      "falling back to 'dirint'")
        decomposition = "dirint"

    if decomposition == "era5":
        # ERA5's beam share of its own global, applied to the frame's GHI: the
        # finer ERA5-Land global keeps its say on how much light there was,
        # and ERA5's clouds decide how much of it came straight from the sun.
        global_era5 = values["GHI_ERA5"].to_numpy(dtype=float)
        direct_era5 = values["directHorizontal"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(global_era5 > 0.0, direct_era5 / global_era5, 0.0)
        fraction = np.clip(np.nan_to_num(fraction, nan=0.0), 0.0, 1.0)
        cos_zenith = np.cos(np.radians(zenith.to_numpy()))
        beam = fraction * ghi.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            dni = np.where(cos_zenith > 1e-3, beam / np.maximum(cos_zenith, 1e-3), 0.0)
        # An hourly mean over a sunrise hour, divided by the cosine at one
        # instant of it, can exceed what reaches the top of the atmosphere;
        # nothing at the surface does.
        dni = np.minimum(dni, extra.to_numpy())
    elif decomposition == "dirint":
        dew = values["dewAirTemperature"] if "dewAirTemperature" in values else None
        dni = pvlib.irradiance.dirint(
            ghi, zenith, target, pressure=pressure, temp_dew=dew)
    elif decomposition == "disc":
        dni = pvlib.irradiance.disc(ghi, zenith, target, pressure=pressure)["dni"]
    else:
        dni = pvlib.irradiance.erbs(ghi, zenith, target)["dni"]

    if decomposition == "era5" and "totalCloudCover" in values:
        result_df["totalCloudCover"] = values["totalCloudCover"].to_numpy()

    cos_zenith = np.cos(np.radians(zenith.to_numpy()))
    dni = np.nan_to_num(np.asarray(dni, dtype=float), nan=0.0)
    dni = np.where(cos_zenith > 0.0, np.clip(dni, 0.0, None), 0.0)

    # The beam can never deliver more onto the horizontal than the global
    # does: where a model overshoots - near the horizon, mostly - the beam is
    # cut back and the diffuse is what is left, never negative.
    beam = np.minimum(dni * np.clip(cos_zenith, 0.0, None), ghi.to_numpy())
    dni = np.where(cos_zenith > 1e-3, beam / np.maximum(cos_zenith, 1e-3), 0.0)

    result_df["GHI"] = ghi.to_numpy()
    result_df["DNI"] = dni
    result_df["DHI"] = ghi.to_numpy() - beam

    return result_df


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
    irradiance_decomposition: str = "disc",
    weather_time_offset_minutes: float = -30.0,
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
        irradiance_decomposition: How GHI is split into DNI and DHI at each
            sun position; see ``_attach_weather``.
        weather_time_offset_minutes: Shift from a weather record's stamp to
            the middle of the period it averages; -30 for ERA5(-Land).
    
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
        _attach_weather(
            result_df, weather_df,
            lat=lat, lon=lon,
            decomposition=irradiance_decomposition,
            time_offset_minutes=weather_time_offset_minutes,
        )
    
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
    irradiance_decomposition: str = "disc",
    weather_time_offset_minutes: float = -30.0,
) -> Dict[str, Any]:
    """
    Generate azimuth/altitude pairs for solar positions across a full year.

    Includes optional weather integration (same semantics as daily function,
    including ``irradiance_decomposition`` and ``weather_time_offset_minutes``).
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
    # WEATHER INTEGRATION (SAME AS THE DAILY FUNCTION)
    # ==================================================
    if weather_df is not None:
        _attach_weather(
            result_df, weather_df,
            lat=lat, lon=lon,
            decomposition=irradiance_decomposition,
            time_offset_minutes=weather_time_offset_minutes,
        )

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

    with rasterio.open(dsm_path) as dsm_src:
        dsm = dsm_src.read(1).astype(np.float32)
        raster_crs = dsm_src.crs
        dsm_transform = dsm_src.transform
        if dsm_src.nodata is not None:
            dsm[dsm == dsm_src.nodata] = np.nan

    # The DEM may cover a different extent; resample it onto the DSM grid
    # (which is also the hillshade grid). Its nodata becomes NaN.
    dem = np.full(dsm.shape, np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as dem_src:
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            src_nodata=dem_src.nodata,
            dst_transform=dsm_transform,
            dst_crs=raster_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

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

            if not overwrite and geotiff_is_complete(out_path):
                return str(out_path)

            with rasterio.open(path) as src:
                hs = src.read(1).astype(np.float32)
                profile = src.profile.copy()

            if hs.shape != obstacle_mask.shape:
                failed.append(f"{path.name}: shape {hs.shape}, DSM {obstacle_mask.shape}")
                return None

            hs[obstacle_mask] = 0.0
            hs[~aoi_mask] = nodata_value
            np.clip(hs, 0.0, 1.0, out=hs, where=aoi_mask)

            profile.update(
                dtype="float32",
                nodata=nodata_value,
                **HILLSHADE_COMPRESSION,
            )

            tmp_path = f"{out_path}.part"
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(hs, 1)
            _finish_geotiff(tmp_path, out_path)

            return str(out_path)

        except Exception as e:
            failed.append(f"{path.name}: {e}")
            return None

    failed = []

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            res = list(tqdm(ex.map(process, files), total=len(files)))
    else:
        res = [process(f) for f in tqdm(files)]

    # A missing elevated hillshade would only surface later, in whatever
    # reads the set; say which ones and why here instead.
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(files)} elevated-shadow hillshades failed:\n  "
            + "\n  ".join(sorted(failed))
        )

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


def _resolve_hillshade_path(hillshade_dir: Path, azimuth: float, altitude: float) -> Path:
    """Find the hillshade file for one (azimuth, altitude) bin."""
    az_i = int(azimuth)
    alt_i = int(altitude)
    prefix = f"{az_i:03d}_{alt_i:02d}"

    candidates = sorted(
        p for p in Path(hillshade_dir).glob(f"{prefix}*.tif")
        if not p.name.endswith("_raw.tif")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No valid hillshade for az={az_i}, alt={alt_i} in {hillshade_dir}"
        )

    fn = candidates[0]
    if fn.stat().st_size == 0:
        raise ValueError(f"Corrupt hillshade (0 bytes): {fn}")

    return fn


def _read_hillshade_bands(path: Path):
    """
    Read one hillshade file.

    Returns ``(base, path_lengths)`` where ``base`` is the [0, 1] hillshade with
    opaque shadows applied and nodata zeroed, and ``path_lengths`` is the
    (3, H, W) canopy path length in metres, or None for a plain hillshade.
    """
    # Deflate decoding is the slow part of a hillshade, not the disk: on a
    # 10k x 11k canopy hillshade one thread takes ~10 s, all cores ~0.6 s.
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(path) as ds:
        hs = ds.read(1).astype(np.float32, copy=False)
        nodata = ds.nodata
        has_canopy = (
            "GREENSHADOW_CANOPY" in ds.tags()
            and ds.count >= 1 + N_VEGETATION_CLASSES
        )
        path_lengths = (
            ds.read(list(range(2, 2 + N_VEGETATION_CLASSES))).astype(np.float32, copy=False)
            if has_canopy
            else None
        )

    if nodata is not None:
        hs[hs == nodata] = 0.0
        if path_lengths is not None:
            path_lengths[path_lengths == nodata] = 0.0

    return hs, path_lengths


def load_hillshade(
    hillshade_dir: Path,
    azimuth: float,
    altitude: float,
    *,
    month: Optional[int] = None,
    leaf_type: int = 0,
    reference_thickness_m: Optional[Sequence[float]] = None,
    ignore_canopy: bool = False,
) -> np.ndarray:
    """
    Load a hillshade generated by ``generate_multiple_hillshades_gdal``.

    A canopy hillshade carries three extra path-length bands. They only become a
    beam fraction once a month picks the leaf-on / leaf-off state, so reading one
    without saying which month is an error rather than a silent loss of all
    vegetation shading.

    Args:
        hillshade_dir: Directory holding the hillshades.
        azimuth: Solar azimuth in degrees.
        altitude: Solar elevation in degrees.
        month: 1-12. Required for canopy hillshades, ignored for plain ones.
        leaf_type: 0 deciduous, 1 evergreen.
        reference_thickness_m: Passed through to ``vegetation_extinction``.
        ignore_canopy: Read band 1 only from a canopy hillshade, i.e. opaque
            obstacles but fully transparent vegetation. Deliberate escape hatch.

    Returns:
        float32 array with values clipped to [0, 1].
    """

    fn = _resolve_hillshade_path(hillshade_dir, azimuth, altitude)
    hs, path_lengths = _read_hillshade_bands(fn)
    if ignore_canopy:
        path_lengths = None

    if path_lengths is not None:
        if month is None:
            raise ValueError(
                f"{fn.name} is a canopy hillshade: pass month=1..12 so the leaf-on "
                f"state can be applied, or ignore_canopy=True to read band 1 alone "
                f"(which leaves vegetation fully transparent)"
            )
        hs = apply_canopy_transmittance(
            hs, path_lengths, month,
            leaf_type=leaf_type,
            reference_thickness_m=reference_thickness_m,
        )

    return np.clip(hs, 0.0, 1.0).astype(np.float32)

SOLAR_ENERGY_MODES = (
    "total", "monthly", "fourier_decomposition", "fourier_poa",
    "vertical_basis", "tracking_basis",
)

# Bands each mode accumulates. Monthly keeps 12 direct and 12 diffuse; its
# two annual bands are their sums, added when the raster is written.
_ACCUMULATOR_BANDS = {
    "total": 2,
    "monthly": 24,
    "fourier_decomposition": 6,
    "fourier_poa": 5,
    "vertical_basis": 4,
    "tracking_basis": 6,
}


def _solar_energy_band_names(mode: str) -> tuple:
    if mode == "total":
        return ("Direct", "Diffuse")
    if mode == "monthly":
        months = calendar.month_name[1:]
        return (
            tuple(f"Direct - {m}" for m in months)
            + tuple(f"Diffuse - {m}" for m in months)
            + ("Direct - Annual", "Diffuse - Annual")
        )
    if mode == "fourier_decomposition":
        return ("A0 direct", "A1c cos(az)", "A1s sin(az)", "A2 sin(alt)",
                "D0 diffuse", "D1 sin(alt)")
    if mode == "fourier_poa":
        return ("B_z direct cos(tilt)", "B_east direct sin(tilt)sin(az)",
                "B_north direct sin(tilt)cos(az)", "D_sky diffuse horizontal",
                "GHI global horizontal")
    if mode == "vertical_basis":
        return VERTICAL_BASIS_BANDS
    return TRACKING_BASIS_BANDS


def _iter_hillshade_bands(files):
    """
    ``_read_hillshade_bands`` over ``files``, one file ahead in the background.

    GDAL fetches a tiled GeoTIFF tile by tile, which on these rasters gets a
    third of the disk's sequential speed (1.3 s against 0.45 s for a 350 MB
    canopy hillshade). So while one file is decoded the next is pulled into
    the page cache with a plain sequential read, and the caller sums the
    previous one meanwhile.
    """
    def warm(path):
        with open(path, "rb", buffering=0) as f:
            while f.read(64 << 20):
                pass

    def read(k):
        if k + 1 < len(files):
            warmer.submit(warm, files[k + 1])
        return _read_hillshade_bands(files[k])

    with ThreadPoolExecutor(max_workers=1) as warmer, \
            ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read, 0) if files else None
        for k in range(len(files)):
            bands = pending.result()
            if k + 1 < len(files):
                pending = reader.submit(read, k + 1)
            yield bands
            del bands


def _solar_energy_bins(df: pd.DataFrame) -> list:
    """
    The (azimuth, altitude) bins of a solar table and the energy each brings.

    Returns ``[(az, alt, whole, by_month)]``: ``whole`` is the bin's single
    ``(None, dni_E, dhi_E)``, ``by_month`` the same split by month, which is
    what a canopy hillshade (whose attenuation changes with the leaves) and
    the monthly mode need. Bins and months without energy are left out, so
    their hillshade is never read.
    """
    df = df.assign(
        dni_wh=df.DNI * df.dt_hours,
        dhi_wh=df.DHI * df.dt_hours,
        month=df.datetime_utc.dt.month,
    )
    bins = []
    for (az, alt), g in df.groupby(["az_bin", "alt_bin"]):
        dni_E, dhi_E = np.float32(g.dni_wh.sum()), np.float32(g.dhi_wh.sum())
        if dni_E == 0 and dhi_E == 0:
            continue
        by_month = [
            (int(month), np.float32(m.dni_wh.sum()), np.float32(m.dhi_wh.sum()))
            for month, m in g.groupby("month")
        ]
        by_month = [v for v in by_month if v[1] != 0 or v[2] != 0]
        bins.append((int(az), int(alt), [(None, dni_E, dhi_E)], by_month))
    return bins


def _single_axis_constants(theta: float, phi: float, max_rotation: float):
    """``(cos AOI, cos tilt)`` of a horizontal N-S tracker, as ``_accumulate_tracking_basis``."""
    s_e = math.cos(phi) * math.sin(theta)
    s_up = math.sin(phi)
    rotation = min(max(math.atan2(s_e, s_up), -max_rotation), max_rotation)
    cos_aoi = max(s_e * math.sin(rotation) + s_up * math.cos(rotation), 0.0)
    return cos_aoi, math.cos(rotation)


class _CpuSolarAccumulator:
    """The bands of every requested mode, accumulated with the Numba CPU kernels."""

    def __init__(self, modes, svf, valid, pixel_area, max_rotation):
        self.acc = {
            m: np.zeros((_ACCUMULATOR_BANDS[m],) + svf.shape, np.float32) for m in modes
        }
        self.svf = svf
        self.valid = valid
        self.pixel_area = pixel_area
        self.max_rotation = max_rotation
        self.beam = np.empty(svf.shape, np.float32)

    def load(self, base, path_lengths):
        self.base, self.path_lengths = base, path_lengths
        if path_lengths is None:
            np.clip(base, 0.0, 1.0, out=self.beam)

    def add(self, month, mu, dni_E, dhi_E, theta, phi):
        if mu is not None:
            _canopy_beam(self.base, self.path_lengths, mu, self.beam)
        hill, svf, valid = self.beam, self.svf, self.valid

        for mode, a in self.acc.items():
            if mode == "total":
                _accumulate_energy_split(a[0], a[1], svf, hill, dni_E, dhi_E,
                                         valid, self.pixel_area)
            elif mode == "monthly":
                _accumulate_energy_split(a[month - 1], a[month + 11], svf, hill,
                                         dni_E, dhi_E, valid, self.pixel_area)
            elif mode == "fourier_decomposition":
                _accumulate_fourier(*a, hill, svf, dni_E, dhi_E, theta, phi,
                                    valid, self.pixel_area)
            elif mode == "fourier_poa":
                _accumulate_poa_basis(*a, hill, svf, dni_E, dhi_E, theta, phi, valid)
            elif mode == "vertical_basis":
                _accumulate_vertical_basis(*a, hill, dni_E, theta, phi, valid)
            else:
                _accumulate_tracking_basis(*a, hill, svf, dni_E, dhi_E, theta, phi,
                                           self.max_rotation, valid)

    def result(self):
        return self.acc


@njit(parallel=True, fastmath=True, cache=True)
def _canopy_beam(base, path_lengths, mu, out):
    """``apply_canopy_transmittance`` clipped to [0, 1], in one pass."""
    H, W = base.shape
    for i in prange(H):
        for j in range(W):
            b = base[i, j] * np.exp(-(
                mu[0] * path_lengths[0, i, j]
                + mu[1] * path_lengths[1, i, j]
                + mu[2] * path_lengths[2, i, j]
            ))
            out[i, j] = min(max(b, 0.0), 1.0)


# ======================================================================================
# CUDA SOLAR ENERGY
#
# The same sums as the Numba accumulators. Each is a few multiply-adds per pixel, so
# on the GPU they cost next to nothing; the accumulators stay on the device for the
# whole year and each hillshade crosses the bus once, however many modes and months
# it feeds.
# ======================================================================================

_ENERGY_THREADS = (32, 8)


@cuda.jit(device=True, inline=True)
def _lit_fraction_gpu(hill, s_phi):
    if s_phi <= 1e-6:
        return 0.0
    return min(max(hill / s_phi, 0.0), 1.0)


@cuda.jit(cache=True)
def _canopy_beam_kernel(base, path_lengths, mu0, mu1, mu2, out):
    j, i = cuda.grid(2)
    if i >= out.shape[0] or j >= out.shape[1]:
        return
    b = base[i, j] * math.exp(-(
        mu0 * path_lengths[0, i, j]
        + mu1 * path_lengths[1, i, j]
        + mu2 * path_lengths[2, i, j]
    ))
    out[i, j] = min(max(b, 0.0), 1.0)


@cuda.jit(cache=True)
def _energy_split_kernel(acc, k_dir, k_dif, hill, svf, valid, dir_scale, dif_scale):
    j, i = cuda.grid(2)
    if i >= hill.shape[0] or j >= hill.shape[1] or not valid[i, j]:
        return
    acc[k_dir, i, j] += dir_scale * hill[i, j]
    acc[k_dif, i, j] += dif_scale * svf[i, j]


@cuda.jit(cache=True)
def _fourier_kernel(acc, hill, svf, valid, dir_scale, dif_scale, c_theta, s_theta, s_phi):
    j, i = cuda.grid(2)
    if i >= hill.shape[0] or j >= hill.shape[1] or not valid[i, j]:
        return
    e_dir = dir_scale * hill[i, j]
    e_dif = dif_scale * svf[i, j]
    acc[0, i, j] += e_dir
    acc[1, i, j] += e_dir * c_theta
    acc[2, i, j] += e_dir * s_theta
    acc[3, i, j] += e_dir * s_phi
    acc[4, i, j] += e_dif
    acc[5, i, j] += e_dif * s_phi


@cuda.jit(cache=True)
def _poa_basis_kernel(acc, hill, svf, valid, dni_E, dhi_E, s_phi, s_east, s_north):
    j, i = cuda.grid(2)
    if i >= hill.shape[0] or j >= hill.shape[1] or not valid[i, j]:
        return
    e_dir = dni_E * _lit_fraction_gpu(hill[i, j], s_phi)
    e_dif = dhi_E * svf[i, j]
    acc[0, i, j] += e_dir * s_phi
    acc[1, i, j] += e_dir * s_east
    acc[2, i, j] += e_dir * s_north
    acc[3, i, j] += e_dif
    acc[4, i, j] += e_dir * s_phi + e_dif


@cuda.jit(cache=True)
def _vertical_basis_kernel(acc, hill, valid, dni_E, s_phi, east, west, north, south):
    j, i = cuda.grid(2)
    if i >= hill.shape[0] or j >= hill.shape[1] or not valid[i, j]:
        return
    e_dir = dni_E * _lit_fraction_gpu(hill[i, j], s_phi)
    acc[0, i, j] += e_dir * east
    acc[1, i, j] += e_dir * west
    acc[2, i, j] += e_dir * north
    acc[3, i, j] += e_dir * south


@cuda.jit(cache=True)
def _tracking_basis_kernel(acc, hill, svf, valid, dni_E, dhi_E, s_phi, cos_aoi_1, cos_tilt_1):
    j, i = cuda.grid(2)
    if i >= hill.shape[0] or j >= hill.shape[1] or not valid[i, j]:
        return
    e_dir = dni_E * _lit_fraction_gpu(hill[i, j], s_phi)
    e_dif = dhi_E * svf[i, j]
    g = e_dir * s_phi + e_dif
    acc[0, i, j] += e_dir
    acc[1, i, j] += e_dif * (1.0 + s_phi) * 0.5
    acc[2, i, j] += g * (1.0 - s_phi) * 0.5
    acc[3, i, j] += e_dir * cos_aoi_1
    acc[4, i, j] += e_dif * (1.0 + cos_tilt_1) * 0.5
    acc[5, i, j] += g * (1.0 - cos_tilt_1) * 0.5


def _cuda_solar_bytes(modes, shape) -> int:
    """Device memory ``_CudaSolarAccumulator`` needs: accumulators, hillshade, beam, SVF, mask."""
    n = shape[0] * shape[1]
    planes = sum(_ACCUMULATOR_BANDS[m] for m in modes) + 1 + N_VEGETATION_CLASSES + 1 + 1
    return n * (4 * planes + 1)


class _CudaSolarAccumulator:
    """``_CpuSolarAccumulator`` on the GPU; same calls, same bands."""

    def __init__(self, modes, svf, valid, pixel_area, max_rotation):
        shape = svf.shape
        self.acc = {}
        for m in modes:
            d = cuda.device_array((_ACCUMULATOR_BANDS[m],) + shape, dtype=np.float32)
            cuda.driver.device_memset(d, 0, d.nbytes)
            self.acc[m] = d
        self.svf = cuda.to_device(np.ascontiguousarray(svf, dtype=np.float32))
        self.valid = cuda.to_device(valid.astype(np.uint8))
        self.base = cuda.device_array(shape, dtype=np.float32)
        self.beam = cuda.device_array(shape, dtype=np.float32)
        self.path_lengths = None
        self.pixel_area = pixel_area
        self.max_rotation = max_rotation
        self.grid = (
            (shape[1] + _ENERGY_THREADS[0] - 1) // _ENERGY_THREADS[0],
            (shape[0] + _ENERGY_THREADS[1] - 1) // _ENERGY_THREADS[1],
        )

    def load(self, base, path_lengths):
        if path_lengths is None:
            self.beam.copy_to_device(np.clip(base, 0.0, 1.0))
            return
        self.base.copy_to_device(base)
        if self.path_lengths is None:
            self.path_lengths = cuda.device_array(path_lengths.shape, dtype=np.float32)
        self.path_lengths.copy_to_device(np.ascontiguousarray(path_lengths))

    def add(self, month, mu, dni_E, dhi_E, theta, phi):
        launch = (self.grid, _ENERGY_THREADS)
        if mu is not None:
            _canopy_beam_kernel[launch](
                self.base, self.path_lengths,
                float(mu[0]), float(mu[1]), float(mu[2]), self.beam,
            )
        hill, svf, valid = self.beam, self.svf, self.valid
        dni_E, dhi_E = float(dni_E), float(dhi_E)
        s_phi, c_phi = math.sin(phi), math.cos(phi)
        s_theta, c_theta = math.sin(theta), math.cos(theta)

        for mode, a in self.acc.items():
            if mode in ("total", "monthly"):
                k = 0 if mode == "total" else month - 1
                _energy_split_kernel[launch](
                    a, k, k + (1 if mode == "total" else 12), hill, svf, valid,
                    dni_E * self.pixel_area, dhi_E * self.pixel_area,
                )
            elif mode == "fourier_decomposition":
                _fourier_kernel[launch](
                    a, hill, svf, valid,
                    dni_E * self.pixel_area, dhi_E * self.pixel_area,
                    c_theta, s_theta, s_phi,
                )
            elif mode == "fourier_poa":
                _poa_basis_kernel[launch](
                    a, hill, svf, valid, dni_E, dhi_E,
                    s_phi, c_phi * s_theta, c_phi * c_theta,
                )
            elif mode == "vertical_basis":
                s_e, s_n = c_phi * s_theta, c_phi * c_theta
                _vertical_basis_kernel[launch](
                    a, hill, valid, dni_E, s_phi,
                    max(s_e, 0.0), max(-s_e, 0.0), max(s_n, 0.0), max(-s_n, 0.0),
                )
            else:
                cos_aoi_1, cos_tilt_1 = _single_axis_constants(theta, phi, self.max_rotation)
                _tracking_basis_kernel[launch](
                    a, hill, svf, valid, dni_E, dhi_E, s_phi, cos_aoi_1, cos_tilt_1,
                )

    def result(self):
        cuda.synchronize()
        return {m: d.copy_to_host() for m, d in self.acc.items()}


def _write_solar_energy(path, mode, bands, profile, aoi_mask, nodata_value,
                        tracker_max_rotation_deg):
    if mode == "monthly":
        bands = np.concatenate(
            [bands, bands[:12].sum(axis=0)[None], bands[12:].sum(axis=0)[None]]
        )
    bands[:, ~aoi_mask] = nodata_value
    names = _solar_energy_band_names(mode)

    # via a temporary name, so an interrupted run never leaves a raster the
    # next run would take as finished
    tmp_path = f"{path}.part"
    with rasterio.open(tmp_path, "w", **dict(profile, count=len(bands))) as dst:
        for band, (arr, name) in enumerate(zip(bands, names), 1):
            dst.write(arr, band)
            dst.set_band_description(band, name)
        if mode == "tracking_basis":
            dst.update_tags(TRACKER_MAX_ROTATION_DEG=str(tracker_max_rotation_deg))
    _finish_geotiff(tmp_path, path)


def compute_surface_solar_energy(
    solar_df,
    *,
    hillshade_dir,
    svf_path,
    out_path,
    mode: str = "total",   # "total" | "monthly" | "fourier_decomposition" | ...
    aoi=None,
    aoi_crs: str = "EPSG:4326",
    nodata_value: float = -9999.0,
    overwrite: bool = False,
    canopy_leaf_type: Optional[int] = None,
    canopy_reference_thickness_m: Optional[Sequence[float]] = None,
    tracker_max_rotation_deg: float = 60.0,
    cuda: Optional[bool] = None,
) -> str:
    """
    Compute surface solar energy with different aggregation strategies.

    For more than one mode use compute_surface_solar_energies, which reads
    the hillshades once for all of them. cuda runs the sums on the GPU
    (None: when usable and the accumulators fit).

    ``mode="vertical_basis"`` and ``mode="tracking_basis"`` extend
    ``"fourier_poa"`` to the panels it cannot represent: vertical panels,
    whose beam has to be split by the side of the panel it falls on, and
    trackers, whose orientation changes with every sun position. They are
    read back with ``reconstruct_vertical_poa`` and
    ``reconstruct_tracking_poa``, and both write a ``*_sun.csv`` next to the
    raster: the sun-position bins and their beam and diffuse energy, from
    which a planner works out the shading between rows of a given layout.
    ``tracker_max_rotation_deg`` is the single-axis tracker's rotation limit.

    ``canopy_leaf_type`` (0 deciduous, 1 evergreen) is required when the
    hillshades were generated with a ``CanopyModel``: it selects the leaf-on /
    leaf-off column of ``VEGETATION_SHADOWING``, and every (azimuth, altitude)
    bin is then resolved per month so a single sun position attenuates correctly
    in both February and July. Leave it ``None`` for plain opaque hillshades.

    Returns the path actually written, which is *not* ``out_path``: the mode is
    appended to the stem (``foo.tif`` -> ``foo_total.tif``). Always use the
    return value rather than rebuilding the name at the call site.

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

    fourier_poa:
        Bands:
          1 : B_z    (direct, pairs with cos(tilt))
          2 : B_east (direct, pairs with sin(tilt)*sin(azimuth))
          3 : B_north(direct, pairs with sin(tilt)*cos(azimuth))
          4 : D_sky  (diffuse on the horizontal, SVF-weighted)
          5 : GHI    (global horizontal, for the ground-reflected term)

        Reconstruct the annual plane-of-array irradiation of a panel at tilt
        beta and azimuth gamma (0 = North, clockwise) with
        ``reconstruct_poa_from_basis``, or directly::

            POA = B_z*cos(beta)
                + sin(beta) * (B_north*cos(gamma) + B_east*sin(gamma))
                + D_sky * (1 + cos(beta)) / 2
                + GHI * albedo * (1 - cos(beta)) / 2

        This is exact for the direct component, because
        cos(AOI) = cos(beta)*sin(phi) + sin(beta)*cos(phi)*cos(theta - gamma)
        expands into precisely those three basis terms. At beta = 0 it collapses
        to B_z + D_sky == GHI, which is the cheapest self-check.

        Unlike every other mode, these bands are in **Wh/m2**, not Wh per pixel:
        a panel's energy is the band value times the module area.

        Limitation: the lit fraction is recovered as hill / sin(altitude), which
        assumes a near-horizontal surface. That is exactly where this basis is
        meant to be used - flat roofs carrying tilted racks. A module mounted
        coplanar with an inclined roof already has the roof's own orientation,
        so for those the ``total`` mode raster *is* the POA and should be used
        instead.
    """
    return compute_surface_solar_energies(
        solar_df,
        modes=(mode,),
        hillshade_dir=hillshade_dir,
        svf_path=svf_path,
        out_path=out_path,
        aoi=aoi,
        aoi_crs=aoi_crs,
        nodata_value=nodata_value,
        overwrite=overwrite,
        canopy_leaf_type=canopy_leaf_type,
        canopy_reference_thickness_m=canopy_reference_thickness_m,
        tracker_max_rotation_deg=tracker_max_rotation_deg,
        cuda=cuda,
    )[mode]


def compute_surface_solar_energies(
    solar_df,
    *,
    modes: Sequence[str],
    hillshade_dir,
    svf_path,
    out_path,
    aoi=None,
    aoi_crs: str = "EPSG:4326",
    nodata_value: float = -9999.0,
    overwrite: bool = False,
    canopy_leaf_type: Optional[int] = None,
    canopy_reference_thickness_m: Optional[Sequence[float]] = None,
    tracker_max_rotation_deg: float = 60.0,
    cuda: Optional[bool] = None,
) -> Dict[str, str]:
    """
    ``compute_surface_solar_energy`` for several modes in one read of the hillshades.

    Reading the hillshades is most of the cost, and every mode is a sum over
    the same ones, so asking for ``("total", "fourier_poa",
    "fourier_decomposition")`` here takes about as long as one of them alone.
    Modes, bands and arguments are those of ``compute_surface_solar_energy``.

    ``cuda`` runs the sums on the GPU: ``None`` uses it when CUDA is usable and
    the accumulators fit in its free memory, ``False`` keeps them on the CPU.
    The next hillshade is read in the background either way.

    Returns ``{mode: path written}``. A mode whose raster already exists is
    returned as is unless ``overwrite``.
    """
    from shapely.ops import transform as shp_transform

    modes = list(dict.fromkeys(modes))
    unknown = [m for m in modes if m not in SOLAR_ENERGY_MODES]
    if unknown or not modes:
        raise ValueError(f"modes must be among {SOLAR_ENERGY_MODES}, got {unknown or modes}")

    out_path = str(out_path)
    paths = {m: out_path.replace(".tif", f"_{m}.tif") for m in modes}
    todo = [m for m in modes if overwrite or not geotiff_is_complete(paths[m])]
    if not todo:
        return paths

    hillshade_dir = Path(hillshade_dir)

    # ==========================================================
    # SOLAR DATA PREPARATION
    # ==========================================================
    df = solar_df.copy()

    df["az_bin"] = df.azimuth.astype(int)
    df["alt_bin"] = df.elevation.astype(int)

    # Every row stands for one step of the table. Taking the gap to the next
    # row instead is wrong wherever the table has holes - a daylight-only
    # table jumps the whole night after each day's last hour, which then
    # counted for twelve to sixteen hours of evening sun, inflated the annual
    # direct and diffuse totals by a fifth and a half respectively, and leaned
    # the year towards the West.
    steps = df.datetime_utc.diff().dt.total_seconds()
    steps = steps[steps > 0]
    df["dt_hours"] = (float(steps.median()) / 3600.0) if len(steps) else 1.0

    df = df[df.elevation > 0]
    bins = _solar_energy_bins(df)
    # resolved up front, so a missing hillshade fails now and not hours in
    files = [_resolve_hillshade_path(hillshade_dir, az, alt) for az, alt, _, _ in bins]

    # ==========================================================
    # GRID, SVF & AOI
    # ==========================================================
    with rasterio.open(svf_path) as src:
        profile = src.profile.copy()
        H, W = src.height, src.width
        raster_crs = src.crs
        pixel_area = abs(src.transform.a * src.transform.e)
        svf = src.read(1).astype(np.float32)

    if aoi is not None:
        transformer = pyproj.Transformer.from_crs(
            aoi_crs, raster_crs, always_xy=True
        ).transform
        aoi_mask = rasterize_aoi_mask(shp_transform(transformer, aoi), svf_path)
    else:
        aoi_mask = np.ones((H, W), dtype=bool)

    profile.update(
        dtype="float32",
        nodata=nodata_value,
        compress="zstd",
        zstd_level=9,
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
        num_threads="ALL_CPUS",
    )

    # ==========================================================
    # BACKEND
    # ==========================================================
    if cuda is None:
        cuda = has_cuda()
        if cuda:
            from numba import cuda as _cuda
            free = _cuda.current_context().get_memory_info().free
            if _cuda_solar_bytes(todo, (H, W)) > 0.9 * free:
                warning(
                    f"Solar energy accumulators ({_cuda_solar_bytes(todo, (H, W)) / 1e9:.1f} GB) "
                    f"do not fit in the GPU's {free / 1e9:.1f} GB free; summing on the CPU."
                )
                cuda = False
    elif cuda and not has_cuda():
        warning(
            "CUDA requested but not usable (no GPU, driver or libNVVM); "
            "summing the solar energy on the CPU."
        )
        cuda = False

    backend = _CudaSolarAccumulator if cuda else _CpuSolarAccumulator
    acc = backend(todo, svf, aoi_mask, pixel_area, np.deg2rad(tracker_max_rotation_deg))
    monthly = "monthly" in todo

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    hillshades = _iter_hillshade_bands(files)
    for (az, alt, whole, by_month), fn, (base, path_lengths) in zip(tqdm(
        bins,
        desc=f"Solar energy ({', '.join(todo)}{', CUDA' if cuda else ''})",
        unit="group",
        dynamic_ncols=True,
    ), files, hillshades):
        canopy = path_lengths is not None
        if canopy and canopy_leaf_type is None:
            hillshades.close()
            raise ValueError(
                f"{fn.name} is a canopy hillshade: pass canopy_leaf_type=0 "
                f"(deciduous) or 1 (evergreen) to compute_surface_solar_energy, "
                f"or regenerate the hillshades with canopy=None"
            )

        acc.load(base, path_lengths)
        del base, path_lengths
        # a canopy attenuates the same sun differently in February and July,
        # so it is resolved per month like the monthly bands
        for month, dni_E, dhi_E in (by_month if monthly or canopy else whole):
            mu = (
                vegetation_extinction(
                    month, leaf_type=canopy_leaf_type,
                    reference_thickness_m=canopy_reference_thickness_m,
                )
                if canopy else None
            )
            acc.add(month, mu, dni_E, dhi_E, np.deg2rad(az), np.deg2rad(alt))

    # ==========================================================
    # WRITE
    # ==========================================================
    for mode, bands in acc.result().items():
        Path(paths[mode]).parent.mkdir(parents=True, exist_ok=True)
        _write_solar_energy(paths[mode], mode, bands, profile, aoi_mask,
                            nodata_value, tracker_max_rotation_deg)
        if mode in ("vertical_basis", "tracking_basis"):
            write_sun_table(df, sun_table_path(paths[mode]))

    return paths


def sun_table_path(basis_path) -> str:
    """Where the sun table of a vertical or tracking basis is written."""
    return str(basis_path).replace(".tif", "_sun.csv")


def write_sun_table(solar_df: pd.DataFrame, path) -> str:
    """
    The sun-position bins of a solar table, with their annual energy.

    One row per (azimuth, altitude) bin, as the hillshades are discretised:
    ``azimuth_deg``, ``altitude_deg``, and the beam (``dni_wh_m2``, normal to
    the sun) and diffuse (``dhi_wh_m2``, horizontal) energy the year delivers
    from it, unobstructed. It is what shading between rows is computed from:
    that depends on the layout, not on the pixel, and needs the sun only.
    """
    table = solar_df[solar_df.elevation > 0].copy()
    if "dt_hours" not in table:
        steps = table.datetime_utc.diff().dt.total_seconds()
        steps = steps[steps > 0]
        table["dt_hours"] = (float(steps.median()) / 3600.0) if len(steps) else 1.0
    table["dni_wh_m2"] = table.DNI * table.dt_hours
    table["dhi_wh_m2"] = table.DHI * table.dt_hours
    bins = (table.groupby([table.azimuth.astype(int), table.elevation.astype(int)])
            [["dni_wh_m2", "dhi_wh_m2"]].sum().reset_index())
    bins.columns = ["azimuth_deg", "altitude_deg", "dni_wh_m2", "dhi_wh_m2"]
    bins.to_csv(path, index=False)
    return str(path)


def read_sun_table(basis_path) -> pd.DataFrame:
    """The sun table written next to a vertical or tracking basis."""
    return pd.read_csv(sun_table_path(basis_path))


#: Band order of ``mode="vertical_basis"``: the lit beam on a vertical panel
#: facing each way, Wh/m2.
VERTICAL_BASIS_BANDS = (
    "H_east direct on an East-facing vertical",
    "H_west direct on a West-facing vertical",
    "H_north direct on a North-facing vertical",
    "H_south direct on a South-facing vertical",
)

#: Band order of ``mode="tracking_basis"``, Wh/m2. The ground bands are the
#: global the panel sees reflected, before the albedo.
TRACKING_BASIS_BANDS = (
    "T2_dir dual-axis direct",
    "T2_sky dual-axis sky diffuse",
    "T2_gnd dual-axis ground view",
    "T1_dir single-axis N-S direct",
    "T1_sky single-axis N-S sky diffuse",
    "T1_gnd single-axis N-S ground view",
)

_VERTICAL_FACES = {90.0: 0, 270.0: 1, 0.0: 2, 180.0: 3}


def reconstruct_vertical_poa(vertical_bands, d_sky, ghi, *, facing_azimuth_deg,
                             albedo: float = 0.2):
    """
    Annual ``(direct, sky, ground)`` on a vertical panel, Wh/m2.

    ``vertical_bands`` are the four bands of ``mode="vertical_basis"``;
    ``d_sky`` and ``ghi`` bands 4 and 5 of ``mode="fourier_poa"`` built on the
    same hillshades. The facing must be one of the four cardinal points the
    basis was split along - which is what North-South and East-West rows give.
    A vertical panel sees half the sky and half the ground.
    """
    key = float(facing_azimuth_deg) % 360.0
    if key not in _VERTICAL_FACES:
        raise ValueError("a vertical panel must face 0, 90, 180 or 270 degrees")
    direct = vertical_bands[_VERTICAL_FACES[key]]
    return direct, 0.5 * d_sky, 0.5 * albedo * ghi


def reconstruct_tracking_poa(tracking_bands, *, kind: str, albedo: float = 0.2):
    """
    Annual ``(direct, sky, ground)`` on a tracker, Wh/m2.

    ``tracking_bands`` are the six bands of ``mode="tracking_basis"``;
    ``kind`` is ``"dual_axis"`` or ``"single_axis"`` (horizontal North-South
    axis). Shading between trackers is not in it: multiply the direct (and,
    for backtracking, the sky) by the layout's factors.
    """
    if kind == "dual_axis":
        direct, sky, ground = tracking_bands[0], tracking_bands[1], tracking_bands[2]
    elif kind == "single_axis":
        direct, sky, ground = tracking_bands[3], tracking_bands[4], tracking_bands[5]
    else:
        raise ValueError("kind must be 'dual_axis' or 'single_axis'")
    return direct, sky, albedo * ground


def reconstruct_poa_from_basis(
    b_z, b_east, b_north, d_sky, ghi,
    *,
    tilt_deg,
    azimuth_deg,
    albedo: float = 0.2,
):
    """
    Annual plane-of-array irradiation (Wh/m2) for an arbitrary panel orientation.

    Takes the five bands written by ``compute_surface_solar_energy`` in
    ``mode="fourier_poa"``. Each may be a scalar or a numpy array; ``tilt_deg``
    and ``azimuth_deg`` may likewise be scalars or arrays broadcastable against
    them, so a whole tilt sweep can be evaluated in one call.

    Parameters
    ----------
    b_z, b_east, b_north, d_sky, ghi : float or np.ndarray
        Bands 1-5 of the ``fourier_poa`` raster, in Wh/m2.
    tilt_deg : float or np.ndarray
        Panel tilt from the horizontal, degrees.
    azimuth_deg : float or np.ndarray
        Panel azimuth, degrees, 0 = North and increasing clockwise, so 180 is
        South. This matches the solar azimuth convention the basis was
        accumulated with.
    albedo : float, default=0.2
        Ground reflectance for the isotropic reflected term. The default is the
        usual value for mixed urban surfaces; it only matters at steep tilts.
    """
    import numpy as np

    beta = np.deg2rad(tilt_deg)
    gamma = np.deg2rad(azimuth_deg)

    cos_b = np.cos(beta)
    sin_b = np.sin(beta)

    direct = b_z * cos_b + sin_b * (b_north * np.cos(gamma) + b_east * np.sin(gamma))
    diffuse = d_sky * (1.0 + cos_b) * 0.5
    reflected = ghi * albedo * (1.0 - cos_b) * 0.5

    # A panel can never receive negative direct beam: the expansion goes negative
    # when the sun is behind the plane, which physically means zero, not a debit.
    return np.maximum(direct, 0.0) + diffuse + reflected


@njit(parallel=True, fastmath=True, cache=True)
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


@njit(parallel=True, fastmath=True, cache=True)
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


@njit(parallel=True, fastmath=True, cache=True)
def _accumulate_poa_basis(
    B_z, B_e, B_n,
    D_sky, GHI,
    hill, svf,
    dni_E, dhi_E,
    theta, phi,
    valid_mask,
):
    """
    Exact plane-of-array basis for an arbitrary panel tilt/azimuth.

    Unlike ``_accumulate_fourier``, the hillshade is not used as a weight but as
    the source of the lit fraction: on a near-horizontal surface the GDAL
    hillshade equals sin(phi) where the pixel is lit and ~0 where it is shadowed,
    so ``hill / sin(phi)`` recovers the shadow mask itself. Dividing the surface
    cos-incidence back out is what makes the stored coefficients independent of
    the roof orientation, and therefore reusable for any rack tilt.

    Bands accumulate in Wh/m2 (no pixel_area factor), so panel energy is a plain
    multiplication by the module area.
    """
    cφ = np.cos(phi)
    sφ = np.sin(phi)
    cθ = np.cos(theta)
    sθ = np.sin(theta)

    H, W = B_z.shape
    for i in prange(H):
        for j in range(W):
            if not valid_mask[i, j]:
                continue

            if sφ > 1e-6:
                lit = hill[i, j] / sφ
                if lit > 1.0:
                    lit = 1.0
                elif lit < 0.0:
                    lit = 0.0
            else:
                lit = 0.0

            E_dir = dni_E * lit
            E_dif = dhi_E * svf[i, j]

            B_z[i, j]   += E_dir * sφ
            B_e[i, j]   += E_dir * cφ * sθ
            B_n[i, j]   += E_dir * cφ * cθ

            D_sky[i, j] += E_dif
            GHI[i, j]   += E_dir * sφ + E_dif


@njit(parallel=True, fastmath=True, cache=True)
def _accumulate_vertical_basis(
    H_ep, H_em, H_np, H_nm,
    hill,
    dni_E,
    theta, phi,
    valid_mask,
):
    """
    The lit beam split by which side of a vertical panel it reaches.

    A vertical panel facing East takes the beam only while the sun is East of
    it, so its annual direct is the sum of ``E_dir * max(0, s_east)`` - which
    the signed ``B_e`` of ``_accumulate_poa_basis`` cannot give back once the
    mornings and afternoons have been added together. One band per half axis,
    in Wh/m2: East, West, North, South faces.
    """
    cφ = np.cos(phi)
    sφ = np.sin(phi)
    s_e = cφ * np.sin(theta)
    s_n = cφ * np.cos(theta)

    east = s_e if s_e > 0.0 else 0.0
    west = -s_e if s_e < 0.0 else 0.0
    north = s_n if s_n > 0.0 else 0.0
    south = -s_n if s_n < 0.0 else 0.0

    H, W = H_ep.shape
    for i in prange(H):
        for j in range(W):
            if not valid_mask[i, j]:
                continue

            if sφ > 1e-6:
                lit = hill[i, j] / sφ
                if lit > 1.0:
                    lit = 1.0
                elif lit < 0.0:
                    lit = 0.0
            else:
                lit = 0.0

            E_dir = dni_E * lit

            H_ep[i, j] += E_dir * east
            H_em[i, j] += E_dir * west
            H_np[i, j] += E_dir * north
            H_nm[i, j] += E_dir * south


@njit(parallel=True, fastmath=True, cache=True)
def _accumulate_tracking_basis(
    T2_dir, T2_sky, T2_gnd,
    T1_dir, T1_sky, T1_gnd,
    hill, svf,
    dni_E, dhi_E,
    theta, phi,
    max_rotation,
    valid_mask,
):
    """
    Plane-of-array sums for trackers, whose orientation follows the sun.

    Dual axis: the panel faces the sun, so the beam arrives at normal
    incidence and the panel's tilt is the zenith angle. Single axis,
    horizontal and North-South: the panel rotates East-West by
    ``R = atan2(s_east, s_up)``, held within ``max_rotation``, so the beam
    meets it at ``cos AOI = s_east sin R + s_up cos R`` and its tilt is
    ``|R|``. Each gets its direct, its isotropic sky diffuse and its ground
    view (``GHI * (1 - cos tilt) / 2``, to be multiplied by the albedo), in
    Wh/m2. Shading of one tracker by the next depends on the layout and is
    applied afterwards (``tracker_row_factors`` in ``parcel_pv``).
    """
    cφ = np.cos(phi)
    sφ = np.sin(phi)
    s_e = cφ * np.sin(theta)

    rotation = np.arctan2(s_e, sφ)
    if rotation > max_rotation:
        rotation = max_rotation
    elif rotation < -max_rotation:
        rotation = -max_rotation

    cos_aoi_1 = s_e * np.sin(rotation) + sφ * np.cos(rotation)
    if cos_aoi_1 < 0.0:
        cos_aoi_1 = 0.0
    cos_tilt_1 = np.cos(rotation)

    H, W = T2_dir.shape
    for i in prange(H):
        for j in range(W):
            if not valid_mask[i, j]:
                continue

            if sφ > 1e-6:
                lit = hill[i, j] / sφ
                if lit > 1.0:
                    lit = 1.0
                elif lit < 0.0:
                    lit = 0.0
            else:
                lit = 0.0

            E_dir = dni_E * lit
            E_dif = dhi_E * svf[i, j]
            G = E_dir * sφ + E_dif

            T2_dir[i, j] += E_dir
            T2_sky[i, j] += E_dif * (1.0 + sφ) * 0.5
            T2_gnd[i, j] += G * (1.0 - sφ) * 0.5

            T1_dir[i, j] += E_dir * cos_aoi_1
            T1_sky[i, j] += E_dif * (1.0 + cos_tilt_1) * 0.5
            T1_gnd[i, j] += G * (1.0 - cos_tilt_1) * 0.5

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

    # The GPU path needs a working driver *and* libNVVM; if either is
    # missing, numba only fails at the first kernel launch, so check first.
    if cuda is None:
        cuda = has_cuda()
    elif cuda and not has_cuda():
        warning(
            "CUDA requested but not usable (no GPU, driver or libNVVM); "
            "falling back to the CPU SVF. Install the toolkit with "
            "`pip install \"numba-cuda[cu13]\"`."
        )
        cuda = False

    if hybrid and not cuda:
        info("Hybrid SVF is CUDA-only; computing full-resolution SVF on CPU")

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


@njit(parallel=True, cache=True)
def _svf_kernel_rows(
    dsm,
    valid_mask,
    svf_acc,
    dxs,
    dys,
    pixel_size,
    max_steps,
    r0,
    r1,
):
    """
    CPU twin of ``svf_horizon_kernel``: accumulates cos² of the horizon
    angle over every azimuth for rows ``r0:r1``.
    """
    rows, cols = dsm.shape
    for i in prange(r0, r1):
        for j in range(cols):
            if not valid_mask[i, j]:
                continue
            z0 = dsm[i, j]
            acc = 0.0
            for a in range(dxs.size):
                dx = dxs[a]
                dy = dys[a]
                max_angle = 0.0
                for step in range(1, max_steps + 1):
                    ii = int(i + step * dy)
                    jj = int(j + step * dx)
                    if ii < 0 or ii >= rows or jj < 0 or jj >= cols:
                        break
                    ang = math.atan2(dsm[ii, jj] - z0, step * pixel_size)
                    if ang > max_angle:
                        max_angle = ang
                acc += math.cos(max_angle) ** 2
            svf_acc[i, j] = acc


def _calculate_svf_cpu(
    dsm_path: str,
    out_path: str,
    search_radius: float,
    angular_resolution: float,
    nodata_value: float,
    row_block_size: int = 64,
) -> str:
    """
    CPU/Numba SVF, same horizon model as ``_calculate_svf_cuda``.
    """

    info("Calculating Sky View Factor (CPU / Numba)")

    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        pixel_size = abs(src.transform[0])

    height, width = dsm.shape
    info(f"DSM size: {width} × {height}, pixel size: {pixel_size:.2f} m")

    if nodata is not None:
        valid_mask = (dsm != nodata) & np.isfinite(dsm)
    else:
        valid_mask = np.isfinite(dsm)

    # float32 angles like the CUDA path, so rays land on the same pixels
    azimuths = np.deg2rad(
        np.arange(0.0, 360.0, angular_resolution, dtype=np.float32)
    ).astype(np.float64)
    dxs = np.cos(azimuths)
    dys = -np.sin(azimuths)

    max_steps = int(search_radius / pixel_size)
    if max_steps < 2:
        raise ValueError("search_radius too small for DSM resolution")

    info(f"Azimuth directions: {len(azimuths)}")
    info(f"Ray steps: {max_steps}")

    svf_acc = np.zeros(dsm.shape, dtype=np.float32)
    n_blocks = (height + row_block_size - 1) // row_block_size

    with tqdm(total=n_blocks, desc="SVF CPU", unit="block",
              dynamic_ncols=True) as pbar:
        for b in range(n_blocks):
            r0 = b * row_block_size
            r1 = min(r0 + row_block_size, height)
            _svf_kernel_rows(
                dsm, valid_mask, svf_acc, dxs, dys,
                pixel_size, max_steps, r0, r1,
            )
            pbar.update(1)

    svf = np.where(
        valid_mask, np.clip(svf_acc / len(azimuths), 0.0, 1.0), nodata_value
    ).astype(np.float32)

    profile.update(dtype="float32", count=1, nodata=nodata_value, driver="GTiff")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(svf, 1)
        dst.update_tags(
            units="dimensionless",
            description="Sky View Factor",
            method="cpu_2d_angular_horizon",
            angular_resolution=f"{angular_resolution}deg",
            search_radius=f"{search_radius}m",
        )

    info(f"Output written to {out_path}")
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
    canopy_leaf_type: Optional[int] = None,
    canopy_reference_thickness_m: Optional[Sequence[float]] = None,
):
    """
    Generate an MP4 animation of daily terrain shadows with:
      - bounded LRU hillshade cache
      - OpenCV resize
      - timestamp overlay
      - solar altitude / azimuth overlay (1 decimal)

    ``canopy_leaf_type`` (0 deciduous, 1 evergreen) is required when the
    hillshades carry canopy path-length bands; the month comes from each frame's
    own timestamp, so a clip spanning a leaf-out date shows the canopy filling in.
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
    month_of = df.datetime_utc.dt.month.to_numpy(dtype=np.int32)

    # ----------------------------------------------------------
    # LRU Hillshade Cache (bounded)
    # ----------------------------------------------------------
    @lru_cache(maxsize=8)
    def get_hillshade_cached(a: int, h: int, month: int) -> np.ndarray:
        hs = load_hillshade(
            hillshade_dir,
            azimuth=a,
            altitude=h,
            month=month if canopy_leaf_type is not None else None,
            leaf_type=canopy_leaf_type or 0,
            reference_thickness_m=canopy_reference_thickness_m,
        )
        return hs.astype(np.float32)

    # ----------------------------------------------------------
    # Geometry reference
    # ----------------------------------------------------------
    ref = get_hillshade_cached(int(az_bin[0]), int(alt_bin[0]), int(month_of[0]))
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
    # Written under a temporary name and renamed once complete: a video cut
    # short by an error would otherwise be skipped as done by the next run.
    tmp_mp4 = out_mp4.with_name(out_mp4.stem + ".part.mp4")
    with imageio.get_writer(
        tmp_mp4,
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
            hs0 = get_hillshade_cached(
                int(az_bin[idx]), int(alt_bin[idx]), int(month_of[idx])
            )[y0:y1, x0:x1]
            hs1 = get_hillshade_cached(
                int(az_bin[idx + 1]), int(alt_bin[idx + 1]), int(month_of[idx + 1])
            )[y0:y1, x0:x1]

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

    os.replace(tmp_mp4, out_mp4)
    info(f"Shadow MP4 written to: {out_mp4}")


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
    """
    Coarse per-pixel PV yield from the ``fourier_decomposition`` basis.

    Note that this basis weights every coefficient by the hillshade, which
    already contains the surface's own cos-incidence, so it cannot separate the
    panel orientation from the ground orientation. For panel-level work at a
    tilt that differs from the underlying surface - tilted racks on a flat roof -
    use ``mode="fourier_poa"`` and ``reconstruct_poa_from_basis`` instead, which
    is exact. This function is retained for quick whole-raster screening where
    tilt and azimuth track the terrain.
    """
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

@njit(parallel=True, fastmath=True, cache=True)
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


@njit(parallel=True, cache=True)
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

