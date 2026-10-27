# -*- coding: utf-8 -*-
"""
greenshadow.roof_obstacles

Roof obstacles - chimneys, vents, air-conditioning units, parapets, skylights,
light wells - and the area of each roof plane left for PV once they, and the
setbacks around them and along the roof edge, are taken out.

    obstacles, qc = detect_lidar_obstacles(planes, laz_paths)
    svf = detect_svf_obstacles(planes, "svf.tif")
    fused, usable = fuse_obstacles(planes, [obstacles, svf])
    write_obstacle_layers("roof_obstacles.gpkg", obstacles=fused,
                          usable_area=usable, planes_qc=qc)

Runs after ``detect_roof_planes(_tiled)`` and before
``plan_pv_strings_in_buildings``, which takes the usable area through its
``buildable_areas`` argument. The obstacles stay in every DSM: they still
cast shade, and only the area a module may be placed on changes.

**Why the raw returns.** The refined roof DSM the planes were fitted to is
built to make facets clean - TIN sampling and Delaunay smoothing - and that
flattens the clutter into the roof: on the digitised Alpicat obstacles the
median one cleared its own plane by 0.09 m there (docs/roof_inference.md).
The residuals are therefore measured on the LiDAR returns themselves, against
each plane refitted to them. A DSM (``dsm_path``) is the fallback where the
returns are too sparse, or where there are none.

Every threshold is a field of ``ObstacleConfig``.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window
from shapely.geometry import MultiPoint, shape
from shapely.ops import unary_union

try:
    from .logging_utils import tqdm
    from . import buildings_infer as _bi
    from .buildings_infer import (_fit_plane_ransac, _fit_plane_ls, _plane_coefficients,
                                  _dbscan_grid, _detect_roof_obstructions, _fork_map)
except ImportError:
    from greenshadow.logging_utils import tqdm
    from greenshadow import buildings_infer as _bi
    from greenshadow.buildings_infer import (_fit_plane_ransac, _fit_plane_ls, _plane_coefficients,
                                             _dbscan_grid, _detect_roof_obstructions, _fork_map)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ObstacleConfig:
    """
    Every threshold of the obstacle detection and of the usable area.

    Lengths in metres, areas in m2 measured in plan unless stated otherwise.
    """

    # --- LiDAR residual detector -------------------------------------------
    # Classes of the returns read. None: the pipeline's region decides
    # (ASPRS 1 unclassified and 6 building by default; the ICGC adds 17 and
    # 77, "other constructions", which is where most rooftop structures are).
    lidar_classes: Optional[Tuple[int, ...]] = None
    # A return is an obstacle candidate when |residual| > max(k * sigma,
    # residual_min_m), sigma the MAD-based spread of its plane's residuals.
    residual_k: float = 3.0
    residual_min_m: float = 0.20
    # Refit of each plane on the raw returns: RANSAC inlier band and draws,
    # then least squares on the inliers. A refit leaning more than
    # max_refit_angle_deg off the detector's plane has locked onto something
    # else; the detector's orientation is kept and the plane flagged.
    fit_threshold_m: float = 0.05
    fit_iterations: int = 200
    max_refit_angle_deg: float = 10.0
    # Band along every plane boundary (ridges, hips, eaves, steps) where the
    # returns belong to two surfaces or to a facade. A return there counts
    # only if it stands above every plane around it by more than
    # max(threshold, boundary_keep_above_m) - a chimney on a ridge - and
    # never when it is below them (eave drop, wall of a step). None: the band
    # is ignored entirely.
    boundary_buffer_m: float = 0.3
    boundary_keep_above_m: Optional[float] = 0.5
    # Gaps narrower than this between a roof's planes are closed when its
    # outline is formed (unassigned ridge pixels).
    outline_close_m: float = 0.25
    # DBSCAN in plan: eps = clip(cluster_eps_factor * point spacing,
    # cluster_eps_min_m, cluster_eps_max_m), spacing = 1 / sqrt(density).
    cluster_eps_factor: float = 2.0
    cluster_eps_min_m: float = 0.3
    cluster_eps_max_m: float = 1.0
    # Smallest cluster, in returns. With density rho the smallest obstacle
    # found is about cluster_min_samples / rho m2: 0.2 m2 at 15 pts/m2,
    # 1.5-3 m2 at 1-2 pts/m2.
    cluster_min_samples: int = 3
    # Cluster outline: concave hull (0 = tightest, 1 = convex), widened by
    # hull_buffer_factor * spacing so that a footprint covers its returns.
    hull_ratio: float = 0.3
    hull_buffer_factor: float = 0.5
    min_obstacle_area_m2: float = 0.0
    max_obstacle_area_m2: Optional[float] = None

    # --- Raster residual (DSM - plane), for sparse or missing returns ------
    # "fallback": only planes with fewer than raster_min_density pts/m2 (or
    # no returns); "both": every plane, unioned with the returns; "off".
    raster_mode: str = "fallback"
    raster_min_density: float = 3.0
    raster_min_pixels: int = 2

    # --- Plane quality -----------------------------------------------------
    # needs_review when the fit is poor or the plane is mostly clutter.
    max_plane_rmse_m: float = 0.08
    max_outlier_fraction: float = 0.25
    min_plane_points: int = 20

    # --- Sky view factor detector (docs/roof_inference.md section 3.4b) ----
    svf: bool = True
    svf_resolution_m: float = 0.2
    svf_drop: float = 0.12
    svf_close_m: float = 0.6
    svf_min_area_m2: float = 0.5
    svf_max_area_m2: float = 15.0
    svf_max_elongation: float = 2.5
    svf_max_planes: int = 1

    # --- Usable area -------------------------------------------------------
    # Clearance around every obstacle.
    obstacle_setback_m: float = 0.3
    # Clearance along the roof's outer edge (access, wind uplift, fire code).
    edge_setback_m: float = 0.3
    # Clearance along the seams between two planes of the same roof.
    seam_setback_m: float = 0.05
    # Pieces narrower than this (an opening at half of it) or smaller than
    # min_fragment_area_m2 (true area, on the roof) cannot take a module.
    # None: the narrowest and the smallest of the PV modules considered.
    min_fragment_width_m: Optional[float] = None
    min_fragment_area_m2: Optional[float] = None

    # --- Execution ---------------------------------------------------------
    n_workers: Optional[int] = None
    # The pipeline reuses roof_obstacles.gpkg unless this is set.
    recalculate: bool = False


DETECTION_COLUMNS = [
    "part_id", "plane_id", "source", "method", "sign", "max_height_above_plane",
    "mean_residual", "n_points", "area",
]

QC_COLUMNS = [
    "part_id", "plane_id", "n_points", "density_pts_m2", "rmse_m", "sigma_m",
    "threshold_m", "outlier_fraction", "refit_angle_deg", "fit", "method",
    "needs_review",
]


# ============================================================
# LIDAR RESIDUAL DETECTOR
# ============================================================

def detect_lidar_obstacles(
    planes,
    laz_paths=None,
    config: Optional[ObstacleConfig] = None,
    *,
    dsm_path=None,
    lidar_classes=None,
    tile_size_m=1000.0,
    progress=False,
):
    """
    Obstacles standing on (or sunk into) each roof plane, from the LiDAR
    returns' distance to the plane, and a quality record per plane.

    Parameters
    ----------
    planes : GeoDataFrame or str
        Roof planes from ``detect_roof_planes(_tiled)``: ``part_id``,
        ``plane_id``, ``tilt_deg``, ``azimuth_deg`` and the facet polygon.
    laz_paths : sequence of str, optional
        Classified LAZ tiles covering the roofs. Without them only the
        raster residual on ``dsm_path`` is available.
    config : ObstacleConfig, optional
    dsm_path : str, optional
        A DSM that still holds the clutter - the LiDAR DSM at its own
        resolution with ``max`` aggregation, not the refined roof DSM.
        Used per ``config.raster_mode``.
    lidar_classes : sequence of int, optional
        Overrides ``config.lidar_classes``; default (1, 6).
    tile_size_m : float
        Roofs are read and processed in groups of this size, one forked
        process each; the planes' ``tile_id`` column is used when present.

    Returns
    -------
    (obstacles, planes_qc)
        ``obstacles``: one polygon per cluster, ``source='lidar'``,
        ``method`` 'points' or 'raster', ``sign`` +1 (standing out) or -1
        (sunk in), ``max_height_above_plane`` (signed, the extreme
        residual), ``mean_residual``, ``n_points``, ``area``.
        ``planes_qc``: per plane, the plane polygon with ``n_points``,
        ``density_pts_m2``, ``rmse_m`` (of the roof returns, obstacles
        excluded), ``sigma_m``, ``threshold_m``, ``outlier_fraction``,
        ``refit_angle_deg``, ``fit`` ('refit', 'detector' or 'none'),
        ``method`` and ``needs_review``.
    """
    cfg = config or ObstacleConfig()
    if isinstance(planes, str):
        planes = gpd.read_file(planes)
    if cfg.raster_mode not in {"fallback", "both", "off"}:
        raise ValueError("raster_mode must be 'fallback', 'both' or 'off'")

    classes = lidar_classes if lidar_classes is not None else cfg.lidar_classes
    if classes is None:
        classes = (1, 6)

    crs = planes.crs
    if planes.empty:
        return _empty_detections(crs), _empty_qc(crs)

    parts = planes.groupby("part_id", sort=False)
    groups = _part_groups(planes, tile_size_m)

    # The bounding box of every roof, to keep only the returns under one.
    boxes = planes.bounds.groupby(planes["part_id"].values).agg(
        {"minx": "min", "miny": "min", "maxx": "max", "maxy": "max"}).to_numpy()
    boxes = boxes + np.array([-1.0, -1.0, 1.0, 1.0])

    laz = [(path, _laz_bounds(path)) for path in (laz_paths or [])]
    laz = [(path, b) for path, b in laz if b is not None and _boxes_touching(boxes, b).any()]

    with tempfile.TemporaryDirectory(prefix="roof_obstacles_") as scratch:
        # Pass 1: each LAZ file decoded once, its returns under a roof kept.
        # A roof group near a tile corner touches up to nine files, which
        # it would otherwise decode whole for a strip of each.
        extract_job = {"laz": laz, "boxes": boxes, "scratch": scratch,
                       "classes": tuple(int(c) for c in classes)}
        extracts = _fork_map(_laz_extract_worker, list(range(len(laz))), extract_job,
                             n_workers=cfg.n_workers, progress=progress,
                             desc="Roof returns", chunksize=1)

        # Pass 2: the roofs, a tile group per process.
        job = {
            "parts": {pid: frame for pid, frame in parts},
            "groups": groups,
            "extracts": [e for e in extracts if e is not None],
            "dsm_path": dsm_path,
            "cfg": cfg,
        }
        results = _fork_map(_lidar_group_worker, list(range(len(groups))), job,
                            n_workers=cfg.n_workers, progress=progress,
                            desc="Roof obstacles (LiDAR)", chunksize=1)

    detections = [row for rows, _ in results for row in rows]
    qc = [row for _, rows in results for row in rows]

    obstacles = _detections_frame(detections, crs)

    qc = pd.DataFrame(qc, columns=QC_COLUMNS)
    keys = planes[["part_id", "plane_id", "geometry"]]
    planes_qc = gpd.GeoDataFrame(keys.merge(qc, on=["part_id", "plane_id"], how="left"),
                                 geometry="geometry", crs=crs)
    planes_qc["needs_review"] = planes_qc["needs_review"].fillna(True).astype(bool)
    planes_qc["fit"] = planes_qc["fit"].fillna("none")
    planes_qc["method"] = planes_qc["method"].fillna("none")

    return obstacles, planes_qc


def _part_groups(planes, tile_size_m):
    """Part ids grouped by tile: the planes' tile_id, or a grid of their centroids."""
    if "tile_id" in planes.columns and planes["tile_id"].notna().all():
        keys = planes["tile_id"].astype(str)
    else:
        c = planes.geometry.centroid
        keys = (np.floor(c.x / tile_size_m).astype(np.int64).astype(str) + "_"
                + np.floor(c.y / tile_size_m).astype(np.int64).astype(str))
    by_part = pd.DataFrame({"part_id": planes["part_id"].values, "key": keys.values})
    by_part = by_part.drop_duplicates("part_id")
    return [list(ids) for _, ids in by_part.groupby("key", sort=True)["part_id"]]


def _laz_bounds(path):
    """(minx, miny, maxx, maxy) of a LAZ file from its header, or None."""
    import laspy
    try:
        with laspy.open(path) as reader:
            mins, maxs = reader.header.mins, reader.header.maxs
        return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])
    except Exception:
        return None


def _boxes_touching(boxes, bounds):
    """Which of the (n, 4) ``boxes`` intersect ``bounds``."""
    x0, y0, x1, y1 = bounds
    return (boxes[:, 0] <= x1) & (boxes[:, 2] >= x0) & (boxes[:, 1] <= y1) & (boxes[:, 3] >= y0)


def _laz_extract_worker(i, chunk_size=2_000_000):
    """
    The returns of one LAZ file that are of the classes asked for and under
    a roof's bounding box, to an .npz in the scratch directory:
    ``(path, bounds)``, or None when there are none.
    """
    import laspy

    job = _bi._FORK_JOB
    path, bounds = job["laz"][i]
    x0, y0, x1, y1 = bounds
    boxes = job["boxes"][_boxes_touching(job["boxes"], bounds)]

    # Roof boxes burnt into a grid over the file, so a return is tested by
    # one lookup instead of against every box.
    cell = max(1.0, max(x1 - x0, y1 - y0) / 4000.0)
    nx, ny = int(np.ceil((x1 - x0) / cell)) + 1, int(np.ceil((y1 - y0) / cell)) + 1
    grid = np.zeros((ny, nx), dtype=bool)
    for bx0, by0, bx1, by1 in boxes:
        c0, c1 = int(max((bx0 - x0) // cell, 0)), int(min((bx1 - x0) // cell, nx - 1))
        r0, r1 = int(max((by0 - y0) // cell, 0)), int(min((by1 - y0) // cell, ny - 1))
        grid[r0:r1 + 1, c0:c1 + 1] = True

    classes = np.asarray(job["classes"])
    kept = []
    # The serial backend, always: a forked process decoding LAZ with laspy's
    # default (rayon) backend deadlocks once its parent has used it. Only
    # x, y, z and classification are decoded (LAS 1.4 layered compression).
    with laspy.open(path, laz_backend=laspy.LazBackend.Lazrs,
                    decompression_selection=(laspy.DecompressionSelection.XY_RETURNS_CHANNEL
                                             | laspy.DecompressionSelection.Z
                                             | laspy.DecompressionSelection.CLASSIFICATION)) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            c = np.asarray(chunk.classification, dtype=np.uint8)
            keep = np.isin(c, classes)
            if not keep.any():
                continue
            x = np.asarray(chunk.x, dtype=np.float64)[keep]
            y = np.asarray(chunk.y, dtype=np.float64)[keep]
            col = np.clip(((x - x0) // cell).astype(np.int64), 0, nx - 1)
            row = np.clip(((y - y0) // cell).astype(np.int64), 0, ny - 1)
            under = grid[row, col]
            if not under.any():
                continue
            z = np.asarray(chunk.z, dtype=np.float64)[keep][under]
            kept.append(np.column_stack((x[under], y[under], z, c[keep][under])))

    if not kept:
        return None
    out = os.path.join(job["scratch"], f"{i}.npy")
    np.save(out, np.vstack(kept))
    return out, bounds


def _lidar_group_worker(k):
    """Obstacles and QC of one group of roofs, from the returns extracted under them."""
    job = _bi._FORK_JOB
    cfg = job["cfg"]
    part_ids = job["groups"][k]
    frames = [job["parts"][pid] for pid in part_ids]

    minx, miny, maxx, maxy = pd.concat(frames).total_bounds
    margin = 1.0
    bounds = (minx - margin, miny - margin, maxx + margin, maxy + margin)

    chunks = []
    for path, (lx0, ly0, lx1, ly1) in job["extracts"]:
        if lx1 < bounds[0] or lx0 > bounds[2] or ly1 < bounds[1] or ly0 > bounds[3]:
            continue
        pts = np.load(path, mmap_mode="r")
        inside = ((pts[:, 0] >= bounds[0]) & (pts[:, 0] <= bounds[2])
                  & (pts[:, 1] >= bounds[1]) & (pts[:, 1] <= bounds[3]))
        if inside.any():
            chunks.append(np.asarray(pts[inside]))
    points = np.vstack(chunks) if chunks else np.empty((0, 4))

    dsm = rasterio.open(job["dsm_path"]) if job["dsm_path"] else None
    detections, qc = [], []
    try:
        for part_planes in frames:
            rows, qc_rows = _part_lidar_obstacles(part_planes, points, cfg, dsm)
            detections.extend(rows)
            qc.extend(qc_rows)
    finally:
        if dsm is not None:
            dsm.close()
    return detections, qc


def _roof_outline(geometries, close_m):
    """The roof as one surface: its planes unioned, hairline gaps closed."""
    outline = unary_union(list(geometries))
    if close_m > 0:
        outline = outline.buffer(close_m / 2, join_style="mitre").buffer(
            -close_m / 2, join_style="mitre")
    return outline


def _signed_residual(a, b, c, x, y, z):
    """Orthogonal distance to z = a x + b y + c, positive above the plane."""
    return (z - (a * x + b * y + c)) / np.sqrt(1.0 + a * a + b * b)


def _normal_angle_deg(a0, b0, a1, b1):
    n0 = np.array([-a0, -b0, 1.0]) / np.sqrt(1.0 + a0 * a0 + b0 * b0)
    n1 = np.array([-a1, -b1, 1.0]) / np.sqrt(1.0 + a1 * a1 + b1 * b1)
    return float(np.degrees(np.arccos(np.clip(n0 @ n1, -1.0, 1.0))))


def _robust_sigma(values):
    if values.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _part_lidar_obstacles(part_planes, points, cfg, dsm):
    """
    Detections and QC rows of one roof (one ``part_id``).

    Every return inside the roof outline is given a home plane - the one it
    falls in, or the nearest - and measured against it. Inside a plane and
    clear of its boundary, a return is a candidate when it is off the plane
    either way by more than the plane's threshold. In the boundary band, and
    between planes, it must stand above *every* plane within reach, so a
    ridge, an eave or the wall of a step does not become an obstacle while
    a chimney on the ridge still does.
    """
    part_id = part_planes["part_id"].iloc[0]
    geoms = list(part_planes.geometry.values)
    plane_ids = list(part_planes["plane_id"].values)
    n_planes = len(geoms)

    # Plane equations as the detector reported them; the offset comes from
    # the returns.
    orient = [_plane_coefficients(t, az) for t, az in
              zip(part_planes["tilt_deg"].values, part_planes["azimuth_deg"].values)]

    outline = _roof_outline(geoms, cfg.outline_close_m)
    minx, miny, maxx, maxy = outline.bounds
    if len(points):
        box_mask = ((points[:, 0] >= minx) & (points[:, 0] <= maxx)
                    & (points[:, 1] >= miny) & (points[:, 1] <= maxy))
        pts = points[box_mask]
        pts = pts[shapely.contains_xy(outline, pts[:, 0], pts[:, 1])]
    else:
        pts = points
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    n = len(pts)

    # Home plane and distance to it (0 inside).
    if n:
        tree = shapely.STRtree(geoms)
        pgeoms = shapely.points(x, y)
        (pt_idx, home_idx), home_dist = tree.query_nearest(
            pgeoms, return_distance=True, all_matches=False)
        home = np.full(n, -1, dtype=np.int64)
        home[pt_idx] = home_idx
        dist_home = np.full(n, np.inf)
        dist_home[pt_idx] = home_dist
    else:
        pgeoms = np.empty(0, dtype=object)
        home = np.empty(0, dtype=np.int64)
        dist_home = np.empty(0)
    inside = dist_home == 0.0

    buffer_m = cfg.boundary_buffer_m
    band = np.zeros(n, dtype=bool)
    # Distance of each inside return to its own plane's boundary.
    for i, g in enumerate(geoms):
        sel = np.flatnonzero(inside & (home == i))
        if sel.size and buffer_m > 0:
            band[sel] = shapely.distance(g.boundary, pgeoms[sel]) < buffer_m

    core = inside & ~band
    unassigned_far = ~inside & (dist_home > buffer_m)
    edge = ~core & ~unassigned_far          # band, or between planes

    coeffs = [None] * n_planes
    thresholds = np.full(n_planes, np.nan)
    qc_rows = []
    residual = np.full(n, np.nan)

    for i, g in enumerate(geoms):
        a0, b0 = orient[i]
        own_core = core & (home == i)
        own_inside = inside & (home == i)
        # The interior returns, unless too few to fit: a narrow plane is
        # mostly boundary band.
        fit_sel = own_core if own_core.sum() >= cfg.min_plane_points else own_inside
        n_inside = int(own_inside.sum())
        area = g.area
        density = n_inside / area if area > 0 else 0.0

        fit = "none"
        a = b = c = None
        angle = np.nan
        if fit_sel.sum() >= max(3, cfg.min_plane_points):
            sample = pts[fit_sel, :3]
            ransac = _fit_plane_ransac(sample, threshold=cfg.fit_threshold_m,
                                       max_iterations=cfg.fit_iterations, seed=0)
            if ransac is not None and len(ransac[3]) >= 3:
                a, b, c = _fit_plane_ls(*sample[ransac[3]].T)
                angle = _normal_angle_deg(a0, b0, a, b)
                fit = "refit"
                if angle > cfg.max_refit_angle_deg:
                    a = b = c = None
        if a is None and fit_sel.sum() >= 3:
            # Too few returns for a refit, or a refit that tilted away: the
            # detector's orientation, at the returns' median height.
            sample = pts[fit_sel]
            a, b = a0, b0
            c = float(np.median(sample[:, 2] - a * sample[:, 0] - b * sample[:, 1]))
            fit = "detector"
        if a is not None:
            coeffs[i] = (a, b, c)

        sigma = rmse = np.nan
        threshold = np.nan
        outlier_fraction = np.nan
        if coeffs[i] is not None:
            own = home == i
            residual[own] = _signed_residual(a, b, c, x[own], y[own], z[own])
            r_core = residual[own_core]
            sigma = _robust_sigma(r_core) if r_core.size else np.nan
            threshold = max(cfg.residual_k * (sigma if np.isfinite(sigma) else 0.0),
                            cfg.residual_min_m)
            thresholds[i] = threshold
            if r_core.size:
                off = np.abs(r_core) > threshold
                outlier_fraction = float(off.mean())
                roof = r_core[~off]
                rmse = float(np.sqrt(np.mean(roof ** 2))) if roof.size else np.nan

        method = "points" if fit != "none" else "none"
        sparse = density < cfg.raster_min_density
        if dsm is not None and (cfg.raster_mode == "both"
                                or (cfg.raster_mode == "fallback" and sparse)):
            method = "raster" if method == "none" or cfg.raster_mode == "fallback" else "both"

        needs_review = bool(
            fit == "none"
            or n_inside < cfg.min_plane_points
            or (np.isfinite(rmse) and rmse > cfg.max_plane_rmse_m)
            or (np.isfinite(outlier_fraction) and outlier_fraction > cfg.max_outlier_fraction)
            or (np.isfinite(angle) and angle > cfg.max_refit_angle_deg)
        )
        qc_rows.append({
            "part_id": part_id, "plane_id": plane_ids[i], "n_points": n_inside,
            "density_pts_m2": density, "rmse_m": rmse, "sigma_m": sigma,
            "threshold_m": threshold, "outlier_fraction": outlier_fraction,
            "refit_angle_deg": angle, "fit": fit, "method": method,
            "needs_review": needs_review,
        })

    # --- candidates ----------------------------------------------------
    has_plane = np.array([cf is not None for cf in coeffs])
    use_points = np.array([row["method"] in ("points", "both") for row in qc_rows])
    flagged = np.zeros(n, dtype=bool)
    if n:
        valid = (home >= 0) & has_plane[np.maximum(home, 0)] & use_points[np.maximum(home, 0)]
        thr = thresholds[np.maximum(home, 0)]
        flagged |= valid & (core | unassigned_far) & (np.abs(residual) > thr)

        if cfg.boundary_keep_above_m is not None and buffer_m > 0:
            sel = np.flatnonzero(valid & edge)
            if sel.size:
                lowest = residual[sel].copy()
                xs, ys, zs = x[sel], y[sel], z[sel]
                for i, g in enumerate(geoms):
                    if coeffs[i] is None:
                        continue
                    gx0, gy0, gx1, gy1 = g.bounds
                    near = np.flatnonzero((xs >= gx0 - buffer_m) & (xs <= gx1 + buffer_m)
                                          & (ys >= gy0 - buffer_m) & (ys <= gy1 + buffer_m))
                    if near.size == 0:
                        continue
                    near = near[shapely.dwithin(g, pgeoms[sel[near]], buffer_m)]
                    if near.size:
                        ai, bi, ci = coeffs[i]
                        r_i = _signed_residual(ai, bi, ci, xs[near], ys[near], zs[near])
                        lowest[near] = np.minimum(lowest[near], r_i)
                floor = np.maximum(thr[sel], cfg.boundary_keep_above_m)
                flagged[sel] |= lowest > floor

    rows = []
    if flagged.any():
        density = n / max(outline.area, 1e-9)
        spacing = 1.0 / np.sqrt(max(density, 1e-9))
        eps = float(np.clip(cfg.cluster_eps_factor * spacing,
                            cfg.cluster_eps_min_m, cfg.cluster_eps_max_m))
        for sign in (1, -1):
            sel = np.flatnonzero(flagged & (np.sign(residual) == sign))
            if sel.size < cfg.cluster_min_samples:
                continue
            xyz = np.column_stack((x[sel], y[sel], np.zeros(sel.size)))
            labels = _dbscan_grid(xyz, eps, cfg.cluster_min_samples)
            for lab in np.unique(labels[labels >= 0]):
                members = sel[labels == lab]
                polygon = _cluster_polygon(x[members], y[members], spacing, cfg)
                if polygon is None:
                    continue
                area = polygon.area
                if area < cfg.min_obstacle_area_m2:
                    continue
                if cfg.max_obstacle_area_m2 is not None and area > cfg.max_obstacle_area_m2:
                    continue
                r = residual[members]
                homes = home[members]
                rows.append({
                    "part_id": part_id,
                    "plane_id": plane_ids[int(np.bincount(homes).argmax())],
                    "source": "lidar", "method": "points", "sign": sign,
                    "max_height_above_plane": float(r.max() if sign > 0 else r.min()),
                    "mean_residual": float(r.mean()), "n_points": int(members.size),
                    "area": float(area), "geometry": polygon,
                })

    if dsm is not None:
        for i, g in enumerate(geoms):
            if qc_rows[i]["method"] not in ("raster", "both"):
                continue
            rows.extend(_plane_raster_obstacles(
                g, part_id, plane_ids[i], orient[i], coeffs[i], dsm, cfg))

    return rows, qc_rows


def _cluster_polygon(xs, ys, spacing, cfg):
    """Footprint of a cluster of returns: concave hull, widened by half a spacing."""
    pts = MultiPoint(np.column_stack((xs, ys)))
    hull = shapely.concave_hull(pts, ratio=cfg.hull_ratio) if len(xs) >= 3 else pts
    if hull.geom_type != "Polygon" or hull.area <= 0:
        hull = pts.convex_hull
    polygon = hull.buffer(max(cfg.hull_buffer_factor * spacing, 1e-3))
    return None if polygon.is_empty else polygon


def _plane_raster_obstacles(geometry, part_id, plane_id, orientation, coeffs, dsm, cfg):
    """
    Obstacles of one plane from DSM - plane: the pixels off it by more than
    max(k * sigma, residual_min_m), clear of the boundary band, grouped into
    components of at least ``raster_min_pixels``.
    """
    from scipy.ndimage import label

    minx, miny, maxx, maxy = geometry.bounds
    top, left = rowcol(dsm.transform, minx, maxy, op=np.floor)
    bottom, right = rowcol(dsm.transform, maxx, miny, op=np.floor)
    top, left = max(int(top), 0), max(int(left), 0)
    bottom, right = min(int(bottom) + 1, dsm.height), min(int(right) + 1, dsm.width)
    if bottom <= top or right <= left:
        return []
    window = Window(left, top, right - left, bottom - top)
    block = dsm.read(1, window=window).astype(np.float64)
    transform = dsm.window_transform(window)

    rows, cols = np.indices(block.shape)
    gx = transform.c + (cols + 0.5) * transform.a
    gy = transform.f + (rows + 0.5) * transform.e

    valid = np.isfinite(block) & (block > -9000)
    if dsm.nodata is not None:
        valid &= block != dsm.nodata
    interior = geometry.buffer(-cfg.boundary_buffer_m) if cfg.boundary_buffer_m > 0 else geometry
    if interior.is_empty:
        return []
    inside = valid & shapely.contains_xy(interior, gx, gy)
    if inside.sum() < 3:
        return []

    # The orientation of the refit when there is one, the height always from
    # the DSM itself: a max-aggregated cell sits above the plane at its
    # centre by up to half a pixel times the slope.
    a, b = coeffs[:2] if coeffs is not None else orientation
    c = float(np.median(block[inside] - a * gx[inside] - b * gy[inside]))

    residual = _signed_residual(a, b, c, gx, gy, block)
    sigma = _robust_sigma(residual[inside])
    threshold = max(cfg.residual_k * sigma, cfg.residual_min_m)

    out = []
    pixel_area = abs(transform.a * transform.e)
    for sign in (1, -1):
        mask = inside & (sign * residual > threshold)
        if not mask.any():
            continue
        labels, n_comp = label(mask)
        counts = np.bincount(labels.ravel())
        for comp in range(1, n_comp + 1):
            if counts[comp] < cfg.raster_min_pixels:
                continue
            blob = labels == comp
            polygon = unary_union([shape(s) for s, _ in shapes(
                blob.astype(np.uint8), mask=blob, transform=transform)])
            r = residual[blob]
            if polygon.area < cfg.min_obstacle_area_m2:
                continue
            if cfg.max_obstacle_area_m2 is not None and polygon.area > cfg.max_obstacle_area_m2:
                continue
            out.append({
                "part_id": part_id, "plane_id": plane_id, "source": "lidar",
                "method": "raster", "sign": sign,
                "max_height_above_plane": float(r.max() if sign > 0 else r.min()),
                "mean_residual": float(r.mean()), "n_points": int(counts[comp]),
                "area": float(counts[comp] * pixel_area), "geometry": polygon,
            })
    return out


# ============================================================
# SKY VIEW FACTOR DETECTOR
# ============================================================

def detect_svf_obstacles(planes, svf_path, config: Optional[ObstacleConfig] = None, *,
                         progress=False):
    """
    Obstacles from the ring of sky view an object steals from the roof
    around it: ``buildings_infer._detect_roof_obstructions`` (the detector
    of ``detect_roof_planes(svf_path=...)``, docs/roof_inference.md
    section 3.4b) run on finished planes, roof by roof.

    The planes are rasterised at ``svf_resolution_m`` and the SVF resampled
    onto that grid, as the roof detector does. ``source='svf'``.
    """
    cfg = config or ObstacleConfig()
    if isinstance(planes, str):
        planes = gpd.read_file(planes)
    if planes.empty:
        return _empty_detections(planes.crs)

    job = {"parts": [frame for _, frame in planes.groupby("part_id", sort=False)],
           "svf_path": svf_path, "cfg": cfg}
    results = _fork_map(_svf_part_worker, list(range(len(job["parts"]))), job,
                        n_workers=cfg.n_workers, progress=progress,
                        desc="Roof obstacles (SVF)")
    return _detections_frame([row for rows in results for row in rows], planes.crs)


_SVF_SOURCE = None


def _svf_part_worker(k):
    global _SVF_SOURCE
    job = _bi._FORK_JOB
    cfg = job["cfg"]
    part_planes = job["parts"][k]

    # One handle per process: GDAL handles do not survive fork.
    if _SVF_SOURCE is None or _SVF_SOURCE[0] != (os.getpid(), job["svf_path"]):
        _SVF_SOURCE = ((os.getpid(), job["svf_path"]), rasterio.open(job["svf_path"]))
    src = _SVF_SOURCE[1]

    res = cfg.svf_resolution_m
    margin = cfg.svf_close_m + 1.0
    minx, miny, maxx, maxy = part_planes.total_bounds
    minx, miny = np.floor((minx - margin) / res) * res, np.floor((miny - margin) / res) * res
    maxx, maxy = np.ceil((maxx + margin) / res) * res, np.ceil((maxy + margin) / res) * res
    width, height = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
    if width < 3 or height < 3:
        return []
    transform = from_origin(minx, maxy, res, res)

    index = np.arange(1, len(part_planes) + 1)
    plane_raster = rasterize(zip(part_planes.geometry.values, index), out_shape=(height, width),
                             transform=transform, fill=0, dtype="int32")
    if not plane_raster.any():
        return []

    svf = np.full((height, width), np.nan, dtype=np.float32)
    reproject(source=rasterio.band(src, 1), destination=svf,
              src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
              dst_transform=transform, dst_crs=src.crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    svf[(svf < 0.0) | (svf > 1.0)] = np.nan

    mask = _detect_roof_obstructions(
        plane_raster, [], None, svf, transform,
        svf_drop=cfg.svf_drop, min_height_m=0.0, min_area_m2=cfg.svf_min_area_m2,
        max_area_m2=cfg.svf_max_area_m2, max_elongation=cfg.svf_max_elongation,
        max_planes=cfg.svf_max_planes, close_m=cfg.svf_close_m, buffer_m=0.0)
    if mask is None:
        return []

    from scipy.ndimage import label
    labels, n_comp = label(mask)
    part_id = part_planes["part_id"].iloc[0]
    plane_ids = part_planes["plane_id"].values
    rows = []
    for comp in range(1, n_comp + 1):
        blob = labels == comp
        polygon = unary_union([shape(s) for s, _ in shapes(
            blob.astype(np.uint8), mask=blob, transform=transform)])
        under = plane_raster[blob]
        under = under[under > 0]
        if under.size:
            plane_id = plane_ids[np.bincount(under).argmax() - 1]
        else:
            plane_id = plane_ids[int(np.argmin(part_planes.geometry.distance(polygon).values))]
        rows.append({
            "part_id": part_id, "plane_id": plane_id, "source": "svf", "method": "svf",
            "sign": 1, "max_height_above_plane": np.nan, "mean_residual": np.nan,
            "n_points": int(blob.sum()), "area": float(polygon.area), "geometry": polygon,
        })
    return rows


# ============================================================
# FUSION AND USABLE AREA
# ============================================================

def fuse_obstacles(planes, detections: Sequence, config: Optional[ObstacleConfig] = None, *,
                   planes_qc=None, modules=None, progress=False):
    """
    Merge the detections of every source and cut the usable area out of
    each plane.

    Overlapping detections of one roof become one obstacle, whatever their
    source; ``sources`` lists them (``"lidar,svf"``). The usable area of a
    plane is::

        plane shrunk by seam_setback_m
        ∩ roof outline shrunk by edge_setback_m
        − obstacles buffered by obstacle_setback_m

    opened at half ``min_fragment_width_m`` and stripped of pieces under
    ``min_fragment_area_m2`` (true area), so what is left can hold a module.

    Parameters
    ----------
    planes : GeoDataFrame
        The roof planes the detections were made on.
    detections : sequence of GeoDataFrame
        Outputs of ``detect_lidar_obstacles``, ``detect_svf_obstacles`` ...
    planes_qc : GeoDataFrame, optional
        Carries ``needs_review`` over to the usable area.
    modules : sequence of PVModuleSpec, optional
        The modules the layout will try; they set the fragment defaults.
        ``dimensionate.DEFAULT_MODULES`` when omitted.

    Returns
    -------
    (obstacles, usable_area)
        ``obstacles``: ``part_id``, ``plane_id`` (the plane holding most of
        it), ``sources``, ``n_detections``, ``max_height_above_plane``,
        ``n_points``, ``area``. ``usable_area``: one row per plane with
        anything left, the plane's columns plus ``roof_area_m2``,
        ``usable_area_m2`` (both on the roof surface), ``obstacle_area_m2``,
        ``n_obstacles``.
    """
    cfg = config or ObstacleConfig()
    crs = planes.crs

    if modules is None:
        from .dimensionate import DEFAULT_MODULES
        modules = DEFAULT_MODULES
    width = cfg.min_fragment_width_m
    if width is None:
        width = min(min(m.length_m, m.width_m) for m in modules)
    min_area = cfg.min_fragment_area_m2
    if min_area is None:
        min_area = min(m.length_m * m.width_m for m in modules)

    frames = [d for d in detections if d is not None and len(d)]
    if frames:
        detected = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                    geometry="geometry", crs=crs)
    else:
        detected = _empty_detections(crs)
    by_part = {pid: frame for pid, frame in detected.groupby("part_id", sort=False)}

    review = {}
    if planes_qc is not None and "needs_review" in planes_qc.columns:
        review = {(p, q): bool(r) for p, q, r in
                  zip(planes_qc["part_id"], planes_qc["plane_id"], planes_qc["needs_review"])}

    obstacle_rows, usable_rows = [], []
    groups = planes.groupby("part_id", sort=False)
    if progress:
        groups = tqdm(groups, total=planes["part_id"].nunique(), desc="Usable roof area")

    for part_id, part_planes in groups:
        part_obstacles = _fuse_part(part_id, part_planes, by_part.get(part_id))
        obstacle_rows.extend(part_obstacles)
        usable_rows.extend(_part_usable_area(part_planes, part_obstacles, cfg,
                                             width, min_area, review))

    obstacles = gpd.GeoDataFrame(
        obstacle_rows, columns=["part_id", "plane_id", "sources", "n_detections",
                                "max_height_above_plane", "n_points", "area", "geometry"],
        geometry="geometry", crs=crs)

    extra = ["roof_area_m2", "usable_area_m2", "obstacle_area_m2", "n_obstacles", "needs_review"]
    columns = [c for c in planes.columns if c != planes.geometry.name] + extra + ["geometry"]
    usable = gpd.GeoDataFrame(usable_rows, columns=columns, geometry="geometry", crs=crs)
    return obstacles, usable


def _fuse_part(part_id, part_planes, detected):
    """One obstacle per connected group of overlapping detections."""
    if detected is None or detected.empty:
        return []
    geoms = list(detected.geometry.values)
    merged = unary_union(geoms)
    pieces = list(getattr(merged, "geoms", [merged]))
    tree = shapely.STRtree(geoms)
    plane_geoms = list(part_planes.geometry.values)
    plane_ids = part_planes["plane_id"].values

    rows = []
    for piece in pieces:
        if piece.is_empty or piece.area <= 0:
            continue
        members = detected.iloc[tree.query(piece, predicate="intersects")]
        overlap = [piece.intersection(g).area for g in plane_geoms]
        heights = members["max_height_above_plane"].to_numpy(dtype=float)
        finite = heights[np.isfinite(heights)]
        rows.append({
            "part_id": part_id,
            "plane_id": plane_ids[int(np.argmax(overlap))],
            "sources": ",".join(sorted(set(members["source"]))),
            "n_detections": len(members),
            "max_height_above_plane": (float(finite[np.argmax(np.abs(finite))])
                                       if finite.size else np.nan),
            "n_points": int(members["n_points"].sum()),
            "area": float(piece.area),
            "geometry": piece,
        })
    return rows


# The overlays of the usable area snap-round to this grid (m). The mitred
# buffers leave slivers and near-coincident edges on which the floating
# overlay can fail ("Unable to determine overlay result geometry dimension").
_GRID_M = 1e-3


def _part_usable_area(part_planes, part_obstacles, cfg, width, min_area, review):
    outline = _roof_outline(part_planes.geometry.values, cfg.outline_close_m)
    inner = outline.buffer(-cfg.edge_setback_m, join_style="mitre") \
        if cfg.edge_setback_m > 0 else outline

    obstacle_geoms = [row["geometry"] for row in part_obstacles]
    obstacles = unary_union(obstacle_geoms) if obstacle_geoms else None
    keep_out = (obstacles.buffer(cfg.obstacle_setback_m)
                if obstacles is not None and cfg.obstacle_setback_m > 0 else obstacles)

    rows = []
    geom_name = part_planes.geometry.name
    for _, plane in part_planes.iterrows():
        geometry = plane.geometry
        if geometry is None or geometry.is_empty:
            continue
        cos_tilt = np.cos(np.radians(float(plane["tilt_deg"])))
        usable = geometry.buffer(-cfg.seam_setback_m, join_style="mitre") \
            if cfg.seam_setback_m > 0 else geometry
        usable = _overlay(shapely.intersection, usable, inner)
        if keep_out is not None and not usable.is_empty:
            usable = _overlay(shapely.difference, usable, keep_out)
        if width > 0 and not usable.is_empty:
            opened = usable.buffer(-width / 2, join_style="mitre").buffer(
                width / 2, join_style="mitre")
            usable = _overlay(shapely.intersection, opened, usable)
        usable = _drop_small_pieces(usable, min_area * cos_tilt)

        n_obstacles = 0
        obstacle_area = 0.0
        if obstacles is not None:
            on_plane = _overlay(shapely.intersection, obstacles, geometry)
            obstacle_area = on_plane.area / cos_tilt
            n_obstacles = sum(1 for g in obstacle_geoms if g.intersects(geometry))

        if usable.is_empty:
            continue
        row = {c: plane[c] for c in part_planes.columns if c != geom_name}
        row.update({
            "roof_area_m2": geometry.area / cos_tilt,
            "usable_area_m2": usable.area / cos_tilt,
            "obstacle_area_m2": obstacle_area,
            "n_obstacles": n_obstacles,
            "needs_review": review.get((plane["part_id"], plane["plane_id"]), False),
            "geometry": usable,
        })
        rows.append(row)
    return rows


def _overlay(operation, a, b):
    """``operation(a, b)`` snap-rounded to ``_GRID_M``, its polygons only: the
    rounding can collapse a sliver into a line, which the next overlay
    would refuse as mixed-dimension input."""
    result = operation(a, b, grid_size=_GRID_M)
    if result.geom_type in ("Polygon", "MultiPolygon"):
        return result
    polygons = _polygons(result)
    if not polygons:
        return shapely.Polygon()
    return polygons[0] if len(polygons) == 1 else shapely.MultiPolygon(polygons)


def _drop_small_pieces(geometry, min_area):
    """The polygons of ``geometry`` of at least ``min_area`` (plan), as one geometry."""
    polygons = [g for g in _polygons(geometry) if g.area >= min_area]
    if not polygons:
        return shapely.Polygon()
    return polygons[0] if len(polygons) == 1 else shapely.MultiPolygon(polygons)


def _polygons(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [g for part in getattr(geometry, "geoms", []) for g in _polygons(part)]


# ============================================================
# OUTPUT
# ============================================================

def write_obstacle_layers(path, *, obstacles=None, usable_area=None, planes_qc=None,
                          detections=None, coreg_log=None):
    """
    Write the obstacle products to one GeoPackage, from scratch: layers
    ``obstacles`` (fused), ``usable_area``, ``planes_qc``, ``detections``
    (per source, before fusion) and ``coreg_log``, each when given.
    """
    if os.path.exists(path):
        os.remove(path)
    layers = {"obstacles": obstacles, "usable_area": usable_area, "planes_qc": planes_qc,
              "detections": detections, "coreg_log": coreg_log}
    for name, frame in layers.items():
        if frame is None:
            continue
        frame = frame[~frame.geometry.isna() & ~frame.geometry.is_empty]
        if len(frame):
            frame.to_file(path, layer=name, driver="GPKG")


def _empty_detections(crs):
    return gpd.GeoDataFrame({c: [] for c in DETECTION_COLUMNS}, geometry=[], crs=crs)


def _empty_qc(crs):
    return gpd.GeoDataFrame({c: [] for c in QC_COLUMNS}, geometry=[], crs=crs)


def _detections_frame(rows, crs):
    if not rows:
        return _empty_detections(crs)
    return gpd.GeoDataFrame(rows, columns=DETECTION_COLUMNS + ["geometry"],
                            geometry="geometry", crs=crs)
