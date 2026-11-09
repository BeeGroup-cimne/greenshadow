# -*- coding: utf-8 -*-
"""
greenshadow.roof_obstacles

Roof obstacles - chimneys, vents, air-conditioning units, parapets, skylights,
light wells, existing PV, walkable terraces - and the area of each roof plane
left for PV once they, and the setbacks around them and along the roof edge,
are taken out.

    obstacles, qc = detect_lidar_obstacles(planes, laz_paths)   # if wanted
    qc = planes_qc_without_lidar(planes)                         # otherwise
    svf = detect_svf_obstacles(planes, "svf.tif")
    # Optional imagery: register the orthophoto on the LiDAR, then SAM 3.
    log = coregister_orthophoto(planes, ortho, "dsm.tif")
    sam3 = detect_sam3_obstacles(planes, ortho, log, dsm_path="dsm.tif")
    shadows = detect_shadow_obstacles(planes, ortho, log, "dsm.tif")
    terraces = detect_terraces(planes, "dsm.tif", ortho, log)
    fused, usable = fuse_obstacles(planes, [obstacles, svf, sam3, shadows, terraces])
    write_obstacle_layers("roof_obstacles.gpkg", obstacles=fused,
                          usable_area=usable, planes_qc=qc, coreg_log=log)

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
import warnings
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
    from .logging_utils import tqdm, warning
    from . import buildings_infer as _bi
    from .buildings_infer import (_fit_plane_ransac, _fit_plane_ls, _plane_coefficients,
                                  _dbscan_grid, _detect_roof_obstructions, _fork_map)
except ImportError:
    from greenshadow.logging_utils import tqdm, warning
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
    # Run it (and its DSM raster fallback). Off: no LiDAR obstacles, and
    # planes_qc carries no fit measures (fit and method 'off', needs_review
    # False); the sky view factor, SAM 3 and the terraces still run.
    lidar: bool = False
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

    # --- Orthophoto and co-registration ------------------------------------
    # Read an orthophoto (the pipeline's load_orthophoto) and register it on
    # the LiDAR, building by building. Off: LiDAR and SVF only.
    imagery: bool = False
    # Largest displacement searched, each way: building lean (height x
    # tan(off-nadir)) plus the orthophoto's own error.
    coreg_max_shift_m: float = 5.0
    # Margin around the roof outline matched; the edges just outside it
    # (eaves against the ground, walls) are what registers.
    coreg_pad_m: float = 2.0
    # Gaussian scale of the edges of both images, about one orthophoto
    # pixel: wider blurs let cast shadows pull the match (a 0.1 m bias at
    # 0.4 m on synthetic roofs), with no gain in repeatability on Alpicat.
    coreg_edge_sigma_m: float = 0.25
    # The LiDAR edges: DSM slope (rise / run), clipped so that every wall
    # weighs alike, plus the boundaries of the roof planes.
    coreg_slope_clip: float = 1.0
    # Neighbour consensus: the correlation surfaces of the roofs within
    # coreg_neighbour_radius_m are averaged, and each roof's own peak is
    # sought within coreg_local_radius_m of the consensus peak (lean grows
    # with height, so neighbours differ a little). With fewer than
    # coreg_min_neighbours, or a consensus peak under coreg_min_score, the
    # consensus of every roof stands in.
    coreg_neighbour_radius_m: float = 150.0
    coreg_local_radius_m: float = 1.5
    coreg_min_neighbours: int = 5
    # Below this normalised cross-correlation the roof's imagery is skipped.
    coreg_min_score: float = 0.15
    # Roofs smaller than this (plan m2) carry too few edges to register on
    # their own and take the consensus shift.
    coreg_min_area_m2: float = 20.0
    # Pixel size of the registration; None: the orthophoto's native one.
    coreg_resolution_m: Optional[float] = None
    # Warn when the orthophoto and the LiDAR were flown further apart: roofs
    # change (new panels, extensions, demolitions).
    max_date_gap_years: float = 2.0

    # --- SAM 3 on the orthophoto ------------------------------------------
    # With imagery: segment every registered roof with SAM 3 text prompts,
    # for what the LiDAR misses - flat skylights and roof windows, panels
    # already there. Needs torch and transformers >= 5 (here, or in the
    # interpreter sam3_python) and access to the gated checkpoint; without
    # them a warning, and LiDAR and SVF only.
    sam3: bool = True
    # Hugging Face id or local directory of the checkpoint.
    sam3_model: str = "facebook/sam3"
    # Interpreter to run SAM 3 in (a venv with torch and transformers);
    # None: this one.
    sam3_python: Optional[str] = None
    # None: CUDA when available, bfloat16 there and float32 on the CPU.
    sam3_device: Optional[str] = None
    sam3_dtype: Optional[str] = None
    # "solar panels" outlines whole arrays where "solar panel" often takes
    # single modules or rows (on Alpicat, 3x the area at the same score).
    sam3_prompts: Tuple[str, ...] = ("chimney", "skylight", "roof window", "roof hatch",
                                     "solar panel", "solar panels", "solar PV",
                                     "air conditioning unit", "HVAC system", "vent",
                                     "antenna", "dormer")
    # The orthophoto is resampled (cubic) this much finer than native and
    # cut into windows of the model's input size (1008 px: 84 m at 0.25 m
    # x 3) around each roof plus sam3_pad_m. A 0.6 m chimney is 2-3 native
    # pixels; the model sees 7-8. Roofs larger than a window are tiled with
    # sam3_tile_overlap_m between tiles.
    sam3_upsample: float = 3.0
    sam3_pad_m: float = 2.0
    sam3_tile_overlap_m: float = 6.0
    # Windows per forward pass.
    sam3_batch_size: int = 4
    # Lowest instance score kept, and the mask probability that is inside.
    sam3_min_score: float = 0.4
    sam3_mask_threshold: float = 0.5
    # Mask area limits (plan m2, the whole mask before clipping to the roof).
    sam3_min_area_m2: float = 0.1
    sam3_max_area_m2: float = 40.0
    # A mask mostly off the roof belongs to the ground or a neighbour; one
    # covering most of it is the roof itself.
    sam3_min_inside_fraction: float = 0.5
    sam3_max_roof_fraction: float = 0.5
    # A mask that is mostly shadow (pixels darker than this fraction of full
    # brightness) is the shade of an obstacle, not one. None: not checked.
    sam3_shadow_brightness: Optional[float] = 0.15
    sam3_max_shadow_fraction: float = 0.6
    # A little shadow on a roof is cast by something on it - most often a
    # small chimney the LiDAR barely resolves - so it is an obstacle, class
    # "shadow": a dark mask (above) or a would-be PV mask in the DSM's
    # shade (below) of at most sam3_shadow_obstacle_max_area_m2 and compact
    # (minimum rotated rectangle at most sam3_shadow_obstacle_max_elongation
    # long for its width), widened toward the sun over its caster: as far
    # as the DSM puts the caster (0 when the DSM does not hold it), plus the
    # shadow's width across the sun - a compact caster is about as deep -
    # or sam3_shadow_caster_m if more. A shade the DSM casts from off the
    # roof (a neighbour, a tree), or a long one (a wall, a higher part of
    # the roof), is no obstacle. On Alpicat the
    # shadows of domes on an industrial roof were 5.4-6.3 m2 at 1.1-1.5;
    # those of eaves and parapets 9-24 m2 at 2.7-5.8.
    sam3_shadow_obstacle_max_area_m2: float = 8.0
    sam3_shadow_obstacle_max_elongation: float = 2.0
    sam3_shadow_caster_m: float = 0.5
    # Prompts for what may cover much of a roof and is dark by design (PV
    # and solar thermal arrays): no sam3_max_area_m2 and no shadow check,
    # but at most sam3_extensive_max_roof_fraction of the roof, and a 25th
    # percentile of brightness (mean RGB, 0-1) at most
    # sam3_extensive_max_brightness. On Alpicat, "solar panels" also takes
    # whole light metal, fibre-cement and white roofs (>= 0.96 of the roof,
    # brightness 0.58-0.94); the arrays seen covered 0.04-0.39 of theirs at
    # 0.23-0.53. None: no brightness test.
    sam3_extensive_prompts: Tuple[str, ...] = ("solar panel", "solar panels", "solar PV")
    sam3_extensive_max_roof_fraction: float = 0.8
    sam3_extensive_max_brightness: Optional[float] = 0.55
    # A cast shadow is dark, flat and often rectangular, and SAM 3 takes it
    # for panels: the shade of a parapet or of a higher roof on a lower one.
    # Given a DSM (detect_sam3_obstacles' dsm_path), the shadows it casts
    # with the sun of the orthophoto are predicted, and an extensive mask
    # more than sam3_extensive_max_cast_shadow in them is no PV: dropped,
    # or a "shadow" obstacle when little and cast from the roof (above). On
    # Alpicat the shadows SAM 3 took for PV lay 0.70-1.0 in the predicted
    # shade, the real arrays at most 0.46 (one under a wall 0.38). None: no
    # test.
    sam3_extensive_max_cast_shadow: Optional[float] = 0.5
    # A pixel is in shade where the DSM toward the sun rises above the sun
    # ray by more than sam3_shadow_min_height_m (DSM noise), within
    # sam3_shadow_max_distance_m. Every shadow comes out this over
    # tan(altitude) short: 0.3 m halved that of a 2 m chimney.
    sam3_shadow_min_height_m: float = 0.15
    sam3_shadow_max_distance_m: float = 40.0
    # The sun of the orthophoto (azimuth from north, altitude; degrees).
    # None: estimated from the image, per flight date of coreg_log, as the
    # position whose DSM shadows best match its dark pixels (the correlation
    # at least sam3_sun_min_correlation, else no shadow test), on up to
    # sam3_sun_sample_roofs roofs and their surroundings (sam3_sun_pad_m).
    # Alpicat 2022: 168 / 56, correlation 0.32.
    sam3_sun: Optional[Tuple[float, float]] = None
    sam3_sun_sample_roofs: int = 80
    sam3_sun_pad_m: float = 15.0
    sam3_sun_min_correlation: float = 0.1
    # Their masks on one roof closer than this are one array, named after
    # the first extensive prompt: the gaps between rows (kept free of shade)
    # and between modules are not free roof. 0: masks as segmented.
    sam3_array_close_m: float = 1.0
    # Two masks of one roof overlapping more than this (IoU) are one object
    # under two prompts; the higher score is kept.
    sam3_duplicate_iou: float = 0.5
    # Douglas-Peucker tolerance of the mask outlines.
    sam3_simplify_m: float = 0.05

    # --- Shadows on the orthophoto -----------------------------------------
    # With imagery: what stands on a roof shows by its shadow, even when the
    # LiDAR barely resolves it and SAM 3 names nothing (a vent pipe, a small
    # chimney, an antenna). The registered orthophoto of each roof is read
    # against the brightness of each plane - its shadow_reference_percentile,
    # as each plane faces the sun differently - and within shadow_edge_m of
    # a seam against the darkest plane there (the registration leaves a
    # pixel or two of a dark plane on its neighbour); the roof's outer
    # shadow_edge_m is not read, its holes are (a chimney is often one). A
    # shadow is the pixels under shadow_grow_ratio connected to one under
    # shadow_dark_ratio, none green (canopy over the roof: 2G - R - B above
    # shadow_max_green): shadow_min_area_m2 - shadow_max_area_m2, at least
    # half of it shadow_min_inset_m inside the roof's outer edge, and
    # compact (minimum rotated rectangle at most shadow_max_elongation long
    # for its width) or a pole's: along the sun (within
    # shadow_along_sun_deg) and at most shadow_pole_max_width_m across. A
    # shadow the DSM casts from off the roof, or from further than
    # shadow_max_caster_distance_m across it (a higher part of the building;
    # sam3_shadow_* settings), is dropped. The rest are obstacles, class
    # "shadow", widened over their casters as SAM 3's shadows are; the
    # caster's height is read from the shadow's length, the sun's altitude
    # and the plane's slope (max_height_above_plane, within about 0.4 m on
    # synthetic roofs). On Alpicat 20 of 30 shadows no other source had
    # were vents and chimneys by eye, 3 inside PV arrays, 5 wall or eave
    # shade before the pole width and caster distance limits.
    shadows: bool = True
    shadow_edge_m: float = 0.5
    shadow_reference_percentile: float = 60.0
    shadow_dark_ratio: float = 0.6
    shadow_grow_ratio: float = 0.8
    shadow_min_inset_m: float = 0.5
    shadow_max_green: float = 0.04
    shadow_min_area_m2: float = 0.25
    shadow_max_area_m2: float = 4.0
    shadow_max_elongation: float = 2.5
    shadow_along_sun_deg: float = 20.0
    shadow_pole_max_width_m: float = 0.5
    shadow_max_caster_distance_m: float = 3.0

    # --- Walkable terraces, balconies and patios ---------------------------
    # Flat planes people walk on take no module. A plane is flat up to
    # terrace_max_tilt_deg and looked at from terrace_min_area_m2 (plan).
    # Its guard is the fraction of its outline along which the DSM, within
    # terrace_guard_search_m, rises terrace_guard_height_m above the plane:
    # a parapet or railing, or the wall of a higher floor. On Alpicat a
    # pitched tile roof the plane detector fitted flat across its ridge, and
    # the top roof of a block behind a low parapet, mostly read below 0.5.
    terraces: bool = True
    terrace_max_tilt_deg: float = 5.0
    terrace_min_area_m2: float = 2.0
    terrace_guard_height_m: float = 0.8
    terrace_guard_search_m: float = 1.0
    # Enclosed: guarded along at least terrace_enclosed_min_guard of the
    # outline and at most terrace_enclosed_max_area_m2 - a patio, or a
    # terrace among higher floors, whatever its finish (light tiles, in
    # shade). A larger flat roof behind a parapet stays usable.
    terrace_enclosed_min_guard: float = 0.75
    terrace_enclosed_max_area_m2: float = 60.0
    # Tiled (needs the imagery): guarded along terrace_tiled_min_guard, and
    # at least terrace_min_finish_fraction of the plane a walkable finish:
    # ceramic tile (HSV hue within terrace_tile_hue_deg, saturation and
    # value at least terrace_tile_min_saturation / _value), garden or pool
    # water. Red metal sheet sits below 8 degrees of hue, gravel and
    # membranes below the saturation.
    terrace_tiled_min_guard: float = 0.5
    terrace_min_finish_fraction: float = 0.5
    terrace_tile_hue_deg: Tuple[float, float] = (8.0, 38.0)
    terrace_tile_min_saturation: float = 0.22
    terrace_tile_min_value: float = 0.3

    # --- Usable area -------------------------------------------------------
    # Clearance around every obstacle.
    obstacle_setback_m: float = 0.1
    # Clearance along the roof's outer edge (access, wind uplift, fire code).
    edge_setback_m: float = 0.1
    # Clearance along the seams between two planes of the same roof.
    seam_setback_m: float = 0.05
    # Pieces narrower than this (an opening at half of it) or smaller than
    # min_fragment_area_m2 (true area, on the roof) cannot take a module.
    # None: the narrowest and the smallest of the PV modules considered.
    min_fragment_width_m: Optional[float] = None
    min_fragment_area_m2: Optional[float] = None
    # The opening runs on the pieces simplified by this much (Douglas-
    # Peucker; the result is cut back to the unsimplified piece). The planes
    # are pixel outlines, and on their stairs the mitred opening acts as an
    # axis-aligned square: it took a triangle out of every corner of a roof
    # turned against the grid (0.8 m legs at 45 degrees), where an aligned
    # module fits. On Alpicat the opening took 4.7% of the planes, 3.4% with
    # 0.2 m (one roof pixel).
    fragment_simplify_m: float = 0.2

    # --- Validation and debug figures --------------------------------------
    # A detected and a labelled obstacle are the same object when their
    # intersection over union reaches this (one to one, best pairs first).
    # Low, as footprints of 1 m objects differ by tens of centimetres
    # between LiDAR hulls, masks and hand labels.
    validation_match_iou: float = 0.2
    # ... or when they lie this close (m). A label is drawn where the
    # orthophoto shows an object's top, which leans further than the roof it
    # was registered by: a 1 m chimney stands 0.3-0.6 m from its LiDAR
    # footprint, and at 0.6 m across their IoU is near zero.
    validation_match_distance_m: float = 0.5
    # The pipeline draws this many roofs (a fixed random sample), plus
    # debug_part_ids, to roof_obstacles_debug/: the orthophoto as published
    # and registered, the planes, every detection by source, and the usable
    # area. debug_pad_m: margin around the roof.
    debug_pngs: int = 0
    debug_part_ids: Optional[Tuple] = None
    debug_pad_m: float = 3.0

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

SAM3_COLUMNS = DETECTION_COLUMNS + ["class", "score", "image_date"]

COREG_COLUMNS = [
    "part_id", "dx", "dy", "shift_m", "score", "consensus_dx", "consensus_dy",
    "consensus_score", "n_neighbours", "status", "use_imagery", "image", "image_date",
    "lidar_date", "date_gap_years",
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


def planes_qc_without_lidar(planes):
    """
    ``planes_qc`` when the LiDAR residual detector is off: every plane, no
    fit measures, ``fit`` and ``method`` 'off', ``needs_review`` False.
    """
    if isinstance(planes, str):
        planes = gpd.read_file(planes)
    qc = gpd.GeoDataFrame(planes[["part_id", "plane_id", "geometry"]].copy(),
                          geometry="geometry", crs=planes.crs)
    for column in QC_COLUMNS[2:]:
        qc[column] = np.nan
    qc["fit"] = "off"
    qc["method"] = "off"
    qc["needs_review"] = False
    return qc[["part_id", "plane_id"] + QC_COLUMNS[2:] + ["geometry"]]


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
# ORTHOPHOTO CO-REGISTRATION
# ============================================================

def coregister_orthophoto(planes, ortho, dsm_path, config: Optional[ObstacleConfig] = None, *,
                          lidar_date=None, progress=False):
    """
    Where the orthophoto shows each roof, relative to the LiDAR.

    An orthophoto is rectified on the terrain, not on the roofs: a roof
    stands displaced by its lean, height x tan(off-nadir), away from the
    photo's nadir, on top of the orthophoto's own error (0.5-1 m). Per roof
    (``part_id``):

    1. Edges of the orthophoto (gradient magnitude of its brightness) and
       edges of the LiDAR (DSM slope, clipped, plus the plane boundaries) on
       the same grid, the LiDAR's over the roof outline plus
       ``coreg_pad_m``, the orthophoto's ``coreg_max_shift_m`` wider.
    2. Their normalised cross-correlation at every displacement within
       +-``coreg_max_shift_m``.
    3. Consensus: the surfaces of the roofs within
       ``coreg_neighbour_radius_m`` averaged. One roof's surface is often
       flat or has several peaks; the neighbourhood's has one, since the
       orthophoto's error varies slowly and lean differs little between
       neighbours. The roof's shift is its own peak within
       ``coreg_local_radius_m`` of the consensus peak (sub-pixel, parabolic).
    4. Roofs below ``coreg_min_area_m2`` take the consensus shift; roofs
       whose peak scores under ``coreg_min_score`` are left without imagery.

    Args:
        planes: Roof planes (``part_id``, ``plane_id``, polygons).
        ortho: ``greenshadow.orthophoto.Orthophoto``.
        dsm_path: DSM with the buildings (the LiDAR DSM without vegetation).
        config: ``ObstacleConfig``.
        lidar_date: LiDAR flight date, for ``date_gap_years``.

    Returns:
        GeoDataFrame, one row per roof (its outline), ``COREG_COLUMNS``:
        ``dx``, ``dy`` (m, east and north) is where the orthophoto shows the
        roof relative to the LiDAR - ``read_orthophoto(..., shift=(dx, dy))``
        returns it aligned, and a polygon drawn on the orthophoto maps back
        by (-dx, -dy). ``status``: ok, consensus, low_score, at_limit (own
        peak on the edge of the window, no consensus), too_small, no_image.
        ``use_imagery`` for ok and consensus.
    """
    from scipy.spatial import cKDTree

    cfg = config or ObstacleConfig()
    crs = planes.crs
    groups = [frame for _, frame in planes.groupby("part_id", sort=False)]
    if not groups:
        return gpd.GeoDataFrame({c: [] for c in COREG_COLUMNS}, geometry=[], crs=crs)
    outlines = [_roof_outline(frame.geometry.values, cfg.outline_close_m) for frame in groups]
    res = cfg.coreg_resolution_m or ortho.resolution(crs)
    reach = int(round(cfg.coreg_max_shift_m / res))

    own = [k for k, outline in enumerate(outlines) if outline.area >= cfg.coreg_min_area_m2]
    job = {"parts": groups, "outlines": outlines, "ortho": ortho, "dsm_path": dsm_path,
           "cfg": cfg, "res": res, "reach": reach, "crs": crs}
    results = _fork_map(_coreg_part_worker, own, job, n_workers=cfg.n_workers,
                        progress=progress, desc="Orthophoto co-registration")
    surfaces = {k: surface for k, (surface, _) in zip(own, results) if surface is not None}
    no_image = {k for k, (surface, why) in zip(own, results) if why == "no_image"}

    # Distances in metres on the surface grid, row 0 the northernmost
    # displacement: index (i, j) is dx = (j - reach) * res, dy = (reach - i) * res.
    offsets = (np.arange(2 * reach + 1) - reach) * res
    grid_dx, grid_dy = np.meshgrid(offsets, -offsets)

    centres = np.array([[o.centroid.x, o.centroid.y] if not o.is_empty else [np.nan, np.nan]
                        for o in outlines])
    keys = np.array(sorted(surfaces), dtype=int)
    stack = np.stack([surfaces[k] for k in keys]) if len(keys) else None
    tree = cKDTree(centres[keys]) if len(keys) else None

    # A roof with too few neighbours (a farm, a warehouse out of town), or
    # whose neighbours do not agree on a peak, falls back on the consensus
    # of every roof.
    overall, overall_peak = None, None
    if stack is not None and len(keys) >= cfg.coreg_min_neighbours:
        overall = stack.mean(axis=0)
        overall_peak = _correlation_peak(overall, res)[:3]

    rows = []
    for k, frame in enumerate(groups):
        row = {"part_id": frame["part_id"].iloc[0], "dx": np.nan, "dy": np.nan,
               "score": np.nan, "consensus_dx": np.nan, "consensus_dy": np.nan,
               "consensus_score": np.nan, "n_neighbours": 0}
        consensus = None
        if tree is not None and np.isfinite(centres[k]).all():
            near = [i for i in tree.query_ball_point(centres[k], cfg.coreg_neighbour_radius_m)
                    if keys[i] != k]
            row["n_neighbours"] = len(near)
            if len(near) >= cfg.coreg_min_neighbours:
                consensus = stack[near].mean(axis=0)
                peak = _correlation_peak(consensus, res)[:3]
                if peak[2] < cfg.coreg_min_score:
                    consensus = None
        if consensus is None and overall is not None:
            consensus, peak = overall, overall_peak
        if consensus is not None:
            row.update(consensus_dx=peak[0], consensus_dy=peak[1], consensus_score=peak[2])
        trusted = consensus is not None and row["consensus_score"] >= cfg.coreg_min_score

        if k in surfaces:
            if consensus is not None:
                window = np.hypot(grid_dx - row["consensus_dx"],
                                  grid_dy - row["consensus_dy"]) <= cfg.coreg_local_radius_m
                dx, dy, score, _ = _correlation_peak(surfaces[k], res, window)
                at_limit = False
            else:
                dx, dy, score, at_limit = _correlation_peak(surfaces[k], res)
            row.update(dx=dx, dy=dy, score=score)
            if not at_limit and score >= cfg.coreg_min_score:
                status = "ok"
            elif trusted:
                # Too few edges of its own to say; the neighbourhood's shift.
                row.update(dx=row["consensus_dx"], dy=row["consensus_dy"])
                status = "consensus"
            else:
                status = "at_limit" if at_limit else "low_score"
        elif k in no_image:
            status = "no_image"
        elif trusted:
            row.update(dx=row["consensus_dx"], dy=row["consensus_dy"])
            status = "consensus"
        else:
            status = "too_small"
        row["status"] = status
        row["use_imagery"] = status in ("ok", "consensus")
        rows.append(row)

    log = gpd.GeoDataFrame(rows, geometry=outlines, crs=crs)
    log["shift_m"] = np.hypot(log.dx, log.dy)
    log["image"] = ortho.name
    log["image_date"] = ortho.dates_at(log.geometry).values
    log["lidar_date"] = pd.Timestamp(lidar_date) if lidar_date is not None else pd.NaT
    log["lidar_date"] = pd.to_datetime(log["lidar_date"])
    log["date_gap_years"] = ((log.image_date - log.lidar_date).dt.days.abs() / 365.25)
    log = log[COREG_COLUMNS + ["geometry"]]

    far = log.date_gap_years > cfg.max_date_gap_years
    if far.any():
        warning(f"Orthophoto {ortho.name}: {int(far.sum())} of {len(log)} roofs were flown "
                f"{log.date_gap_years[far].median():.1f} years from the LiDAR (more than "
                f"{cfg.max_date_gap_years:g}); roofs may have changed in between")
    return log


def _coreg_part_worker(k):
    job = _bi._FORK_JOB
    try:
        return _coreg_surface(job["outlines"][k], job["parts"][k], job)
    except Exception as exc:                                    # one bad roof is not fatal
        return None, f"error: {exc}"


def _coreg_surface(outline, part_planes, job):
    """Normalised cross-correlation of the LiDAR edges over the orthophoto's."""
    from scipy import ndimage as ndi
    from skimage.feature import match_template
    try:
        from .orthophoto import read_orthophoto, _cached_open
    except ImportError:
        from greenshadow.orthophoto import read_orthophoto, _cached_open

    cfg, res, reach, crs = job["cfg"], job["res"], job["reach"], job["crs"]
    minx, miny, maxx, maxy = outline.buffer(cfg.coreg_pad_m).bounds
    minx, miny = np.floor(minx / res) * res, np.floor(miny / res) * res
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    maxx, maxy = minx + width * res, miny + height * res
    transform = from_origin(minx, maxy, res, res)

    image, _ = read_orthophoto(job["ortho"], (minx - reach * res, miny - reach * res,
                                              maxx + reach * res, maxy + reach * res),
                               crs=crs, resolution=res)
    gray = image.mean(axis=0)[:height + 2 * reach, :width + 2 * reach]
    if gray.shape != (height + 2 * reach, width + 2 * reach) or np.isnan(gray).mean() > 0.05:
        return None, "no_image"
    gray = np.where(np.isnan(gray), np.nanmean(gray), gray)
    sigma = cfg.coreg_edge_sigma_m / res
    moving = _edge_magnitude(gray, sigma)

    dsm = _cached_open(job["dsm_path"])
    height_m = np.full((height, width), np.nan, dtype=np.float32)
    reproject(source=rasterio.band(dsm, 1), destination=height_m,
              src_transform=dsm.transform, src_crs=dsm.crs, src_nodata=dsm.nodata,
              dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    reference = np.zeros((height, width), dtype=np.float32)
    if np.isfinite(height_m).any():
        height_m = np.where(np.isfinite(height_m), height_m, np.nanmin(height_m))
        slope = np.minimum(_edge_magnitude(height_m, sigma) / res, cfg.coreg_slope_clip)
        reference += _standardised(slope)
    labels = rasterize(zip(part_planes.geometry.values, range(1, len(part_planes) + 1)),
                       out_shape=(height, width), transform=transform, fill=0, dtype="int32")
    seams = (ndi.maximum_filter(labels, 3) != ndi.minimum_filter(labels, 3)).astype(np.float32)
    reference += _standardised(ndi.gaussian_filter(seams, sigma))
    if not reference.std() > 0 or not moving.std() > 0:
        return None, "no_reference"
    return match_template(moving, reference).astype(np.float32), None


def _edge_magnitude(image, sigma):
    from scipy import ndimage as ndi
    gx = ndi.gaussian_filter(image, sigma, order=(0, 1))
    gy = ndi.gaussian_filter(image, sigma, order=(1, 0))
    return np.hypot(gx, gy)


def _standardised(values):
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)


def _correlation_peak(surface, res, window=None):
    """
    (dx, dy, score, at_limit) of the highest point of a correlation surface,
    within ``window`` when given; sub-pixel by a parabola through the peak
    and its neighbours on each axis.
    """
    reach = surface.shape[0] // 2
    masked = surface if window is None else np.where(window, surface, -np.inf)
    i, j = np.unravel_index(int(np.argmax(masked)), surface.shape)

    def vertex(before, at, after):
        # Only at a true maximum: on the rim of the window the surface may
        # still be climbing, and the parabola's vertex would lie far off.
        curvature = before - 2 * at + after
        if at < before or at < after or not curvature < 0:
            return 0.0
        return 0.5 * (before - after) / curvature

    n = surface.shape[0] - 1
    di = vertex(surface[i - 1, j], surface[i, j], surface[i + 1, j]) if 0 < i < n else 0.0
    dj = vertex(surface[i, j - 1], surface[i, j], surface[i, j + 1]) if 0 < j < n else 0.0
    at_limit = i in (0, n) or j in (0, n)
    return ((j + dj - reach) * res, (reach - i - di) * res, float(surface[i, j]), at_limit)


# ============================================================
# SAM 3 ON THE ORTHOPHOTO
# ============================================================

class Sam3Unavailable(RuntimeError):
    """SAM 3 cannot run here: no torch / transformers, or no checkpoint."""


def sam3_segmenter(config: Optional[ObstacleConfig] = None):
    """
    The SAM 3 model ``config`` asks for: in this process when
    ``sam3_python`` is None, else served by that interpreter. Raises
    ``Sam3Unavailable`` with the reason.
    """
    cfg = config or ObstacleConfig()
    try:
        from . import sam3_worker
    except ImportError:
        from greenshadow import sam3_worker
    try:
        if cfg.sam3_python:
            return sam3_worker.Sam3Client(cfg.sam3_python, cfg.sam3_model, cfg.sam3_device,
                                          cfg.sam3_dtype)
        return sam3_worker.Sam3Segmenter(cfg.sam3_model, cfg.sam3_device, cfg.sam3_dtype)
    except ImportError as exc:
        raise Sam3Unavailable(
            f"SAM 3 needs torch and transformers>=5 ({exc}); install them, or point "
            f"ObstacleConfig.sam3_python at an interpreter that has them") from exc
    except Exception as exc:
        text = str(exc).splitlines()
        gated = any(k in str(exc) for k in ("gated", "401", "403", "awaiting a review"))
        hint = (f"; {cfg.sam3_model} is gated: request access on its Hugging Face page, "
                f"then `hf auth login`" if gated else "")
        raise Sam3Unavailable(f"SAM 3 could not load {cfg.sam3_model}: "
                              f"{text[-1] if text else exc}{hint}") from exc


def detect_sam3_obstacles(planes, ortho, coreg_log, config: Optional[ObstacleConfig] = None, *,
                          segmenter=None, cache_dir=None, dsm_path=None, progress=False):
    """
    Obstacles SAM 3 finds on the registered orthophoto of each roof.

    For every roof ``coreg_log`` lets use the imagery, windows of the
    model's input size, ``sam3_upsample`` times finer than the orthophoto,
    cover the roof outline plus ``sam3_pad_m``; each is read undoing the
    roof's shift, so that the masks fall on the LiDAR planes. Every prompt of
    ``sam3_prompts`` runs on every window. A mask is kept when it scores at
    least ``sam3_min_score``; its area is within ``sam3_min_area_m2`` -
    ``sam3_max_area_m2``; at least ``sam3_min_inside_fraction`` of it lies on
    the roof; it covers at most ``sam3_max_roof_fraction`` of the roof; and
    it is not mostly shadow. It is clipped to the roof outline, and of two
    masks of a roof overlapping more than ``sam3_duplicate_iou`` the higher
    scoring one stays. With ``dsm_path``, an extensive mask (existing PV)
    lying more than ``sam3_extensive_max_cast_shadow`` in the shadows the
    DSM casts under the orthophoto's sun (``orthophoto_suns``) is a
    shadow. A little shadow cast from the roof itself, or a little dark
    mask the DSM does not explain, is the shade of an obstacle on the roof:
    class "shadow", widened toward the sun over its caster. Other shadows
    are dropped.

    Args:
        planes: Roof planes (``part_id``, ``plane_id``, polygons).
        ortho: ``greenshadow.orthophoto.Orthophoto``.
        coreg_log: ``coregister_orthophoto`` output.
        segmenter: a ``Sam3Segmenter`` / ``Sam3Client`` (or anything with
            their ``segment`` and ``image_size``); None builds the one
            ``config`` describes (``sam3_segmenter``).
        cache_dir: SAM 3 outputs are kept here per window, keyed on the
            pixels, the model and the prompts; a rerun only runs the filters.
        dsm_path: DSM with what casts shadows on the roofs (buildings and
            trees), for the cast-shadow test; None: no test.

    Returns:
        GeoDataFrame, ``SAM3_COLUMNS``: ``source='sam3'``, ``method='image'``,
        ``class`` (the prompt), ``score``, ``image_date``. ``attrs``:
        ``sam3_windows``, ``sam3_inferred``, ``sam3_cast_shadows`` (masks
        dropped as shadows) and ``sun`` ({part_id: (azimuth, altitude,
        correlation)}, when tested).
    """
    import hashlib
    import pickle
    from concurrent.futures import ThreadPoolExecutor
    from shapely import affinity
    try:
        from .orthophoto import read_orthophoto, _cached_open
    except ImportError:
        from greenshadow.orthophoto import read_orthophoto, _cached_open

    cfg = config or ObstacleConfig()
    crs = planes.crs
    empty = gpd.GeoDataFrame({c: [] for c in SAM3_COLUMNS}, geometry=[], crs=crs)
    if coreg_log is None or not len(coreg_log):
        return empty
    use = coreg_log[coreg_log.use_imagery.astype(bool)]
    if not len(use):
        return empty
    if segmenter is None:
        segmenter = sam3_segmenter(cfg)

    size = int(segmenter.image_size)
    res = ortho.resolution(crs) / cfg.sam3_upsample
    span = size * res
    prompts = tuple(cfg.sam3_prompts)
    parts = {pid: frame for pid, frame in planes.groupby("part_id", sort=False)}
    outlines = {pid: _roof_outline(parts[pid].geometry.values, cfg.outline_close_m)
                for pid in use.part_id if pid in parts}

    windows = []                                  # (part_id, (minx, miny), (dx, dy))
    for row in use.itertuples():
        if row.part_id not in outlines:
            continue
        for origin in _sam3_windows(outlines[row.part_id].buffer(cfg.sam3_pad_m).bounds,
                                    span, cfg.sam3_tile_overlap_m, res):
            windows.append((row.part_id, origin, (float(row.dx), float(row.dy))))

    def read(window):
        _, (minx, miny), shift = window
        # Half a pixel inside, so that snapping keeps exactly size x size.
        image, transform = read_orthophoto(
            ortho, (minx + res / 2, miny + res / 2, minx + span - res / 2, miny + span - res / 2),
            crs=crs, resolution=res, shift=shift, resampling=Resampling.cubic)
        image = image[:, :size, :size]
        if image.shape[1:] != (size, size) or np.isnan(image).mean() > 0.5:
            return None, transform
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        rgb = np.clip(np.nan_to_num(image[:3], nan=0.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb.transpose(1, 2, 0)), transform

    def key(rgb):
        digest = hashlib.sha1(repr((cfg.sam3_model, prompts, cfg.sam3_min_score,
                                    cfg.sam3_mask_threshold, rgb.shape)).encode())
        digest.update(rgb.tobytes())
        return digest.hexdigest()

    cache = os.path.join(cache_dir, "sam3") if cache_dir else None
    if cache:
        os.makedirs(cache, exist_ok=True)

    found = {}                                    # part_id -> [(class, score, geometry)]
    n_run = 0
    batches = [windows[i:i + max(1, cfg.sam3_batch_size)]
               for i in range(0, len(windows), max(1, cfg.sam3_batch_size))]
    bar = tqdm(total=len(windows), desc="SAM 3", disable=not progress)
    # One reader thread a batch ahead of the model: reading and resampling
    # the next windows overlaps inference on these. The windows carry their
    # transforms, yet rasterio's in-memory datasets now and then warn that
    # they have none, through the bar; the filter holds for the reader too.
    with warnings.catch_warnings(), ThreadPoolExecutor(max_workers=1) as reader:
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        pending = reader.submit(lambda b: [read(w) for w in b], batches[0]) if batches else None
        for b, batch in enumerate(batches):
            crops = pending.result()
            pending = (reader.submit(lambda b: [read(w) for w in b], batches[b + 1])
                       if b + 1 < len(batches) else None)
            todo, results = [], [None] * len(batch)
            for i, (rgb, _) in enumerate(crops):
                if rgb is None:
                    results[i] = []
                    continue
                path = os.path.join(cache, key(rgb) + ".pkl") if cache else None
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        results[i] = pickle.load(f)
                else:
                    todo.append((i, rgb, path))
            if todo:
                masks = segmenter.segment([rgb for _, rgb, _ in todo], prompts,
                                          threshold=cfg.sam3_min_score,
                                          mask_threshold=cfg.sam3_mask_threshold)
                n_run += len(todo)
                for (i, _, path), image_masks in zip(todo, masks):
                    # Outlines in pixel coordinates: the cache holds what the
                    # model said, whatever the filters.
                    results[i] = [(prompts[p], score, shapely.to_wkb(_mask_polygon(mask)))
                                  for p, score, mask in image_masks]
                    if path:
                        with open(path + ".part", "wb") as f:
                            pickle.dump(results[i], f)
                        os.replace(path + ".part", path)
            for (part_id, _, _), (rgb, transform), masks in zip(batch, crops, results):
                kept = found.setdefault(part_id, [])
                if not masks:
                    continue
                matrix = [transform.a, transform.b, transform.d, transform.e,
                          transform.c, transform.f]
                brightness = rgb.mean(axis=2) / 255
                for name, score, wkb in masks:
                    pixels = shapely.from_wkb(wkb)
                    passed = _sam3_filter(affinity.affine_transform(pixels, matrix), pixels,
                                          outlines[part_id], brightness, cfg,
                                          extensive=name in cfg.sam3_extensive_prompts)
                    if passed is not None:
                        # A dark mask is the shade of an obstacle, not one.
                        kept.append(("shadow" if passed[1] else name, score, passed[0]))
            bar.update(len(batch))
    bar.close()

    # Cast shadows taken for panels: dark, flat, often rectangular.
    sun, n_shadows = None, 0
    if cfg.sam3_extensive_max_cast_shadow is not None and dsm_path:
        sun = orthophoto_suns(ortho, coreg_log, dsm_path, cfg, progress=progress)
        if not any(np.isfinite(a) and np.isfinite(h) for a, h, _ in sun.values()):
            warning(f"Orthophoto {ortho.name}: its sun could not be told from its shadows; "
                    "existing PV is not tested against cast shadows (set sam3_sun)")
        dsm = _cached_open(dsm_path)
        for part_id, masks in found.items():
            azimuth, altitude, _ = sun.get(part_id, (np.nan, np.nan, np.nan))
            tested = [k for k, m in enumerate(masks)
                      if m[0] in cfg.sam3_extensive_prompts or m[0] == "shadow"]
            if not tested or not (np.isfinite(azimuth) and np.isfinite(altitude)):
                continue
            shades = _cast_shadows(dsm, outlines[part_id], [masks[k][2] for k in tested],
                                   (azimuth, altitude), crs, cfg)
            kept = {k: m for k, m in enumerate(masks)}
            for k, (fraction, from_roof, distance) in zip(tested, shades):
                name, score, geometry = masks[k]
                cast = fraction > cfg.sam3_extensive_max_cast_shadow
                if name != "shadow" and not cast:
                    continue                              # panels in the sun
                if name != "shadow":
                    n_shadows += 1
                if (cast and from_roof < 0.5) or \
                        geometry.area > cfg.sam3_shadow_obstacle_max_area_m2 or \
                        _elongation(geometry) > cfg.sam3_shadow_obstacle_max_elongation:
                    del kept[k]                # shade from off the roof, or of a wall
                    continue
                # A compact caster is about as deep along the sun as its
                # shadow is wide across it.
                reach = (max(cfg.sam3_shadow_caster_m, _width_across(geometry, azimuth))
                         + (distance if cast and np.isfinite(distance) else 0.0))
                kept[k] = ("shadow", score, _toward_caster(geometry, outlines[part_id],
                                                           azimuth, reach))
            found[part_id] = list(kept.values())
        # Without the sun, a dark mask stays as found.

    dates = dict(zip(use.part_id, use.image_date)) if "image_date" in use else {}
    rows = []
    for part_id, masks in found.items():
        part_planes = parts[part_id]
        # Arrays are not weighed against the other prompts: a block of
        # panels is also a "roof window" or a "vent" to SAM 3, and often with
        # the better score.
        arrays = _sam3_arrays([m for m in masks if m[0] in cfg.sam3_extensive_prompts],
                              outlines[part_id], cfg)
        # Nor shadows, widened over their casters: SAM 3 often has the same
        # dark patch as a "vent" with the better score, and the fusion joins
        # the two.
        shadows = [m for m in masks if m[0] == "shadow"]
        others = [m for m in masks if m[0] not in cfg.sam3_extensive_prompts and m[0] != "shadow"]
        for name, score, geometry in (_sam3_unique(arrays, cfg.sam3_duplicate_iou)
                                      + _sam3_unique(shadows, cfg.sam3_duplicate_iou)
                                      + _sam3_unique(others, cfg.sam3_duplicate_iou)):
            overlap = [geometry.intersection(g).area for g in part_planes.geometry.values]
            date = dates.get(part_id)
            rows.append({
                "part_id": part_id, "plane_id": part_planes["plane_id"].values[int(np.argmax(overlap))],
                "source": "sam3", "method": "image", "sign": 0,
                "max_height_above_plane": np.nan, "mean_residual": np.nan, "n_points": 0,
                "area": float(geometry.area), "class": name, "score": float(score),
                "image_date": (pd.Timestamp(date).date().isoformat()
                               if date is not None and not pd.isna(date) else None),
                "geometry": geometry,
            })
    out = (gpd.GeoDataFrame(rows, columns=SAM3_COLUMNS + ["geometry"], geometry="geometry",
                            crs=crs) if rows else empty)
    out.attrs["sam3_windows"] = len(windows)
    out.attrs["sam3_inferred"] = n_run
    out.attrs["sam3_cast_shadows"] = n_shadows
    if sun is not None:
        out.attrs["sun"] = sun
    return out


def _sam3_windows(bounds, span, overlap, res):
    """Lower-left corners (on the res grid) of span-wide windows covering bounds."""
    minx, miny, maxx, maxy = bounds
    origins = []
    for lo, hi in ((minx, maxx), (miny, maxy)):
        extent = hi - lo
        if extent <= span:
            starts = [(lo + hi) / 2 - span / 2]
        else:
            n = int(np.ceil((extent - overlap) / (span - overlap)))
            step = (extent - span) / (n - 1) if n > 1 else 0.0
            starts = [lo + k * step for k in range(n)]
        origins.append([np.floor(s / res) * res for s in starts])
    return [(x, y) for y in origins[1] for x in origins[0]]


def _mask_polygon(mask):
    """A boolean mask as (multi)polygons in pixel coordinates (col, row)."""
    with warnings.catch_warnings():
        # Pixel coordinates are the point here.
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        pieces = [shape(geom) for geom, value in
                  shapes(mask.astype(np.uint8), mask=mask, connectivity=8) if value]
    return shapely.make_valid(unary_union(pieces)) if pieces else shapely.Polygon()


def _sam3_filter(geometry, pixels, outline, brightness, cfg, extensive=False):
    """
    (the mask clipped to the roof, whether it is mostly shadow), or None
    when it fails a filter; a shadow larger than
    sam3_shadow_obstacle_max_area_m2 fails.
    brightness: the window's mean RGB, 0-1, at the mask's pixels.
    """
    area = geometry.area
    if area < cfg.sam3_min_area_m2 or (not extensive and area > cfg.sam3_max_area_m2):
        return None
    on_roof = _overlay(shapely.intersection, geometry, outline)
    if on_roof.is_empty or on_roof.area < cfg.sam3_min_inside_fraction * area:
        return None
    max_fraction = (cfg.sam3_extensive_max_roof_fraction if extensive
                    else cfg.sam3_max_roof_fraction)
    if outline.area > 0 and on_roof.area > max_fraction * outline.area:
        return None
    shadow = cfg.sam3_shadow_brightness is not None and not extensive
    bright = extensive and cfg.sam3_extensive_max_brightness is not None
    dark = False
    if shadow or bright:
        inside = rasterize([pixels], out_shape=brightness.shape, fill=0, dtype="uint8",
                           transform=rasterio.Affine.identity()).astype(bool)
        values = brightness[inside]
        if values.size and shadow and \
                (values < cfg.sam3_shadow_brightness).mean() > cfg.sam3_max_shadow_fraction:
            if on_roof.area > cfg.sam3_shadow_obstacle_max_area_m2:
                return None
            dark = True
        if values.size and bright and \
                np.percentile(values, 25) > cfg.sam3_extensive_max_brightness:
            return None
    pieces = _polygons(shapely.make_valid(on_roof.simplify(cfg.sam3_simplify_m)))
    if not pieces:
        return None
    on_roof = pieces[0] if len(pieces) == 1 else shapely.MultiPolygon(pieces)
    return (on_roof, dark) if on_roof.area > 0 else None


def _sam3_arrays(masks, outline, cfg):
    """
    The extensive prompts' masks of one roof as arrays: closed over gaps up
    to sam3_array_close_m (between rows, between modules of one row),
    clipped to the roof, one entry per array under the first extensive
    prompt, with its best score.
    """
    extensive = [m for m in masks if m[0] in cfg.sam3_extensive_prompts]
    if not extensive or not cfg.sam3_array_close_m:
        return masks
    d = cfg.sam3_array_close_m / 2
    merged = unary_union([g for _, _, g in extensive])
    closed = merged.buffer(d, join_style="mitre").buffer(-d, join_style="mitre")
    closed = _overlay(shapely.intersection, unary_union([closed, merged]), outline)
    rows = [m for m in masks if m[0] not in cfg.sam3_extensive_prompts]
    for piece in _polygons(shapely.make_valid(closed)):
        scores = [score for _, score, g in extensive if g.intersects(piece)]
        if scores and piece.area > 0:
            rows.append((cfg.sam3_extensive_prompts[0], max(scores), piece))
    return rows


def _sam3_unique(masks, max_iou):
    """Highest score first; a mask overlapping a kept one beyond max_iou is dropped."""
    kept = []
    for name, score, geometry in sorted(masks, key=lambda m: -m[1]):
        duplicate = False
        for _, _, other in kept:
            if not geometry.intersects(other):
                continue
            inter = geometry.intersection(other).area
            union = geometry.area + other.area - inter
            if union > 0 and inter / union > max_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append((name, score, geometry))
    return kept


# ============================================================
# CAST SHADOWS ON THE ORTHOPHOTO
# ============================================================

def estimate_orthophoto_sun(ortho, coreg_log, dsm_path,
                            config: Optional[ObstacleConfig] = None, *, progress=False):
    """
    The sun the orthophoto was flown under, from its shadows.

    For each flight date of ``coreg_log`` (one for an older ICGC edition,
    one per sheet of the current one) up to ``sam3_sun_sample_roofs`` roofs
    are read with ``sam3_sun_pad_m`` around them, the orthophoto registered
    as the roof and the DSM on its grid. The sun is the position whose cast
    shadows on the DSM best correlate with the image's dark pixels: a grid
    of 15 degrees of azimuth and 5 of altitude, then 3 and 1 around the best.
    A date with fewer than a quarter of the sample takes the estimate of
    every roof together.

    Args:
        dsm_path: DSM with what casts the shadows (buildings and trees).

    Returns:
        DataFrame per roof of ``coreg_log``: ``part_id``, ``sun_azimuth``,
        ``sun_altitude`` (degrees; NaN when the best correlation is below
        ``sam3_sun_min_correlation``), ``sun_correlation``.
    """
    try:
        from .orthophoto import read_orthophoto, _cached_open
    except ImportError:
        from greenshadow.orthophoto import read_orthophoto, _cached_open

    cfg = config or ObstacleConfig()
    columns = ["part_id", "sun_azimuth", "sun_altitude", "sun_correlation"]
    if coreg_log is None or not len(coreg_log):
        return pd.DataFrame(columns=columns)
    use = coreg_log[coreg_log.use_imagery.astype(bool)]
    dsm = _cached_open(dsm_path)
    res = abs(dsm.transform.a)
    reach = min(cfg.sam3_shadow_max_distance_m, cfg.sam3_sun_pad_m)

    def windows(rows):
        out = []
        for row in rows.itertuples():
            image, transform = read_orthophoto(ortho, row.geometry.buffer(cfg.sam3_sun_pad_m).bounds,
                                               crs=coreg_log.crs, resolution=res,
                                               shift=(float(row.dx), float(row.dy)))
            z = _dsm_on_grid(dsm, transform, image.shape[1:], coreg_log.crs)
            brightness = np.nanmean(image[:3], axis=0) / 255 if image.shape[0] >= 3 else image[0] / 255
            valid = np.isfinite(brightness) & np.isfinite(z)
            if valid.mean() > 0.5:
                out.append((z, brightness, valid))
        return out

    def fit(sample):
        if not sample:
            return np.nan, np.nan, np.nan
        brightness = np.concatenate([b[v] for _, b, v in sample])
        cache = {}

        def horizons(azimuth):
            if azimuth not in cache:
                cache[azimuth] = np.concatenate([
                    _horizon(z, res, azimuth, cfg.sam3_shadow_min_height_m, reach)[v]
                    for z, _, v in sample])
            return cache[azimuth]

        def best(azimuths, altitudes):
            top = (-np.inf, np.nan, np.nan)
            for azimuth in azimuths:
                rise = horizons(float(azimuth % 360))
                for altitude in altitudes:
                    shade = rise > np.tan(np.radians(altitude))
                    if 0 < shade.mean() < 1:
                        r = -np.corrcoef(shade, brightness)[0, 1]
                        if r > top[0]:
                            top = (r, float(azimuth % 360), float(altitude))
            return top

        r, azimuth, altitude = best(np.arange(0, 360, 15), np.arange(15, 86, 5))
        if np.isfinite(azimuth):
            cache.clear()
            r, azimuth, altitude = best(np.arange(azimuth - 12, azimuth + 12.1, 3),
                                        np.arange(max(5, altitude - 4), min(89, altitude + 4) + 0.1, 1))
        if not r >= cfg.sam3_sun_min_correlation:
            return np.nan, np.nan, r
        return azimuth, altitude, r

    rng = np.random.default_rng(0)
    n = max(1, cfg.sam3_sun_sample_roofs)

    def sample_of(rows):
        if len(rows) > n:
            rows = rows.iloc[np.sort(rng.choice(len(rows), n, replace=False))]
        return rows

    # Flight dates as keys; undated roofs are one group.
    dates = (pd.to_datetime(use["image_date"]).dt.strftime("%Y-%m-%d").fillna("")
             if "image_date" in use else pd.Series("", index=use.index))
    groups = list(use.groupby(dates.values, sort=True))
    pooled = None
    estimates = {}
    bar = tqdm(groups, desc="Orthophoto sun", disable=not progress)
    for date, rows in bar:
        if len(rows) >= max(1, n // 4) or len(groups) == 1:
            estimates[date] = fit(windows(sample_of(rows)))
        else:
            if pooled is None:
                pooled = fit(windows(sample_of(use)))
            estimates[date] = pooled

    out = pd.DataFrame({"part_id": coreg_log.part_id.values})
    keys = dict(zip(use.part_id, dates.values))
    sun = [estimates.get(keys.get(p), (np.nan, np.nan, np.nan)) for p in out.part_id]
    out["sun_azimuth"] = [s[0] for s in sun]
    out["sun_altitude"] = [s[1] for s in sun]
    out["sun_correlation"] = [s[2] for s in sun]
    return out


def _elongation(geometry):
    """Length over width of the minimum rotated rectangle."""
    corners = np.asarray(geometry.minimum_rotated_rectangle.exterior.coords)
    if len(corners) < 4:
        return np.inf
    sides = np.hypot(*np.diff(corners[:3], axis=0).T)
    return float(sides.max() / max(sides.min(), 1e-9))


def _width_across(geometry, azimuth):
    """Extent of geometry across the direction of azimuth (m)."""
    a = np.radians(azimuth)
    xy = np.asarray(shapely.get_coordinates(geometry))
    across = xy[:, 0] * np.cos(a) - xy[:, 1] * np.sin(a)
    return float(np.ptp(across)) if len(across) else 0.0


def _toward_caster(geometry, outline, azimuth, distance):
    """A shadow widened toward the sun by distance (m) over its caster, on the roof."""
    from shapely import affinity
    a = np.radians(azimuth)
    moved = affinity.translate(geometry, distance * np.sin(a), distance * np.cos(a))
    grown = _overlay(shapely.intersection, unary_union([geometry, moved]).convex_hull, outline)
    pieces = _polygons(shapely.make_valid(grown))
    if not pieces:
        return geometry
    return pieces[0] if len(pieces) == 1 else shapely.MultiPolygon(pieces)


def _dsm_on_grid(dsm, transform, shape, crs):
    """The DSM on a north-up grid (nearest), NaN outside it and at nodata."""
    z = np.full(shape, np.nan, dtype=np.float32)
    reproject(source=rasterio.band(dsm, 1), destination=z,
              src_transform=dsm.transform, src_crs=dsm.crs, src_nodata=dsm.nodata,
              dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
              resampling=Resampling.nearest)
    return z


def _horizon(z, res, azimuth, min_height, max_distance, *, altitude=None):
    """
    Per pixel of z (north-up, res m), the steepest rise toward azimuth of
    what stands more than min_height above it within max_distance, as a
    tangent: in shade from a sun at altitude a where it exceeds tan(a).

    With ``altitude``: (shade, distance) instead, the pixels in shade and,
    for each, how far toward the sun the nearest caster stands (NaN in sun).
    """
    from scipy import ndimage as ndi
    rows, cols = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float32)
    ground = np.where(np.isfinite(z), z, -1e4).astype(np.float32)
    step_r, step_c = -np.cos(np.radians(azimuth)) / res, np.sin(np.radians(azimuth)) / res
    rise = np.full(z.shape, -np.inf, dtype=np.float32)
    tan = np.tan(np.radians(altitude)) if altitude is not None else None
    distance = np.full(z.shape, np.nan, dtype=np.float32) if altitude is not None else None
    for d in np.arange(res, max_distance + res / 2, res):
        # Nearest pixel: interpolating would lower the edges of a chimney
        # two pixels wide, and shorten its shadow.
        ahead = ndi.map_coordinates(ground, [rows + d * step_r, cols + d * step_c],
                                    order=0, mode="constant", cval=-1e4)
        step = (ahead - ground - min_height) / d
        if distance is not None:
            distance[(step > tan) & np.isnan(distance)] = d
        np.maximum(rise, step, out=rise)
    rise[~np.isfinite(z)] = -np.inf
    if distance is None:
        return rise
    distance[~np.isfinite(z)] = np.nan
    return np.isfinite(distance), distance


def _cast_shadows(dsm, outline, geometries, sun, crs, cfg):
    """
    How the DSM shades each geometry from sun = (azimuth, altitude): per
    geometry (fraction in shade, fraction of its shade cast from the roof
    itself, median distance toward the sun of the casters, m).
    """
    res = abs(dsm.transform.a)
    pad = cfg.sam3_shadow_max_distance_m
    minx, miny, maxx, maxy = outline.bounds
    minx, miny = np.floor((minx - pad) / res) * res, np.floor((miny - pad) / res) * res
    width = int(np.ceil((maxx + pad - minx) / res))
    height = int(np.ceil((maxy + pad - miny) / res))
    transform = from_origin(minx, miny + height * res, res, res)
    z = _dsm_on_grid(dsm, transform, (height, width), crs)
    if not np.isfinite(z).any():
        return [(0.0, 0.0, np.nan)] * len(geometries)
    tan = np.tan(np.radians(sun[1]))
    # No shadow reaches further than the relief over the sun's rise.
    reach = min(pad, max(res, (np.nanmax(z) - np.nanmin(z)) / max(tan, 1e-3)))
    shade, distance = _horizon(z, res, sun[0], cfg.sam3_shadow_min_height_m, reach,
                               altitude=sun[1])
    # Where each caster stands - half a pixel into it, past the edge the
    # ray first meets - and whether that is on the roof: within its outline,
    # holes filled (a chimney is often a hole in the planes), not on a
    # neighbour against it.
    rows, cols = np.nonzero(shade)
    d = distance[rows, cols] / res + 0.5
    caster_r = np.round(rows - d * np.cos(np.radians(sun[0]))).astype(int)
    caster_c = np.round(cols + d * np.sin(np.radians(sun[0]))).astype(int)
    filled = unary_union([shapely.Polygon(p.exterior) for p in _polygons(outline)])
    roof = rasterize([filled], out_shape=z.shape, transform=transform).astype(bool)
    on_roof = np.zeros(z.shape, dtype=bool)
    ok = (caster_r >= 0) & (caster_r < height) & (caster_c >= 0) & (caster_c < width)
    on_roof[rows[ok], cols[ok]] = roof[caster_r[ok], caster_c[ok]]
    out = []
    for geometry in geometries:
        inside = rasterize([geometry], out_shape=z.shape, transform=transform).astype(bool)
        if not inside.any():
            inside = rasterize([geometry], out_shape=z.shape, transform=transform,
                               all_touched=True).astype(bool)
        shaded = inside & shade
        if not shaded.any():
            out.append((0.0, 0.0, np.nan))
            continue
        out.append((float(shaded.sum() / inside.sum()), float(on_roof[shaded].mean()),
                    float(np.median(distance[shaded]))))
    return out


# ============================================================
# SHADOWS ON THE ORTHOPHOTO
# ============================================================

def orthophoto_suns(ortho, coreg_log, dsm_path=None, config: Optional[ObstacleConfig] = None, *,
                    progress=False):
    """
    The sun of the orthophoto at each roof, {part_id: (azimuth, altitude,
    correlation)}: ``coreg_log``'s ``sun_azimuth``, ``sun_altitude`` and
    ``sun_correlation`` when it has them (``estimate_orthophoto_sun``, as the
    pipeline stores it), else ``sam3_sun``, else estimated with ``dsm_path``;
    None without either.
    """
    cfg = config or ObstacleConfig()
    if coreg_log is None or not len(coreg_log):
        return {}
    if {"sun_azimuth", "sun_altitude"} <= set(coreg_log.columns):
        r = (coreg_log["sun_correlation"] if "sun_correlation" in coreg_log
             else pd.Series(np.nan, index=coreg_log.index))
        return {p: (float(a), float(h), float(c)) for p, a, h, c in
                zip(coreg_log.part_id, coreg_log.sun_azimuth, coreg_log.sun_altitude, r)}
    if cfg.sam3_sun is not None:
        return {p: (float(cfg.sam3_sun[0]), float(cfg.sam3_sun[1]), np.nan)
                for p in coreg_log.part_id}
    if not dsm_path:
        return None
    estimate = estimate_orthophoto_sun(ortho, coreg_log, dsm_path, cfg, progress=progress)
    return {p: (a, h, r) for p, a, h, r in
            zip(estimate.part_id, estimate.sun_azimuth, estimate.sun_altitude,
                estimate.sun_correlation)}


def detect_shadow_obstacles(planes, ortho, coreg_log, dsm_path=None,
                            config: Optional[ObstacleConfig] = None, *, progress=False):
    """
    Obstacles found by their shadows on the registered orthophoto.

    A vent pipe, a small chimney or an antenna may be too small for the
    LiDAR and unnamed by SAM 3, but it casts a shadow. Each plane of each
    roof ``coreg_log`` lets use the imagery is read undoing the roof's
    shift and shrunk by ``shadow_edge_m``; its pixels darker than
    ``shadow_dark_ratio`` of the plane's own brightness (its
    ``shadow_reference_percentile``) and not green are grouped. A group of
    ``shadow_min_area_m2`` - ``shadow_max_area_m2``, compact or elongated
    along the sun, is the shadow of something on the roof unless the DSM
    (``dsm_path``) casts it from off the roof. Its caster stands on its sun
    side: the obstacle is the shadow widened toward the sun as far as the
    DSM puts the caster, plus the shadow's width across the sun (a compact
    caster is about as deep), at least ``sam3_shadow_caster_m``. The
    caster's height is the shadow's length along the sun (less half its
    width across, for the caster's sides it wraps around) times
    tan(altitude) plus the plane's rise along the shadow.

    The sun is ``orthophoto_suns``'s; without one, nothing is found.

    Returns:
        GeoDataFrame, ``SAM3_COLUMNS``: ``source='shadow'``,
        ``method='image'``, ``class='shadow'``, ``score`` (the shadow's
        darkness, 1 - its brightness over the plane's),
        ``max_height_above_plane`` (the caster's estimated height),
        ``n_points`` (its pixels), ``image_date``. ``attrs['sun']`` as
        ``orthophoto_suns``.
    """
    cfg = config or ObstacleConfig()
    crs = planes.crs
    empty = gpd.GeoDataFrame({c: [] for c in SAM3_COLUMNS}, geometry=[], crs=crs)
    if coreg_log is None or not len(coreg_log):
        return empty
    use = coreg_log[coreg_log.use_imagery.astype(bool)]
    sun = orthophoto_suns(ortho, coreg_log, dsm_path, cfg, progress=progress) or {}
    parts = {pid: frame for pid, frame in planes.groupby("part_id", sort=False)}
    dates = dict(zip(use.part_id, use.image_date)) if "image_date" in use else {}
    jobs = []
    for row in use.itertuples():
        azimuth, altitude, _ = sun.get(row.part_id, (np.nan, np.nan, np.nan))
        if row.part_id in parts and np.isfinite(azimuth) and np.isfinite(altitude):
            date = dates.get(row.part_id)
            jobs.append((parts[row.part_id], (float(row.dx), float(row.dy)),
                         (float(azimuth), float(altitude)),
                         pd.Timestamp(date).date().isoformat()
                         if date is not None and not pd.isna(date) else None))
    if not jobs:
        if len(use):
            warning("No sun for the orthophoto's shadows; no shadow obstacles")
        empty.attrs["sun"] = sun
        return empty
    job = {"parts": jobs, "ortho": ortho, "dsm_path": dsm_path, "cfg": cfg}
    results = _fork_map(_shadow_part_worker, list(range(len(jobs))), job,
                        n_workers=cfg.n_workers, progress=progress,
                        desc="Roof obstacles (shadows)")
    rows = [row for part_rows in results for row in part_rows]
    out = (gpd.GeoDataFrame(rows, columns=SAM3_COLUMNS + ["geometry"], geometry="geometry",
                            crs=crs) if rows else empty)
    out.attrs["sun"] = sun
    return out


def _shadow_part_worker(k):
    from scipy import ndimage as ndi
    from shapely import affinity
    try:
        from .orthophoto import read_orthophoto, _cached_open
    except ImportError:
        from greenshadow.orthophoto import read_orthophoto, _cached_open
    job = _bi._FORK_JOB
    cfg = job["cfg"]
    part_planes, shift, (azimuth, altitude), date = job["parts"][k]
    part_id = part_planes["part_id"].iloc[0]
    crs = part_planes.crs
    outline = _roof_outline(part_planes.geometry.values, cfg.outline_close_m)

    image, transform = read_orthophoto(job["ortho"], outline.buffer(1.0).bounds, crs=crs,
                                       shift=shift)
    if image.shape[0] < 3:
        return []
    rgb = image[:3] / 255.0
    brightness = rgb.mean(axis=0)
    green = 2 * rgb[1] - rgb[0] - rgb[2]
    res = abs(transform.a)
    index = np.arange(1, len(part_planes) + 1)
    plane_raster = rasterize(zip(part_planes.geometry.values, index), out_shape=brightness.shape,
                             transform=transform, fill=0, dtype="int32")

    # Each plane against its own brightness: a north face is darker all over.
    # Within shadow_edge_m of a seam, against the darkest plane there: the
    # registration leaves a pixel or two of a dark plane on its brighter
    # neighbour, while a chimney's shadow on a ridge is darker than either.
    # Holes are read too (a chimney is often a hole in the planes, and its
    # shadow lies against it); the roof's outer edge is not.
    edge = max(1, int(round(cfg.shadow_edge_m / res)))
    references = np.full(brightness.shape, np.inf, dtype=np.float32)
    for i in index:
        own = (plane_raster == i) & np.isfinite(brightness)
        if own.sum() >= 8:
            references[own] = max(np.percentile(brightness[own], cfg.shadow_reference_percentile),
                                  1e-3)
    references = ndi.minimum_filter(references, size=2 * edge + 1)
    filled = unary_union([shapely.Polygon(p.exterior) for p in _polygons(outline)])
    roof = ndi.binary_erosion(
        rasterize([filled], out_shape=brightness.shape, transform=transform).astype(bool),
        iterations=edge)
    relative = np.where(roof & np.isfinite(references), brightness / references,
                        np.nan).astype(np.float32)
    # Hysteresis: a shadow is the pixels under shadow_grow_ratio connected
    # to one under shadow_dark_ratio, so that a long shade broken up by the
    # texture of the roof is still one long shade.
    weak = (relative < cfg.shadow_grow_ratio) & ~(green > cfg.shadow_max_green)
    labels, n = ndi.label(weak, structure=np.ones((3, 3)))
    if not n:
        return []
    seeded = np.zeros(n + 1, dtype=bool)
    seeded[np.unique(labels[weak & (relative < cfg.shadow_dark_ratio)])] = True
    seeded[0] = False
    labels[~seeded[labels]] = 0
    # Mostly away from the roof's outer edge: what the registration leaves
    # of the eave, the ground or a tree's crown beyond it.
    inner = filled.buffer(-cfg.shadow_min_inset_m) if cfg.shadow_min_inset_m > 0 else filled

    sun_axis = np.radians(azimuth)
    candidates = []
    for comp, window in enumerate(ndi.find_objects(labels), start=1):
        if window is None:
            continue
        blob = labels[window] == comp
        area = blob.sum() * res * res
        if not cfg.shadow_min_area_m2 <= area <= cfg.shadow_max_area_m2:
            continue
        r0, c0 = window[0].start, window[1].start
        polygon = unary_union([shape(g) for g, _ in shapes(
            blob.astype(np.uint8), mask=blob,
            transform=transform * transform.translation(c0, r0))])
        if polygon.intersection(inner).area < 0.5 * polygon.area:
            continue
        if _elongation(polygon) > cfg.shadow_max_elongation:
            # Long: a pole's shadow runs along the sun and is narrow; an
            # eave's or a seam's runs any way, a wall's is wider.
            corners = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)
            sides = np.diff(corners[:3], axis=0)
            long_side = sides[np.argmax(np.hypot(*sides.T))]
            bearing = np.arctan2(long_side[0], long_side[1])
            off = np.degrees(abs((bearing - sun_axis + np.pi / 2) % np.pi - np.pi / 2))
            if (off > cfg.shadow_along_sun_deg
                    or _width_across(polygon, azimuth) > cfg.shadow_pole_max_width_m):
                continue
        under = plane_raster[window][blob]
        under = under[under > 0]
        plane = (int(np.bincount(under).argmax()) - 1 if under.size else
                 int(np.argmin(part_planes.geometry.distance(polygon).values)))
        candidates.append((polygon, plane, int(blob.sum()),
                           float(1 - np.nanmean(relative[window][blob]))))
    if not candidates:
        return []

    # A shade the DSM casts from off the roof - a neighbour, a tree - is no
    # obstacle of this roof.
    if job["dsm_path"]:
        shades = _cast_shadows(_cached_open(job["dsm_path"]), outline,
                               [c[0] for c in candidates], (azimuth, altitude), crs, cfg)
    else:
        shades = [(0.0, 0.0, np.nan)] * len(candidates)

    rows = []
    tan_altitude = np.tan(np.radians(altitude))
    for (polygon, plane, pixels, darkness), (fraction, from_roof, distance) in zip(candidates,
                                                                                   shades):
        cast = fraction > 0.5
        # Shade from off the roof, or from far across it (a higher part of
        # the building), not from something standing by it.
        if cast and (from_roof < 0.5 or distance > cfg.shadow_max_caster_distance_m):
            continue
        width = _width_across(polygon, azimuth)
        reach = (max(cfg.sam3_shadow_caster_m, width)
                 + (distance if cast and np.isfinite(distance) else 0.0))
        # Height of the caster: the shadow's length along the sun over the
        # sun's descent per metre, plus what the plane rises (or minus what
        # it falls) along the shadow. Under an oblique sun the shadow wraps
        # around the caster's sides, which adds up to its depth to the
        # length; half its width across is taken off.
        length = max(0.0, _width_across(affinity.rotate(polygon, 90, origin="centroid"),
                                        azimuth) - width / 2)
        tilt = np.radians(float(part_planes["tilt_deg"].iloc[plane]))
        aspect = np.radians(float(part_planes["azimuth_deg"].iloc[plane]))
        rise = np.tan(tilt) * np.cos(sun_axis - aspect)
        geometry = _toward_caster(polygon, outline, azimuth, reach)
        rows.append({
            "part_id": part_id, "plane_id": part_planes["plane_id"].iloc[plane],
            "source": "shadow", "method": "image", "sign": 1,
            "max_height_above_plane": float(max(0.0, length * (tan_altitude + rise))),
            "mean_residual": np.nan, "n_points": pixels, "area": float(geometry.area),
            "class": "shadow", "score": darkness, "image_date": date, "geometry": geometry,
        })
    return rows


# ============================================================
# WALKABLE TERRACES
# ============================================================

TERRACE_COLUMNS = DETECTION_COLUMNS + ["class", "score", "guard_fraction", "finish_fraction"]

# HSV (hue in degrees) of the walkable finishes other than tile: lawn,
# artificial grass and planters; pool water.
_GARDEN_HSV = ((60.0, 170.0), 0.15, 0.15)
_WATER_HSV = ((170.0, 215.0), 0.30, 0.45)


def detect_terraces(planes, dsm_path, ortho=None, coreg_log=None,
                    config: Optional[ObstacleConfig] = None, *, progress=False):
    """
    Walkable terraces, balconies and patios: flat planes people use, which
    take no module.

    A plane flat to ``terrace_max_tilt_deg`` is a terrace when it is

    * enclosed: guarded (parapet, railing, the walls of higher floors)
      along ``terrace_enclosed_min_guard`` of its outline, and no larger
      than ``terrace_enclosed_max_area_m2``; or
    * tiled: guarded along ``terrace_tiled_min_guard``, and at least
      ``terrace_min_finish_fraction`` of it ceramic tile, garden or pool
      water on the registered orthophoto (``ortho`` with ``coreg_log``;
      without them only the enclosed test runs).

    The guard is measured on ``dsm_path`` (the DSM without vegetation: a
    tree is no railing) against the plane's median height.

    Returns:
        GeoDataFrame, ``TERRACE_COLUMNS``: the whole plane, ``source`` and
        ``class`` ``'terrace'``, ``method`` ``'enclosed'`` or ``'tiled'``.
        ``attrs["measures"]``: ``part_id``, ``plane_id``,
        ``terrace_guard``, ``terrace_finish``, ``terrace`` of every flat
        plane looked at.
    """
    cfg = config or ObstacleConfig()
    crs = planes.crs
    flat = planes[(planes["tilt_deg"] <= cfg.terrace_max_tilt_deg)
                  & (planes.geometry.area >= cfg.terrace_min_area_m2)]
    shifts = {}
    if ortho is not None and coreg_log is not None and len(coreg_log):
        use = coreg_log[coreg_log.use_imagery.astype(bool)]
        shifts = {p: (float(dx), float(dy)) for p, dx, dy in zip(use.part_id, use.dx, use.dy)}
    rows, measures = [], []
    if len(flat):
        job = {"parts": [frame for _, frame in flat.groupby("part_id", sort=False)],
               "dsm_path": dsm_path, "ortho": ortho if shifts else None, "shifts": shifts,
               "cfg": cfg}
        results = _fork_map(_terrace_part_worker, list(range(len(job["parts"]))), job,
                            n_workers=cfg.n_workers, progress=progress,
                            desc="Roof terraces")
        for part_rows, part_measures in results:
            rows.extend(part_rows)
            measures.extend(part_measures)
    if rows:
        out = gpd.GeoDataFrame(rows, columns=TERRACE_COLUMNS + ["geometry"],
                               geometry="geometry", crs=crs)
    else:
        out = gpd.GeoDataFrame({c: [] for c in TERRACE_COLUMNS}, geometry=[], crs=crs)
    out.attrs["measures"] = pd.DataFrame(
        measures, columns=["part_id", "plane_id", "terrace_guard", "terrace_finish", "terrace"])
    return out


_TERRACE_DSM = None


def _terrace_part_worker(k):
    global _TERRACE_DSM
    from scipy.ndimage import maximum_filter
    try:
        from .orthophoto import read_orthophoto
    except ImportError:
        from greenshadow.orthophoto import read_orthophoto
    job = _bi._FORK_JOB
    cfg = job["cfg"]
    part_planes = job["parts"][k]
    part_id = part_planes["part_id"].iloc[0]

    # One handle per process: GDAL handles do not survive fork.
    if _TERRACE_DSM is None or _TERRACE_DSM[0] != (os.getpid(), job["dsm_path"]):
        _TERRACE_DSM = ((os.getpid(), job["dsm_path"]), rasterio.open(job["dsm_path"]))
    src = _TERRACE_DSM[1]
    minx, miny, maxx, maxy = part_planes.total_bounds
    pad = cfg.terrace_guard_search_m + 1.0
    window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad,
                                          transform=src.transform)
    window = window.round_offsets().round_lengths()
    z = src.read(1, window=window, boundless=True, masked=True).astype(float).filled(np.nan)
    transform = src.window_transform(window)
    res = abs(transform.a)
    # Highest surface within the search distance of every pixel.
    reach = int(np.ceil(cfg.terrace_guard_search_m / res))
    z_max = maximum_filter(np.where(np.isfinite(z), z, -np.inf), size=2 * reach + 1)

    hsv = None
    shift = job["shifts"].get(part_id)
    if job["ortho"] is not None and shift is not None:
        image, ortho_transform = read_orthophoto(job["ortho"], (minx, miny, maxx, maxy),
                                                 crs=part_planes.crs, shift=shift)
        if image.shape[0] >= 3:
            hsv = _hsv(np.moveaxis(image[:3], 0, -1) / 255.0)

    rows, measures = [], []
    for plane_id, geometry in zip(part_planes["plane_id"].values, part_planes.geometry.values):
        guard = _terrace_guard(geometry, z, z_max, transform, cfg)
        if guard is None:
            continue
        finish = (_terrace_finish(geometry, hsv, ortho_transform, cfg)
                  if hsv is not None else np.nan)
        enclosed = (guard >= cfg.terrace_enclosed_min_guard
                    and geometry.area <= cfg.terrace_enclosed_max_area_m2)
        tiled = (np.isfinite(finish) and finish >= cfg.terrace_min_finish_fraction
                 and guard >= cfg.terrace_tiled_min_guard)
        measures.append((part_id, plane_id, guard, finish, bool(enclosed or tiled)))
        if not (enclosed or tiled):
            continue
        rows.append({
            "part_id": part_id, "plane_id": plane_id, "source": "terrace",
            "method": "tiled" if tiled else "enclosed", "sign": 0,
            "max_height_above_plane": np.nan, "mean_residual": np.nan, "n_points": 0,
            "area": float(geometry.area), "class": "terrace", "score": np.nan,
            "guard_fraction": guard, "finish_fraction": finish, "geometry": geometry,
        })
    return rows, measures


def _terrace_guard(geometry, z, z_max, transform, cfg):
    """Fraction of the plane's outline guarded, or None without DSM under it."""
    inner = geometry.buffer(-0.5)
    inside = rasterize([(inner if not inner.is_empty else geometry, 1)], out_shape=z.shape,
                       transform=transform, all_touched=True).astype(bool)
    level = z[inside]
    level = level[np.isfinite(level)]
    if not level.size:
        return None
    level = float(np.median(level))
    ring = max(_polygons(geometry), key=lambda g: g.area).exterior
    n = max(8, int(np.ceil(ring.length / abs(transform.a))))
    points = shapely.line_interpolate_point(ring, np.arange(n) / n, normalized=True)
    r, c = rowcol(transform, shapely.get_x(points), shapely.get_y(points))
    r = np.clip(np.asarray(r), 0, z.shape[0] - 1)
    c = np.clip(np.asarray(c), 0, z.shape[1] - 1)
    return float(np.mean(z_max[r, c] - level >= cfg.terrace_guard_height_m))


def _terrace_finish(geometry, hsv, transform, cfg):
    """Fraction of the plane (0.25 m in from its edge) a walkable finish."""
    hue, sat, val = hsv
    inner = geometry.buffer(-0.25)
    mask = rasterize([(inner if not inner.is_empty else geometry, 1)], out_shape=hue.shape,
                     transform=transform).astype(bool) & np.isfinite(val)
    if not mask.any():
        return np.nan
    h, s, v = hue[mask], sat[mask], val[mask]
    lo, hi = cfg.terrace_tile_hue_deg
    walkable = (h >= lo) & (h <= hi) & (s >= cfg.terrace_tile_min_saturation) \
        & (v >= cfg.terrace_tile_min_value)
    for (h0, h1), s0, v0 in (_GARDEN_HSV, _WATER_HSV):
        walkable |= (h >= h0) & (h <= h1) & (s >= s0) & (v >= v0)
    return float(walkable.mean())


def _hsv(rgb):
    """Hue (degrees), saturation and value of an (..., 3) RGB array in 0-1;
    NaN where a band is."""
    top = rgb.max(axis=-1)
    span = top - rgb.min(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sat = np.where(top > 0, span / top, 0.0)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        hue = np.where(top == r, ((g - b) / span) % 6,
                       np.where(top == g, (b - r) / span + 2, (r - g) / span + 4))
    hue = np.where(span > 0, hue * 60.0, 0.0)
    invalid = ~np.isfinite(rgb).all(axis=-1)
    return (np.where(invalid, np.nan, hue), np.where(invalid, np.nan, sat),
            np.where(invalid, np.nan, top))


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

    opened at half ``min_fragment_width_m`` (on the outlines simplified by
    ``fragment_simplify_m``) and stripped of pieces under
    ``min_fragment_area_m2`` (true area), so what is left can hold a module.

    Parameters
    ----------
    planes : GeoDataFrame
        The roof planes the detections were made on.
    detections : sequence of GeoDataFrame
        Outputs of ``detect_lidar_obstacles``, ``detect_svf_obstacles``,
        ``detect_sam3_obstacles``, ``detect_shadow_obstacles``.
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
        ``n_points``, ``area``, ``classes`` and ``score`` (SAM 3's). ``usable_area``: one row per plane with
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
                                "max_height_above_plane", "n_points", "area", "classes",
                                "score", "geometry"],
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
            # What SAM 3 called it, and its best score.
            "classes": (",".join(sorted(set(members["class"].dropna())))
                        if "class" in members else ""),
            "score": (float(members["score"].max())
                      if "score" in members and members["score"].notna().any() else np.nan),
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
            smooth = (usable.simplify(cfg.fragment_simplify_m, preserve_topology=True)
                      if cfg.fragment_simplify_m > 0 else usable)
            opened = smooth.buffer(-width / 2, join_style="mitre").buffer(
                width / 2, join_style="mitre")
            # The simplified rings may touch themselves, and the mitred
            # buffers of them come out invalid.
            opened = _drop_small_pieces(shapely.make_valid(opened), 0.0)
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
