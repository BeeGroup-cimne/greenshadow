# -*- coding: utf-8 -*-
"""
Roof plane inference: from a DSM and LiDAR returns to vector roof facets.

Everything the roof pipeline needs lives here. There is one detector and
one regularizer, and they are meant to be used in that order:

``detect_roof_planes``
    The raster detector. It reads a DSM - optionally rebuilt at a finer
    resolution from the LiDAR returns, with the returns denoised over their
    Delaunay graph first - segments each building's pixels into planes by
    normal clustering plus RANSAC, hands every pixel to a plane, merges the
    planes that are one surface, and polygonizes the result. Given a sky
    view factor raster it also cuts the roof's clutter - chimneys, aerials,
    air-conditioning units - out of the facets.

``regularize_roof_planes``
    Turns either detector's pixel-staircase facets into straight-edged
    ones. Its default "wireframe" method rebuilds each roof of touching
    facets as one planar wireframe - a simplified outline, junction nodes
    shared by the facets meeting there, and one straight edge per ridge or
    hip, placed on the line where the two planes intersect.

The point-native detector that used to live here, ``detect_roof_planes_v2``
- Delaunay-graph segmentation of the returns with no rasterization at all -
has been removed, along with the face-polygon stage it wrapped
(``delaunay_canopy_roof_planes``): it kept crisper creases but stopped at
every thin patch, and the raster detector was the one every municipality
run used.

What survives of that work is the part the raster detector depends on. The
Delaunay graph is still built, still scored for creases
(``delaunay_corner_scores``) and still segmented into planes, because
``smooth_points_to_planes`` needs the segmentation: ``delaunay_smooth_strength``
denoises each return toward the plane of its own segment, which is how the
noise goes without the creases going with it.

See ``docs/roof_inference.md`` for how the pipeline works end to end, what
each parameter does, and what the accuracy measurements say.
"""

from typing import List, Dict
import inspect
import os
import sys
import warnings
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import (sobel, label, gaussian_filter, generate_binary_structure, binary_dilation,
                           binary_closing, binary_erosion, binary_fill_holes, convolve,
                           distance_transform_edt, find_objects)
try:
    from .logging_utils import tqdm
except ImportError:
    from greenshadow.logging_utils import tqdm
try:
    from .lidar import _read_laz
    from .dsm_dem_estimation import _aligned_grid, build_digital_model_from_lidar_pure
    from .utils import _write_array_to_tiff, _filter_by_class_priority_grid
    from .logging_utils import info
except ImportError:
    from greenshadow.lidar import _read_laz
    from greenshadow.dsm_dem_estimation import _aligned_grid, build_digital_model_from_lidar_pure
    from greenshadow.utils import _write_array_to_tiff, _filter_by_class_priority_grid
    from greenshadow.logging_utils import info
from rasterio.features import shapes, rasterize
from rasterio.transform import rowcol, from_origin
from shapely.geometry import shape, Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from rasterio.warp import reproject
from rasterio.enums import Resampling
from shapely import contains_xy
from shapely.strtree import STRtree
from numba import njit, prange
from skimage.morphology import skeletonize
import numpy as np
from collections import defaultdict

# ==========================================================
# Public API
# ==========================================================

def detect_roof_planes(
    dsm_path,
    *,
    slope_path,
    aspect_path,
    dsm_class_path,
    dem_path=None,
    building_class_value=(6,77),
    min_height_m=1.5,
    normal_tolerance_deg=15,
    ransac_threshold=0.15,
    smooth_sigma=0.0,
    min_roof_plane_area=5,
    dsm_max_slope_filter=0.4,
    buffer_m=1,
    test_window=None,
    output_path=None,
    lidar_refinement=True,
    lidar_class_values=(1,2,6,8,17,77),
    lidar_class_priority=(6,77,17,2,8,1),
    lidar_class_window_diameter=0.5,
    laz_dir=None,
    laz_tiles=None,
    resolution_refined=0.25,
    preclustering_roof_planes=False,
    ridge_splitting_roof_planes=False,
    # Delaunay smoothing of the LiDAR points before the refined DSM is built
    delaunay_smooth_strength=0.0,
    delaunay_normal_tolerance_deg=15.0,
    delaunay_plane_offset_tolerance=0.2,
    delaunay_min_plane_points=30,
    refined_aggregation="smart_for_roofs",
    assignment_normal_weight=1.0,
    merge_normal_angle_deg=2.5,
    merge_max_offset_m=0.3,
    recover_distance_m=0.0,
    recover_max_tilt_deg=60.0,
    coplanar_merge_angle_deg=0.0,
    coplanar_merge_step_m=0.15,
    skirt_assign_distance_m=0.0,
    # Chimneys, aerials and the rest of the roof clutter, cut out of the facets
    svf_path=None,
    svf_obstruction_drop=0.12,
    svf_obstruction_min_height_m=0.0,
    svf_obstruction_min_area_m2=0.5,
    svf_obstruction_max_area_m2=15.0,
    svf_obstruction_max_elongation=2.5,
    svf_obstruction_max_planes=1,
    svf_obstruction_close_m=0.6,
    svf_obstruction_buffer_m=0.0,
    # Tiling: see detect_roof_planes_tiled
    building_owner_mask=None,
    refined_output_paths=None,
    n_workers=1,
    debug_mode=False
):
    """
    Detect roof planes on the DSM grid, optionally on Delaunay-smoothed points.

    ``delaunay_smooth_strength`` denoises the LiDAR returns before they are
    rasterized into the refined roof DSM this detector reads. Unlike
    ``smooth_sigma``, which gaussian-blurs the DSM afterwards and rounds off
    every ridge with it, the points are first segmented into planes over
    their Delaunay graph and each point is then pulled only toward the plane
    of its own segment - so the noise goes and the creases stay. 0 disables
    it and nothing changes; 0.7 is a good starting point; 1.0 flattens every
    facet exactly onto its fitted plane.

    It needs the LiDAR branch: ``lidar_refinement`` with ``laz_dir`` and
    ``laz_tiles`` set. Without points there is nothing to smooth, and a
    warning is issued.

    ``refined_aggregation`` chooses how those points become the refined DSM.
    The default bins them per pixel, which leaves every pixel with no return
    empty for the gap filler to invent - at a resolution finer than the point
    spacing that is most of them. ``"tin"`` instead samples each pixel centre
    off the Delaunay triangulation of the returns, so the surface is
    interpolated along measured triangles rather than guessed afterwards.
    The two pair up: on raw returns a triangle's gradient is set by three
    noisy corners, but on smoothed ones the triangles within a roof face are
    near-coplanar and the surface inherits clean gradients.

    ``assignment_normal_weight`` controls how much a pixel's own
    orientation counts when it is handed to a plane. Distance alone cannot
    tell two facets apart where they meet - both planes pass through the
    surface there - so the boundary wanders and facets merge. 0 restores
    the old distance-only behaviour; 1 weighs a full ``normal_tolerance_deg``
    of angular disagreement the same as a full ``ransac_threshold`` of
    vertical error.

    ``recover_distance_m`` recovers roof the region growing left behind.
    Pixels inside a building that no plane claimed are first grown into
    the neighbouring plane they fit within that vertical distance; patches
    still unclaimed and at least ``min_roof_plane_area`` large then get a
    plane of their own, if one fits most of the patch and is no steeper
    than ``recover_max_tilt_deg`` (steeper is a wall, not a roof). On the
    digitised Alpicat roofs 7% of the real roof area was inside the
    building mask but in no plane, often a whole facet. 0 disables it.

    ``coplanar_merge_angle_deg`` merges, as the very last step, touching
    planes that are one surface split in two: normals within that angle and
    no height step larger than ``coplanar_merge_step_m`` along the boundary
    they share. The earlier graph merge misses these - one ridge pixel on
    the boundary vetoes it, and it compares heights at the pair's centroid,
    where a small slope difference is extrapolated into a false step. Pairs
    merge closest-first and the plane is refitted after each merge, so a
    chain of slightly different planes cannot creep into one. 0 disables it.

    ``skirt_assign_distance_m`` keeps the roof's skirt. Along the building
    edge the refined DSM drops steeply over its last one or two pixels -
    the triangulation reaching down from the eave to returns on the facade -
    so those pixels fit no plane, stay unassigned, and leave the outline
    ragged and short of the real eave. Measured on the digitised Alpicat
    roofs they sit at a median 60 deg slope, 0.2 m from the nearest plane.
    As the very last step, every building pixel still in no plane and
    within this distance of one is given to the nearest plane. Planes are
    not refitted with them, and the planarity metrics leave them out, so
    only the footprint changes. 0 disables it.

    ``svf_path`` cuts the roof's clutter out of the facets. Chimneys,
    aerials, lift overruns and air-conditioning units sit on a roof plane
    but are not roof: a panel cannot go there, and the plane polygon should
    say so. Height above the fitted plane alone does not identify them - at
    0.25 m a noisy return, a raised ridge tile and a real chimney all clear
    the same threshold - so the sky view factor is asked to confirm. An
    object standing on a roof steals sky from the roof around its base,
    which is a signature nothing flat produces: the SVF raster shows a ring
    of depressed values around every chimney in it, while the chimney's own
    top, seeing the whole sky, does not stand out at all. So the ring is
    what is searched for, and the object is recovered as what the ring
    encloses.

    Give it the raster ``calculate_sky_view_factor`` writes, in this CRS;
    it is resampled onto whichever grid the detection runs on, refined or
    not. Leave it None and nothing changes.

    A pixel joins a candidate when its SVF sits ``svf_obstruction_drop``
    below the median of its own plane - per plane, because a north facet
    and a south facet do not share a baseline - and, once the ring is
    closed over what it encloses, when the DSM is more than
    ``svf_obstruction_min_height_m`` above the fitted plane. Both
    conditions are needed: SVF alone also flags the foot of a taller
    neighbour and the shaded side of a ridge, which are real occlusions but
    still roof, and the height test drops them because they lie on their
    plane. Set the height to 0 to cut on SVF alone.

    Candidates smaller than ``svf_obstruction_min_area_m2`` are noise;
    larger than ``svf_obstruction_max_area_m2`` they are a structure the
    detector should have planed in its own right, not a hole, and are left
    alone.

    Two shape tests then do most of the work, because the sky view factor
    turned out not to discriminate on real roofs. Measured over the Alpicat
    test window, 65% of what the SVF and height tests agreed on was a thin
    sliver along a ridge, an eave or a step between two roof levels, and 87%
    of it sat on a boundary between facets. That is not a failure of the SVF
    reading: a step between two roof levels really does shadow the roof
    below it, printing the same depressed ring a chimney prints. Sky view
    cannot tell a small isolated object from a long linear one, so:

    ``svf_obstruction_max_elongation`` drops candidates whose pixel cloud is
    longer than this many times its width - a chimney is roughly as wide as
    it is long, a ridge artefact is not. ``svf_obstruction_max_planes``
    drops candidates with more than this many distinct planes around them -
    a chimney stands in the middle of one roof face, while a ridge or a step
    has a different face on either side. Together they take Alpicat from 100
    candidates, of which 5 were chimney-shaped, to the 5. Raise either to
    loosen; ``svf_obstruction_max_planes=0`` disables that test.

    ``svf_obstruction_buffer_m`` widens each hole, for the strip of
    roof beside a chimney that is technically clear but unusable.

    The holes become interior rings of the plane polygon, so ``area`` and
    ``footprint_area`` exclude them and the PV layout in ``pv`` skips them
    without being told. ``n_obstructions`` and ``obstruction_area`` report
    what was removed per facet.

    ``building_owner_mask`` and ``refined_output_paths`` are what
    ``detect_roof_planes_tiled`` runs each tile with. The owner mask is a
    boolean array on the ``test_window`` grid: building pixels outside it
    belong to another tile and are dropped, both on the input grid and on
    the refined one. ``refined_output_paths`` is (dsm, slope, aspect,
    class) for the refined rasters, which otherwise go next to the inputs
    with a ``_roofs`` suffix.

    ``n_workers`` buildings are smoothed, and roof parts detected, at once
    in forked processes; the result does not depend on it.

    See ``smooth_points_to_planes`` below for the smoother, and
    the remaining parameters' behaviour is unchanged.
    """
    # print(inspect.getsource(_read_laz))
    # =====================================================
    # Set paths from DSMs when detection is LIDAR refined
    # =====================================================

    lidar_dsm_path = None
    lidar_slope_path = None
    lidar_aspect_path = None
    lidar_classification_path = None

    if lidar_refinement and refined_output_paths is not None:
        (lidar_dsm_path, lidar_slope_path,
         lidar_aspect_path, lidar_classification_path) = refined_output_paths
    elif lidar_refinement:
        lidar_dsm_path = dsm_path.replace(".tif", "_roofs.tif")
        lidar_slope_path = slope_path.replace(".tif", "_roofs.tif")
        lidar_aspect_path = aspect_path.replace(".tif", "_roofs.tif")
        lidar_classification_path = dsm_class_path.replace(".tif", "_roofs.tif")

    # =====================================================
    # Load DSM + slope + aspect
    # =====================================================

    with rasterio.open(dsm_path) as d_src:

        if test_window:
            r0, r1, c0, c1 = test_window
            window = Window(c0, r0, c1 - c0, r1 - r0)
            dsm = d_src.read(1, window=window).astype(np.float32)
            transform = d_src.window_transform(window)
        else:
            window = None
            dsm = d_src.read(1).astype(np.float32)
            transform = d_src.transform
        crs = d_src.crs

    with rasterio.open(slope_path) as s_src:
        slope = s_src.read(1, window=window)

    with rasterio.open(aspect_path) as a_src:
        aspect = a_src.read(1, window=window)

    with rasterio.open(dsm_class_path) as c_src:
        dsm_class = c_src.read(1, window=window)

    buildings_mask = np.isin(dsm_class, building_class_value)

    if dem_path:
        with rasterio.open(dem_path) as dem_src:
            dem = dem_src.read(1, window=window).astype(np.float32)
        buildings_mask &= (dsm - dem > min_height_m)

    if building_owner_mask is not None:
        building_owner_mask = np.asarray(building_owner_mask, dtype=bool)
        if building_owner_mask.shape != buildings_mask.shape:
            raise ValueError(
                f"building_owner_mask is {building_owner_mask.shape}, "
                f"the DSM window is {buildings_mask.shape}"
            )
        owner_transform = transform
        buildings_mask &= building_owner_mask

    # =====================================================
    # Optional LIDAR refinement
    # =====================================================
    if delaunay_smooth_strength > 0 and not (
            lidar_refinement
            and laz_dir is not None
            and laz_tiles is not None
    ):
        warnings.warn(
            "delaunay_smooth_strength > 0 has no effect without the LiDAR "
            "branch: it smooths the returns before they are rasterized, so "
            "it needs lidar_refinement=True with laz_dir and laz_tiles set.",
            stacklevel=2
        )

    if (
            lidar_refinement
            and laz_dir is not None
            and laz_tiles is not None
    ):
        building_geom = _building_mask_to_buffered_geometry(
            buildings_mask,
            transform,
            epsg=crs.to_epsg(),
            buffer_m=1.0,
        )

        # ---------------------------------------------
        # 1️⃣ Define refined grid from current DSM extent
        # ---------------------------------------------
        minx = transform.c
        maxx = transform.c + dsm.shape[1] * transform.a
        maxy = transform.f
        miny = transform.f + dsm.shape[0] * transform.e

        width_ref, height_ref, transform_ref = _aligned_grid(
            minx,
            miny,
            maxx,
            maxy,
            resolution_refined
        )

        # ---------------------------------------------
        # 2️⃣ Resample DEM to refined grid (if exists)
        # ---------------------------------------------
        if dem_path:

            dem_ref = np.full(
                (height_ref, width_ref),
                -9999.0,
                dtype=np.float32
            )

            with rasterio.open(dem_path) as dem_src:
                reproject(
                    source=rasterio.band(dem_src, 1),
                    destination=dem_ref,
                    src_transform=dem_src.transform,
                    src_crs=dem_src.crs,
                    dst_transform=transform_ref,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                )

            dem = dem_ref

        # ---------------------------------------------
        # 3️⃣ Extract LiDAR points
        # ---------------------------------------------
        building_pts_struct = _get_building_lidar_points(
            dem=dem,
            min_height_m=min_height_m,
            transform=transform_ref,
            laz_dir=laz_dir,
            tiles=laz_tiles,
            aoi=building_geom,
            lidar_allowed_classes=lidar_class_values
        )

        # float64, not float32: a float32 northing near 4.6e6 m is rounded
        # to 0.5 m, which put the returns on a lattice of rows 2.5 refined
        # pixels apart. The lattice's co-circular points leave the Delaunay
        # triangulation behind the TIN without a unique answer, so the
        # surface changed with whatever other points were triangulated
        # alongside - up to metres at single pixels.
        if building_pts_struct is None:
            building_pts = np.empty((0, 4), dtype=np.float64)
        else:
            building_pts = np.column_stack([
                building_pts_struct["x"],
                building_pts_struct["y"],
                building_pts_struct["z"],
                building_pts_struct["classification"],
            ]).astype(np.float64)

        building_pts_geom = _filter_by_class_priority_grid(
            building_pts,
            lidar_class_priority,
            lidar_class_window_diameter
        )

        building_geom = _lidar_points_to_building_geom(
            building_pts_geom,
            transform_ref,
            width_ref,
            height_ref,
            buffer_m=0.0
        )

        # ---------------------------------------------
        # Delaunay smoothing, before the points become a DSM
        # ---------------------------------------------
        # Done here rather than on the finished raster because once the
        # returns are averaged into pixels the crease and the noise are
        # indistinguishable. One building at a time: the triangulation
        # behind the segmentation fills the convex volume of whatever it is
        # handed, so a whole window at once would both crawl and build
        # edges from one roof across to the next.
        if delaunay_smooth_strength > 0 and len(building_pts) > 0:
            building_labels, _ = label(
                buildings_mask,
                structure=generate_binary_structure(2, 1)
            )

            point_labels = _points_to_building_labels(
                building_pts,
                building_labels,
                transform
            )

            building_pts = smooth_roof_points_by_building(
                building_pts,
                point_labels,
                strength=delaunay_smooth_strength,
                normal_tolerance_deg=delaunay_normal_tolerance_deg,
                plane_offset_tolerance=delaunay_plane_offset_tolerance,
                min_plane_points=delaunay_min_plane_points,
                progress=True,
                n_workers=n_workers
            )
        
        #gpd.GeoDataFrame({"geometry":[building_geom]}, crs=f"EPSG:{crs.to_epsg()}").to_file("test.gpkg",driver="GPKG")
        if building_pts is not None and len(building_pts) > 0:

            # ---------------------------------------------
            # 2️⃣ Build refined DSM from in-memory points
            # ---------------------------------------------
            dsm_ref, class_ref, slope_ref, aspect_ref, transform_ref = \
                build_digital_model_from_lidar_pure(
                    laz_points=building_pts,
                    out_path=lidar_dsm_path,
                    out_slope_path=lidar_slope_path,
                    out_aspect_path=lidar_aspect_path,
                    out_classification_path=lidar_classification_path,
                    resolution=resolution_refined,
                    bounds=(
                        transform_ref.c,
                        transform_ref.f + height_ref * transform_ref.e,
                        transform_ref.c + width_ref * transform_ref.a,
                        transform_ref.f,
                    ),
                    epsg=crs.to_epsg(),
                    aoi=building_geom,
                    aoi_epsg=crs.to_epsg(),
                    aggregation=refined_aggregation,
                    classification=lidar_class_values,
                    vegetation_classes=(),
                    terrain_classes=(),
                    building_classes=lidar_class_values,
                    fill_gaps="smart_for_roofs",
                    lidar_class_priority=lidar_class_priority,
                    lidar_class_window_diameter=lidar_class_window_diameter,
                    return_arrays=True,
                    cache_rasterization=False
                )

            # ---------------------------------------------
            # 4️⃣ Replace base rasters
            # ---------------------------------------------
            dsm = dsm_ref
            slope = slope_ref
            aspect = aspect_ref
            dsm_class = class_ref
            transform = transform_ref

            # Re-align DEM onto the refined DSM's actual grid. The pure model
            # derives its own grid from bounds+resolution, which can differ by
            # ±1 px from the _aligned_grid prediction used above for filtering.
            if dem_path:
                dem = np.full(dsm.shape, -9999.0, dtype=np.float32)
                with rasterio.open(dem_path) as dem_src:
                    reproject(
                        source=rasterio.band(dem_src, 1),
                        destination=dem,
                        src_transform=dem_src.transform,
                        src_crs=dem_src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.bilinear,
                    )

            buildings_mask = np.isin(dsm_class, building_class_value)
            if dem_path:
                buildings_mask &= (dsm - dem > min_height_m)

            if building_owner_mask is not None:
                owner_ref = np.zeros(dsm.shape, dtype=np.uint8)
                reproject(
                    source=building_owner_mask.astype(np.uint8),
                    destination=owner_ref,
                    src_transform=owner_transform,
                    src_crs=crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest,
                )
                buildings_mask &= owner_ref.astype(bool)

    # Set the number of minimum pixels per roof, using the digital models resolution and the minimum roof plane area.
    min_pixels = np.ceil(min_roof_plane_area / (abs(transform[0]) ** 2))

    # =====================================================
    # Sky view factor, on the grid detection actually runs on
    # =====================================================
    # Done here and not earlier: the LiDAR branch above replaces dsm and
    # transform with the refined grid, and the SVF has to follow whichever
    # of the two survived. Nodata is carried as NaN so the resampling
    # cannot smear -9999 into the roof edge.

    svf = None

    if svf_path:
        svf = np.full(dsm.shape, np.nan, dtype=np.float32)

        with rasterio.open(svf_path) as svf_src:
            reproject(
                source=rasterio.band(svf_src, 1),
                destination=svf,
                src_transform=svf_src.transform,
                src_crs=svf_src.crs,
                src_nodata=svf_src.nodata,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

        # A valid SVF is a fraction of visible sky; anything outside that is
        # nodata that survived, and must not become a plane's baseline.
        svf[(svf < 0.0) | (svf > 1.0)] = np.nan

    # =====================================================
    # Smooth DSM (only inside building)
    # =====================================================

    if smooth_sigma > 0:
        tmp = dsm.copy()
        tmp[~buildings_mask] = np.nan
        tmp = gaussian_filter(
            np.nan_to_num(tmp, nan=np.nanmean(tmp)),
            smooth_sigma
        )

        dsm[buildings_mask] = tmp[buildings_mask]

        # Recompute slope and aspect from smoothed DSM
        slope[:] = -9999
        aspect[:] = -9999

        _compute_sobel_slope_aspect(
            dsm,
            slope,
            aspect,
            nodata=-9999,
            resolution=abs(transform.a)
        )

    # Filter the extreme slopes
    buildings_mask &= ((slope <= dsm_max_slope_filter) & (slope >= 0))

    # =====================================================
    # Convert slope/aspect → normals (radians)
    # =====================================================

    nx = np.sin(slope) * np.sin(aspect)
    ny = np.sin(slope) * np.cos(aspect)
    nz = np.cos(slope)

    # =====================================================
    # Analysis building by building
    # =====================================================

    # Initial labeling
    s = generate_binary_structure(2, 1)
    labeled_buildings, n_buildings = label(buildings_mask, structure=s)

    # Remove small components
    counts = np.bincount(labeled_buildings.ravel())

    remove = counts < min_pixels
    remove[0] = False  # keep background

    labeled_buildings[remove[labeled_buildings]] = 0

    # Relabel to force 1..N consecutive labels
    labeled_buildings, n_buildings = label(labeled_buildings > 0, structure=s)
    if debug_mode:
        _write_array_to_tiff(
            array=labeled_buildings,
            out_path=output_path.replace(".gpkg","_continuous_areas.tif"),
            transform=transform,
            crs=crs,
            nodata=-9999.0,
            dtype="float32"
        )

    # Detect the roof planes
    planes_list = []
    ridge_mask = np.zeros(dsm.shape, dtype=np.uint8)
    clustering_mask = np.zeros(dsm.shape, dtype=np.uint32)
    obstruction_mask = np.zeros(dsm.shape, dtype=bool)
    cluster_global_id = 1

    # Each part's bounding box, found once: comparing the whole raster
    # against every label in turn grows with parts x pixels.
    part_slices = find_objects(labeled_buildings)

    # Parts are independent, so they are spread over forked processes,
    # largest first so a big block is not the last one left running. Results
    # are gathered in part order: the output does not depend on n_workers.
    sizes = np.bincount(labeled_buildings.ravel(), minlength=n_buildings + 1)
    order = sorted(range(1, n_buildings + 1), key=lambda k: -sizes[k])

    frame = locals()
    job = {name: frame[name] for name in _PART_JOB_NAMES}
    results = _fork_map(
        _detect_part_planes, order, job, n_workers=n_workers, progress=True,
        desc="Processing building roof parts (only continuous)", chunksize=1
    )
    by_part = dict(zip(order, results))

    for part_id in range(1, n_buildings + 1):
        result = by_part[part_id]
        if result is None:
            continue
        gdf_planes, ridge, clusters, n_clusters, holes = result
        row_slice, col_slice = part_slices[part_id - 1]

        if debug_mode:
            ridge_mask[row_slice, col_slice] |= ridge
            inside = clusters > 0
            clustering_mask[row_slice, col_slice][inside] = (
                clusters[inside] + (cluster_global_id - 1)
            )
            if holes is not None:
                obstruction_mask[row_slice, col_slice] |= holes
        cluster_global_id += n_clusters

        planes_list.append(gdf_planes)

    if planes_list:
        gdf_all_planes = pd.concat(planes_list, ignore_index=True)
    else:
        gdf_all_planes = gpd.GeoDataFrame(geometry=[], crs=crs)

    if debug_mode:
    
        _write_array_to_tiff(
            array=ridge_mask,
            out_path=output_path.replace(".gpkg", f"_ridge_mask.tif"),
            transform=transform,
            crs=crs,
            nodata=0,
            dtype="uint8"
        )
    
        _write_array_to_tiff(
            array=clustering_mask,
            out_path=output_path.replace(".gpkg", "_clusters.tif"),
            transform=transform,
            crs=crs,
            nodata=0,
            dtype="uint32"
        )

        if svf is not None:
            _write_array_to_tiff(
                array=obstruction_mask.astype(np.uint8),
                out_path=output_path.replace(".gpkg", "_obstructions.tif"),
                transform=transform,
                crs=crs,
                nodata=0,
                dtype="uint8"
            )

    if output_path:
        gdf_all_planes.to_file(output_path)

    return gdf_all_planes


def detect_roof_planes_tiled(
    dsm_path,
    *,
    slope_path,
    aspect_path,
    dsm_class_path,
    dem_path=None,
    output_path=None,
    test_window=None,
    tile_size_m=1000.0,
    cluster_gap_m=2.0,
    n_workers=None,
    worker_ram_gb=6.0,
    tiles_dir=None,
    keep_tiles=False,
    aoi=None,
    aoi_crs="EPSG:4326",
    **detect_kwargs
):
    """
    ``detect_roof_planes`` over a whole municipality, tile by tile and in
    parallel, in bounded memory.

    Run on a municipality in one go the detector holds the refined grid of
    the whole extent - at 0.20 m, Alpicat's 7 x 8 km is 1.4 billion pixels
    per array, with a dozen arrays - and runs out of memory. Here the extent
    is cut into tiles of ``tile_size_m`` and each tile is a separate call of
    the detector, in a process of its own, so a run needs the memory of a
    few tiles however large the municipality is.

    Tiles never cut a roof. The buildings are first grouped into clusters -
    building pixels closer than about ``cluster_gap_m`` belong to the same
    one - and every cluster goes, whole, to the tile its centre falls in.
    That tile's window is stretched to cover its clusters, and each tile is
    told which building pixels are its own (``building_owner_mask``), so a
    building straddling two tiles is detected once, complete, and the
    windows may overlap without anything being counted twice. A cluster is
    a closed set: every roof part the detector could form, refined LiDAR
    footprint included, lies inside one. The planes therefore come out the
    same as from one untiled call; only ``part_id`` is renumbered, to stay
    unique across tiles, and a ``tile_id`` column says where each was found.

    ``cluster_gap_m`` must exceed the 1 m the detector buffers buildings by
    when it collects their LiDAR returns; the 2 m default leaves margin. A
    contiguous built-up block is one cluster however large, and its tile's
    window grows with it.

    Each tile runs in a fresh process, which hands all its memory back when
    the tile is done. At most ``n_workers`` run at once (default: as many as
    the free memory allows at ``worker_ram_gb`` each, capped at the CPU
    count), and a new one only starts while ``worker_ram_gb`` is still free.
    A tile whose process is killed - the kernel's OOM killer - is retried
    alone once the others are done. On platforms without ``fork`` the tiles
    run one after the other in this process.

    The result of every finished tile is kept in ``tiles_dir`` (default:
    ``<output_path stem>_tiles``) together with its log, under a key derived
    from the parameters. A run that dies part-way can simply be repeated:
    tiles already done are read back, not recomputed. Once every tile is
    done the results are merged, the refined rasters are mosaicked into the
    ``*_roofs.tif`` files the untiled detector writes, and the tile files
    are deleted unless ``keep_tiles``.

    ``aoi`` (a shapely geometry in ``aoi_crs``) keeps only the buildings
    that touch it - typically the municipality itself, while the rasters
    cover it plus a buffer for the shadows. Clusters clear of it are never
    detected, and of the roofs found, those (``part_id``) with no facet
    touching it are dropped; a roof straddling the boundary is kept whole.

    ``test_window`` (row0, row1, col0, col1) restricts the run to that part
    of the DSM, as in ``detect_roof_planes``. Every other keyword is passed
    on to ``detect_roof_planes`` unchanged.
    """
    import hashlib
    import json
    import shutil
    import tempfile
    import time

    for reserved in ("building_owner_mask", "refined_output_paths"):
        if reserved in detect_kwargs:
            raise TypeError(f"{reserved} is set per tile by detect_roof_planes_tiled")
    detect_kwargs.pop("debug_mode", None)

    defaults = {
        name: p.default
        for name, p in inspect.signature(detect_roof_planes).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    params = {**defaults, **detect_kwargs}
    # How many processes a tile uses does not change what it finds.
    params.pop("n_workers", None)

    # -----------------------------------------------------
    # Building clusters, on a coarse grid of cluster_gap_m
    # -----------------------------------------------------
    with rasterio.open(dsm_path) as src:
        full_transform = src.transform
        crs = src.crs
        res = abs(full_transform.a)
        if test_window:
            r_off, r_end, c_off, c_end = test_window
        else:
            r_off, r_end, c_off, c_end = 0, src.height, 0, src.width

    height, width = r_end - r_off, c_end - c_off
    factor = max(1, int(round(cluster_gap_m / res)))
    clusters, n_clusters = _building_clusters(
        dsm_path,
        dsm_class_path,
        dem_path,
        window=(r_off, c_off, height, width),
        factor=factor,
        building_class_value=params["building_class_value"],
        min_height_m=params["min_height_m"],
    )

    aoi_geom = None
    if aoi is not None:
        aoi_geom = _aoi_to_crs(aoi, aoi_crs, crs)
        # Every building pixel lies in a cell of its cluster, so a building
        # touching the AOI has a cluster cell that does too.
        coarse_transform = (
            full_transform
            * rasterio.Affine.translation(c_off, r_off)
            * rasterio.Affine.scale(factor)
        )
        inside = rasterize(
            [(aoi_geom, 1)], out_shape=clusters.shape,
            transform=coarse_transform, fill=0, all_touched=True, dtype="uint8",
        ).astype(bool)
        touching = np.unique(clusters[inside & (clusters > 0)])
        info(f"{len(touching)} of {n_clusters} building clusters touch the AOI")
        clusters, n_clusters = label(
            np.isin(clusters, touching), structure=np.ones((3, 3), dtype=bool)
        )

    if n_clusters == 0:
        warnings.warn("No building pixels found: no roof planes to detect.", stacklevel=2)
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # -----------------------------------------------------
    # Each cluster to the tile its centre falls in
    # -----------------------------------------------------
    tile_clusters = defaultdict(list)
    tile_extent = {}

    for cluster_id, slc in enumerate(find_objects(clusters), start=1):
        if slc is None:
            continue
        rs, cs = slc
        row = r_off + (rs.start + rs.stop) / 2 * factor
        col = c_off + (cs.start + cs.stop) / 2 * factor
        x, y = full_transform * (col, row)
        key = (int(np.floor(x / tile_size_m)), int(np.floor(y / tile_size_m)))
        tile_clusters[key].append(cluster_id)

        box = (rs.start, rs.stop, cs.start, cs.stop)
        if key in tile_extent:
            old = tile_extent[key]
            box = (min(old[0], box[0]), max(old[1], box[1]),
                   min(old[2], box[2]), max(old[3], box[3]))
        tile_extent[key] = box

    # -----------------------------------------------------
    # Cache keyed by the parameters
    # -----------------------------------------------------
    fingerprint = repr((
        os.path.abspath(dsm_path), os.path.abspath(slope_path),
        os.path.abspath(aspect_path), os.path.abspath(dsm_class_path),
        os.path.abspath(dem_path) if dem_path else None,
        (r_off, r_end, c_off, c_end), tile_size_m, cluster_gap_m,
        sorted((k, repr(v)) for k, v in params.items()),
        None if aoi_geom is None else hashlib.sha1(aoi_geom.wkb).hexdigest(),
    ))
    run_key = hashlib.sha1(fingerprint.encode()).hexdigest()[:10]

    own_tiles_dir = tiles_dir is None
    if tiles_dir is None:
        tiles_dir = (
            os.path.splitext(output_path)[0] + "_tiles"
            if output_path else tempfile.mkdtemp(prefix="roof_tiles_")
        )
    os.makedirs(tiles_dir, exist_ok=True)

    refine = bool(
        params["lidar_refinement"]
        and params["laz_dir"] is not None
        and params["laz_tiles"] is not None
    )

    tasks = []
    for key, ids in tile_clusters.items():
        cr0, cr1, cc0, cc1 = tile_extent[key]
        r0 = r_off + cr0 * factor
        r1 = min(r_end, r_off + cr1 * factor)
        c0 = c_off + cc0 * factor
        c1 = min(c_end, c_off + cc1 * factor)
        owned = np.isin(clusters[cr0:cr1, cc0:cc1], ids)
        name = f"{run_key}_{key[0]}_{key[1]}"
        base = os.path.join(tiles_dir, name)
        tasks.append({
            "tile_id": f"{key[0]}_{key[1]}",
            "window": (r0, r1, c0, c1),
            "owned": owned,
            "factor": factor,
            "result_path": base + ".parquet",
            "status_path": base + ".json",
            "log_path": base + ".log",
            "refined_paths": tuple(
                f"{base}_{kind}.tif" for kind in ("dsm", "slope", "aspect", "class")
            ) if refine else None,
        })

    # Largest windows first, so the last tiles to finish are small ones.
    tasks.sort(key=lambda t: -(t["window"][1] - t["window"][0]) * (t["window"][3] - t["window"][2]))

    common = {
        "dsm_path": dsm_path,
        "slope_path": slope_path,
        "aspect_path": aspect_path,
        "dsm_class_path": dsm_class_path,
        "dem_path": dem_path,
        "detect_kwargs": detect_kwargs,
    }

    todo = [t for t in tasks if not os.path.exists(t["result_path"])]
    largest = max(
        (t["window"][1] - t["window"][0]) * (t["window"][3] - t["window"][2]) * res * res
        for t in tasks
    )
    info(
        f"{n_clusters} building clusters in {len(tasks)} tiles of {tile_size_m:.0f} m "
        f"(largest window {largest / 1e6:.2f} km2); "
        f"{len(tasks) - len(todo)} already done in {tiles_dir}"
    )

    # -----------------------------------------------------
    # Run the tiles
    # -----------------------------------------------------
    import multiprocessing as mp
    import psutil

    can_fork = sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods()
    cpus = os.cpu_count() or 1

    if n_workers is None:
        free_gb = _memory_gb()[0]
        n_workers = max(1, min(cpus, int(free_gb // worker_ram_gb)))
    n_workers = max(1, min(n_workers, len(todo) or 1))

    started = time.time()

    if todo and not can_fork:
        for task in tqdm(todo, desc="Roof tiles", unit="tile"):
            _run_roof_tile(task, common, threads=cpus)
    elif todo:
        failed = _run_roof_tiles_parallel(todo, common, n_workers, worker_ram_gb, cpus)

        if failed:
            lines = "\n".join(
                f"  tile {t['tile_id']} ({'out of memory' if code == 'memory' else f'exit code {code}'}): "
                f"see {t['log_path']}"
                for t, code in failed
            )
            raise RuntimeError(
                f"{len(failed)} of {len(tasks)} roof tiles failed:\n{lines}\n"
                "Finished tiles are kept; running again only recomputes the failed ones."
            )

    peaks = []
    for task in tasks:
        if os.path.exists(task["status_path"]):
            with open(task["status_path"]) as f:
                peaks.append(json.load(f).get("max_rss_gb", 0.0))
    if todo:
        info(
            f"{len(todo)} tiles in {time.time() - started:.0f} s with {n_workers} workers; "
            f"peak memory per tile {max(peaks, default=0):.1f} GB"
        )

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------
    frames = []
    for task in sorted(tasks, key=lambda t: t["tile_id"]):
        gdf = gpd.read_parquet(task["result_path"])
        if len(gdf):
            frames.append(gdf)

    if frames:
        gdf_all_planes = pd.concat(frames, ignore_index=True)
        gdf_all_planes = gpd.GeoDataFrame(gdf_all_planes, geometry="geometry", crs=crs)
        if aoi_geom is not None:
            gdf_all_planes = filter_roof_planes_by_aoi(
                gdf_all_planes, aoi_geom, aoi_crs=crs, part_columns=("tile_id", "part_id")
            )
        gdf_all_planes["part_id"] = (
            gdf_all_planes.groupby(["tile_id", "part_id"], sort=True).ngroup() + 1
        )
    else:
        gdf_all_planes = gpd.GeoDataFrame(geometry=[], crs=crs)

    if refine:
        region = rasterio.windows.bounds(
            Window(c_off, r_off, width, height), full_transform
        )
        mosaic_targets = (
            dsm_path.replace(".tif", "_roofs.tif"),
            slope_path.replace(".tif", "_roofs.tif"),
            aspect_path.replace(".tif", "_roofs.tif"),
            dsm_class_path.replace(".tif", "_roofs.tif"),
        )
        for k, out in enumerate(mosaic_targets):
            parts = [
                t["refined_paths"][k] for t in tasks
                if os.path.exists(t["refined_paths"][k])
            ]
            if parts:
                _mosaic_aligned_rasters(parts, out, region, params["resolution_refined"])

    if output_path:
        gdf_all_planes.to_file(output_path)

    if not keep_tiles:
        for task in tasks:
            for path in (task["result_path"], task["status_path"], task["log_path"],
                         *(task["refined_paths"] or ())):
                if os.path.exists(path):
                    os.remove(path)
        if own_tiles_dir and not os.listdir(tiles_dir):
            shutil.rmtree(tiles_dir, ignore_errors=True)

    return gdf_all_planes


def _aoi_to_crs(aoi, aoi_crs, crs):
    """A shapely geometry, or a GeoDataFrame/GeoSeries, as one geometry in ``crs``."""
    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return aoi.to_crs(crs).union_all()
    return gpd.GeoSeries([aoi], crs=aoi_crs).to_crs(crs).iloc[0]


def filter_roof_planes_by_aoi(roof_planes, aoi, *, aoi_crs="EPSG:4326",
                              part_columns=("part_id",)):
    """
    Keep the roofs that touch ``aoi``, whole.

    A roof is the facets sharing ``part_columns`` (the detector's
    ``part_id``, one per contiguous block of building pixels). It is kept
    when any of its facets intersects the AOI, so a building on the
    boundary keeps the facets that fall outside it. Works on the detector's
    output and on the regularized one, which carries ``part_id`` through.
    """
    roof_planes = _load_plane_frame(roof_planes)
    if len(roof_planes) == 0:
        return roof_planes

    aoi_geom = _aoi_to_crs(aoi, aoi_crs, roof_planes.crs)
    keys = list(part_columns)
    touching = roof_planes.intersects(aoi_geom)
    kept = roof_planes.loc[touching, keys].drop_duplicates()
    mask = (
        roof_planes[keys].merge(kept, on=keys, how="left", indicator=True)["_merge"]
        .eq("both").to_numpy()
    )
    return roof_planes[mask].reset_index(drop=True)


def _building_clusters(
    dsm_path, dsm_class_path, dem_path, *, window, factor,
    building_class_value, min_height_m, strip_rows=2048
):
    """
    Label the building clusters on a grid ``factor`` times coarser than the DSM.

    A coarse cell is occupied when any DSM pixel in it is building - the same
    test ``detect_roof_planes`` starts from - and every occupied cell claims
    its eight neighbours. Clusters are the 8-connected components of what is
    claimed, so two clusters never come closer than one coarse cell, and
    everything within one cell of a building pixel belongs to its cluster.
    Read in strips: the full-resolution mask is never held.
    """
    r_off, c_off, height, width = window
    hc = -(-height // factor)
    wc = -(-width // factor)
    occupied = np.zeros((hc, wc), dtype=bool)
    strip = factor * max(1, strip_rows // factor)

    with rasterio.open(dsm_path) as d_src, rasterio.open(dsm_class_path) as c_src:
        dem_src = rasterio.open(dem_path) if dem_path else None
        try:
            for r in range(0, height, strip):
                h = min(strip, height - r)
                win = Window(c_off, r_off + r, width, h)
                mask = np.isin(c_src.read(1, window=win), building_class_value)
                if dem_src is not None:
                    dsm = d_src.read(1, window=win).astype(np.float32)
                    dem = dem_src.read(1, window=win).astype(np.float32)
                    mask &= (dsm - dem > min_height_m)
                    del dsm, dem

                hp = -(-h // factor) * factor
                padded = np.zeros((hp, wc * factor), dtype=bool)
                padded[:h, :width] = mask
                rc = r // factor
                occupied[rc:rc + hp // factor] |= padded.reshape(
                    hp // factor, factor, wc, factor
                ).any(axis=(1, 3))
        finally:
            if dem_src is not None:
                dem_src.close()

    eight = np.ones((3, 3), dtype=bool)
    claimed = binary_dilation(occupied, structure=eight)
    clusters, n_clusters = label(claimed, structure=eight)

    return clusters, n_clusters


def _detect_part_planes(part_id):
    """
    One roof part's planes for ``detect_roof_planes``, as ``(gdf_planes,
    ridge, clusters, n_clusters, holes)`` - the last four only feed the
    debug rasters - or None when the part is empty.

    Everything else comes from ``_FORK_JOB``, so forked workers inherit
    the rasters instead of having them pickled per part.
    """
    job = _FORK_JOB
    labeled_buildings = job["labeled_buildings"]
    part_slices = job["part_slices"]
    dsm = job["dsm"]
    nx = job["nx"]
    ny = job["ny"]
    nz = job["nz"]
    svf = job["svf"]
    transform = job["transform"]
    crs = job["crs"]
    min_pixels = job["min_pixels"]
    ridge_splitting_roof_planes = job["ridge_splitting_roof_planes"]
    preclustering_roof_planes = job["preclustering_roof_planes"]
    normal_tolerance_deg = job["normal_tolerance_deg"]
    smooth_sigma = job["smooth_sigma"]
    ransac_threshold = job["ransac_threshold"]
    buffer_m = job["buffer_m"]
    assignment_normal_weight = job["assignment_normal_weight"]
    recover_distance_m = job["recover_distance_m"]
    recover_max_tilt_deg = job["recover_max_tilt_deg"]
    merge_normal_angle_deg = job["merge_normal_angle_deg"]
    merge_max_offset_m = job["merge_max_offset_m"]
    coplanar_merge_angle_deg = job["coplanar_merge_angle_deg"]
    coplanar_merge_step_m = job["coplanar_merge_step_m"]
    skirt_assign_distance_m = job["skirt_assign_distance_m"]
    svf_obstruction_drop = job["svf_obstruction_drop"]
    svf_obstruction_min_height_m = job["svf_obstruction_min_height_m"]
    svf_obstruction_min_area_m2 = job["svf_obstruction_min_area_m2"]
    svf_obstruction_max_area_m2 = job["svf_obstruction_max_area_m2"]
    svf_obstruction_max_elongation = job["svf_obstruction_max_elongation"]
    svf_obstruction_max_planes = job["svf_obstruction_max_planes"]
    svf_obstruction_close_m = job["svf_obstruction_close_m"]
    svf_obstruction_buffer_m = job["svf_obstruction_buffer_m"]
    debug_mode = job["debug_mode"]


    if part_slices[part_id - 1] is None:
        return None

    # --------------------------------------------
    # Bounding box
    # --------------------------------------------
    row_slice, col_slice = part_slices[part_id - 1]
    min_row, min_col = row_slice.start, col_slice.start

    local_mask = labeled_buildings[row_slice, col_slice] == part_id
    local_dsm = dsm[row_slice, col_slice]
    local_nx = nx[row_slice, col_slice]
    local_ny = ny[row_slice, col_slice]
    local_nz = nz[row_slice, col_slice]
    local_svf = None if svf is None else svf[row_slice, col_slice]
    local_transform = transform * rasterio.Affine.translation(min_col, min_row)
    local_clusters = np.zeros(local_mask.shape, dtype=np.uint32) if debug_mode else None
    n_local_clusters = 0

    # --------------------------------------------
    # Roof plane detection
    # --------------------------------------------
    # Roof segmentation
    if ridge_splitting_roof_planes:

        local_ridge_mask, normal_clusters_init = _ridge_mask_from_normal_clustering(
            local_nx,
            local_ny,
            local_nz,
            local_mask,
            eps_deg=normal_tolerance_deg,
            smooth_sigma=smooth_sigma
        )
        structure = generate_binary_structure(2, 2)
        local_ridge_mask = binary_dilation(local_ridge_mask, structure=structure)
        structure = generate_binary_structure(2, 1)
        mask_eroded = binary_erosion(local_mask, structure=structure)
        mask_contour = local_mask & (~mask_eroded)
        local_ridge_mask |= mask_contour
        local_ridge_mask &= local_mask
        local_ridge_mask = skeletonize(local_ridge_mask)
        # segmentation = connected components without ridges
        split_mask = local_mask & (~local_ridge_mask)
        segments, n_segments = label(split_mask, structure=structure)
    else:
        segments = np.zeros_like(local_mask, dtype=np.int32)
        segments[local_mask] = 1
        n_segments = 1
        local_ridge_mask = np.zeros_like(local_mask, dtype=np.uint8)

    ridge_snapshot = local_ridge_mask.astype(np.uint8) if debug_mode else None
    building_planes_dsm = []

    for sid in range(1, n_segments + 1):

        segment_mask = segments == sid

        if segment_mask.sum() < min_pixels:
            continue

        # ----------------------------------------
        # OPTIONAL normal clustering
        # ----------------------------------------
        if preclustering_roof_planes:

            normal_clusters = _cluster_normals(
                local_nx,
                local_ny,
                local_nz,
                segment_mask,
                ridge_mask=local_ridge_mask,
                eps_deg=normal_tolerance_deg,
                smooth_sigma=smooth_sigma
            )

            # filter debris clusters
            normal_clusters = _filter_normal_clusters(
                normal_clusters,
                segment_mask,
                min_pixels
            )
            rejected = (normal_clusters < 0) & segment_mask
            local_ridge_mask |= rejected
            cluster_ids = np.unique(normal_clusters)

        else:

            normal_clusters = np.full(local_mask.shape, -1, dtype=np.int32)
            normal_clusters[segment_mask] = 0
            cluster_ids = [0]

        valid_clusters = np.unique(normal_clusters)

        for cid in valid_clusters:

            if cid < 0:
                continue

            cluster_pixels = (normal_clusters == cid) & segment_mask

            if not np.any(cluster_pixels):
                continue

            # Numbered within the part; the caller offsets them into the
            # global debug raster.
            if debug_mode:
                local_clusters[cluster_pixels] = n_local_clusters + 1
            n_local_clusters += 1

        planes = []

        for cid in cluster_ids:

            if cid < 0:
                continue

            cluster_mask = normal_clusters == cid

            if cluster_mask.sum() < min_pixels:
                continue

            cluster_planes = _detect_planes_from_dsm(
                building_mask=cluster_mask,
                nx=local_nx,
                ny=local_ny,
                nz=local_nz,
                dsm=local_dsm,
                transform=local_transform,
                normal_tolerance_deg=normal_tolerance_deg,
                ransac_threshold=ransac_threshold,
                min_pixels=min_pixels,
            )

            planes.extend(cluster_planes)

        building_planes_dsm.extend(planes)
        
    polygons_gdf = _dsm_planes_to_buffered_polygons(
        building_planes_dsm,
        local_transform,
        buffer_m=buffer_m
    )
    
    plane_id_raster = _assign_pixels_to_planes_3d(
        local_mask,
        local_dsm,
        building_planes_dsm,
        polygons_gdf,
        local_transform,
        local_ridge_mask,
        nx=local_nx,
        ny=local_ny,
        nz=local_nz,
        distance_tolerance=ransac_threshold,
        normal_tolerance_deg=normal_tolerance_deg,
        normal_weight=assignment_normal_weight
    )
    
    # Enforce continuity
    plane_id_raster = _enforce_plane_continuity(
        plane_id_raster,
        local_mask
    )

    updated_planes = _refit_planes_from_assignment(
        plane_id_raster,
        local_dsm,
        local_transform
    )

    plane_id_raster = _reassign_unassigned_pixels_local(
        plane_id_raster,
        local_mask,
        local_dsm,
        updated_planes,
        local_transform,
        ridge_mask=local_ridge_mask
,
        max_search_radius=1
    )

    updated_planes = _refit_planes_from_assignment(
        plane_id_raster,
        local_dsm,
        local_transform
    )

    if recover_distance_m and recover_distance_m > 0:
        plane_id_raster = _recover_unassigned_roof(
            plane_id_raster,
            local_mask,
            local_dsm,
            updated_planes,
            local_transform,
            distance_tolerance=recover_distance_m,
            min_pixels=min_pixels,
            max_tilt_deg=recover_max_tilt_deg
        )

        updated_planes = _refit_planes_from_assignment(
            plane_id_raster,
            local_dsm,
            local_transform
        )

    # Graph-based planes merging
    updated_planes, plane_id_raster = _merge_similar_adjacent_planes_graph(
        plane_id_raster,
        updated_planes,
        local_dsm,
        local_transform,
        normal_angle_threshold_deg=merge_normal_angle_deg,
        ridge_mask=local_ridge_mask,
        max_offset_m=merge_max_offset_m
    )

    # Remove small planes and reassign them
    plane_id_raster = _remove_small_planes(
        plane_id_raster,
        min_pixels=min_pixels
    )

    updated_planes = _refit_planes_from_assignment(
        plane_id_raster,
        local_dsm,
        local_transform
    )

    plane_id_raster = _reassign_unassigned_pixels_local(
        plane_id_raster,
        local_mask,
        local_dsm,
        updated_planes,
        local_transform,
        ridge_mask=local_ridge_mask
,
        max_search_radius=1
    )

    updated_planes = _refit_planes_from_assignment(
        plane_id_raster,
        local_dsm,
        local_transform
    )

    if coplanar_merge_angle_deg and coplanar_merge_angle_deg > 0:
        updated_planes, plane_id_raster = _merge_coplanar_planes(
            plane_id_raster,
            updated_planes,
            local_dsm,
            local_transform,
            max_angle_deg=coplanar_merge_angle_deg,
            max_step_m=coplanar_merge_step_m
        )

    fit_mask = None

    if skirt_assign_distance_m and skirt_assign_distance_m > 0:
        fit_mask = plane_id_raster > 0
        plane_id_raster = _assign_roof_skirts(
            plane_id_raster,
            local_mask,
            max_distance_px=skirt_assign_distance_m / abs(local_transform.a)
        )

    # Chimneys, aerials and the rest of the clutter standing on the
    # finished facets. Last, so it sees the planes the polygons will
    # actually carry - skirts included, since clutter sits on those too.
    hole_mask = None

    if local_svf is not None:
        hole_mask = _detect_roof_obstructions(
            plane_id_raster,
            updated_planes,
            local_dsm,
            local_svf,
            local_transform,
            svf_drop=svf_obstruction_drop,
            min_height_m=svf_obstruction_min_height_m,
            min_area_m2=svf_obstruction_min_area_m2,
            max_area_m2=svf_obstruction_max_area_m2,
            max_elongation=svf_obstruction_max_elongation,
            max_planes=svf_obstruction_max_planes,
            close_m=svf_obstruction_close_m,
            buffer_m=svf_obstruction_buffer_m
        )


    # Roof planes vectorisation
    gdf_planes = _polygonize_and_compute_parameters(
        plane_id_raster,
        updated_planes,
        local_dsm,
        local_transform,
        part_id,
        crs,
        simplify_tolerance=None,
        fit_mask=fit_mask,
        hole_mask=hole_mask
    )


    return (
        gdf_planes,
        ridge_snapshot,
        local_clusters,
        n_local_clusters,
        hole_mask if debug_mode else None,
    )


# What _detect_part_planes reads from detect_roof_planes' frame.
_PART_JOB_NAMES = (
    "labeled_buildings", "part_slices", "dsm", "nx", "ny", "nz", "svf",
    "transform", "crs", "min_pixels", "ridge_splitting_roof_planes",
    "preclustering_roof_planes", "normal_tolerance_deg", "smooth_sigma",
    "ransac_threshold", "buffer_m", "assignment_normal_weight",
    "recover_distance_m", "recover_max_tilt_deg", "merge_normal_angle_deg",
    "merge_max_offset_m", "coplanar_merge_angle_deg", "coplanar_merge_step_m",
    "skirt_assign_distance_m", "svf_obstruction_drop",
    "svf_obstruction_min_height_m", "svf_obstruction_min_area_m2",
    "svf_obstruction_max_area_m2", "svf_obstruction_max_elongation",
    "svf_obstruction_max_planes", "svf_obstruction_close_m",
    "svf_obstruction_buffer_m", "debug_mode",
)


def _memory_gb():
    """
    (available, total) memory in GB for this process: the machine's, or
    its cgroup's when that is tighter - a container, a Slurm job or a
    systemd scope with a memory limit is killed at its own limit, however
    much the machine has free.
    """
    import psutil

    vm = psutil.virtual_memory()
    available, total = float(vm.available), float(vm.total)

    try:
        with open("/proc/self/cgroup") as f:
            path = next((line[3:].strip() for line in f if line.startswith("0::")), None)
    except OSError:
        path = None

    # cgroup v2: every ancestor's limit applies, so take the tightest.
    while path is not None:
        base = os.path.join("/sys/fs/cgroup", path.lstrip("/"))
        try:
            with open(os.path.join(base, "memory.max")) as f:
                limit = f.read().strip()
            if limit != "max":
                with open(os.path.join(base, "memory.current")) as f:
                    used = float(f.read())
                # Page cache the kernel can drop is not in the way.
                with open(os.path.join(base, "memory.stat")) as f:
                    for line in f:
                        if line.startswith("inactive_file "):
                            used -= float(line.split()[1])
                            break
                available = min(available, float(limit) - used)
                total = min(total, float(limit))
        except (OSError, ValueError):
            pass
        if path in ("", "/"):
            break
        path = os.path.dirname(path)

    gb = 1024 ** 3
    return available / gb, total / gb


def _run_roof_tiles_parallel(tasks, common, n_workers, worker_ram_gb, cpus,
                             reserve_gb=None):
    """
    One forked process per tile, at most ``n_workers`` at a time, never
    letting the machine run out of memory.

    Processes rather than a pool: a pool whose worker dies loses the task
    and waits for it forever, while here a dead process is just an exit
    code - and a fresh process per tile hands all its memory back.

    Memory is watched, not just checked at the start: a tile grows long
    after it is launched. A new tile starts only while the free memory,
    less what the running tiles may still grow into (``worker_ram_gb`` each,
    minus what they already hold), leaves room for it. And should free
    memory fall under ``reserve_gb`` (default: 10% of RAM, at least 4 GB)
    anyway, the most recently started tile is killed before the kernel's
    OOM killer picks something - the desktop session, say - and is put back
    to run later on its own. A tile that crosses the reserve while running
    alone needs more memory than the machine has; it is reported as failed.

    Returns the (task, exit code) pairs that did not finish; exit code
    "memory" marks a tile stopped for lack of memory while alone.
    """
    import itertools
    import multiprocessing as mp
    import signal
    from multiprocessing.connection import wait
    import psutil

    gb = 1024 ** 3
    total_gb = _memory_gb()[1]
    if reserve_gb is None:
        reserve_gb = max(4.0, 0.10 * total_gb)

    ctx = mp.get_context("fork")
    counter = itertools.count()
    pending = list(tasks)
    deferred = []
    running = {}       # sentinel -> (proc, task)
    stopped = set()    # pids killed by the watchdog
    failed = []

    def rss_gb(proc):
        try:
            return psutil.Process(proc.pid).memory_info().rss / gb
        except psutil.Error:
            return 0.0

    def kill_tree(proc):
        # The tile's own workers first: killed alone they would be orphans
        # waiting forever on a queue nobody writes to, holding their memory.
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except psutil.Error:
            children = []
        for p in [*children, proc]:
            try:
                os.kill(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def can_start(limit):
        if len(running) >= limit:
            return False
        if not running:
            return True
        free = _memory_gb()[0]
        growth = sum(max(0.0, worker_ram_gb - rss_gb(p)) for p, _ in running.values())
        return free - growth - worker_ram_gb > reserve_gb

    try:
        with tqdm(total=len(tasks), desc="Roof tiles", unit="tile") as bar:
            while pending or deferred or running:

                # Deferred tiles go once the shared phase is over, one at a time.
                queue, limit = (pending, n_workers) if pending else (deferred, 1)
                while queue and can_start(limit):
                    task = queue.pop(0)
                    threads = max(1, cpus // max(1, min(limit, len(queue) + len(running) + 1)))
                    # Not daemonic: a tile forks its own workers per building,
                    # and daemonic processes may not have children. The
                    # finally below stops them instead if this loop dies.
                    proc = ctx.Process(
                        target=_roof_tile_process,
                        args=(task, common, threads)
                    )
                    proc._start_order = next(counter)
                    proc.start()
                    running[proc.sentinel] = (proc, task)

                ready = wait(list(running), timeout=0.2)

                for sentinel in ready:
                    proc, task = running.pop(sentinel)
                    proc.join()

                    if proc.pid in stopped:
                        if task.get("_alone"):
                            failed.append((task, "memory"))
                            tqdm.write(
                                f"Roof tile {task['tile_id']} needs more memory than "
                                f"is free even on its own; stopped"
                            )
                            bar.update(1)
                        else:
                            task["_alone"] = True
                            deferred.append(task)
                            tqdm.write(
                                f"Roof tile {task['tile_id']} stopped to keep "
                                f"{reserve_gb:.0f} GB free; it will run on its own later"
                            )
                        continue

                    if proc.exitcode != 0 or not os.path.exists(task["result_path"]):
                        failed.append((task, proc.exitcode))
                        tqdm.write(
                            f"Roof tile {task['tile_id']} failed "
                            f"(exit code {proc.exitcode}): see {task['log_path']}"
                        )
                    bar.update(1)

                # Watchdog: give memory back before the kernel takes it. As many
                # tiles as it takes, youngest first, counting what the ones
                # already killed but not yet reaped are about to return.
                if running:
                    projected = _memory_gb()[0] + sum(
                        rss_gb(p) for p, _ in running.values() if p.pid in stopped
                    )
                    victims = sorted(
                        (p for p, _ in running.values() if p.pid not in stopped),
                        key=lambda p: -p._start_order
                    )
                    for victim in victims:
                        if projected >= reserve_gb:
                            break
                        projected += rss_gb(victim)
                        stopped.add(victim.pid)
                        kill_tree(victim)
                        # A tile alone when killed was given the whole machine.
                        if len(running) == 1:
                            running[victim.sentinel][1]["_alone"] = True
    finally:
        # Interrupted or failed here: no tile, nor any of its workers,
        # is left running on its own.
        for proc, _ in running.values():
            kill_tree(proc)

    for task in tasks:
        task.pop("_alone", None)

    return failed


def _roof_tile_process(task, common, threads):
    """Body of a tile's process: output to the tile's log, then ``_run_roof_tile``."""
    code = 1
    try:
        log = open(task["log_path"], "w", buffering=1)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)

        # The progress bars of a dozen tiles at once say nothing useful.
        global tqdm
        tqdm = _QuietTqdm
        try:
            from . import dsm_dem_estimation as _dde
        except ImportError:
            import greenshadow.dsm_dem_estimation as _dde
        _dde.tqdm = _QuietTqdm

        _run_roof_tile(task, common, threads=threads)
        code = 0
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Skip the handlers inherited from the parent at fork.
        os._exit(code)


class _QuietTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def _run_roof_tile(task, common, *, threads):
    """Detect one tile's roof planes and write its result and status files."""
    import json
    import resource
    import time

    try:
        import numba
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    except ImportError:
        pass
    from threadpoolctl import threadpool_limits

    started = time.time()
    r0, r1, c0, c1 = task["window"]
    factor = task["factor"]
    owner = np.repeat(np.repeat(task["owned"], factor, axis=0), factor, axis=1)
    owner = owner[:r1 - r0, :c1 - c0]

    with threadpool_limits(limits=threads):
        gdf = detect_roof_planes(
            common["dsm_path"],
            slope_path=common["slope_path"],
            aspect_path=common["aspect_path"],
            dsm_class_path=common["dsm_class_path"],
            dem_path=common["dem_path"],
            test_window=task["window"],
            building_owner_mask=owner,
            refined_output_paths=task["refined_paths"],
            output_path=None,
            n_workers=threads,
            **common["detect_kwargs"]
        )

    gdf["tile_id"] = task["tile_id"]

    # Written under a temporary name and renamed: the result file existing
    # is what marks the tile done, so it must never exist half-written.
    tmp = task["result_path"] + ".tmp"
    gdf.to_parquet(tmp)
    os.replace(tmp, task["result_path"])

    with open(task["status_path"], "w") as f:
        json.dump({
            "tile_id": task["tile_id"],
            "window": list(task["window"]),
            "n_planes": int(len(gdf)),
            "seconds": round(time.time() - started, 1),
            "max_rss_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2
            ),
        }, f)


def _mosaic_aligned_rasters(paths, out_path, bounds, resolution, chunk=4096):
    """
    Mosaic tile rasters that share one global grid into a single GeoTIFF.

    The refined grids are all snapped to multiples of ``resolution``, so each
    tile sits at a whole pixel offset of the region's grid and nothing is
    resampled. Valid pixels overwrite nodata; tiles own disjoint buildings,
    so valid pixels do not compete. Written chunk by chunk, each once, with
    empty chunks left sparse.
    """
    minx, miny, maxx, maxy = bounds
    width, height, transform = _aligned_grid(minx, miny, maxx, maxy, resolution)

    sources = []
    for path in paths:
        src = rasterio.open(path)
        row = int(round((transform.f - src.transform.f) / resolution))
        col = int(round((src.transform.c - transform.c) / resolution))
        sources.append((src, row, col))

    first = sources[0][0]
    nodata = first.nodata
    profile = {
        "driver": "GTiff",
        "dtype": first.dtypes[0],
        "count": 1,
        "width": width,
        "height": height,
        "transform": transform,
        "crs": first.crs,
        "nodata": nodata,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
        "SPARSE_OK": "TRUE",
    }

    try:
        with rasterio.open(out_path, "w", **profile) as dst:
            for r in range(0, height, chunk):
                h = min(chunk, height - r)
                for c in range(0, width, chunk):
                    w = min(chunk, width - c)
                    block = None

                    for src, row, col in sources:
                        # Overlap of the tile with this chunk, in chunk coordinates
                        a0, a1 = max(r, row), min(r + h, row + src.height)
                        b0, b1 = max(c, col), min(c + w, col + src.width)
                        if a0 >= a1 or b0 >= b1:
                            continue

                        data = src.read(1, window=Window(b0 - col, a0 - row, b1 - b0, a1 - a0))
                        if block is None:
                            block = np.full((h, w), nodata if nodata is not None else 0,
                                            dtype=profile["dtype"])
                        if nodata is None:
                            valid = np.ones(data.shape, dtype=bool)
                        elif np.isnan(nodata):
                            valid = ~np.isnan(data)
                        else:
                            valid = data != nodata
                        target = block[a0 - r:a1 - r, b0 - c:b1 - c]
                        target[valid] = data[valid]

                    if block is not None:
                        dst.write(block, 1, window=Window(c, r, w, h))
    finally:
        for src, _, _ in sources:
            src.close()


# ============================================================
# Private utility functions
# ============================================================

def _ridge_mask_from_normal_clustering(
    nx,
    ny,
    nz,
    mask,
    eps_deg=10,
    min_samples=20,
    smooth_sigma=1.0
):

    clusters = _cluster_normals(
        nx,
        ny,
        nz,
        mask,
        ridge_mask=None,
        eps_deg=eps_deg,
        min_samples=min_samples,
        smooth_sigma=smooth_sigma
    )

    ridge_mask = np.zeros(mask.shape, dtype=bool)

    valid = clusters >= 0

    if not np.any(valid):
        return ridge_mask, clusters

    ids, counts = np.unique(clusters[valid], return_counts=True)

    main_cluster = ids[np.argmax(counts)]

    ridge_mask = (clusters != main_cluster) & mask

    return ridge_mask, clusters

def _filter_normal_clusters(
    clusters,
    segment_mask,
    min_pixels
):

    structure = generate_binary_structure(2,1)

    filtered = np.full(clusters.shape, -1, dtype=np.int32)

    next_id = 0

    for cid in np.unique(clusters):

        if cid < 0:
            continue

        mask = (clusters == cid) & segment_mask

        if not np.any(mask):
            continue

        # enforce spatial connectivity
        labeled, n = label(mask, structure=structure)

        for i in range(1, n+1):

            comp = labeled == i

            if comp.sum() < min_pixels:
                continue

            filtered[comp] = next_id
            next_id += 1

    return filtered


def _remove_small_planes(
    plane_id_raster: np.ndarray,
    min_pixels: int
):
    """
    Remove planes smaller than min_pixels and set them to 0.
    """

    counts = np.bincount(plane_id_raster.ravel())

    remove = counts < min_pixels
    remove[0] = False  # keep background

    plane_id_raster[remove[plane_id_raster]] = 0

    return plane_id_raster

def _compute_plane_adjacency_pairs(plane_id_raster):

    H, W = plane_id_raster.shape

    pairs = set()

    # horizontal neighbors
    a = plane_id_raster[:, :-1]
    b = plane_id_raster[:, 1:]

    mask = (a != b) & (a > 0) & (b > 0)

    for p, q in zip(a[mask], b[mask]):
        pairs.add(tuple(sorted((p, q))))

    # vertical neighbors
    a = plane_id_raster[:-1, :]
    b = plane_id_raster[1:, :]

    mask = (a != b) & (a > 0) & (b > 0)

    for p, q in zip(a[mask], b[mask]):
        pairs.add(tuple(sorted((p, q))))

    return list(pairs)

def _boundary_has_ridge(pid1, pid2, plane_id_raster, ridge_mask):

    if ridge_mask is None:
        return False

    boundary = (
        ((plane_id_raster == pid1) & np.roll(plane_id_raster == pid2, 1, 0)) |
        ((plane_id_raster == pid1) & np.roll(plane_id_raster == pid2, -1, 0)) |
        ((plane_id_raster == pid1) & np.roll(plane_id_raster == pid2, 1, 1)) |
        ((plane_id_raster == pid1) & np.roll(plane_id_raster == pid2, -1, 1))
    )

    return np.any(boundary & ridge_mask)

def _cluster_normals(
    nx,
    ny,
    nz,
    mask,
    ridge_mask=None,
    eps_deg=10,
    min_samples=20,
    smooth_sigma=1.0
):

    # --------------------------------
    # smooth normals
    # --------------------------------
    if smooth_sigma > 0:

        nx, ny, nz = _smooth_normals_preserve_ridges(
            nx, ny, nz,
            mask,
            ridge_mask=ridge_mask,
            sigma=smooth_sigma
        )

    normals = np.stack([nx[mask], ny[mask], nz[mask]], axis=1)

    # normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)

    # orientation invariance
    normals[normals[:,2] < 0] *= -1

    eps = np.sin(np.radians(eps_deg / 2))

    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = _dbscan_grid(normals, eps, min_samples)

    return labels


def _dbscan_grid(points, eps, min_samples):
    """
    DBSCAN labels identical to ``sklearn.cluster.DBSCAN(eps, min_samples)``,
    in memory linear in the number of points.

    sklearn materialises every point's neighbour list before it clusters.
    Roof normals are the worst case for that: every flat roof points the
    same way, so a building's pixels are nearly all each other's
    neighbours, and one terraced block of a town centre asked for tens of
    GB. Here the points are binned into cubic cells whose diagonal is
    shorter than ``eps`` - so the points of one cell are all neighbours of
    one another - and only cells within reach are visited. A cell lying
    wholly inside a point's ``eps`` ball is counted without looking at its
    points; one wholly outside is skipped.

    The result is the same partition with the same numbers: core points are
    those with at least ``min_samples`` neighbours within ``eps`` (counting
    themselves), clusters are the connected components of the core points
    numbered in order of their lowest index, and a border point takes the
    lowest-numbered cluster with a core point within ``eps`` of it - which is
    the order sklearn's expansion reaches them in. Distances are compared
    squared in float64, as sklearn's tree does.
    """
    X = np.ascontiguousarray(points, dtype=np.float64)
    n = X.shape[0]

    if n == 0:
        return np.empty(0, dtype=np.int64)

    eps = float(eps)
    # A hair under eps / sqrt(3), so that two points in one cell are within
    # eps even after rounding.
    side = eps / np.sqrt(3.0) * (1.0 - 1e-6)
    reach = int(np.ceil(eps / side))

    origin = X.min(axis=0)
    ijk = np.floor((X - origin) / side).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    cell_keys, starts = np.unique(sorted_keys, return_index=True)
    ends = np.append(starts[1:], n)
    cell_ijk = ijk[order[starts]]
    Xs = X[order]

    offsets = np.array(
        [(a, b, c)
         for a in range(-reach, reach + 1)
         for b in range(-reach, reach + 1)
         for c in range(-reach, reach + 1)],
        dtype=np.int64
    )

    eps2 = eps * eps

    counts = _dbscan_count(
        Xs, cell_keys, starts, ends, cell_ijk, dims, origin, side, eps2, offsets
    )
    is_core = counts >= min_samples

    labels_sorted = _dbscan_label(
        Xs, order, is_core, cell_keys, starts, ends, cell_ijk, dims,
        origin, side, eps2, offsets
    )

    labels = np.empty(n, dtype=np.int64)
    labels[order] = labels_sorted

    return labels


@njit(cache=True, inline="always")
def _dbscan_find_cell(cell_keys, key):
    k = np.searchsorted(cell_keys, key)
    if k < cell_keys.shape[0] and cell_keys[k] == key:
        return k
    return -1


@njit(cache=True, inline="always")
def _dbscan_box_d2(Xs, p, lo0, lo1, lo2, side):
    """Squared min and max distance from point p to a cell box (padded)."""
    pad = 1e-9
    mind2 = 0.0
    maxd2 = 0.0
    for axis in range(3):
        lo = (lo0, lo1, lo2)[axis] - pad
        hi = (lo0, lo1, lo2)[axis] + side + pad
        x = Xs[p, axis]
        if x < lo:
            d = lo - x
        elif x > hi:
            d = x - hi
        else:
            d = 0.0
        mind2 += d * d
        far = max(abs(x - lo), abs(x - hi))
        maxd2 += far * far
    return mind2, maxd2


@njit(cache=True, inline="always")
def _dbscan_d2(Xs, p, q):
    d = Xs[p, 0] - Xs[q, 0]
    t = d * d
    d = Xs[p, 1] - Xs[q, 1]
    t += d * d
    d = Xs[p, 2] - Xs[q, 2]
    t += d * d
    return t


@njit(cache=True, parallel=True)
def _dbscan_count(Xs, cell_keys, starts, ends, cell_ijk, dims, origin, side, eps2, offsets):
    n = Xs.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    inside2 = eps2 * (1.0 - 1e-9)

    for c in prange(cell_keys.shape[0]):
        for o in range(offsets.shape[0]):
            i = cell_ijk[c, 0] + offsets[o, 0]
            j = cell_ijk[c, 1] + offsets[o, 1]
            k = cell_ijk[c, 2] + offsets[o, 2]
            if i < 0 or j < 0 or k < 0 or i >= dims[0] or j >= dims[1] or k >= dims[2]:
                continue
            d = _dbscan_find_cell(cell_keys, (i * dims[1] + j) * dims[2] + k)
            if d < 0:
                continue
            lo0 = origin[0] + i * side
            lo1 = origin[1] + j * side
            lo2 = origin[2] + k * side
            size = ends[d] - starts[d]

            for p in range(starts[c], ends[c]):
                mind2, maxd2 = _dbscan_box_d2(Xs, p, lo0, lo1, lo2, side)
                if mind2 > eps2:
                    continue
                if maxd2 <= inside2:
                    counts[p] += size
                    continue
                for q in range(starts[d], ends[d]):
                    if _dbscan_d2(Xs, p, q) <= eps2:
                        counts[p] += 1

    return counts


@njit(cache=True)
def _dbscan_root(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True)
def _dbscan_label(Xs, order, is_core, cell_keys, starts, ends, cell_ijk, dims,
                  origin, side, eps2, offsets):
    n = Xs.shape[0]
    n_cells = cell_keys.shape[0]
    inside2 = eps2 * (1.0 - 1e-9)

    has_core = np.zeros(n_cells, dtype=np.bool_)
    for c in range(n_cells):
        for p in range(starts[c], ends[c]):
            if is_core[p]:
                has_core[c] = True
                break

    # The core points of one cell are all within eps of one another, so a
    # cell's cores are one component: union-find over cells.
    parent = np.arange(n_cells)

    for c in range(n_cells):
        if not has_core[c]:
            continue
        for o in range(offsets.shape[0]):
            i = cell_ijk[c, 0] + offsets[o, 0]
            j = cell_ijk[c, 1] + offsets[o, 1]
            k = cell_ijk[c, 2] + offsets[o, 2]
            if i < 0 or j < 0 or k < 0 or i >= dims[0] or j >= dims[1] or k >= dims[2]:
                continue
            d = _dbscan_find_cell(cell_keys, (i * dims[1] + j) * dims[2] + k)
            if d <= c or not has_core[d]:
                continue
            if _dbscan_root(parent, c) == _dbscan_root(parent, d):
                continue
            lo0 = origin[0] + i * side
            lo1 = origin[1] + j * side
            lo2 = origin[2] + k * side
            linked = False
            for p in range(starts[c], ends[c]):
                if not is_core[p]:
                    continue
                mind2, maxd2 = _dbscan_box_d2(Xs, p, lo0, lo1, lo2, side)
                if mind2 > eps2:
                    continue
                for q in range(starts[d], ends[d]):
                    if is_core[q] and _dbscan_d2(Xs, p, q) <= eps2:
                        linked = True
                        break
                if linked:
                    break
            if linked:
                parent[_dbscan_root(parent, d)] = _dbscan_root(parent, c)

    # Each component's seed is its lowest original index; clusters are
    # numbered in seed order, as sklearn discovers them.
    big = n + 1
    seed = np.full(n_cells, big, dtype=np.int64)
    for c in range(n_cells):
        if not has_core[c]:
            continue
        r = _dbscan_root(parent, c)
        for p in range(starts[c], ends[c]):
            if is_core[p] and order[p] < seed[r]:
                seed[r] = order[p]

    seeds = np.sort(seed[seed < big])
    labels = np.full(n, -1, dtype=np.int64)

    for c in range(n_cells):
        if not has_core[c]:
            continue
        lab = np.searchsorted(seeds, seed[_dbscan_root(parent, c)])
        for p in range(starts[c], ends[c]):
            if is_core[p]:
                labels[p] = lab

    # Border points: the lowest-numbered cluster within eps.
    for c in range(n_cells):
        for p in range(starts[c], ends[c]):
            if is_core[p]:
                continue
            best = big
            for o in range(offsets.shape[0]):
                i = cell_ijk[c, 0] + offsets[o, 0]
                j = cell_ijk[c, 1] + offsets[o, 1]
                k = cell_ijk[c, 2] + offsets[o, 2]
                if i < 0 or j < 0 or k < 0 or i >= dims[0] or j >= dims[1] or k >= dims[2]:
                    continue
                d = _dbscan_find_cell(cell_keys, (i * dims[1] + j) * dims[2] + k)
                if d < 0 or not has_core[d]:
                    continue
                s = seed[_dbscan_root(parent, d)]
                if s >= best:
                    continue
                mind2, maxd2 = _dbscan_box_d2(
                    Xs, p, origin[0] + i * side, origin[1] + j * side,
                    origin[2] + k * side, side
                )
                if mind2 > eps2:
                    continue
                if maxd2 <= inside2:
                    best = s
                    continue
                for q in range(starts[d], ends[d]):
                    if is_core[q] and _dbscan_d2(Xs, p, q) <= eps2:
                        best = s
                        break
            if best < big:
                labels[p] = np.searchsorted(seeds, best)

    return labels

def _smooth_normals_preserve_ridges(
    nx, ny, nz,
    mask,
    ridge_mask=None,
    sigma=1
):
    """
    Smooth normals while preventing smoothing across ridges.
    """

    # Apply mask
    nx_f = np.where(mask, nx, np.nan)
    ny_f = np.where(mask, ny, np.nan)
    nz_f = np.where(mask, nz, np.nan)

    # Replace NaNs for filtering
    nx_tmp = np.nan_to_num(nx_f)
    ny_tmp = np.nan_to_num(ny_f)
    nz_tmp = np.nan_to_num(nz_f)

    w = mask.astype(float)

    # Smooth normals
    nx_s = gaussian_filter(nx_tmp, sigma)
    ny_s = gaussian_filter(ny_tmp, sigma)
    nz_s = gaussian_filter(nz_tmp, sigma)

    w_s = gaussian_filter(w, sigma)

    valid = w_s > 1e-6

    nx_s[valid] /= w_s[valid]
    ny_s[valid] /= w_s[valid]
    nz_s[valid] /= w_s[valid]

    nx_s[~valid] = 0
    ny_s[~valid] = 0
    nz_s[~valid] = 0

    # Preserve ridge normals
    if ridge_mask is not None:
        nx_s[ridge_mask] = nx[ridge_mask]
        ny_s[ridge_mask] = ny[ridge_mask]
        nz_s[ridge_mask] = nz[ridge_mask]

    # Renormalize normals
    norm = np.sqrt(nx_s**2 + ny_s**2 + nz_s**2)
    norm = np.maximum(norm, 1e-6)

    nx_s /= norm
    ny_s /= norm
    nz_s /= norm

    return nx_s, ny_s, nz_s

def _merge_similar_adjacent_planes_graph(
    plane_id_raster,
    planes,
    local_dsm,
    local_transform,
    normal_angle_threshold_deg=2.5,
    ridge_mask=None,
    max_offset_m=0.3,
):

    threshold_rad = np.radians(normal_angle_threshold_deg)

    plane_ids = [p["plane_id"] for p in planes]
    id_to_idx = {pid: i for i, pid in enumerate(plane_ids)}
    idx_to_id = {i: pid for pid, i in id_to_idx.items()}

    n_planes = len(planes)

    normals = np.zeros((n_planes, 3))
    for p in planes:
        i = id_to_idx[p["plane_id"]]
        normals[i] = _plane_normal(p["a"], p["b"])

    adjacency = np.zeros((n_planes, n_planes), dtype=bool)

    # fast adjacency detection
    pairs = _compute_plane_adjacency_pairs(plane_id_raster)

    for pid1, pid2 in pairs:

        if pid1 not in id_to_idx or pid2 not in id_to_idx:
            continue

        if _boundary_has_ridge(pid1, pid2, plane_id_raster, ridge_mask):
            continue

        i = id_to_idx[pid1]
        j = id_to_idx[pid2]

        angle = np.arccos(
            np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0)
        )

        if angle >= threshold_rad:
            continue

        # Agreeing on orientation is not enough. Two stacked flat roofs, a
        # terrace and the block beside it, are exactly parallel and would
        # merge into one plane spanning the step between them. Compare the
        # two planes where they actually meet.
        if max_offset_m is not None:
            shared = np.isin(plane_id_raster, (pid1, pid2))

            if np.any(shared):
                rows_s, cols_s = np.where(shared)

                mx = local_transform.c + (cols_s.mean() + 0.5) * local_transform.a
                my = local_transform.f + (rows_s.mean() + 0.5) * local_transform.e

                pa = planes[i]
                pb = planes[j]

                za = pa["a"] * mx + pa["b"] * my + pa["c"]
                zb = pb["a"] * mx + pb["b"] * my + pb["c"]

                if abs(za - zb) > max_offset_m:
                    continue

        adjacency[i, j] = True
        adjacency[j, i] = True

    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(graph, directed=False)

    new_plane_id_raster = plane_id_raster.copy()
    merged_planes = []

    next_plane_id = 1

    for comp in range(n_components):

        indices = np.where(labels == comp)[0]
        comp_plane_ids = [idx_to_id[i] for i in indices]

        merge_mask = np.isin(plane_id_raster, comp_plane_ids)

        if not np.any(merge_mask):
            continue

        rows, cols = np.where(merge_mask)

        xs = local_transform.c + (cols + 0.5) * local_transform.a
        ys = local_transform.f + (rows + 0.5) * local_transform.e
        zs = local_dsm[merge_mask]

        a, b, c = _fit_plane_ls(xs, ys, zs)

        new_plane_id_raster[merge_mask] = next_plane_id

        merged_planes.append({
            "plane_id": next_plane_id,
            "a": a,
            "b": b,
            "c": c,
            "area_pixels": int(merge_mask.sum()),
        })

        next_plane_id += 1

    return merged_planes, new_plane_id_raster

def _fit_plane_ls(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    """
    Fit plane z = ax + by + c via least squares
    using centered coordinates for numerical stability.
    """

    # Center coordinates
    x0 = xs.mean()
    y0 = ys.mean()
    z0 = zs.mean()

    xs_c = xs - x0
    ys_c = ys - y0
    zs_c = zs - z0

    # Solve for centered system
    A = np.c_[xs_c, ys_c, np.ones_like(xs_c)]
    coef, *_ = np.linalg.lstsq(A, zs_c, rcond=None)

    a, b, c_local = coef

    # Recover global intercept
    c = z0 - a*x0 - b*y0

    return float(a), float(b), float(c)


def _fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.2,
    max_iterations: int = 1000,
    min_inliers: int = 3,
    seed: int = 0
):

    if len(points) < min_inliers:
        return None

    best_inliers = None
    best_params = None

    # Seeded, so that the same inputs always give the same planes: with the
    # global generator two identical runs differed by 0.01 in IoU against
    # the digitised roofs, which hid any parameter change smaller than that.
    rng = np.random.default_rng(seed)

    for _ in range(max_iterations):

        sample_indices = rng.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]

        try:
            a, b, c = _fit_plane_ls(
                sample_points[:, 0],
                sample_points[:, 1],
                sample_points[:, 2]
            )
        except:
            continue

        distances = _plane_distance_3d(
            a, b, c,
            points[:, 0],
            points[:, 1],
            points[:, 2]
        )

        inliers = np.where(distances < threshold)[0]

        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = (a, b, c)

    if best_inliers is None or len(best_inliers) < min_inliers:
        return None

    return best_params[0], best_params[1], best_params[2], best_inliers


def _plane_slope_aspect(a, b):

    # slope
    slope_deg = np.degrees(
        np.arctan(np.sqrt(a*a + b*b))
    )

    # steepest descent direction
    aspect_deg = np.degrees(
        np.arctan2(-a, -b)
    )

    aspect_deg = (aspect_deg + 360) % 360

    return slope_deg, aspect_deg

def _plane_normal(a, b):
    n = np.array([-a, -b, 1.0], dtype=np.float64)
    return n / np.linalg.norm(n)

def _compute_sobel_slope_aspect(
    dsm: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    nodata: float,
    resolution: float
):
    """
    Compute slope and aspect using Sobel operator.

    Returns:
        slope  -> radians
        aspect -> radians (GIS convention)
    """

    slope[:] = nodata
    aspect[:] = nodata

    valid = dsm != nodata

    # Replace nodata temporarily to avoid edge contamination
    dsm_filled = dsm.copy()
    mean_val = np.nanmean(np.where(valid, dsm, np.nan))
    dsm_filled[~valid] = mean_val

    # Sobel derivatives
    dzdx = sobel(dsm_filled, axis=1, mode="nearest") / (8.0 * resolution)
    dzdy = sobel(dsm_filled, axis=0, mode="nearest") / (8.0 * resolution)

    # Gradient magnitude
    grad_mag = np.sqrt(dzdx**2 + dzdy**2)

    # ---- SLOPE (radians) ----
    slope_rad = np.arctan(grad_mag)

    # ---- ASPECT (radians, GIS convention) ----
    # Steepest descent vector
    aspect_rad = np.arctan2(-dzdx, -dzdy)
    aspect_rad = (aspect_rad + 2 * np.pi) % (2 * np.pi)

    slope[valid] = slope_rad[valid]
    aspect[valid] = aspect_rad[valid]



def _plane_distance_3d(a, b, c, x, y, z):
    num = np.abs(a*x + b*y - z + c)
    den = np.sqrt(a*a + b*b + 1.0)
    return num / den


# ============================================================
# Private plane detection functions
# ============================================================

def _detect_planes_from_dsm(
    building_mask: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    dsm: np.ndarray,
    transform,
    normal_tolerance_deg: float = 10.0,
    ransac_threshold: float = 0.15,
    min_pixels: int = 20,
) -> List[Dict]:

    visited = np.zeros_like(building_mask, dtype=bool)
    # Which region last tested a pixel, so a region does not re-test its own
    # rejects while they stay available to every later region.
    tested = np.full(building_mask.shape, -1, dtype=np.int32)
    planes = []
    plane_id = 0

    rows_all, cols_all = np.where(building_mask)

    for region, (r, c) in enumerate(zip(rows_all, cols_all)):

        if visited[r, c]:
            continue

        seed_normal = np.array([nx[r, c], ny[r, c], nz[r, c]])
        stack = [(r, c)]
        plane_mask = np.zeros_like(building_mask, dtype=bool)

        while stack:

            rr, cc = stack.pop()

            if visited[rr, cc] or tested[rr, cc] == region:
                continue

            tested[rr, cc] = region

            current = np.array([nx[rr, cc], ny[rr, cc], nz[rr, cc]])
            dot = np.clip(np.dot(seed_normal, current), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))

            if angle > normal_tolerance_deg:
                # Deliberately NOT marked visited. Marking it here was
                # consuming every pixel a region merely reached: a dormer
                # tested against the main slope's seed fails, and used to
                # be discarded for good, so it never got a plane of its
                # own. Small facets were absorbed by their big neighbours.
                continue

            visited[rr, cc] = True

            plane_mask[rr, cc] = True

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = rr+dr, cc+dc
                if (
                    0 <= nr < dsm.shape[0] and
                    0 <= nc < dsm.shape[1] and
                    building_mask[nr, nc] and
                    not visited[nr, nc]
                ):
                    stack.append((nr, nc))

        if np.sum(plane_mask) < min_pixels:
            continue

        rows_p, cols_p = np.where(plane_mask)

        # Pixel centres, as everywhere else in the pipeline: the corner
        # convention put a half-cell bias into every fitted plane.
        xs = transform.c + (cols_p + 0.5) * transform.a
        ys = transform.f + (rows_p + 0.5) * transform.e
        zs = dsm[plane_mask]

        ##### Fit the plane
        # LS Mode
        #a, b, c_plane = _fit_plane_ls(xs, ys, zs)
        # Ransac Mode
        points = np.column_stack((xs, ys, zs))
        ransac_result = _fit_plane_ransac(
            points,
            threshold=ransac_threshold,  # adjust for 0.25m DSM
            max_iterations=500,
            min_inliers=min_pixels
        )
        if ransac_result is None:
            continue
        a, b, c_plane, inliers = ransac_result

        planes.append({
            "plane_id": plane_id,
            "mask": plane_mask.copy(),
            "a": a,
            "b": b,
            "c": c_plane,
            "area_pixels": int(np.sum(plane_mask)),
        })

        plane_id += 1

    return planes


# ============================================================
# Private polygon and assignment functions
# ============================================================

def _dsm_planes_to_buffered_polygons(
    building_planes_dsm,
    transform,
    buffer_m=0.5
):

    polygons = []
    plane_ids = []

    for plane in building_planes_dsm:

        mask = plane["mask"]

        for geom, value in shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=transform
        ):
            poly = shape(geom)
            if not poly.is_empty:
                polygons.append(poly.buffer(buffer_m))
                plane_ids.append(plane["plane_id"])

    gdf = gpd.GeoDataFrame(
        {"plane_id": plane_ids},
        geometry=polygons,
        crs=None
    )

    return gdf


def _assign_pixels_to_planes_3d(
    local_mask,
    local_dsm,
    building_planes_dsm,
    polygons_gdf,
    transform,
    ridge_mask=None,
    nx=None,
    ny=None,
    nz=None,
    distance_tolerance=0.15,
    normal_tolerance_deg=15.0,
    normal_weight=1.0
):
    """
    Give every roof pixel to a plane, on distance *and* orientation.

    Distance alone cannot separate two facets that meet: within a band
    either side of a ridge both planes pass within millimetres of the
    surface, so the winner is decided by noise and the boundary wanders -
    or one facet quietly takes the other's pixels. Measured against the
    digitised Alpicat roofs, a quarter of detected facets were each
    swallowing about three real ones.

    What does separate them is the pixel's own normal, which the detector
    has already computed from slope and aspect: two facets meeting at a
    ridge differ by tens of degrees there even though their heights agree.
    Each candidate is scored on both, each normalised by its own tolerance
    so the two terms are comparable:

        cost = distance / distance_tolerance
             + normal_weight * angle / normal_tolerance_deg

    ``normal_weight=0`` restores the old distance-only behaviour.

    Pixels are still restricted to the planes whose buffered polygon
    contains them, with a fallback to all planes when none does.
    """
    H, W = local_mask.shape
    plane_id_raster = np.zeros((H, W), dtype=np.int32)

    if not building_planes_dsm:
        return plane_id_raster

    rows, cols = np.where(local_mask)

    if rows.size == 0:
        return plane_id_raster

    if ridge_mask is not None:
        keep = ~ridge_mask[rows, cols].astype(bool)
        rows, cols = rows[keep], cols[keep]

        if rows.size == 0:
            return plane_id_raster

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    zs = local_dsm[rows, cols].astype(np.float64)

    coefficients = np.array(
        [[p["a"], p["b"], p["c"]] for p in building_planes_dsm],
        dtype=np.float64
    )

    ids = np.array(
        [p["plane_id"] for p in building_planes_dsm], dtype=np.int64
    )

    a = coefficients[:, 0]
    b = coefficients[:, 1]
    c = coefficients[:, 2]

    denominator = np.sqrt(a * a + b * b + 1.0)

    # (pixels, planes)
    distance = np.abs(
        xs[:, None] * a[None, :] +
        ys[:, None] * b[None, :] -
        zs[:, None] +
        c[None, :]
    ) / denominator[None, :]

    cost = distance / max(distance_tolerance, 1e-6)

    if normal_weight > 0 and nx is not None and ny is not None and nz is not None:
        pixel_normals = np.column_stack([
            nx[rows, cols], ny[rows, cols], nz[rows, cols]
        ]).astype(np.float64)

        lengths = np.linalg.norm(pixel_normals, axis=1, keepdims=True)
        pixel_normals /= np.maximum(lengths, 1e-9)

        plane_normals = np.column_stack([-a, -b, np.ones_like(a)])
        plane_normals /= np.linalg.norm(plane_normals, axis=1, keepdims=True)

        cosine = np.clip(pixel_normals @ plane_normals.T, -1.0, 1.0)

        angle = np.degrees(np.arccos(np.abs(cosine)))

        cost = cost + normal_weight * angle / max(normal_tolerance_deg, 1e-6)

    # Restrict each pixel to the planes whose polygon covers it.
    inside = np.zeros(cost.shape, dtype=bool)

    if polygons_gdf is not None and len(polygons_gdf) > 0:
        for polygon, pid in zip(
            polygons_gdf.geometry.values, polygons_gdf.plane_id.values
        ):
            if polygon is None or polygon.is_empty:
                continue

            column = np.where(ids == pid)[0]

            if column.size == 0:
                continue

            inside[:, column[0]] = contains_xy(polygon, xs, ys)

    constrained = np.where(inside, cost, np.inf)

    best = np.argmin(constrained, axis=1)

    # Pixels no polygon claimed fall back to the best plane overall.
    orphan = ~np.isfinite(constrained[np.arange(best.size), best])

    if np.any(orphan):
        best[orphan] = np.argmin(cost[orphan], axis=1)

    plane_id_raster[rows, cols] = ids[best] + 1

    return plane_id_raster


def _refit_planes_from_assignment(
    plane_id_raster,
    local_dsm,
    transform
):

    # The pixels of every plane in one pass, grouped by a stable sort so
    # each plane keeps them in row-major order - the order np.where gives,
    # and so the same least-squares fit. Comparing the raster against each
    # plane id in turn, and keeping that comparison as a mask per plane,
    # cost planes x pixels in time and in memory; nothing reads the mask.
    rows_all, cols_all = np.nonzero(plane_id_raster > 0)
    ids_all = plane_id_raster[rows_all, cols_all]
    order = np.argsort(ids_all, kind="stable")
    unique_ids, starts = np.unique(ids_all[order], return_index=True)
    ends = np.append(starts[1:], len(order))

    updated_planes = []

    for pid, start, end in zip(unique_ids, starts, ends):

        rows = rows_all[order[start:end]]
        cols = cols_all[order[start:end]]

        xs = transform.c + cols * transform.a
        ys = transform.f + rows * transform.e
        zs = local_dsm[rows, cols]

        a, b, c = _fit_plane_ls(xs, ys, zs)
        #a, b, c = _fit_plane_ransac(xs, ys, zs)

        updated_planes.append({
            "plane_id": pid,
            "a": a,
            "b": b,
            "c": c,
        })

    return updated_planes


def _disk(radius_px):
    """Disk structuring element, so a dilation stays round rather than square."""
    r = int(max(1, round(radius_px)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]

    return (x * x + y * y) <= r * r


def _detect_roof_obstructions(
    plane_id_raster,
    updated_planes,
    local_dsm,
    local_svf,
    transform,
    *,
    svf_drop,
    min_height_m,
    min_area_m2,
    max_area_m2,
    max_elongation,
    max_planes,
    close_m,
    buffer_m
):
    """
    Boolean mask of the clutter standing on the finished roof planes.

    Chimneys, aerials and air-conditioning units are searched for one plane
    at a time, and that is not a loop for tidiness: every quantity here is
    only meaningful relative to a single face. A facet's sky view baseline
    is its own - a north slope under a tree and a clear south slope share no
    level to be depressed below - and the morphology has to stay inside the
    facet too, or a closing bridges over a ridge and welds the shadow on one
    side of it to the shadow on the other.

    **The sky view factor is the signal, not the confirmation.** Measured
    against the 48 digitised obstacles on the Alpicat window, the refined
    DSM barely knows they exist: the median obstacle clears its own fitted
    plane by 0.09 m, and only 9 of 45 clear 0.30 m. The TIN sampling and the
    Delaunay smoothing that make the facets clean also flatten a 0.5 m flue
    into the roof it stands on. The SVF raster is built from the *unrefined*
    DSM, which still has them, and there they are unmistakable: a median
    depression of 0.30 below their own facet's baseline, where the roof at
    large sits at 0.08 for its 90th percentile.

    So ``min_height_m`` defaults to 0 - the height test is available, but
    asking the flattened DSM to confirm what the SVF found mostly deletes
    real obstacles.

    What sky view cannot do is tell a small object from a long one. A step
    between two roof levels shadows the roof below it and prints the same
    depression a chimney prints, so the shape tests do that part:
    ``max_elongation`` rejects the slivers along ridges and eaves, and
    ``max_planes`` rejects what straddles a boundary between two faces.
    """
    if local_svf is None:
        return None

    pixel_size = abs(transform.a)
    pixel_area = pixel_size * abs(transform.e)

    assigned = plane_id_raster > 0

    if not np.any(assigned):
        return None

    plane_dict = {p["plane_id"]: p for p in updated_planes}

    H, W = plane_id_raster.shape
    rows, cols = np.indices((H, W))
    X = transform.c + (cols + 0.5) * transform.a
    Y = transform.f + (rows + 0.5) * transform.e

    finite_svf = np.isfinite(local_svf)

    # Every pixel answered by the nearest plane that holds one, so a facet's
    # territory includes the clutter standing on it: an object a metre off
    # the roof is far outside any RANSAC threshold and usually carries no
    # plane_id of its own.
    reach_px = max(0.5, close_m + buffer_m) / pixel_size

    distance_px, (nearest_row, nearest_col) = distance_transform_edt(
        ~assigned,
        return_indices=True
    )
    nearest_pid = plane_id_raster[nearest_row, nearest_col]
    reach = assigned | (distance_px <= reach_px)

    close_structure = _disk(close_m / pixel_size) if close_m > 0 else None
    neighbourhood = _disk(3)

    obstructions = np.zeros((H, W), dtype=bool)

    plane_ids = np.unique(plane_id_raster[assigned])

    for pid in plane_ids:

        territory = (nearest_pid == pid) & reach

        # The baseline comes from the facet's own assigned pixels only. Let
        # the clutter into the median and a cluttered facet raises its own
        # bar until nothing on it is depressed any more.
        sampled = (plane_id_raster == pid) & finite_svf

        if not np.any(sampled):
            continue

        baseline = np.median(local_svf[sampled])

        candidates = territory & finite_svf & (local_svf < baseline - svf_drop)

        if not np.any(candidates):
            continue

        # Closing then filling recovers an object whose own top is not
        # depressed at all - standing on a chimney the sky is wide open
        # again - from the ring of shadow it casts around its base. Kept
        # inside the facet so it cannot bridge a ridge.
        if close_structure is not None:
            candidates = binary_closing(candidates, structure=close_structure)

        candidates = binary_fill_holes(candidates) & territory

        if min_height_m > 0:
            plane = plane_dict[pid]
            residual = local_dsm - (
                plane["a"] * X + plane["b"] * Y + plane["c"]
            )
            candidates &= residual > min_height_m

        if not np.any(candidates):
            continue

        labeled, n_components = label(
            candidates,
            structure=generate_binary_structure(2, 2)
        )

        if n_components == 0:
            continue

        areas = np.bincount(labeled.ravel()) * pixel_area

        # Below the floor it is noise; above the ceiling it is a structure in
        # its own right - a stairwell, a set-back upper storey - and the
        # answer there is another plane, not a hole in this one.
        keep = (areas >= min_area_m2) & (areas <= max_area_m2)
        keep[0] = False

        for comp in np.flatnonzero(keep):

            blob = labeled == comp
            blob_rows, blob_cols = np.nonzero(blob)

            if max_elongation > 0 and blob_rows.size >= 3:
                cloud = np.column_stack((blob_rows, blob_cols)).astype(np.float64)
                spread = np.linalg.eigvalsh(np.cov((cloud - cloud.mean(0)).T))
                elongation = np.sqrt(
                    max(spread[-1], 0.0) / max(spread[0], 1e-9)
                )

                if elongation > max_elongation:
                    continue

            if max_planes > 0:
                # What the blob is surrounded by, not what it covers: the
                # clutter's own pixels are usually in no plane at all.
                around = binary_dilation(blob, structure=neighbourhood) & ~blob
                neighbours = np.unique(plane_id_raster[around & assigned])

                if neighbours.size > max_planes:
                    continue

            obstructions |= blob

    if not np.any(obstructions):
        return None

    if buffer_m > 0:
        obstructions = binary_dilation(
            obstructions,
            structure=_disk(buffer_m / pixel_size)
        )

    return obstructions & reach


def _polygonize_and_compute_parameters(
    plane_id_raster,
    updated_planes,
    local_dsm,
    transform,
    part_id,
    crs,
    simplify_tolerance=None,
    fit_mask=None,
    hole_mask=None
):

    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Polygon
    from rasterio.features import shapes
    from scipy.ndimage import label

    H, W = plane_id_raster.shape

    polygons = []
    records = []

    plane_dict = {p["plane_id"]: p for p in updated_planes}

    # coordinate grids for plane fitting metrics
    rows, cols = np.indices((H, W))
    X = transform.c + (cols + 0.5) * transform.a
    Y = transform.f + (rows + 0.5) * transform.e
    Z = local_dsm

    plane_ids = np.unique(plane_id_raster)
    plane_ids = plane_ids[plane_ids > 0]

    # The clutter, once, as one geometry. Facets are polygonized whole and
    # the holes are taken out of the polygon afterwards rather than out of
    # the raster first: subtracting from the intact facet is what lets an
    # edge notch and a severing cut be measured, and it keeps the exterior
    # ring - which is all ``shapes`` is read for here - free of them.
    holes = None

    if hole_mask is not None and np.any(hole_mask):
        hole_parts = [
            shape(geom)
            for geom, _ in shapes(
                hole_mask.astype(np.uint8),
                mask=hole_mask,
                transform=transform
            )
        ]

        if hole_parts:
            holes = unary_union(hole_parts)

    for pid in plane_ids:

        mask = plane_id_raster == pid

        # split disconnected regions
        labeled, ncomp = label(mask)

        plane = plane_dict[pid]
        a, b, c_plane = plane["a"], plane["b"], plane["c"]

        slope, aspect = _plane_slope_aspect(a, b)

        for comp in range(1, ncomp + 1):

            comp_mask = labeled == comp

            if comp_mask.sum() < 3:
                continue

            # polygonize this region
            for geom, value in shapes(
                comp_mask.astype(np.uint8),
                mask=comp_mask,
                transform=transform
            ):

                poly = Polygon(geom["coordinates"][0])

                # simplification
                if simplify_tolerance and simplify_tolerance > 0:
                    poly = poly.simplify(
                        simplify_tolerance,
                        preserve_topology=True
                    )

                gross_area = poly.area
                pieces = [poly]

                if holes is not None and poly.intersects(holes):
                    cut = poly.difference(holes)

                    if cut.is_empty:
                        continue

                    # A buffered hole can sever a facet in two. Each surviving
                    # piece is a roof plane polygon in its own right - dropping
                    # the smaller one would lose real roof - so they share the
                    # plane's identity and are emitted separately.
                    pieces = [
                        g
                        for g in (
                            cut.geoms
                            if cut.geom_type in ("MultiPolygon", "GeometryCollection")
                            else [cut]
                        )
                        if g.geom_type == "Polygon" and not g.is_empty
                    ]

                    if not pieces:
                        continue

                # planarity metrics, over the component as a whole: the
                # pieces below all belong to the same fitted plane.
                m = comp_mask

                # Pixels added only for the footprint (the skirt) did not
                # take part in the fit and would inflate the metrics.
                if fit_mask is not None and np.any(comp_mask & fit_mask):
                    m = comp_mask & fit_mask

                if hole_mask is not None and np.any(m & ~hole_mask):
                    m = m & ~hole_mask

                x = X[m]
                y = Y[m]
                z = Z[m]

                dist = np.abs(a*x + b*y - z + c_plane) / np.sqrt(a*a + b*b + 1)

                rmse = np.sqrt(np.mean(dist**2))
                mad = np.median(np.abs(dist - np.median(dist)))

                cut_area = gross_area - sum(g.area for g in pieces)

                for piece in pieces:

                    footprint_area = piece.area
                    roof_area = footprint_area / np.cos(np.radians(slope))

                    polygons.append(piece)

                    records.append({
                        "part_id": part_id,
                        "plane_id": pid,
                        "tilt_deg": slope,
                        "azimuth_deg": aspect,
                        "footprint_area": footprint_area,
                        "area": roof_area,
                        "planarity_rmse": rmse,
                        "planarity_mad": mad,
                        "roughness": mad / (rmse + 1e-6),
                        "n_obstructions": len(piece.interiors),
                        # The severed case has no per-piece split to report,
                        # so the component's loss goes on its largest piece.
                        "obstruction_area": (
                            cut_area
                            if piece is max(pieces, key=lambda g: g.area)
                            else 0.0
                        )
                    })

    gdf = gpd.GeoDataFrame(records, geometry=polygons, crs=crs)

    return gdf

# def _polygonize_and_compute_parameters(
#     plane_id_raster,
#     updated_planes,
#     local_dsm,
#     transform,
#     part_id,
#     crs,
#     simplify_tolerance=None,
#     preserve_holes=True,
# ):
#
#     import geopandas as gpd
#     import numpy as np
#     from shapely.geometry import shape, Polygon
#     from rasterio.features import shapes
#
#     H, W = plane_id_raster.shape
#
#     polygons = []
#     records = []
#
#     plane_dict = {p["plane_id"]: p for p in updated_planes}
#
#     # Precompute coordinate grids
#     rows, cols = np.indices((H, W))
#
#     X = transform.c + (cols + 0.5) * transform.a
#     Y = transform.f + (rows + 0.5) * transform.e
#     Z = local_dsm
#
#     for geom, value in shapes(
#         plane_id_raster.astype(np.int32),
#         mask=plane_id_raster > 0,
#         transform=transform
#     ):
#
#         pid = int(value)
#         poly = shape(geom)
#
#         # -----------------------------------------
#         # Optional geometry simplification
#         # -----------------------------------------
#         if simplify_tolerance is not None and simplify_tolerance > 0:
#
#             if preserve_holes and isinstance(poly, Polygon):
#
#                 exterior = poly.exterior.simplify(
#                     simplify_tolerance,
#                     preserve_topology=True
#                 )
#
#                 holes = [
#                     ring.simplify(
#                         simplify_tolerance,
#                         preserve_topology=True
#                     )
#                     for ring in poly.interiors
#                 ]
#
#                 poly = Polygon(exterior, holes)
#
#             else:
#                 poly = poly.simplify(
#                     simplify_tolerance,
#                     preserve_topology=True
#                 )
#
#         plane = plane_dict[pid]
#
#         a, b, c_plane = plane["a"], plane["b"], plane["c"]
#
#         slope, aspect = _plane_slope_aspect(a, b)
#
#         footprint_area = poly.area
#         roof_area = footprint_area / np.cos(np.radians(slope))
#
#         # -----------------------------------------
#         # Planar fitting metrics
#         # -----------------------------------------
#
#         mask = plane_id_raster == pid
#
#         x = X[mask]
#         y = Y[mask]
#         z = Z[mask]
#
#         # orthogonal distance to plane
#         dist = np.abs(a*x + b*y - z + c_plane) / np.sqrt(a*a + b*b + 1)
#
#         rmse = np.sqrt(np.mean(dist**2))
#         mad = np.median(np.abs(dist - np.median(dist)))
#
#         polygons.append(poly)
#
#         records.append({
#             "part_id": part_id,
#             "plane_id": pid,
#             "tilt_deg": slope,
#             "azimuth_deg": aspect,
#             "footprint_area": footprint_area,
#             "area": roof_area,
#             "planarity_rmse": rmse,
#             "planarity_mad": mad,
#             "roughness": mad/rmse if rmse > 0 else np.nan,
#         })
#
#     gdf = gpd.GeoDataFrame(records, geometry=polygons, crs=crs)
#
#     return gdf


def _reassign_unassigned_pixels_local(
    plane_id_raster,
    local_mask,
    local_dsm,
    updated_planes,
    transform,
    max_search_radius=3,
    min_support=1.5,
    distance_weight=1.0,
    gradient_angle_tol_deg=35,
    ridge_mask=None
):

    import numpy as np

    H, W = plane_id_raster.shape
    plane_dict = {p["plane_id"]: p for p in updated_planes}

    # neighbor offsets and weights
    neighbor_offsets = [
        (-1, 0, 1.0), (1, 0, 1.0),
        (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 0.5), (-1, 1, 0.5),
        (1, -1, 0.5), (1, 1, 0.5),
    ]

    # ------------------------------------------------
    # Helper: enforce linear continuity
    # ------------------------------------------------
    def _has_linear_support(r, c, pid):

        if c > 0 and c < W-1:
            if plane_id_raster[r, c-1] == pid and plane_id_raster[r, c+1] == pid:
                return True

        if r > 0 and r < H-1:
            if plane_id_raster[r-1, c] == pid and plane_id_raster[r+1, c] == pid:
                return True

        if r > 0 and r < H-1 and c > 0 and c < W-1:
            if plane_id_raster[r-1, c-1] == pid and plane_id_raster[r+1, c+1] == pid:
                return True

            if plane_id_raster[r-1, c+1] == pid and plane_id_raster[r+1, c-1] == pid:
                return True

        return False

    # ------------------------------------------------
    # Helper: gradient-plane consistency
    # ------------------------------------------------
    def _gradient_consistent(r, c, pid):

        plane = plane_dict[pid]
        a, b = plane["a"], plane["b"]

        r0 = max(r-1, 0)
        r1 = min(r+1, H-1)
        c0 = max(c-1, 0)
        c1 = min(c+1, W-1)

        gx = (local_dsm[r, c1] - local_dsm[r, c0]) / (2 * abs(transform.a))
        gy = (local_dsm[r1, c] - local_dsm[r0, c]) / (2 * abs(transform.e))

        g_norm = np.sqrt(gx*gx + gy*gy)
        p_norm = np.sqrt(a*a + b*b)

        if g_norm == 0 or p_norm == 0:
            return True

        cosang = (gx*(-a) + gy*(-b)) / (g_norm * p_norm)
        cosang = np.clip(cosang, -1, 1)

        angle = np.degrees(np.arccos(cosang))

        return angle < gradient_angle_tol_deg

    # ------------------------------------------------
    # pixels needing assignment
    # ------------------------------------------------
    unassigned = np.argwhere((local_mask) & (plane_id_raster == 0))

    for r, c in unassigned:

        x = transform.c + c * transform.a
        y = transform.f + r * transform.e
        z = local_dsm[r, c]

        best_plane = 0
        best_score = -np.inf

        for radius in range(1, max_search_radius + 1):

            support = {}

            for dr, dc, w in neighbor_offsets:

                nr = r + dr * radius
                nc = c + dc * radius

                if (
                        0 <= nr < H and
                        0 <= nc < W and
                        local_mask[nr, nc]
                ):

                    if ridge_mask is not None:
                        if ridge_mask[r, c] or ridge_mask[nr, nc]:
                            continue

                    pid = plane_id_raster[nr, nc]

                    if pid > 0:
                        support[pid] = support.get(pid, 0) + w

            if not support:
                continue

            # filter weak planes
            candidates = []

            for pid, w in support.items():

                if w < min_support:
                    continue

                if not _has_linear_support(r, c, pid):
                    continue

                if not _gradient_consistent(r, c, pid):
                    continue

                candidates.append((pid, w))

            if not candidates:
                continue

            # evaluate candidates
            for pid, w in candidates:

                plane = plane_dict[pid]
                a, b, c_plane = plane["a"], plane["b"], plane["c"]

                dist = abs(a*x + b*y - z + c_plane) / np.sqrt(a*a + b*b + 1)

                score = w - distance_weight * dist

                if score > best_score:
                    best_score = score
                    best_plane = pid

            if best_plane != 0:
                plane_id_raster[r, c] = best_plane
                break

    return plane_id_raster

def _recover_unassigned_roof(
    plane_id_raster,
    local_mask,
    local_dsm,
    planes,
    transform,
    *,
    distance_tolerance,
    min_pixels,
    max_tilt_deg=60.0,
    max_grow_rounds=50,
    min_inlier_fraction=0.6,
    max_new_planes=4
):
    """
    Hand the building's unclaimed pixels to a plane, old or new.

    Region growing stops wherever the normals get noisy and never returns,
    so a facet whose seed was rejected - or the band along a ridge - is
    left with no plane at all. First every existing plane grows, one pixel
    ring per round, into unclaimed neighbours lying within
    ``distance_tolerance`` of it; a pixel two planes both reach goes to the
    closer one. What is still unclaimed is split into connected patches,
    and a patch big enough gets a RANSAC plane of its own when that plane
    explains most of it and is not a wall.
    """
    structure = generate_binary_structure(2, 1)

    rows_idx, cols_idx = np.indices(plane_id_raster.shape)

    # Same corner convention as _refit_planes_from_assignment, which is
    # where the planes came from.
    xs = transform.c + cols_idx * transform.a
    ys = transform.f + rows_idx * transform.e

    valid = local_mask & np.isfinite(local_dsm) & (local_dsm > -9000)

    # Residuals are evaluated only where a plane can grow - one pixel ring
    # around it - and the ring is dilated inside the plane's own bounding
    # box. A full-size residual raster per plane cost planes x pixels x 8
    # bytes: ~10 GB for one terraced block of a town centre.
    coefficients = {p["plane_id"]: (p["a"], p["b"], p["c"]) for p in planes}
    height, width = plane_id_raster.shape

    for _ in range(max_grow_rounds):
        unassigned = valid & (plane_id_raster == 0)

        if not np.any(unassigned):
            break

        best_pid = np.zeros_like(plane_id_raster)
        best_res = np.full(plane_id_raster.shape, np.inf)
        extents = find_objects(np.clip(plane_id_raster, 0, None))

        for pid, (a, b, c) in coefficients.items():
            if pid < 1 or pid > len(extents) or extents[pid - 1] is None:
                continue

            rs, cs = extents[pid - 1]
            r0, r1 = max(rs.start - 1, 0), min(rs.stop + 1, height)
            c0, c1 = max(cs.start - 1, 0), min(cs.stop + 1, width)

            reach = binary_dilation(
                plane_id_raster[r0:r1, c0:c1] == pid, structure=structure
            )
            rr, cc = np.nonzero(reach & unassigned[r0:r1, c0:c1])
            rr += r0
            cc += c0

            res = np.abs(a * xs[rr, cc] + b * ys[rr, cc] + c - local_dsm[rr, cc])
            take = (res <= distance_tolerance) & (res < best_res[rr, cc])
            best_pid[rr[take], cc[take]] = pid
            best_res[rr[take], cc[take]] = res[take]

        grown = best_pid > 0

        if not np.any(grown):
            break

        plane_id_raster[grown] = best_pid[grown]

    next_id = int(plane_id_raster.max()) + 1
    max_slope = np.tan(np.radians(max_tilt_deg))

    patches, n_patches = label(
        valid & (plane_id_raster == 0), structure=structure
    )

    for patch_id, box in enumerate(find_objects(patches), start=1):
        if box is None:
            continue

        # Worked inside the patch's bounding box: everything outside it is
        # False in every mask below, so nothing changes but the cost.
        patch = patches[box] == patch_id
        box_xs, box_ys, box_dsm = xs[box], ys[box], local_dsm[box]
        box_ids = plane_id_raster[box]

        for _ in range(max_new_planes):
            # A thin rim of wall or edge pixels is not a facet, however
            # many pixels it has: require the patch's core to be big too.
            core = binary_erosion(patch, structure=structure)

            if patch.sum() < min_pixels or core.sum() < min_pixels / 2:
                break

            rr, cc = np.nonzero(patch)
            points = np.column_stack((box_xs[rr, cc], box_ys[rr, cc], box_dsm[rr, cc]))

            fit = _fit_plane_ransac(
                points,
                threshold=distance_tolerance,
                max_iterations=200,
                min_inliers=int(min_pixels)
            )

            if fit is None:
                break

            a, b, c, inliers = fit

            if np.hypot(a, b) > max_slope:
                break

            if len(inliers) < min_inlier_fraction * len(points):
                break

            inlier_mask = np.zeros_like(patch)
            inlier_mask[rr[inliers], cc[inliers]] = True

            parts, n_parts = label(inlier_mask, structure=structure)

            if n_parts == 0:
                break

            sizes = np.bincount(parts.ravel())
            sizes[0] = 0
            largest = parts == int(np.argmax(sizes))

            if largest.sum() < min_pixels:
                break

            box_ids[largest] = next_id
            next_id += 1
            patch &= ~largest

    return plane_id_raster


def _merge_coplanar_planes(
    plane_id_raster,
    planes,
    local_dsm,
    transform,
    *,
    max_angle_deg,
    max_step_m
):
    """
    Merge touching planes that are the same surface, closest pair first.

    Two planes qualify when their normals are within ``max_angle_deg`` and,
    along the pixels where they touch, the two planes' heights differ by no
    more than ``max_step_m`` (median over the boundary). The best pair is
    merged, its plane refitted by least squares, and the pairs recomputed,
    so every merge is judged against the plane it actually produces.
    """
    planes = {p["plane_id"]: dict(p) for p in planes}

    if len(planes) < 2:
        return list(planes.values()), plane_id_raster

    plane_id_raster = plane_id_raster.copy()

    H, W = plane_id_raster.shape
    rows_idx, cols_idx = np.indices((H, W))

    # Same corner convention as _refit_planes_from_assignment.
    xs = transform.c + cols_idx * transform.a
    ys = transform.f + rows_idx * transform.e

    max_angle = np.radians(max_angle_deg)

    def normal(p):
        n = np.array([-p["a"], -p["b"], 1.0])
        return n / np.linalg.norm(n)

    def boundary_pixels():
        """For each touching pair, the pixels on either side of the seam."""
        seams = defaultdict(list)

        for axis in (0, 1):
            if axis == 0:
                a, b = plane_id_raster[:-1, :], plane_id_raster[1:, :]
                ra, ca = rows_idx[:-1, :], cols_idx[:-1, :]
                rb, cb = rows_idx[1:, :], cols_idx[1:, :]
            else:
                a, b = plane_id_raster[:, :-1], plane_id_raster[:, 1:]
                ra, ca = rows_idx[:, :-1], cols_idx[:, :-1]
                rb, cb = rows_idx[:, 1:], cols_idx[:, 1:]

            touch = (a != b) & (a > 0) & (b > 0)

            for p, q, r1, c1, r2, c2 in zip(
                a[touch], b[touch], ra[touch], ca[touch], rb[touch], cb[touch]
            ):
                key = (min(p, q), max(p, q))
                seams[key].append((r1, c1))
                seams[key].append((r2, c2))

        return seams

    while True:
        best = None

        for (p, q), pixels in boundary_pixels().items():
            if p not in planes or q not in planes:
                continue

            pa, pb = planes[p], planes[q]

            angle = np.arccos(np.clip(normal(pa) @ normal(pb), -1.0, 1.0))

            if angle > max_angle:
                continue

            rr, cc = np.asarray(pixels).T
            px, py = xs[rr, cc], ys[rr, cc]

            step = np.median(np.abs(
                (pa["a"] - pb["a"]) * px + (pa["b"] - pb["b"]) * py
                + (pa["c"] - pb["c"])
            ))

            if step > max_step_m:
                continue

            if best is None or angle < best[0]:
                best = (angle, p, q)

        if best is None:
            break

        _, p, q = best

        merged = (plane_id_raster == p) | (plane_id_raster == q)
        plane_id_raster[merged] = p

        rr, cc = np.nonzero(merged)
        a, b, c = _fit_plane_ls(xs[rr, cc], ys[rr, cc], local_dsm[rr, cc])

        planes[p].update(a=a, b=b, c=c, mask=merged)
        del planes[q]

    return list(planes.values()), plane_id_raster


def _assign_roof_skirts(plane_id_raster, local_mask, *, max_distance_px):
    """
    Give building pixels no plane claimed to the nearest plane.

    Only pixels within ``max_distance_px`` of an assigned pixel are taken,
    so a large unexplained patch - a courtyard wall, a missed structure -
    stays out rather than being stretched over by a distant facet.
    """
    unassigned = local_mask & (plane_id_raster == 0)

    if not np.any(unassigned) or not np.any(plane_id_raster > 0):
        return plane_id_raster

    distance, (near_r, near_c) = distance_transform_edt(
        plane_id_raster == 0, return_indices=True
    )

    take = unassigned & (distance <= max_distance_px)

    plane_id_raster = plane_id_raster.copy()
    plane_id_raster[take] = plane_id_raster[near_r[take], near_c[take]]

    return plane_id_raster


def _enforce_plane_continuity(plane_id_raster, local_mask):

    structure = generate_binary_structure(2, 1)

    new_raster = np.zeros_like(plane_id_raster)

    next_id = 1

    for pid in np.unique(plane_id_raster):

        if pid == 0:
            continue

        mask = (plane_id_raster == pid) & local_mask

        labeled, n = label(mask, structure=structure)

        for i in range(1, n + 1):

            component = labeled == i

            new_raster[component] = next_id
            next_id += 1

    return new_raster


def _get_building_lidar_points(
    dem,
    min_height_m,
    transform,
    laz_dir,
    tiles,
    lidar_allowed_classes,
    aoi=None
):

    if not laz_dir or not tiles:
        return None

    minx = transform.c
    maxx = transform.c + dem.shape[1] * transform.a
    maxy = transform.f
    miny = transform.f + dem.shape[0] * transform.e

    # AOI bbox for fast rejection
    if aoi is not None:
        aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi.bounds

    all_pts = []

    for tile in tqdm(tiles, desc="Loading LiDAR", unit="tile"):

        coords = tile["geometry"]["coordinates"][0]

        tile_minx = min(pt[0] for pt in coords)
        tile_maxx = max(pt[0] for pt in coords)
        tile_miny = min(pt[1] for pt in coords)
        tile_maxy = max(pt[1] for pt in coords)

        # -------------------------------------------------
        # DSM grid bbox rejection
        # -------------------------------------------------
        if (
            tile_maxx <= minx or
            tile_minx >= maxx or
            tile_maxy <= miny or
            tile_miny >= maxy
        ):
            continue

        # -------------------------------------------------
        # AOI bbox rejection
        # -------------------------------------------------
        if aoi is not None:
            if (
                tile_maxx <= aoi_minx or
                tile_minx >= aoi_maxx or
                tile_maxy <= aoi_miny or
                tile_miny >= aoi_maxy
            ):
                continue

        tile_id = tile["properties"]["ID1K"]

        laz_path = os.path.join(
            laz_dir,
            f"lidar-territorial-v3r1-full1km{tile_id}-2021-2023.laz"
        )

        if not os.path.exists(laz_path):
            continue

        # Filtered while it is read: holding a whole decoded tile is what
        # made the municipality-wide run run out of memory.
        pts = _read_laz(
            path=laz_path,
            bounds=(aoi_minx, aoi_miny, aoi_maxx, aoi_maxy) if aoi is not None else None,
            classes=lidar_allowed_classes
        )

        if pts is None or len(pts) == 0:
            continue

        # -------------------------------------------------
        # Classification filter
        # -------------------------------------------------
        if pts.shape[1] >= 4:
            mask_class = np.isin(pts[:, 3], lidar_allowed_classes)
            pts = pts[mask_class]
        else:
            continue

        # -------------------------------------------------
        # AOI geometry filter (very fast)
        # -------------------------------------------------
        if aoi is not None and len(pts) > 0:

            x = pts[:, 0]
            y = pts[:, 1]

            # fast bbox filter
            bbox_mask = (
                (x >= aoi_minx) & (x <= aoi_maxx) &
                (y >= aoi_miny) & (y <= aoi_maxy)
            )

            if not np.any(bbox_mask):
                continue

            x = x[bbox_mask]
            y = y[bbox_mask]
            pts = pts[bbox_mask]

            inside = contains_xy(aoi, x, y)

            if not np.any(inside):
                continue

            pts = pts[inside]

        # -------------------------------------------------
        # Height filtering
        # -------------------------------------------------
        if dem is not None and min_height_m is not None and len(pts) > 0:

            rows, cols = rowcol(
                transform,
                pts[:, 0],
                pts[:, 1],
                op=np.floor
            )

            rows = rows.astype(np.int64)
            cols = cols.astype(np.int64)

            valid_pix = (
                (rows >= 0) & (rows < dem.shape[0]) &
                (cols >= 0) & (cols < dem.shape[1])
            )

            if not np.any(valid_pix):
                continue

            rows = rows[valid_pix]
            cols = cols[valid_pix]
            pts = pts[valid_pix]

            dem_vals = dem[rows, cols]

            valid_dem = np.isfinite(dem_vals) & (dem_vals != -9999.0)

            if not np.any(valid_dem):
                continue

            dem_vals = dem_vals[valid_dem]
            pts = pts[valid_dem]

            height_diff = pts[:, 2] - dem_vals

            keep_height = height_diff > min_height_m
            pts = pts[keep_height]

        if len(pts) > 0:
            all_pts.append(pts)

    if not all_pts:
        return None

    pts = np.vstack(all_pts)

    structured_pts = np.zeros(
        pts.shape[0],
        dtype=[
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("classification", np.uint8)
        ]
    )

    structured_pts["x"] = pts[:, 0]
    structured_pts["y"] = pts[:, 1]
    structured_pts["z"] = pts[:, 2]
    structured_pts["classification"] = pts[:, 3].astype(np.uint8)

    return structured_pts

def _building_mask_to_buffered_geometry(
    building_mask,
    transform,
    epsg: int = 25831,
    buffer_m: float = 1.0,
):
    """
    Vectorize building mask and buffer in meters.

    Parameters
    ----------
    building_mask : np.ndarray (bool)
    transform : rasterio.Affine
    epsg : int
        CRS of raster (default 25831)
    buffer_m : float
        Buffer distance in meters
    """

    if not np.any(building_mask):
        return None

    geoms = []

    for geom, value in shapes(
        building_mask.astype(np.uint8),
        mask=building_mask,
        transform=transform,
    ):
        if value == 1:
            geoms.append(shape(geom))

    if not geoms:
        return None

    geom_union = unary_union(geoms)

    # ------------------------------------------------------------
    # Ensure buffering happens in projected CRS
    # ------------------------------------------------------------
    if epsg != 25831:
        # If input CRS not projected in meters, reproject to 25831
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}",
            "EPSG:25831",
            always_xy=True,
        )
        geom_union = shp_transform(transformer.transform, geom_union)

    if buffer_m > 0:
        geom_union = geom_union.buffer(buffer_m)

    # Reproject back if necessary
    if epsg != 25831:
        transformer_back = Transformer.from_crs(
            "EPSG:25831",
            f"EPSG:{epsg}",
            always_xy=True,
        )
        geom_union = shp_transform(transformer_back.transform, geom_union)

    return geom_union

def _lidar_points_to_building_geom(
    building_pts,
    transform,
    width,
    height,
    buffer_m=0.5,
):

    if building_pts is None or len(building_pts) == 0:
        return None

    inv = ~transform

    px, py = inv * (
        building_pts[:, 0],
        building_pts[:, 1]
    )

    px = np.floor(px).astype(np.int32)
    py = np.floor(py).astype(np.int32)

    valid = (
        (px >= 0) & (px < width) &
        (py >= 0) & (py < height)
    )

    px = px[valid]
    py = py[valid]

    mask = np.zeros((height, width), dtype=bool)
    mask[py, px] = True

    # fill small holes
    mask = binary_closing(mask, iterations=2)

    # slightly enlarge footprint
    mask = binary_dilation(mask, iterations=1)

    geoms = []

    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if value == 1:
            geoms.append(shape(geom))

    if not geoms:
        return None

    geom = unary_union(geoms)

    if buffer_m > 0:
        geom = geom.buffer(buffer_m)

    return geom

# ============================================================
# Point-graph roof tools, and the boundary regularization the
# detector uses. Formerly greenshadow.buildings_infer_v2.
# ============================================================

def _points_to_building_labels(points, labeled_buildings, transform):
    """
    Building label of each point, by lookup on the labelled raster.

    Points outside every building get 0, so a single comparison per
    building splits the cloud without any spatial join.
    """
    rows, cols = rowcol(
        transform,
        points[:, 0],
        points[:, 1],
        op=np.floor
    )

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    labels = np.zeros(points.shape[0], dtype=np.int32)

    inside = (
        (rows >= 0) & (rows < labeled_buildings.shape[0]) &
        (cols >= 0) & (cols < labeled_buildings.shape[1])
    )

    labels[inside] = labeled_buildings[rows[inside], cols[inside]]

    return labels


# ============================================================
# Delaunay Canopy - geometric scoring stage
#
# Reimplementation of stage 1 ("Delaunay Graph Scoring") of
#   Delaunay Canopy: Building Wireframe Reconstruction from Airborne
#   LiDAR Point Clouds via Delaunay Graph (ECCV 2026),
#   https://arxiv.org/abs/2604.02497
#
# This stage is purely geometric and has no trainable parameters: face
# normals -> edge dihedral angles -> vertex corner scores. The paper's
# corner and wire selection modules are Transformers whose code and weights
# are not published (the authors' repository is a project page only).
#
# Two details the paper leaves unspecified are exposed as options:
#   - which faces of the tetrahedralization are kept (`face_mode`);
#   - how theta_e is defined when an edge is shared by more than two faces
#     (the paper's formulation assumes a 2-manifold, where it is always two).
# ============================================================


def delaunay_corner_scores(
    points,
    *,
    face_mode="all",
    normalize="uniform",
    qhull_options=None
):
    """
    Vertex-wise corner scores from the Delaunay graph of a point cloud.

    Follows the paper's stage-1 formulation:
      n_f  = (v_b - v_a) x (v_c - v_a), flipped so that n_f[2] >= 0
      th_e = arccos(n_1 . n_2 / (|n_1| |n_2|)) in [0, pi], pi on boundary edges
      S_c(v) = mean of th_e over the edges incident to v

    Parameters
    ----------
    points : (N, 3) array_like
        Point cloud in map units (x, y, z).
    face_mode : {"all", "hull"}, default="all"
        Which triangular faces of the 3D Delaunay tetrahedralization to
        score. "hull" keeps only the convex-hull surface, which is a true
        2-manifold (every edge has exactly two faces, matching the paper's
        formulation) but discards concave detail. "all" keeps every unique
        tetrahedron face, which preserves the internal corners the paper
        targets at the cost of edges with more than two incident faces.
    normalize : {"uniform", "per_axis", None}, default="uniform"
        The paper scales input clouds "to a range between 0 and 256".
        "uniform" applies a single scale factor to all three axes, which
        leaves the triangulation unchanged and only conditions the
        arithmetic. "per_axis" scales each axis independently, which
        stretches z relative to x/y and does change the triangulation.
    qhull_options : str, optional
        Passed through to scipy.spatial.Delaunay.

    Returns
    -------
    dict or None
        None if the cloud is degenerate (fewer than 4 points, or coplanar,
        so that no tetrahedralization exists). Otherwise:
        ``scores``  (N,) float32, S_c(v) in [0, pi], 0 for unused vertices
        ``edges``   (E, 2) int32, unique vertex-index pairs of the graph
        ``theta``   (E,) float32, dihedral angle per edge in [0, pi]
        ``faces``   (F, 3) int32, the triangular faces that were scored
    """
    pts = np.asarray(points, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] < 3:
        return None

    pts = pts[:, :3]

    n_points = pts.shape[0]

    if n_points < 4:
        return None

    work = _normalize_for_delaunay(pts, normalize)

    try:
        tri = Delaunay(work, qhull_options=qhull_options)
    except Exception:
        # Coplanar / collinear clouds have no 3D tetrahedralization.
        return None

    if face_mode == "hull":
        faces = np.sort(np.asarray(tri.convex_hull, dtype=np.int64), axis=1)
        faces = np.unique(faces, axis=0)
    elif face_mode == "all":
        simplices = np.asarray(tri.simplices, dtype=np.int64)
        faces = np.concatenate([
            simplices[:, [0, 1, 2]],
            simplices[:, [0, 1, 3]],
            simplices[:, [0, 2, 3]],
            simplices[:, [1, 2, 3]],
        ], axis=0)
        faces = np.unique(np.sort(faces, axis=1), axis=0)
    else:
        raise ValueError(f"Unknown face_mode: {face_mode!r}")

    if faces.shape[0] == 0:
        return None

    normals = _delaunay_face_normals(work, faces)

    edges, theta = _delaunay_edge_dihedral_angles(faces, normals)

    scores = _delaunay_corner_scores_from_edges(edges, theta, n_points)

    return {
        "scores": scores,
        "edges": edges.astype(np.int32),
        "theta": theta.astype(np.float32),
        "faces": faces.astype(np.int32),
    }


def _normalize_for_delaunay(pts, normalize):
    if not normalize:
        return pts

    mins = pts.min(axis=0)
    extent = pts.max(axis=0) - mins

    if normalize == "uniform":
        span = float(extent.max())

        if span <= 0:
            return pts

        return (pts - mins) * (256.0 / span)

    if normalize == "per_axis":
        span = np.where(extent > 0, extent, 1.0)

        return (pts - mins) * (256.0 / span)

    raise ValueError(f"Unknown normalize: {normalize!r}")


def _delaunay_face_normals(pts, faces):
    """Face normals oriented upwards, as in the paper (n_f[2] >= 0)."""
    v0 = pts[faces[:, 0]]
    v1 = pts[faces[:, 1]]
    v2 = pts[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)

    flip = normals[:, 2] < 0
    normals[flip] *= -1

    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    return normals / np.maximum(norms, 1e-12)


def _delaunay_edge_dihedral_angles(faces, normals):
    """
    Dihedral angle per undirected edge.

    Boundary edges (a single incident face) get theta = pi, as specified.
    Edges with exactly two incident faces get the paper's formula. Edges
    with more than two - which the paper's 2-manifold formulation does not
    cover, but which occur whenever face_mode="all" - get the mean of the
    angles over all incident face pairs; for two faces this reduces to the
    paper's definition.
    """
    edge_pairs = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [0, 2]],
        faces[:, [1, 2]],
    ], axis=0)

    face_of_edge = np.tile(np.arange(faces.shape[0], dtype=np.int64), 3)

    edges, inverse = np.unique(edge_pairs, axis=0, return_inverse=True)

    inverse = inverse.ravel()

    theta = np.full(edges.shape[0], np.pi, dtype=np.float64)

    # Group incident faces per edge, then handle each group size in one
    # vectorised pass instead of looping over edges.
    order = np.argsort(inverse, kind="stable")

    sorted_face_id = face_of_edge[order]

    counts = np.bincount(inverse, minlength=edges.shape[0])

    offsets = np.concatenate([[0], np.cumsum(counts)])

    for size in np.unique(counts):
        if size < 2:
            # Boundary edge: theta stays pi.
            continue

        edge_ids = np.where(counts == size)[0]

        starts = offsets[edge_ids]

        gather = starts[:, None] + np.arange(size)[None, :]

        group_faces = sorted_face_id[gather]

        group_normals = normals[group_faces]

        if size == 2:
            dots = np.einsum("ij,ij->i", group_normals[:, 0], group_normals[:, 1])
            theta[edge_ids] = np.arccos(np.clip(dots, -1.0, 1.0))
            continue

        dots = np.einsum("ikj,ilj->ikl", group_normals, group_normals)

        angles = np.arccos(np.clip(dots, -1.0, 1.0))

        iu = np.triu_indices(size, k=1)

        theta[edge_ids] = angles[:, iu[0], iu[1]].mean(axis=1)

    return edges, theta


def _delaunay_corner_scores_from_edges(edges, theta, n_points):
    """S_c(v) = mean of theta over the edges incident to v."""
    flat = edges.ravel()

    weights = np.repeat(theta, 2)

    totals = np.bincount(flat, weights=weights, minlength=n_points)
    degrees = np.bincount(flat, minlength=n_points)

    scores = np.zeros(n_points, dtype=np.float64)

    seen = degrees > 0
    scores[seen] = totals[seen] / degrees[seen]

    return scores.astype(np.float32)



# ============================================================
# Delaunay Canopy - plane segmentation over the point graph
#
# Stage 2: regions are grown over the Delaunay 1-ring graph, so adjacency
# is the paper's graph rather than a pixel neighbourhood.
#
# The point-native detector this once fed - it turned these segments into
# face polygons and snapped their shared boundaries onto the exact
# plane-plane intersections - has been removed. What is left is used by
# ``smooth_points_to_planes`` below: the raster detector segments the
# returns here so it can denoise each one toward its own plane without
# rounding off the creases between them.
# ============================================================

def _segment_roof_planes(
    pts,
    *,
    normal_tolerance_deg,
    plane_offset_tolerance,
    crease_angle_deg,
    corner_percentile,
    min_plane_points,
    max_edge_length,
    edge_length_factor,
    face_mode,
    normalize,
    normal_smoothing_rounds=2
):
    """
    Segment a roof point cloud into planes over its Delaunay graph.

    Everything up to, but not including, the polygons: scoring, edge
    pruning, normals, region growing, coplanar merging and crease
    reassignment. Both the polygon extractor and the 3D mesh builder start
    here, so the shapes drawn on screen are the ones actually detected.

    Returns None when the cloud cannot be triangulated.
    """
    graph = delaunay_corner_scores(
        pts,
        face_mode=face_mode,
        normalize=normalize
    )

    if graph is None:
        return None

    edges, max_edge_length = _prune_long_delaunay_edges(
        pts,
        graph["edges"].astype(np.int64),
        max_edge_length,
        edge_length_factor
    )

    normals = _delaunay_point_normals(pts, edges)

    # Floored, not simply tied to the tolerance: with a tight
    # normal_tolerance_deg the smoothing would reject almost every
    # neighbour and denoise nothing, which is precisely the case where the
    # normals most need it. 15 degrees still sits far below a real roof
    # crease, so ridges are unaffected.
    normals = _smooth_point_normals(
        normals,
        edges,
        normal_smoothing_rounds,
        max(normal_tolerance_deg, 15.0)
    )

    labels = _delaunay_plane_segments(
        pts,
        edges,
        normals,
        normal_tolerance_deg=normal_tolerance_deg,
        plane_offset_tolerance=plane_offset_tolerance,
        min_plane_points=min_plane_points,
        corner_scores=graph["scores"],
        crease_angle_deg=crease_angle_deg,
        corner_percentile=corner_percentile
    )

    planes = _fit_segment_planes(pts, labels, min_plane_points)

    # Crease points go back to their plane *before* the coplanar merge.
    # The barrier vertices sit exactly between the fragments of a face that
    # region growing split, and while they are unlabelled the two fragments
    # share no edge, so the merge cannot see that they touch and one roof
    # face is reported as two.
    planes = _assign_crease_points(
        pts,
        edges,
        planes,
        tolerance=plane_offset_tolerance
    )

    labels = _labels_from_planes(planes, pts.shape[0])

    planes = _merge_coplanar_segments(
        pts,
        edges,
        labels,
        planes,
        normal_tolerance_deg=normal_tolerance_deg,
        plane_offset_tolerance=plane_offset_tolerance
    )

    return {
        "planes": planes,
        "scores": graph["scores"],
        "edges": edges,
        "normals": normals,
        "max_edge_length": max_edge_length
    }




# ============================================================
# Delaunay smoothing of a roof point cloud
#
# The segmentation above is reused here for a different purpose: instead of
# turning the planes into polygons, each point is pulled toward the plane its
# own segment fitted. The result is a denoised cloud that can be rasterized
# into a DSM for the raster detector in buildings_infer.detect_roof_planes.
#
# The point of doing it this way, rather than with a gaussian on the raster,
# is that region growing already knows which points belong together. A
# gaussian averages a ridge with the two faces that meet at it and rounds the
# ridge off; averaging within a segment cannot, because the other face is
# simply not in the segment.
# ============================================================

def smooth_points_to_planes(
    points,
    *,
    strength=0.7,
    normal_tolerance_deg=15.0,
    plane_offset_tolerance=0.2,
    crease_angle_deg=None,
    corner_percentile=None,
    min_plane_points=30,
    max_edge_length=None,
    edge_length_factor=3.0,
    normal_smoothing_rounds=2,
    face_mode="all"
):
    """
    Pull each point toward the plane its Delaunay segment fitted.

    Denoises roof elevations without rounding off the ridges, which is what
    a gaussian on the DSM cannot do: it has no idea which side of a crease a
    pixel is on, so it averages across it. Here the points have already been
    grouped into planes, and a point only ever moves toward the plane of its
    own group.

    Only z changes. x, y and any further columns - classification in
    particular - are passed through untouched, so the result can go straight
    back into the rasterizer.

    Parameters
    ----------
    points : (N, >=3) array_like
        LiDAR returns, (x, y, z, ...). Not modified; a copy is returned.
    strength : float, default=0.7
        How far to move each point toward its plane, from 0 to 1. 0 returns
        the input unchanged, 1 snaps z exactly onto the plane and flattens
        every facet completely. In between keeps some of the roof's real
        relief - sag, camber, the odd parapet - while dropping the ranging
        noise.
    normal_tolerance_deg, plane_offset_tolerance, crease_angle_deg,
    corner_percentile, min_plane_points, max_edge_length,
    edge_length_factor, normal_smoothing_rounds, face_mode
        Segmentation parameters, passed to ``_segment_roof_planes``. The
        tolerances matter here the same way they do for detection: too tight
        and the roof fragments into segments too small to fit a stable plane,
        too loose and two roof levels are flattened into one.

    Returns
    -------
    np.ndarray
        A copy of ``points`` with z smoothed. Points that landed on no plane
        keep their measured z exactly - the cloud is never invented, only
        quieted.
    """
    pts = np.asarray(points, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be an (N, >=3) array of x, y, z")

    smoothed = pts.copy()

    if strength <= 0 or pts.shape[0] < min_plane_points:
        return smoothed

    strength = float(min(strength, 1.0))

    segmentation = _segment_roof_planes(
        pts[:, :3],
        normal_tolerance_deg=normal_tolerance_deg,
        plane_offset_tolerance=plane_offset_tolerance,
        crease_angle_deg=crease_angle_deg,
        corner_percentile=corner_percentile,
        min_plane_points=min_plane_points,
        max_edge_length=max_edge_length,
        edge_length_factor=edge_length_factor,
        face_mode=face_mode,
        normalize="uniform",
        normal_smoothing_rounds=normal_smoothing_rounds
    )

    if segmentation is None:
        return smoothed

    for plane in segmentation["planes"]:
        indices = plane["indices"]

        if indices.size == 0:
            continue

        fitted = (
            plane["a"] * pts[indices, 0] +
            plane["b"] * pts[indices, 1] +
            plane["c"]
        )

        smoothed[indices, 2] = (
            pts[indices, 2] + strength * (fitted - pts[indices, 2])
        )

    return smoothed


def smooth_roof_points_by_building(
    points,
    labels,
    *,
    strength=0.7,
    progress=False,
    n_workers=1,
    **kwargs
):
    """
    Smooth a roof point cloud one building at a time.

    Chunking is not an optimisation, it is what makes this runnable: the
    Delaunay triangulation behind the segmentation is superlinear and fills
    the convex volume of whatever it is given, so handing it a whole
    municipality window at once is both far slower and wrong - it would build
    edges from one roof to the next. One building at a time is also the right
    unit semantically, since planes never span two buildings.

    Parameters
    ----------
    points : (N, >=3) array_like
        LiDAR returns, (x, y, z, ...).
    labels : (N,) array_like of int
        Building id per point. 0 means the point is on no building; those are
        passed through untouched.
    strength : float, default=0.7
        As in ``smooth_points_to_planes``.
    progress : bool, default=False
        Show a progress bar over the buildings.
    n_workers : int, default=1
        Buildings smoothed at once, in forked processes. The result does
        not depend on it.
    **kwargs
        Forwarded to ``smooth_points_to_planes``.

    Returns
    -------
    np.ndarray
        A copy of ``points`` with z smoothed building by building.
    """
    pts = np.asarray(points, dtype=np.float64)

    smoothed = pts.copy()

    if strength <= 0:
        return smoothed

    labels = np.asarray(labels)

    building_ids = np.unique(labels)
    building_ids = building_ids[building_ids > 0]

    if building_ids.size == 0:
        return smoothed

    # Each building's points, in ascending order as np.where would give
    # them, from one sort instead of a pass over the cloud per building.
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, building_ids, side="left")
    ends = np.searchsorted(sorted_labels, building_ids, side="right")
    selections = [order[a:b] for a, b in zip(starts, ends)]

    # Largest first, so a big block is not the last one left running.
    items = sorted(range(len(selections)), key=lambda k: -selections[k].size)

    z = _fork_map(
        _smooth_building_points, items, (pts, selections, strength, kwargs),
        n_workers=n_workers, progress=progress,
        desc="Delaunay-smoothing roof points", chunksize=1
    )

    for k, z_k in zip(items, z):
        smoothed[selections[k], 2] = z_k

    return smoothed


def _smooth_building_points(k):
    """One building's smoothed z, for ``smooth_roof_points_by_building``."""
    pts, selections, strength, kwargs = _FORK_JOB
    return smooth_points_to_planes(
        pts[selections[k]], strength=strength, **kwargs
    )[:, 2]


def _delaunay_adjacency(edges, n_points):
    """Symmetric CSR adjacency over the Delaunay 1-ring."""
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])

    data = np.ones(rows.shape[0], dtype=np.uint8)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(n_points, n_points)
    )


def _prune_long_delaunay_edges(pts, edges, max_edge_length, edge_length_factor):
    """
    Drop the edges that span the interior of the tetrahedralization.

    A 3D Delaunay of a roof fills the whole convex volume, so a vertex's
    1-ring also contains points on the far side of the building - on a
    20 x 12 m gable the edges run from 0.3 m to 16 m. Those long edges
    wreck the PCA normals and bridge regions straight across a ridge, so
    only edges up to ``max_edge_length`` are kept. The default threshold is
    a multiple of the median edge length, which adapts to point density
    instead of assuming a resolution.

    Returns the kept edges and the threshold actually used, which the 2D
    triangle filter reuses so both steps cut at the same length.
    """
    lengths = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)

    if max_edge_length is None:
        median = float(np.median(lengths))

        if median <= 0:
            return edges, float(lengths.max()) if lengths.size else 0.0

        max_edge_length = edge_length_factor * median

    keep = lengths <= max_edge_length

    if not np.any(keep):
        return edges, max_edge_length

    return edges[keep], max_edge_length


def _delaunay_point_normals(pts, edges):
    """
    Per-point normals by PCA over the Delaunay 1-ring.

    The 1-ring replaces the usual k-NN neighbourhood, so the support adapts
    to the local point density the way the Delaunay graph does. Normals are
    oriented upwards, matching the face-normal convention of stage 1.
    """
    n_points = pts.shape[0]

    adjacency = _delaunay_adjacency(edges, n_points)

    indptr = adjacency.indptr
    indices = adjacency.indices

    normals = np.zeros((n_points, 3), dtype=np.float64)
    normals[:, 2] = 1.0

    for i in range(n_points):
        neighbours = indices[indptr[i]:indptr[i + 1]]

        if neighbours.size < 3:
            continue

        patch = pts[np.concatenate([neighbours, [i]])]

        centred = patch - patch.mean(axis=0)

        # Smallest principal direction of the patch is its normal.
        try:
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        normal = vh[2]

        if normal[2] < 0:
            normal = -normal

        normals[i] = normal

    return normals


def _adaptive_crease_angle(edge_angles, normal_tolerance_deg):
    """
    Pick the crease threshold from the roof's own noise level.

    A fixed fraction of ``normal_tolerance_deg`` cannot work for both a
    clean cloud and a real one. After edge-preserving smoothing, a quiet
    roof has a median edge deviation near 0.6 degrees and a noisy one near
    1.5, while the ridges in both sit an order of magnitude above. Keying
    the threshold to the median rides that difference: it lands at the 4
    degree floor on clean data, around 7.5 on noisy data, and in both cases
    below the ridge and above the scatter.

    It never exceeds ``normal_tolerance_deg``, since a barrier looser than
    the growing tolerance would never fire, and never drops under 4 degrees,
    so a perfectly flat roof is not carved up by its own rounding.
    """
    if edge_angles.size == 0:
        return 0.7 * normal_tolerance_deg

    median = float(np.median(edge_angles))

    return float(np.clip(5.0 * median, 4.0, normal_tolerance_deg))


def _crease_vertices(
    edges,
    normals,
    edge_angles,
    corner_scores,
    n_points,
    *,
    crease_angle_deg,
    corner_percentile,
    crease_neighbour_fraction
):
    """
    Vertices sitting on a crease, which region growing must not cross.

    Cutting only the edges that straddle a ridge is not enough: the 1-ring
    of a point on the ridge spans both faces, so its PCA normal comes out
    halfway between them, and those intermediate normals form a chain of
    small steps that walks region growing straight over the crease. Marking
    the crease vertices themselves as barriers breaks that chain.

    A vertex is a crease when a real *share* of its neighbours disagree
    with it, not when its worst one does. Taking the worst neighbour is
    what real LiDAR punishes: one noisy return in a 1-ring is enough to
    condemn a vertex, so a flat roof loses a scattering of points all over
    it, and since a triangle needs all three corners in one plane, a few
    percent of lost vertices turns into a face full of holes. At a genuine
    ridge roughly half the 1-ring sits on the other face, which clears the
    fraction easily; a single bad return never does.

    When ``corner_percentile`` is given, the stage-1 corner score S_c(v)
    adds a second barrier, which is the paper's own notion of where
    corners live.
    """
    barrier = np.zeros(n_points, dtype=bool)

    steep = edge_angles > crease_angle_deg

    disagreeing = np.zeros(n_points, dtype=np.int64)
    degree = np.zeros(n_points, dtype=np.int64)

    for column in (0, 1):
        np.add.at(disagreeing, edges[steep, column], 1)
        np.add.at(degree, edges[:, column], 1)

    share = np.zeros(n_points, dtype=np.float64)

    seen = degree > 0

    share[seen] = disagreeing[seen] / degree[seen]

    # Two disagreeing neighbours minimum, so a lone outlier is never a
    # crease no matter how sparse the vertex's 1-ring is.
    barrier |= (share >= crease_neighbour_fraction) & (disagreeing >= 2)

    if corner_percentile is not None and corner_scores is not None:
        finite = corner_scores[np.isfinite(corner_scores)]

        if finite.size > 0:
            threshold = np.percentile(finite, corner_percentile)
            barrier |= corner_scores >= threshold

    return barrier


def _smooth_point_normals(normals, edges, rounds, preserve_angle_deg):
    """
    Average each normal over its 1-ring, without averaging across a crease.

    A PCA normal fitted to six-odd neighbours of a real airborne return is
    a noisy estimate: at 12 points/m2 with 8 cm of ranging noise, two
    points on the same flat roof disagree by 16 degrees at the 97th
    percentile - twice the crease threshold. Left raw, a fifth of the
    edges on a perfectly flat roof look like creases, region growing is
    fenced off everywhere, and the roof comes out as speckle.

    Averaging over the whole ring fixes that and breaks something worse: it
    blurs the ridge too, pulling a 48-degree crease down to about 11, which
    is inside the tolerance that is supposed to cut it - the two faces of a
    gable then merge into one plane. So a neighbour only contributes when
    it already agrees to within ``preserve_angle_deg``. Flat roofs get the
    denoising; ridges keep their angle.
    """
    if rounds <= 0 or edges.shape[0] == 0:
        return normals

    smoothed = normals

    limit = np.cos(np.radians(preserve_angle_deg))

    for _ in range(int(rounds)):
        agreement = np.einsum(
            "ij,ij->i",
            smoothed[edges[:, 0]],
            smoothed[edges[:, 1]]
        )

        same_face = edges[agreement >= limit]

        accumulated = smoothed.copy()

        np.add.at(accumulated, same_face[:, 0], smoothed[same_face[:, 1]])
        np.add.at(accumulated, same_face[:, 1], smoothed[same_face[:, 0]])

        lengths = np.linalg.norm(accumulated, axis=1, keepdims=True)

        smoothed = accumulated / np.maximum(lengths, 1e-12)

    return smoothed


def _delaunay_plane_segments(
    pts,
    edges,
    normals,
    *,
    normal_tolerance_deg,
    plane_offset_tolerance,
    min_plane_points,
    corner_scores=None,
    crease_angle_deg=None,
    corner_percentile=None,
    crease_neighbour_fraction=0.25
):
    """
    Planar regions as connected components of the pruned Delaunay graph.

    An edge survives when its two endpoints agree on normal direction and
    each lies close to the other's tangent plane, and when neither endpoint
    is a crease vertex. Cutting the rest leaves one component per roof face.
    Crease vertices stay unlabelled here and are handed back to their
    nearest plane afterwards, so no roof area is lost at the ridges.
    """
    n_points = pts.shape[0]

    i = edges[:, 0]
    j = edges[:, 1]

    dots = np.einsum("ij,ij->i", normals[i], normals[j])

    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))

    delta = pts[j] - pts[i]

    offset_i = np.abs(np.einsum("ij,ij->i", delta, normals[i]))
    offset_j = np.abs(np.einsum("ij,ij->i", delta, normals[j]))

    if crease_angle_deg is None:
        crease_angle_deg = _adaptive_crease_angle(angles, normal_tolerance_deg)

    barrier = _crease_vertices(
        edges,
        normals,
        angles,
        corner_scores,
        n_points,
        crease_angle_deg=crease_angle_deg,
        corner_percentile=corner_percentile,
        crease_neighbour_fraction=crease_neighbour_fraction
    )

    keep = (
        (angles <= normal_tolerance_deg) &
        (offset_i <= plane_offset_tolerance) &
        (offset_j <= plane_offset_tolerance) &
        (~barrier[i]) &
        (~barrier[j])
    )

    if not np.any(keep):
        return np.full(n_points, -1, dtype=np.int32)

    graph = _delaunay_adjacency(edges[keep], n_points)

    n_components, labels = connected_components(
        graph,
        directed=False
    )

    counts = np.bincount(labels, minlength=n_components)

    too_small = counts < min_plane_points

    labels = labels.astype(np.int32)
    labels[too_small[labels]] = -1
    labels[barrier] = -1

    return labels


def _assign_crease_points(pts, edges, planes, *, tolerance, max_rounds=6):
    """
    Hand the crease vertices back to the plane they actually belong to.

    Barrier vertices are held out of region growing, which would otherwise
    leave a bare strip along every ridge and shrink each face's outline.
    Assignment propagates outwards from the labelled regions one Delaunay
    step at a time, and a point only joins a plane it is adjacent to and
    lies within ``tolerance`` of vertically, so the two sides of a ridge
    each recover their own half of the strip.
    """
    if not planes:
        return planes

    n_points = pts.shape[0]

    owner = np.full(n_points, -1, dtype=np.int32)

    for k, plane in enumerate(planes):
        owner[plane["indices"]] = k

    coeffs = np.array(
        [[p["a"], p["b"], p["c"]] for p in planes],
        dtype=np.float64
    )

    for _ in range(max_rounds):
        assigned = owner >= 0

        i = edges[:, 0]
        j = edges[:, 1]

        # Every edge crossing the labelled/unlabelled frontier, oriented so
        # that `target` is the point still waiting for a plane.
        forward = assigned[i] & ~assigned[j]
        backward = assigned[j] & ~assigned[i]

        target = np.concatenate([j[forward], i[backward]])
        source = np.concatenate([i[forward], j[backward]])

        if target.size == 0:
            break

        candidate = owner[source]

        residual = np.abs(
            coeffs[candidate, 0] * pts[target, 0] +
            coeffs[candidate, 1] * pts[target, 1] +
            coeffs[candidate, 2] -
            pts[target, 2]
        )

        within = residual <= tolerance

        if not np.any(within):
            break

        target = target[within]
        candidate = candidate[within]
        residual = residual[within]

        # Keep, per point, the candidate plane it sits closest to.
        order = np.lexsort((residual, target))

        target = target[order]
        candidate = candidate[order]

        first = np.ones(target.shape[0], dtype=bool)
        first[1:] = target[1:] != target[:-1]

        owner[target[first]] = candidate[first]

    for k, plane in enumerate(planes):
        plane["indices"] = np.where(owner == k)[0]

    return [p for p in planes if p["indices"].size >= 3]


def _labels_from_planes(planes, n_points):
    """Per-point labels matching the current plane list."""
    labels = np.full(n_points, -1, dtype=np.int32)

    for index, plane in enumerate(planes):
        labels[plane["indices"]] = index

    return labels


def _fit_segment_planes(pts, labels, min_plane_points):
    """Least-squares plane z = a x + b y + c per surviving region."""
    planes = []

    for label_id in np.unique(labels):
        if label_id < 0:
            continue

        indices = np.where(labels == label_id)[0]

        if indices.size < min_plane_points:
            continue

        patch = pts[indices]

        try:
            a, b, c = _fit_plane_ls(patch[:, 0], patch[:, 1], patch[:, 2])
        except Exception:
            continue

        planes.append({
            "indices": indices,
            "a": a,
            "b": b,
            "c": c
        })

    return planes


def _plane_unit_normal(plane):
    normal = np.array([-plane["a"], -plane["b"], 1.0])

    return normal / np.linalg.norm(normal)


def _merge_coplanar_segments(
    pts,
    edges,
    labels,
    planes,
    *,
    normal_tolerance_deg,
    plane_offset_tolerance
):
    """
    Merge regions that are coplanar and Delaunay-adjacent.

    Region growing splits a single roof face whenever noise breaks the chain
    of surviving edges, so faces are reassembled here. Adjacency is taken
    from the Delaunay graph, i.e. two regions are neighbours when any edge
    runs between them.
    """
    if len(planes) < 2:
        return planes

    label_of_plane = {}

    for k, plane in enumerate(planes):
        if plane["indices"].size == 0:
            continue

        label_of_plane[int(labels[plane["indices"][0]])] = k

    # Which plane pairs are Delaunay-adjacent.
    li = labels[edges[:, 0]]
    lj = labels[edges[:, 1]]

    valid = (li >= 0) & (lj >= 0) & (li != lj)

    neighbour_pairs = set()

    for a_label, b_label in zip(li[valid], lj[valid]):
        ka = label_of_plane.get(a_label)
        kb = label_of_plane.get(b_label)

        if ka is None or kb is None:
            continue

        neighbour_pairs.add((min(ka, kb), max(ka, kb)))

    parent = list(range(len(planes)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for ka, kb in sorted(neighbour_pairs):
        pa = planes[ka]
        pb = planes[kb]

        dot = float(np.dot(_plane_unit_normal(pa), _plane_unit_normal(pb)))

        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

        if angle > normal_tolerance_deg:
            continue

        # Vertical gap between the two planes, measured where they meet.
        shared = np.concatenate([pa["indices"], pb["indices"]])

        cx = pts[shared, 0].mean()
        cy = pts[shared, 1].mean()

        za = pa["a"] * cx + pa["b"] * cy + pa["c"]
        zb = pb["a"] * cx + pb["b"] * cy + pb["c"]

        if abs(za - zb) > plane_offset_tolerance:
            continue

        ra, rb = find(ka), find(kb)

        if ra != rb:
            parent[ra] = rb

    groups = defaultdict(list)

    for k in range(len(planes)):
        groups[find(k)].append(k)

    merged = []

    for members in groups.values():
        indices = np.concatenate([planes[k]["indices"] for k in members])

        patch = pts[indices]

        try:
            a, b, c = _fit_plane_ls(patch[:, 0], patch[:, 1], patch[:, 2])
        except Exception:
            continue

        merged.append({
            "indices": indices,
            "a": a,
            "b": b,
            "c": c
        })

    return merged



def _fill_face_holes(faces, *, occupied_fraction=0.2):
    """
    Close the holes in every face, except where another face lives.

    A hole is only ever wanted when something else is standing in it. The
    rest - the pinholes left by a stray ridge triangle, the patches where
    the cloud simply thinned out - are not structure, and for siting panels
    a face with a lace of holes in it is worse than a whole one.

    The exception matters: a dormer or a raised block detected as its own
    face sits inside the surrounding roof as a hole in it. Filling that one
    would make the big face swallow the small one, and the disjointness
    pass, which lays the larger face down first, would then delete the
    small one outright. So a hole covered by another face is kept, and
    everything else is closed.
    """
    if not faces:
        return faces

    holed = [
        face for face in faces
        if face["polygon"] is not None and face["polygon"].interiors
    ]

    if not holed:
        return faces

    # A hole is outside its own face, so anything of it covered by the
    # union of all faces is covered by some other face.
    covered = unary_union([
        face["polygon"] for face in faces if face["polygon"] is not None
    ])

    for face in holed:
        poly = face["polygon"]

        kept = []

        for ring in poly.interiors:
            hole = Polygon(ring.coords)

            if hole.area <= 0:
                continue

            try:
                occupied = hole.intersection(covered).area
            except Exception:
                occupied = 0.0

            if occupied > occupied_fraction * hole.area:
                kept.append(list(ring.coords))

        if len(kept) == len(poly.interiors):
            continue

        try:
            filled = Polygon(poly.exterior.coords, kept)
        except Exception:
            continue

        if not filled.is_valid:
            filled = filled.buffer(0)

        if isinstance(filled, Polygon) and not filled.is_empty and filled.area > 0:
            face["polygon"] = filled

    return faces


def _plane_coefficients(tilt_deg, azimuth_deg):
    """
    Recover z = a x + b y + c orientation from tilt and azimuth.

    The exact inverse of ``_plane_slope_aspect``, which both detectors use
    to report their planes: it takes (a, b) to
    ``slope = atan(hypot(a, b))``, ``aspect = atan2(-a, -b)``, so
    ``a = -tan(slope) sin(aspect)`` and ``b = -tan(slope) cos(aspect)``.
    """
    slope = np.tan(np.radians(tilt_deg))

    azimuth = np.radians(azimuth_deg)

    return -slope * np.sin(azimuth), -slope * np.cos(azimuth)


def _anchor_height(geometry, a, b, dsm, dsm_transform, dsm_nodata):
    """
    Find the face's height by sampling the DSM underneath it.

    The polygon and its tilt fix the plane's orientation but not where it
    sits vertically. Every DSM pixel whose centre falls inside the face
    votes, and the median of ``z - (a x + b y)`` wins: a median rather than
    a mean so that a chimney, an aerial or a stray return cannot lift the
    whole face. Returns None when no pixel falls inside - a sliver too thin
    to contain a pixel centre - and that face is then skipped.
    """
    minx, miny, maxx, maxy = geometry.bounds

    top, left = rowcol(dsm_transform, minx, maxy, op=np.floor)
    bottom, right = rowcol(dsm_transform, maxx, miny, op=np.ceil)

    top = max(int(top), 0)
    left = max(int(left), 0)
    bottom = min(int(bottom) + 1, dsm.shape[0])
    right = min(int(right) + 1, dsm.shape[1])

    if bottom <= top or right <= left:
        return None

    rows = np.arange(top, bottom)
    cols = np.arange(left, right)

    xs = dsm_transform.c + (cols + 0.5) * dsm_transform.a
    ys = dsm_transform.f + (rows + 0.5) * dsm_transform.e

    grid_x, grid_y = np.meshgrid(xs, ys)

    window = dsm[top:bottom, left:right]

    valid = np.isfinite(window)

    if dsm_nodata is not None:
        valid &= window != dsm_nodata

    # The builder writes -9999 for nodata even when the profile does not
    # declare it, so guard against obviously absurd elevations too.
    valid &= window > -9000

    if not np.any(valid):
        return None

    inside = np.zeros_like(valid)

    inside[valid] = contains_xy(
        geometry, grid_x[valid], grid_y[valid]
    )

    if not np.any(inside):
        return None

    residual = (
        window[inside] - a * grid_x[inside] - b * grid_y[inside]
    )

    return float(np.median(residual))


# ============================================================
# Boundary regularization for an existing detection
#
# The raster detector polygonizes pixels, so its facets carry a staircase
# edge and every shared ridge is drawn twice, once per side, a little
# apart. This turns that output into straight-edged facets meeting on a
# single ridge, without re-detecting anything.
#
# Measured on the digitised Alpicat roofs, the detector finds 98% of the
# real planes and misses none - what it loses is boundary accuracy, and
# most of all on small facets. That is a shape problem, not a detection
# problem, which is why it is fixed here rather than by more tuning.
# ============================================================

def regularize_roof_planes(
    roof_planes,
    *,
    dsm_path=None,
    simplify_tolerance=0.25,
    angle_snap_deg=12.0,
    ridge_snap_tolerance=0.6,
    fill_holes=True,
    enforce_disjoint=True,
    min_roof_plane_area=0.0,
    tilt_column="tilt_deg",
    azimuth_column="azimuth_deg",
    method="wireframe",
    max_gap=0.3,
    edge_tolerance=0.45,
    node_merge_distance=0.6,
    wall_step_m=0.3,
    resolution=None,
    n_workers=None,
    progress=False
):
    """
    Straighten the facet boundaries of an already-detected roof.

    Takes the GeoDataFrame either detector produced and returns it with
    tidier geometry: pixel staircases simplified away, edges snapped to
    each facet's dominant orientation, boundaries shared by two facets
    snapped onto the exact line where their planes intersect, and holes
    closed. Attributes are carried through, with the areas recomputed.

    Parameters
    ----------
    roof_planes : gpd.GeoDataFrame or str
        Detector output, or a path to the GeoPackage it wrote.
    dsm_path : str, optional
        DSM to sample for each facet's height. Only the ridge snapping
        needs it: the tilt and azimuth columns fix a plane's orientation
        but not where it sits vertically, and two planes' intersection
        line depends on that. Without it the ridge snapping is skipped and
        a warning is issued; everything else still runs.
    simplify_tolerance : float, default=0.25
        Douglas-Peucker tolerance, in metres. Should be around the raster
        cell size - it is there to drop the pixel staircase, not detail.
    angle_snap_deg : float, default=12.0
        Maximum rotation applied when snapping an edge to the facet's
        dominant orientation. Edges further off keep their own direction.
        The candidate directions are the family {theta0 + k * 45 deg}, so
        they are never more than 22.5 degrees apart: at or above that every
        edge snaps and larger values do nothing. Below it, some edges keep
        their own direction and the facet stays less regular.
    ridge_snap_tolerance : float, default=0.6
        Distance in metres within which a vertex is pulled onto a shared
        plane-plane intersection line.
    fill_holes : bool, default=True
        Close holes inside a facet, except where another facet sits in it.
    enforce_disjoint : bool, default=True
        Guarantee the returned facets do not overlap. Snapping moves edges,
        so two neighbours can be pushed into each other; the larger facet
        keeps its shape and the smaller gives up the overlap.
    min_roof_plane_area : float, default=0.0
        Drop facets smaller than this after regularization.
    tilt_column, azimuth_column : str
        Column names carrying the plane orientation.
    method : {"wireframe", "intersections", "facet"}, default="wireframe"
        ``"wireframe"`` rebuilds every group of touching facets as one
        planar wireframe: a single simplified outline, junction nodes
        shared by the facets that meet there, and one straight edge per
        boundary between two facets, placed on their plane intersection.
        Neighbouring facets then share their vertices exactly and the gaps
        between them are closed. ``"intersections"`` builds the same
        wireframe, but turns every boundary into straight segments first -
        the plane intersection where it holds, otherwise a line fitted to
        the pixels and snapped to the roof's dominant orientation - and puts
        each node where the segments meeting at it collide, rather than at
        the pixel junction nudged towards the ridges. Junctions between
        steps and wrap-around boundaries come out as clean corners. On the
        digitised Alpicat roofs it scores within 0.003 mean IoU of
        ``"wireframe"`` at every ``edge_tolerance`` tried (0.2-0.6), and
        moves 15% of the facets by more than 1% of their area. ``"facet"``
        is the older behaviour, which
        regularizes each facet independently; ``simplify_tolerance``,
        ``enforce_disjoint`` only apply to it - wireframe faces are
        disjoint by construction.
    max_gap : float, default=0.3
        Wireframe and intersections only. Facets closer than this, in metres, belong to the
        same roof, and gaps up to this width between them are handed to the
        nearest facet. Wider gaps stay open: they are more often a step or
        a light well than a missed ridge.
    edge_tolerance : float, default=0.45
        Wireframe and intersections only. How far, in metres, a straight edge may stray from
        the pixel boundary it replaces. Governs the outline simplification
        and the smoothing of frayed boundaries between facets. The trade
        between fidelity and simple shapes, measured on the digitised
        Alpicat roofs: 0.3 gives mean IoU 0.698 at ~15 vertices per facet,
        0.45 gives 0.691 at ~11, 0.6 gives 0.684 at ~8. Larger values round
        small facets away first.
    node_merge_distance : float, default=0.6
        Wireframe and intersections only. Junctions closer than this, in metres, are one
        junction; outline nodes this close to an outline corner move onto
        it, which is how a hip ends exactly at the building's corner.
    wall_step_m : float or None, default=0.3
        Wireframe and intersections only. Two nearby facets whose planes are further apart in
        height than this where they meet are separated by a wall, and are
        regularized as separate roofs: their contours are simplified
        independently instead of being joined along a shared edge, and the
        gap between them is not closed. Where the two outlines then
        overlap, the higher roof keeps its shape. Needs ``dsm_path`` for the
        heights; None joins on distance alone. On the digitised Alpicat
        roofs 0.2-0.3 lifted mean IoU by 0.005-0.009 over None; 0.5 and 1.0
        let too many steps through.
    resolution : float, optional
        Wireframe and intersections only. Cell size of the grid the facets are rasterized on.
        Defaults to the DSM's cell size, or 0.2 m without a DSM.
    n_workers : int, optional
        Wireframe and intersections only. Processes the roofs are rebuilt in - each roof is
        independent of the others. Defaults to the CPU count; 1 runs in
        this process. The result does not depend on it.
    progress : bool, default=False
        Show a progress bar.

    Returns
    -------
    gpd.GeoDataFrame
        The same rows and columns, with regularized geometry and
        ``footprint_area`` / ``area`` recomputed. Rows whose geometry did
        not survive are dropped.
    """
    planes = _load_plane_frame(roof_planes)

    for column in (tilt_column, azimuth_column):
        if column not in planes.columns:
            raise ValueError(
                f"roof_planes has no {column!r} column; got "
                f"{sorted(planes.columns)}"
            )

    if len(planes) == 0:
        return planes

    dsm = dsm_transform = dsm_nodata = dsm_source = None

    if dsm_path:
        # Read window by window under each face rather than whole: a
        # municipality's refined DSM at 0.2 m is ~11 GB once in float64.
        dsm_source = rasterio.open(dsm_path)
        dsm = _WindowedBand(dsm_source)
        dsm_transform = dsm_source.transform
        dsm_nodata = dsm_source.nodata
    elif ridge_snap_tolerance > 0:
        warnings.warn(
            "regularize_roof_planes needs dsm_path to snap shared ridges: "
            "a plane-plane intersection depends on each plane's height, "
            "which tilt and azimuth alone do not give. Skipping that step.",
            stacklevel=2
        )

    faces = []

    rows = planes.itertuples()

    if progress:
        rows = tqdm(rows, total=len(planes), desc="Regularizing facets")

    for position, row in enumerate(rows):
        polygon = row.geometry

        if polygon is None or polygon.is_empty:
            continue

        parts = [p for p in _iter_polygons(polygon) if p.area > 0]

        if not parts:
            continue

        # A multipart facet is regularized part by part; they are separate
        # pieces of roof and share no orientation.
        for part in parts:
            a, b = _plane_coefficients(
                float(getattr(row, tilt_column)),
                float(getattr(row, azimuth_column))
            )

            c = None

            if dsm is not None:
                c = _anchor_height(part, a, b, dsm, dsm_transform, dsm_nodata)

            faces.append({
                "row": position,
                "a": a, "b": b,
                "c": 0.0 if c is None else c,
                "has_height": c is not None,
                "polygon": part,
            })

    if dsm_source is not None:
        dsm_source.close()

    if not faces:
        return planes.iloc[0:0]

    if method in ("wireframe", "intersections"):
        if resolution is None:
            resolution = (
                abs(dsm_transform.a) if dsm_transform is not None else 0.2
            )

        return _rebuild_plane_frame(
            planes,
            _regularize_wireframe_faces(
                faces,
                resolution=resolution,
                max_gap=max_gap,
                fill_holes=fill_holes,
                edge_tolerance=edge_tolerance,
                node_merge_distance=node_merge_distance,
                angle_snap_deg=angle_snap_deg,
                ridge_snap_tolerance=ridge_snap_tolerance if dsm is not None
                else 0.0,
                min_facet_area=max(min_roof_plane_area, resolution ** 2),
                wall_step_m=wall_step_m if dsm is not None else None,
                node_placement="intersect" if method == "intersections"
                else "solve",
                n_workers=n_workers,
                progress=progress,
            ),
            min_roof_plane_area
        )

    if method != "facet":
        raise ValueError(
            "method must be 'wireframe', 'intersections' or 'facet', "
            f"got {method!r}"
        )

    for face in faces:
        face["polygon"] = _clean_face_polygon(
            face["polygon"], simplify_tolerance
        )

    faces = [f for f in faces if f["polygon"] is not None]

    for face in faces:
        face["polygon"] = _regularize_polygon(
            face["polygon"], angle_snap_deg=angle_snap_deg
        )

    if dsm is not None and ridge_snap_tolerance > 0:
        _snap_shared_ridges(faces, tolerance=ridge_snap_tolerance)

    for face in faces:
        face["polygon"] = _drop_collinear_vertices(face["polygon"])

    if fill_holes:
        faces = _fill_face_holes(faces)

    if enforce_disjoint:
        faces = _enforce_disjoint_faces(
            faces, min_roof_plane_area=max(min_roof_plane_area, 0.0)
        )

    return _rebuild_plane_frame(planes, faces, min_roof_plane_area)


def _load_plane_frame(roof_planes):
    if isinstance(roof_planes, gpd.GeoDataFrame):
        return roof_planes

    if isinstance(roof_planes, str):
        return gpd.read_file(roof_planes)

    raise TypeError(
        "roof_planes must be a GeoDataFrame or a path to one, got "
        f"{type(roof_planes).__name__}"
    )


def _snap_shared_ridges(faces, *, tolerance):
    """
    Pull the boundary between two facets onto their planes' intersection.

    The same job as ``_snap_shared_edges_to_plane_intersections``, but
    indexed: a detection run covers thousands of facets and comparing every
    pair would dominate the runtime. Only facets whose polygons are within
    ``tolerance`` of each other are considered, which is what the pairwise
    version was testing anyway.
    """
    usable = [f for f in faces if f["has_height"] and f["polygon"] is not None]

    if len(usable) < 2:
        return

    geometries = [f["polygon"] for f in usable]

    tree = STRtree(geometries)

    for index, face in enumerate(usable):
        neighbours = tree.query(face["polygon"].buffer(tolerance))

        for other_index in np.atleast_1d(neighbours):
            other_index = int(other_index)

            if other_index <= index:
                continue

            other = usable[other_index]

            if face["polygon"].distance(other["polygon"]) > tolerance:
                continue

            line = _plane_intersection_line(face, other)

            if line is None:
                continue

            face["polygon"] = _snap_polygon_to_line(
                face["polygon"], line, tolerance
            )
            other["polygon"] = _snap_polygon_to_line(
                other["polygon"], line, tolerance
            )


def _rebuild_plane_frame(planes, faces, min_roof_plane_area):
    """Reassemble the GeoDataFrame, recomputing the areas that changed."""
    records = []
    geometries = []

    for face in faces:
        polygon = face["polygon"]

        if polygon is None or polygon.is_empty or polygon.area <= 0:
            continue

        if polygon.area < min_roof_plane_area:
            continue

        record = planes.iloc[face["row"]].drop(labels="geometry").to_dict()

        footprint = float(polygon.area)

        record["footprint_area"] = footprint

        tilt = float(record.get("tilt_deg", 0.0) or 0.0)

        record["area"] = footprint / max(
            np.cos(np.radians(tilt)), 1e-6
        )

        records.append(record)
        geometries.append(polygon)

    if not records:
        return planes.iloc[0:0]

    return gpd.GeoDataFrame(records, geometry=geometries, crs=planes.crs)


def _iter_polygons(geometry):
    """Yield the polygonal parts of any geometry, skipping the rest."""
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        yield geometry
        return

    if isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            if not part.is_empty:
                yield part
        return

    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def _clean_face_polygon(poly, simplify_tolerance):
    """Drop the triangulation's staircase, keeping the polygon valid."""
    if simplify_tolerance and simplify_tolerance > 0:
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)

    if not poly.is_valid:
        poly = poly.buffer(0)

    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 0:
        return None

    return poly


def _enforce_disjoint_faces(faces, *, min_roof_plane_area):
    """
    Guarantee non-overlapping faces after regularization.

    The triangle partition hands over disjoint faces, but snapping edges to
    dominant orientations moves them a little and can push two neighbours
    back into each other. Larger faces are laid down first and keep their
    regularized shape; each smaller one gives up whatever it shares with
    what is already placed. Overlaps at this point are slivers, so this
    normally changes nothing - it is here to make the guarantee hold
    rather than to hold most of the time.
    """
    if len(faces) < 2:
        return faces

    order = sorted(
        range(len(faces)),
        key=lambda k: faces[k]["polygon"].area,
        reverse=True
    )

    placed = None
    result = []

    for k in order:
        poly = faces[k]["polygon"]

        if placed is not None and poly.intersects(placed):
            try:
                poly = poly.difference(placed)
            except Exception:
                continue

            parts = [
                part for part in _iter_polygons(poly)
                if part.area >= min_roof_plane_area
            ]

            if not parts:
                continue

            # A face cut in two by its neighbour keeps its largest piece.
            poly = max(parts, key=lambda part: part.area)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if not isinstance(poly, Polygon) or poly.is_empty:
            continue

        if poly.area < min_roof_plane_area:
            continue

        faces[k]["polygon"] = poly

        result.append(faces[k])

        placed = poly if placed is None else unary_union([placed, poly])

    return result


def _polygon_with_interiors(shell_coords, source_poly):
    """
    Rebuild a polygon's shell while keeping the holes it already had.

    Every shape pass here works on the exterior ring, so reattaching the
    interiors explicitly is what stops a chimney or a light well from being
    quietly filled in. If a hole no longer fits inside the reshaped shell
    the rebuild is rejected, since a face that swallows its own hole is
    worse than one that was not tidied up.
    """
    interiors = [list(ring.coords) for ring in source_poly.interiors]

    try:
        rebuilt = Polygon(shell_coords, interiors)
    except Exception:
        return None

    if not rebuilt.is_valid:
        rebuilt = rebuilt.buffer(0)

    if not isinstance(rebuilt, Polygon) or rebuilt.is_empty or rebuilt.area <= 0:
        return None

    if len(rebuilt.interiors) != len(interiors):
        return None

    return rebuilt


def _dominant_orientation(coords):
    """
    Length-weighted dominant edge orientation, modulo 90 degrees.

    Angles are quadrupled before averaging so that the four directions of a
    rectangle map onto the same phase, which is the standard way to get a
    building's principal axis without one long edge dictating it alone.
    """
    deltas = np.diff(coords, axis=0)

    lengths = np.hypot(deltas[:, 0], deltas[:, 1])

    keep = lengths > 1e-9

    if not np.any(keep):
        return None

    angles = np.arctan2(deltas[keep, 1], deltas[keep, 0])

    weights = lengths[keep]

    mean_vector = np.sum(weights * np.exp(4j * angles))

    if abs(mean_vector) < 1e-12:
        return None

    return float(np.angle(mean_vector) / 4.0)


def _regularize_polygon(poly, *, angle_snap_deg, max_area_change=0.3):
    """
    Snap polygon edges to the dominant orientation and rebuild the corners.

    Each edge keeps its midpoint but takes the nearest orientation in the
    {theta0 + k * 45 deg} family when that costs less than angle_snap_deg;
    corners are then re-derived as the intersections of consecutive edge
    lines, so the result closes exactly and reads as a clean triangle,
    rectangle or trapezoid. The original polygon is returned unchanged if
    the rebuild fails or moves the area by more than ``max_area_change``.
    """
    if poly is None or poly.is_empty:
        return poly

    coords = np.asarray(poly.exterior.coords, dtype=np.float64)

    if coords.shape[0] < 4:
        return poly

    theta0 = _dominant_orientation(coords)

    if theta0 is None:
        return poly

    snap_limit = np.radians(angle_snap_deg)

    candidates = theta0 + np.arange(8) * (np.pi / 4.0)

    starts = coords[:-1]
    ends = coords[1:]

    deltas = ends - starts

    lines = []

    for k in range(starts.shape[0]):
        dx, dy = deltas[k]

        length = float(np.hypot(dx, dy))

        if length < 1e-9:
            continue

        angle = float(np.arctan2(dy, dx))

        # Smallest signed correction onto any candidate orientation.
        diffs = np.angle(np.exp(1j * (candidates - angle)))

        best = int(np.argmin(np.abs(diffs)))

        if abs(diffs[best]) <= snap_limit:
            angle = angle + float(diffs[best])

        midpoint = (starts[k] + ends[k]) / 2.0

        lines.append((midpoint, np.array([np.cos(angle), np.sin(angle)])))

    if len(lines) < 3:
        return poly

    new_coords = []

    for k in range(len(lines)):
        p0, d0 = lines[k - 1]
        p1, d1 = lines[k]

        cross = d0[0] * d1[1] - d0[1] * d1[0]

        if abs(cross) < 1e-6:
            # Consecutive edges are parallel: keep the corner they share.
            new_coords.append(tuple(starts[k % starts.shape[0]]))
            continue

        diff = p1 - p0

        t = (diff[0] * d1[1] - diff[1] * d1[0]) / cross

        new_coords.append(tuple(p0 + t * d0))

    if len(new_coords) < 3:
        return poly

    rebuilt = _polygon_with_interiors(new_coords, poly)

    if rebuilt is None:
        return poly

    if abs(rebuilt.area - poly.area) > max_area_change * poly.area:
        return poly

    return rebuilt


def _drop_collinear_vertices(
    poly,
    *,
    angle_tol_deg=6.0,
    min_edge_length=0.3,
    max_area_change=0.1
):
    """
    Remove the vertices that carry no shape.

    Snapping edges to a dominant orientation leaves runs of nearly collinear
    corners, and the concave hull contributes very short edges, so a face
    that is geometrically a trapezoid still reports seven vertices. Dropping
    both kinds is what turns it back into a four-corner trapezoid. The
    original is kept if the cleanup moves the area by more than
    ``max_area_change``.
    """
    if poly is None or poly.is_empty:
        return poly

    coords = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]

    if coords.shape[0] < 4:
        return poly

    cos_limit = np.cos(np.radians(180.0 - angle_tol_deg))

    changed = True

    while changed and coords.shape[0] > 3:
        changed = False

        for k in range(coords.shape[0]):
            previous = coords[k - 1]
            current = coords[k]
            following = coords[(k + 1) % coords.shape[0]]

            incoming = current - previous
            outgoing = following - current

            len_in = np.hypot(*incoming)
            len_out = np.hypot(*outgoing)

            if len_in < 1e-9 or len_out < 1e-9:
                drop = True
            else:
                cosine = float(
                    np.dot(incoming, outgoing) / (len_in * len_out)
                )

                # Straight-through corner, or an edge too short to matter.
                drop = (
                    cosine >= -cos_limit or
                    min(len_in, len_out) < min_edge_length
                )

            if not drop:
                continue

            candidate = np.delete(coords, k, axis=0)

            if candidate.shape[0] < 3:
                break

            trial = _polygon_with_interiors(candidate, poly)

            if trial is None:
                continue

            if abs(trial.area - poly.area) > max_area_change * poly.area:
                continue

            coords = candidate
            changed = True
            break

    result = _polygon_with_interiors(coords, poly)

    if result is None:
        return poly

    return result


def _plane_intersection_line(plane_a, plane_b):
    """
    XY line where two roof planes meet, as (point, unit direction).

    Both planes are graphs over xy, so subtracting them leaves the linear
    equation (da) x + (db) y + (dc) = 0. Returns None when the planes are
    close to parallel and no usable ridge exists.
    """
    da = plane_a["a"] - plane_b["a"]
    db = plane_a["b"] - plane_b["b"]
    dc = plane_a["c"] - plane_b["c"]

    norm = np.hypot(da, db)

    if norm < 1e-6:
        return None

    # Closest point of the line to the origin, plus its direction.
    point = np.array([-da * dc, -db * dc]) / (norm ** 2)

    direction = np.array([-db, da]) / norm

    return point, direction


def _snap_polygon_to_line(poly, line, tolerance):
    point, direction = line

    coords = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]

    if coords.shape[0] < 3:
        return poly

    offsets = coords - point

    along = offsets @ direction

    projected = point + along[:, None] * direction

    distances = np.linalg.norm(coords - projected, axis=1)

    near = distances <= tolerance

    if not np.any(near):
        return poly

    snapped = coords.copy()
    snapped[near] = projected[near]

    result = _polygon_with_interiors(snapped, poly)

    if result is None:
        return poly

    return result



# ============================================================
# Wireframe regularization for continuous roofs
#
# Regularizing facet by facet leaves a roof where no two facets share a
# vertex: each one is straightened on its own, and the ridges and hips
# come out as pairs of nearly-coincident lines. For a roof made of several
# facets that touch, this instead rebuilds it as one planar wireframe -
# a single simplified outline, junction nodes where three or more facets
# meet, and one straight edge per boundary between two facets, placed on
# the line where their planes intersect - and cuts the facets out of it.
# ============================================================

def _regularize_wireframe_faces(faces, *, progress, wall_step_m, n_workers=None,
                                **options):
    """
    Run the wireframe rebuild roof by roof; returns face dicts.

    A roof here is a set of facets that are near each other *and* meet
    without a wall between them. Each roof gets its own outline, so two
    roofs separated by a wall are simplified independently and their
    contours are not pulled onto each other. Where the independent outlines
    then overlap, the higher roof keeps its shape - in plan the wall is the
    edge of the upper roof - and the lower one gives up the overlap.

    Roofs are rebuilt independently of one another, so across a
    municipality they are spread over ``n_workers`` processes; the overlap
    resolution after it sees them in the same order either way.
    """
    groups = _group_faces(
        faces, options["max_gap"], wall_step_m=wall_step_m,
        n_workers=n_workers, progress=progress
    )

    results = _fork_map(
        _rebuild_roof_group, groups, (faces, options),
        n_workers=n_workers, progress=progress, desc="Regularizing roofs"
    )

    rebuilt_groups = []

    for members, (rebuilt, error) in zip(groups, results):
        if error is not None:
            # One awkward roof should not sink a municipality: keep its
            # facets as detected and say which.
            warnings.warn(
                f"wireframe regularization failed for a roof of "
                f"{len(members)} facet(s), kept unchanged: {error}",
                stacklevel=3
            )
            rebuilt = [(k, faces[k]["polygon"]) for k in members]

        rebuilt = [
            (k, polygon) for k, polygon in rebuilt
            if polygon is not None and not polygon.is_empty
        ]

        if rebuilt:
            rebuilt_groups.append(rebuilt)

    if len(rebuilt_groups) > 1:
        rebuilt_groups = _resolve_roof_overlaps(faces, rebuilt_groups)

    return [
        dict(faces[k], polygon=polygon)
        for rebuilt in rebuilt_groups
        for k, polygon in rebuilt
    ]


# What the workers of _fork_map read their inputs from: inherited with the
# forked process, so the facets are not pickled once per item.
_FORK_JOB = None


def _fork_map(func, items, job, *, n_workers=None, progress=False, desc=None,
              chunksize=None):
    """
    ``[func(item) for item in items]``, spread over forked processes.

    ``func`` finds ``job`` in ``_FORK_JOB``. Results come back in order, so
    the outcome does not depend on ``n_workers``; 1, or a platform without
    ``fork``, runs it in this process.
    """
    global _FORK_JOB
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(items)))
    can_fork = sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods()

    _FORK_JOB = job

    try:
        if n_workers == 1 or not can_fork:
            results = map(func, items)
            if progress:
                results = tqdm(results, total=len(items), desc=desc)
            return list(results)

        # The workers share this process's threads: n_workers processes each
        # with all of them would oversubscribe the cores n_workers times.
        threads = max(1, _process_threads() // n_workers)
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork"),
                                 initializer=_limit_worker_threads,
                                 initargs=(threads,)) as pool:
            if chunksize is None:
                chunksize = max(1, len(items) // (n_workers * 16))
            results = pool.map(func, items, chunksize=chunksize)
            if progress:
                results = tqdm(results, total=len(items), desc=desc)
            return list(results)
    finally:
        _FORK_JOB = None


def _process_threads():
    """The threads this process may use: numba's setting, or the CPUs it may run on."""
    try:
        import numba
        return numba.get_num_threads()
    except ImportError:
        return len(os.sched_getaffinity(0))


def _limit_worker_threads(threads):
    """Initializer of the _fork_map workers: numba and BLAS to ``threads``."""
    try:
        import numba
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    except ImportError:
        pass
    from threadpoolctl import threadpool_limits
    # Kept for the worker's lifetime: the limits hold while the object lives.
    global _WORKER_THREAD_LIMITS
    _WORKER_THREAD_LIMITS = threadpool_limits(limits=threads)


def _rebuild_roof_group(members):
    """One roof's wireframe rebuild as (rebuilt, None), or (None, error)."""
    faces, options = _FORK_JOB
    try:
        return _regularize_roof_wireframe(faces, members, **options), None
    except Exception as error:
        return None, str(error)


class _WindowedBand:
    """
    A raster band that reads only what is sliced from it.

    Stands in for the DSM array in ``_anchor_height``, which takes one
    small window under each facet: same values, same float64, without the
    whole raster in memory.
    """

    def __init__(self, source, band=1, dtype=np.float64):
        self.source = source
        self.band = band
        self.dtype = dtype
        self.shape = (source.height, source.width)

    def __getitem__(self, key):
        rows, cols = key
        window = Window(
            cols.start, rows.start, cols.stop - cols.start, rows.stop - rows.start
        )
        return self.source.read(self.band, window=window).astype(self.dtype)


def _roof_height(faces, rebuilt):
    """Area-weighted mean height of a rebuilt roof, at its facets' centroids."""
    heights, weights = [], []

    for k, polygon in rebuilt:
        face = faces[k]
        if not face["has_height"]:
            continue
        centroid = polygon.centroid
        heights.append(face["a"] * centroid.x + face["b"] * centroid.y + face["c"])
        weights.append(polygon.area)

    if not heights:
        return -np.inf

    return float(np.average(heights, weights=weights))


def _resolve_roof_overlaps(faces, rebuilt_groups):
    """Higher roofs first; each lower roof loses what overlaps them."""
    order = sorted(
        range(len(rebuilt_groups)),
        key=lambda g: _roof_height(faces, rebuilt_groups[g]),
        reverse=True
    )

    tree_geoms = []
    tree_bounds = []
    resolved = []

    for g in order:
        rebuilt = rebuilt_groups[g]
        outline = unary_union([polygon for _, polygon in rebuilt])

        # Bounding boxes first: testing every placed roof's polygon against
        # every new one grew with the square of the roofs in a municipality.
        # Polygons that intersect have intersecting boxes, so nothing is lost
        # and the blockers keep their order.
        if tree_bounds:
            bx = np.asarray(tree_bounds)
            ox0, oy0, ox1, oy1 = outline.bounds
            near = np.nonzero(
                (bx[:, 0] <= ox1) & (bx[:, 2] >= ox0)
                & (bx[:, 1] <= oy1) & (bx[:, 3] >= oy0)
            )[0]
        else:
            near = []

        blockers = [tree_geoms[i] for i in near if tree_geoms[i].intersects(outline)]

        if blockers:
            blocker = unary_union(blockers)
            kept = []
            for k, polygon in rebuilt:
                if not polygon.intersects(blocker):
                    kept.append((k, polygon))
                    continue
                cut = polygon.difference(blocker)
                parts = [q for q in _iter_polygons(cut) if q.area > 1e-4]
                if not parts:
                    continue
                kept.append((k, max(parts, key=lambda q: q.area)))
            rebuilt = kept

        if rebuilt:
            resolved.append(rebuilt)
            placed = unary_union([polygon for _, polygon in rebuilt])
            tree_geoms.append(placed)
            tree_bounds.append(placed.bounds)

    return resolved


def _group_faces(faces, max_gap, *, wall_step_m=None, n_workers=1, progress=False):
    """
    Union-find over facets that form one continuous roof.

    Two facets belong together when they are closer than ``max_gap`` and,
    along the stretch where they face each other, their planes are at the
    same height - within ``wall_step_m``. A larger step means a wall stands
    between them: a raised block, a lower annex, a parapet. Facets without
    a height estimate, or with ``wall_step_m`` None, are joined on distance
    alone.

    Which neighbours each facet joins is decided independently, over
    ``n_workers`` processes - buffering the staircase outlines for the
    height test is most of a municipality's regularization time. The unions
    are then made here, in facet order, as they always were.
    """
    parent = list(range(len(faces)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    geometries = [f["polygon"] for f in faces]
    tree = STRtree(geometries)

    joins = _fork_map(
        _face_joins, range(len(faces)), (faces, geometries, tree, max_gap, wall_step_m),
        n_workers=n_workers, progress=progress, desc="Grouping facets into roofs"
    )

    for k, others in enumerate(joins):
        for other in others:
            parent[find(other)] = find(k)

    groups = defaultdict(list)
    for k in range(len(faces)):
        groups[find(k)].append(k)

    return list(groups.values())


def _face_joins(k):
    """The later facets facet k forms one roof with: near, and no wall between."""
    faces, geometries, tree, max_gap, wall_step_m = _FORK_JOB
    geometry = geometries[k]
    joins = []

    for other in np.atleast_1d(tree.query(geometry.buffer(max_gap))):
        other = int(other)
        if other <= k:
            continue
        gap = geometry.distance(geometries[other])
        if gap > max_gap:
            continue
        if wall_step_m is not None and _height_step(
            faces[k], faces[other], gap
        ) > wall_step_m:
            continue
        joins.append(other)

    return joins


def _height_step(face_a, face_b, gap, *, reach=0.5, spacing=0.25):
    """
    Median height difference between two facets' planes where they meet.

    Evaluated on points of each facet's boundary lying within ``reach`` of
    the other facet, so a slope difference is only measured where it
    matters and not extrapolated across the whole roof. Returns 0 when
    either plane has no height, which keeps such facets joinable.
    """
    if not (face_a["has_height"] and face_b["has_height"]):
        return 0.0

    samples = []

    for this, other in ((face_a, face_b), (face_b, face_a)):
        near = this["polygon"].boundary.intersection(
            other["polygon"].buffer(gap + reach)
        )
        if near.is_empty or near.length == 0:
            continue
        count = max(int(near.length / spacing), 1)
        for distance in np.linspace(0.0, near.length, count + 1):
            point = near.interpolate(distance)
            samples.append((point.x, point.y))

    if not samples:
        return 0.0

    xy = np.asarray(samples)
    za = face_a["a"] * xy[:, 0] + face_a["b"] * xy[:, 1] + face_a["c"]
    zb = face_b["a"] * xy[:, 0] + face_b["b"] * xy[:, 1] + face_b["c"]

    return float(np.median(np.abs(za - zb)))


def _roof_label_grid(faces, members, *, resolution, max_gap, fill_holes):
    """
    Rasterize one roof's facets and close the gaps between them.

    Gaps narrower than max_gap (unassigned ridge pixels, stray returns) are
    handed to the nearest facet, so that the facets tile the roof and every
    boundary between two of them becomes a shared line.
    """
    union = unary_union([faces[k]["polygon"] for k in members])
    minx, miny, maxx, maxy = union.bounds
    pad = max_gap + 3 * resolution

    origin_x = np.floor((minx - pad) / resolution) * resolution
    origin_y = np.ceil((maxy + pad) / resolution) * resolution
    width = int(np.ceil((maxx + pad - origin_x) / resolution))
    height = int(np.ceil((origin_y - (miny - pad)) / resolution))

    transform = from_origin(origin_x, origin_y, resolution, resolution)

    order = sorted(members, key=lambda k: faces[k]["polygon"].area,
                   reverse=True)

    labels = rasterize(
        [(faces[k]["polygon"], k + 1) for k in order],
        out_shape=(height, width), transform=transform, fill=0,
        dtype="int32",
    )

    mask = labels > 0

    radius = max(int(round(max_gap / (2.0 * resolution))), 1)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx ** 2 + yy ** 2) <= radius ** 2

    mask = binary_closing(mask, structure=disk) | mask

    if fill_holes:
        mask = binary_fill_holes(mask)

    _, nearest = distance_transform_edt(labels == 0, return_indices=True)
    filled = labels[nearest[0], nearest[1]]
    labels = np.where(mask, filled, 0)

    return labels, transform


def _drop_label_specks(labels, min_pixels):
    """Hand pixel islands too small to be a facet to their neighbours."""
    speck = np.zeros(labels.shape, dtype=bool)

    for value in np.unique(labels):
        if value == 0:
            continue
        components, n = label(labels == value)
        if n <= 1:
            continue
        sizes = np.bincount(components.ravel())
        small = np.where(sizes < min_pixels)[0]
        small = small[small > 0]
        if small.size:
            speck |= np.isin(components, small)

    if not np.any(speck):
        return labels

    inside = labels > 0
    source = np.where(speck, 0, labels)
    _, nearest = distance_transform_edt(source == 0, return_indices=True)
    refilled = source[nearest[0], nearest[1]]
    return np.where(inside, np.where(speck, refilled, labels), 0)


def _pixel_boundary_chains(labels):
    """
    Trace the boundaries between labels as chains of pixel corners.

    Returns the junction corners (where three or more labels meet) and, for
    every pair of touching labels, the chains of corners separating them,
    each ending at a junction or closing on itself.
    """
    padded = np.pad(labels, 1)
    h, w = padded.shape

    # Corner (i, j) sits between padded rows i, i+1 and columns j, j+1.
    a = padded[:-1, :-1]
    b = padded[:-1, 1:]
    c = padded[1:, :-1]
    d = padded[1:, 1:]

    stacked = np.sort(np.stack([a, b, c, d]), axis=0)
    distinct = 1 + np.count_nonzero(np.diff(stacked, axis=0), axis=0)
    diagonal = (distinct == 2) & (a == d) & (b == c) & (a != b)
    junction = (distinct >= 3) | diagonal

    cw = w - 1

    def cid(i, j):
        return i * cw + j

    edges = defaultdict(list)

    # Vertical boundary between padded[i, j] and padded[i, j+1]:
    # from corner (i-1, j) to corner (i, j).
    ii, jj = np.nonzero(padded[:, :-1] != padded[:, 1:])
    for i, j in zip(ii, jj):
        p, q = padded[i, j], padded[i, j + 1]
        edges[(min(p, q), max(p, q))].append((cid(i - 1, j), cid(i, j)))

    # Horizontal boundary between padded[i, j] and padded[i+1, j]:
    # from corner (i, j-1) to corner (i, j).
    ii, jj = np.nonzero(padded[:-1, :] != padded[1:, :])
    for i, j in zip(ii, jj):
        p, q = padded[i, j], padded[i + 1, j]
        edges[(min(p, q), max(p, q))].append((cid(i, j - 1), cid(i, j)))

    junction_ids = set(int(v) for v in np.flatnonzero(junction.ravel()))

    chains = []

    for pair, pair_edges in edges.items():
        adjacency = defaultdict(list)
        for u, v in pair_edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        breaks = {
            n for n, nbrs in adjacency.items()
            if n in junction_ids or len(nbrs) != 2
        }
        junction_ids |= breaks

        used = set()

        def walk(start, first):
            path = [start, first]
            used.add((min(start, first), max(start, first)))
            previous, current = start, first
            while current not in breaks and current != start:
                step = None
                for nxt in adjacency[current]:
                    key = (min(current, nxt), max(current, nxt))
                    if key not in used:
                        step = nxt
                        break
                if step is None:
                    break
                used.add((min(current, step), max(current, step)))
                previous, current = current, step
                path.append(current)
            return path

        for start in breaks:
            for nxt in adjacency[start]:
                if (min(start, nxt), max(start, nxt)) not in used:
                    chains.append((pair, walk(start, nxt)))

        for u, v in pair_edges:
            if (min(u, v), max(u, v)) not in used:
                chains.append((pair, walk(u, v)))

    return chains, junction_ids, cw


def _douglas_peucker_indices(points, tolerance):
    """Indices of the vertices Douglas-Peucker keeps on an open polyline."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        start, end = points[s], points[e]
        seg = end - start
        norm = np.hypot(*seg)
        rel = points[s + 1:e] - start
        if norm < 1e-9:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / norm
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            mid = s + 1 + k
            keep.add(mid)
            stack.append((s, mid))
            stack.append((mid, e))
    return sorted(keep)


def _regularize_roof_wireframe(
    faces, members, *, resolution, max_gap, fill_holes, edge_tolerance,
    node_merge_distance, angle_snap_deg, ridge_snap_tolerance,
    min_facet_area, node_placement="solve", debug=None,
):
    labels, transform = _roof_label_grid(
        faces, members, resolution=resolution, max_gap=max_gap,
        fill_holes=fill_holes,
    )
    labels = _majority_filter(labels, radius=edge_tolerance / 2.0,
                              resolution=resolution)
    labels = _drop_label_specks(
        labels, max(int(min_facet_area / resolution ** 2), 1)
    )

    ox, oy = transform.c, transform.f
    results = []

    components, n_components = label(labels > 0)

    for component in range(1, n_components + 1):
        roof = np.where(components == component, labels, 0)
        results.extend(_wireframe_component(
            faces, roof, ox, oy, resolution=resolution,
            edge_tolerance=edge_tolerance,
            node_merge_distance=node_merge_distance,
            angle_snap_deg=angle_snap_deg,
            ridge_snap_tolerance=ridge_snap_tolerance,
            node_placement=node_placement,
            debug=debug,
        ))

    return results


def _wireframe_component(
    faces, roof, ox, oy, *, resolution, edge_tolerance, node_merge_distance,
    angle_snap_deg, ridge_snap_tolerance, node_placement="solve", debug=None,
):
    transform = from_origin(ox, oy, resolution, resolution)

    # ---- outline ------------------------------------------------------
    outline = unary_union([
        shape(geom) for geom, value in shapes(
            (roof > 0).astype("uint8"), mask=roof > 0, transform=transform
        ) if value == 1
    ])
    outline = max(_iter_polygons(outline), key=lambda p: p.area)

    outline = _regularize_outline(
        outline, edge_tolerance=edge_tolerance,
        angle_snap_deg=angle_snap_deg, min_edge_length=node_merge_distance,
    )

    present = [int(v) for v in np.unique(roof) if v > 0]

    if len(present) == 1:
        return [(present[0] - 1, outline)]

    # ---- boundary graph -----------------------------------------------
    chains, junction_ids, cw = _pixel_boundary_chains(roof)

    def corner_xy(ids):
        ids = np.asarray(ids)
        i, j = ids // cw, ids % cw
        # Padded grid is shifted one pixel up-left of the roof grid.
        return np.column_stack([ox + j * resolution, oy - i * resolution])

    endpoints = sorted({
        n for _, path in chains for n in (path[0], path[-1])
        if n in junction_ids
    })
    endpoint_index = {n: k for k, n in enumerate(endpoints)}
    endpoint_xy = corner_xy(endpoints) if endpoints else np.zeros((0, 2))

    # Cluster junction corners that belong to one physical junction.
    parent = list(range(len(endpoints)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    if len(endpoints) > 1:
        for p, q in cKDTree(endpoint_xy).query_pairs(node_merge_distance):
            parent[find(p)] = find(q)

    cluster_of = {}
    nodes = []  # dict(xy, labels, fixed, line)
    for k in range(len(endpoints)):
        root = find(k)
        if root not in cluster_of:
            cluster_of[root] = len(nodes)
            nodes.append({"members": []})
        nodes[cluster_of[root]]["members"].append(k)

    for node in nodes:
        node["xy0"] = endpoint_xy[node["members"]].mean(axis=0)
        node["labels"] = set()

    def node_of(corner):
        return cluster_of[find(endpoint_index[corner])]

    edges = []  # dict(u, v, pair, raw)
    for pair, path in chains:
        xy = corner_xy(path)
        if pair[0] == 0:
            for end in (path[0], path[-1]):
                if end in endpoint_index:
                    nodes[node_of(end)]["labels"].update(pair)
            continue

        if path[0] not in endpoint_index:
            # Closed loop: a facet enclosed by one other facet.
            ring = Polygon(xy)
            ring = _clean_face_polygon(ring, edge_tolerance)
            if ring is not None:
                ring = _regularize_polygon(
                    ring, angle_snap_deg=angle_snap_deg
                )
                coords = np.asarray(ring.exterior.coords)
                u = len(nodes)
                for p in coords[:-1]:
                    nodes.append({"xy0": p, "labels": set(pair)})
                count = len(coords) - 1
                for s in range(count):
                    edges.append({
                        "chain": [u + s, u + (s + 1) % count],
                        "pair": pair, "raw": xy,
                        "lines": [_line_through(coords[s], coords[s + 1])],
                    })
            continue

        u, v = node_of(path[0]), node_of(path[-1])
        nodes[u]["labels"].update(pair)
        nodes[v]["labels"].update(pair)

        # Two planes meet along a straight line, so a boundary that follows
        # their intersection is one segment however frayed its pixels are.
        # Only a boundary that does not (a step, a facet wrapping round
        # another) keeps intermediate vertices.
        if _ridge_line(faces, pair, xy, ridge_snap_tolerance) is not None:
            keep = [0, len(xy) - 1]
        else:
            keep = _douglas_peucker_indices(xy, 2.0 * edge_tolerance)
        inner = []
        breaks = [0]
        for index in keep[1:-1]:
            point = xy[index]
            if (np.hypot(*(point - nodes[u]["xy0"])) < node_merge_distance or
                    np.hypot(*(point - nodes[v]["xy0"])) < node_merge_distance):
                continue
            inner.append(len(nodes))
            breaks.append(index)
            nodes.append({"xy0": point, "labels": set(pair)})
        breaks.append(len(xy) - 1)

        if u == v and not inner:
            continue

        edges.append({"chain": [u] + inner + [v], "pair": pair, "raw": xy,
                      "breaks": breaks})

    # Two chains of the same pair joining the same nodes are one ridge.
    unique = {}
    for edge in edges:
        chain = edge["chain"]
        key = (edge["pair"], frozenset((chain[0], chain[-1])), len(chain))
        if len(chain) > 2:
            key = id(edge)
        if key not in unique or len(edge["raw"]) > len(unique[key]["raw"]):
            unique[key] = edge
    edges = list(unique.values())

    # ---- constraints and solve -----------------------------------------
    ring_coords = np.asarray(outline.exterior.coords)

    if node_placement == "intersect":
        # Every boundary is first a set of straight segments - the ridge
        # line where it holds, a line fitted to the pixels elsewhere - and
        # the nodes are then put where those segments collide.
        theta0 = _dominant_orientation(ring_coords)
        for edge in edges:
            if "lines" not in edge:
                edge["lines"] = _boundary_segment_lines(
                    faces, edge, ridge_snap_tolerance=ridge_snap_tolerance,
                    theta0=theta0, angle_snap_deg=angle_snap_deg,
                )
        solve = dict(w_data=0.05, max_shift=2.0 * edge_tolerance)
    else:
        for edge in edges:
            edge.pop("lines", None)
        solve = {}

    for _ in range(4):
        _attach_outline_constraints(nodes, ring_coords, node_merge_distance)
        _solve_node_positions(
            nodes, edges, faces, ridge_snap_tolerance=ridge_snap_tolerance,
            **solve
        )
        if not _merge_close_nodes(nodes, edges, node_merge_distance):
            break

    # ---- faces ---------------------------------------------------------
    lines = [LineString(ring_coords)]
    lines += [LineString(r.coords) for r in outline.interiors]

    for edge in edges:
        coords = [nodes[k]["xy"] for k in edge["chain"]]
        coords = np.asarray(coords, dtype=float)
        # Push ends that stop on the outline a hair past it, so the line
        # work is noded there even with rounding.
        for end, prev in ((0, 1), (-1, -2)):
            if 0 in nodes[edge["chain"][end]]["labels"]:
                step = coords[end] - coords[prev]
                length = np.hypot(*step)
                if length > 1e-9:
                    coords[end] = coords[end] + step / length * 0.05
        lines.append(LineString(coords))

    pieces = []
    for poly in polygonize(unary_union(lines)):
        clipped = poly.intersection(outline)
        for part in _iter_polygons(clipped):
            if part.area > 1e-4:
                pieces.append(part)

    h, w = roof.shape
    xs = ox + (np.arange(w) + 0.5) * resolution
    ys = oy - (np.arange(h) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys)
    inside = roof > 0
    px, py, pl = gx[inside], gy[inside], roof[inside]

    owner = []
    for piece in pieces:
        hit = contains_xy(piece, px, py)
        if np.any(hit):
            values, counts = np.unique(pl[hit], return_counts=True)
            owner.append(int(values[np.argmax(counts)]))
        else:
            owner.append(None)

    # Pieces holding no pixel centre go to the neighbour they touch most.
    for k, piece in enumerate(pieces):
        if owner[k] is not None:
            continue
        best, best_len = None, 0.0
        for m, other in enumerate(pieces):
            if m == k or owner[m] is None:
                continue
            shared = piece.boundary.intersection(other.boundary).length
            if shared > best_len:
                best, best_len = owner[m], shared
        owner[k] = best

    by_label = defaultdict(list)
    for piece, value in zip(pieces, owner):
        if value is not None:
            by_label[value].append(piece)

    if debug is not None:
        debug.update(nodes=nodes, edges=edges, outline=outline, roof=roof,
                     pieces=pieces, owner=owner)

    # A facet cut in two by the wireframe keeps its largest piece; each
    # stray piece goes to the facet it shares the longest boundary with.
    merged = {}
    strays = []
    for value, parts in by_label.items():
        geometry = unary_union(parts)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        polygons = sorted(_iter_polygons(geometry), key=lambda q: q.area,
                          reverse=True)
        if not polygons:
            continue
        merged[value] = polygons[0]
        strays.extend(polygons[1:])

    for stray in strays:
        best, best_length = None, 0.0
        for value, polygon in merged.items():
            length = stray.boundary.intersection(polygon.boundary).length
            if length > best_length:
                best, best_length = value, length
        if best is None:
            continue
        joined = unary_union([merged[best], stray])
        if isinstance(joined, Polygon):
            merged[best] = joined

    # A facet whose boundaries did not close into a cell of the wireframe
    # (a dangling edge, two junctions merged across it) would otherwise
    # vanish into a neighbour. Put it back from its own pixels.
    min_pixels = max(node_merge_distance ** 2 / resolution ** 2, 4)
    counts = dict(zip(*np.unique(pl, return_counts=True)))

    for value, count in counts.items():
        if value in merged or count < min_pixels:
            continue
        own = unary_union([
            shape(geom) for geom, v in shapes(
                (roof == value).astype("uint8"), mask=roof == value,
                transform=transform
            ) if v == 1
        ])
        own = max(_iter_polygons(own), key=lambda q: q.area, default=None)
        if own is None:
            continue
        own = _regularize_outline(
            own, edge_tolerance=edge_tolerance,
            angle_snap_deg=angle_snap_deg, min_edge_length=node_merge_distance,
        ).intersection(outline)
        own = max(_iter_polygons(own), key=lambda q: q.area, default=None)
        if own is None:
            continue
        for other in list(merged):
            cut = merged[other].difference(own)
            cut = max(_iter_polygons(cut), key=lambda q: q.area, default=None)
            if cut is None:
                del merged[other]
            else:
                merged[other] = cut
        merged[value] = own

    out = []
    for value, polygon in merged.items():
        out.append((value - 1, _drop_collinear_vertices(
            polygon, min_edge_length=0.0
        )))

    return out


def _attach_outline_constraints(nodes, ring_coords, corner_distance):
    """Pin outline nodes to an outline corner, or let them slide along it."""
    corners = ring_coords[:-1]

    for node in nodes:
        node["fixed"] = None
        node["slide"] = None

        if 0 not in node["labels"]:
            continue

        p = node["xy0"]
        dist = np.hypot(*(corners - p).T)
        k = int(np.argmin(dist))

        if dist[k] <= corner_distance:
            node["fixed"] = corners[k]
            continue

        best = None
        for s0, s1 in zip(ring_coords[:-1], ring_coords[1:]):
            seg = s1 - s0
            length = np.hypot(*seg)
            if length < 1e-9:
                continue
            t = np.clip(np.dot(p - s0, seg) / length ** 2, 0.0, 1.0)
            gap = np.hypot(*(p - (s0 + t * seg)))
            if best is None or gap < best[0]:
                normal = np.array([-seg[1], seg[0]]) / length
                best = (gap, normal, float(normal @ s0))

        if best is not None:
            node["slide"] = best[1:]


def _solve_node_positions(nodes, edges, faces, *, ridge_snap_tolerance,
                          w_data=0.15, w_ridge=1.0, w_hard=50.0,
                          max_shift=None):
    """
    Least-squares placement of the wireframe nodes.

    Every node is pulled weakly towards where the pixels put it, strongly
    onto the outline when it sits on it, and every node of a boundary
    between two facets onto the line where their planes intersect - when
    that line is where the boundary actually runs. Three facets meeting at
    a node therefore meet at the point their three planes share.

    An edge carrying ``lines`` - one (normal, offset) per segment of its
    chain, None where there is none - has both ends of every segment pulled
    onto that segment's line instead, so each node lands where the segments
    meeting at it collide. Nearly parallel segments collide far away; a node
    that moves more than ``max_shift`` from its pixel position is held there
    and the solve repeated.
    """
    n = len(nodes)
    anchored = set()

    for _ in range(3):
        rows, rhs, weights = [], [], []

        def add(k, coef, value, weight):
            row = np.zeros(2 * n)
            row[2 * k:2 * k + 2] = coef
            rows.append(row)
            rhs.append(value)
            weights.append(weight)

        for k, node in enumerate(nodes):
            if node["fixed"] is not None:
                add(k, (1.0, 0.0), node["fixed"][0], w_hard)
                add(k, (0.0, 1.0), node["fixed"][1], w_hard)
                continue
            w = w_ridge if k in anchored else w_data
            add(k, (1.0, 0.0), node["xy0"][0], w)
            add(k, (0.0, 1.0), node["xy0"][1], w)
            if node["slide"] is not None:
                normal, offset = node["slide"]
                add(k, normal, offset, w_hard)

        for edge in edges:
            chain = edge["chain"]
            lines = edge.get("lines")
            if lines is None:
                line = _ridge_line(faces, edge["pair"], edge["raw"],
                                   ridge_snap_tolerance)
                lines = [line] * (len(chain) - 1)
            for s, line in enumerate(lines):
                if line is None:
                    continue
                normal, offset = line
                for k in chain[s:s + 2]:
                    if nodes[k]["fixed"] is None:
                        add(k, normal, offset, w_ridge)

        A = np.asarray(rows) * np.asarray(weights)[:, None]
        B = np.asarray(rhs) * np.asarray(weights)
        solution = np.linalg.lstsq(A, B, rcond=None)[0].reshape(n, 2)

        if max_shift is None:
            break

        far = {
            k for k, node in enumerate(nodes)
            if node["fixed"] is None and k not in anchored
            and np.hypot(*(solution[k] - node["xy0"])) > max_shift
        }
        if not far:
            break
        anchored |= far

    for k, node in enumerate(nodes):
        node["xy"] = solution[k]


def _line_through(p, q):
    """The line through two points as (unit normal, offset)."""
    direction = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
    length = np.hypot(*direction)
    if length < 1e-9:
        return None
    normal = np.array([-direction[1], direction[0]]) / length
    return normal, float(normal @ np.asarray(p, dtype=float))


def _boundary_segment_lines(faces, edge, *, ridge_snap_tolerance, theta0,
                            angle_snap_deg):
    """
    One straight line per segment of a boundary between two facets.

    Where the boundary runs along the two planes' intersection that line is
    the segment - two planes meet along a straight line. Elsewhere (a step,
    a facet wrapping round another) each segment is the line fitted to the
    stretch of pixel corners it replaces, turned onto the roof's dominant
    orientation {theta0 + k * 45 deg} when that is within ``angle_snap_deg``.
    """
    pieces = len(edge["chain"]) - 1
    ridge = _ridge_line(faces, edge["pair"], edge["raw"], ridge_snap_tolerance)

    if ridge is not None:
        return [ridge] * pieces

    raw = edge["raw"]
    breaks = edge.get("breaks", [0, len(raw) - 1])
    if len(breaks) != pieces + 1:
        return [None] * pieces

    snap = np.radians(angle_snap_deg)
    lines = []

    for b0, b1 in zip(breaks[:-1], breaks[1:]):
        points = raw[b0:b1 + 1]
        if len(points) < 2:
            lines.append(None)
            continue

        centre = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
        angle = float(np.arctan2(vt[0, 1], vt[0, 0]))

        if theta0 is not None and snap > 0:
            diffs = np.angle(np.exp(1j * (theta0 + np.arange(8) * np.pi / 4 - angle)))
            best = int(np.argmin(np.abs(diffs)))
            if abs(diffs[best]) <= snap:
                angle += float(diffs[best])

        direction = np.array([np.cos(angle), np.sin(angle)])
        lines.append(_line_through(centre, centre + direction))

    return lines


def _merge_close_nodes(nodes, edges, distance):
    """
    Collapse wireframe edges shorter than ``distance`` into one node.

    One physical junction often shows up in the pixels as a few corners a
    metre apart; once the solve has pulled them together they are merged
    and the solve runs again. Returns whether anything was merged.
    """
    parent = list(range(len(nodes)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    live = sorted({k for edge in edges for k in edge["chain"]})
    merged = False

    if len(live) > 1:
        xy = np.asarray([nodes[k]["xy"] for k in live])
        for p, q in sorted(cKDTree(xy).query_pairs(distance)):
            ru, rv = find(live[p]), find(live[q])
            if ru == rv:
                continue
            if nodes[ru]["fixed"] is not None and \
                    nodes[rv]["fixed"] is not None:
                continue
            if nodes[rv]["fixed"] is not None:
                ru, rv = rv, ru
            parent[rv] = ru
            merged = True

    if not merged:
        return False

    groups = defaultdict(list)
    for k in range(len(nodes)):
        groups[find(k)].append(k)

    for root, members in groups.items():
        if len(members) == 1:
            continue
        nodes[root]["xy0"] = np.mean([nodes[k]["xy"] for k in members],
                                     axis=0)
        fixed = [nodes[k]["fixed"] for k in members
                 if nodes[k]["fixed"] is not None]
        if fixed:
            nodes[root]["xy0"] = fixed[0]
        labels = set()
        for k in members:
            labels |= nodes[k]["labels"]
        nodes[root]["labels"] = labels

    kept = []
    for edge in edges:
        chain = []
        old_lines = edge.get("lines")
        lines = []
        for s, k in enumerate(edge["chain"]):
            r = find(k)
            if not chain or chain[-1] != r:
                if chain and old_lines is not None:
                    # Segments collapsed into this one: the last stands.
                    lines.append(old_lines[s - 1])
                chain.append(r)
        if len(chain) < 2:
            continue
        edge["chain"] = chain
        if old_lines is not None:
            edge["lines"] = lines
        kept.append(edge)
    edges[:] = kept

    return True


def _majority_filter(labels, *, radius, resolution):
    """
    Give each roof pixel the label most common in a disk around it.

    Boundaries between two facets arrive frayed - fingers of one facet
    reaching into the next a pixel or two wide - and every finger would
    become a vertex. The vote irons them out without moving the outline:
    pixels outside the roof never change, and none inside become outside.
    """
    cells = int(round(radius / resolution))
    values = [v for v in np.unique(labels) if v > 0]

    if cells < 1 or len(values) < 2:
        return labels

    yy, xx = np.mgrid[-cells:cells + 1, -cells:cells + 1]
    disk = ((xx ** 2 + yy ** 2) <= cells ** 2).astype(np.float32)

    votes = np.stack([
        convolve((labels == v).astype(np.float32), disk, mode="constant")
        for v in values
    ])

    winner = np.asarray(values)[np.argmax(votes, axis=0)]

    return np.where(labels > 0, winner, 0)


def _regularize_outline(outline, *, edge_tolerance, angle_snap_deg,
                        min_edge_length):
    """
    Simplify the roof outline and square it up where that is honest.

    Snapping to the dominant orientation suits rectilinear buildings and
    distorts the rest, so each snap strength is only accepted while the
    result stays within twice ``edge_tolerance`` of the simplified outline;
    otherwise a weaker one is tried, down to no snapping at all.
    """
    simplified = _clean_face_polygon(outline, edge_tolerance) or outline

    for snap in (angle_snap_deg, min(angle_snap_deg, 8.0), 0.0):
        candidate = simplified
        if snap > 0:
            candidate = _regularize_polygon(simplified, angle_snap_deg=snap)
        candidate = _drop_collinear_vertices(
            candidate, min_edge_length=min_edge_length
        )
        if candidate.hausdorff_distance(simplified) <= 2.0 * edge_tolerance:
            return candidate

    return simplified


def _ridge_line(faces, pair, raw_xy, tolerance):
    """
    The two facets' plane intersection, as (normal, offset), when it runs
    where their pixel boundary does; None otherwise.

    A line far from the boundary means the planes' heights are off or the
    facets do not actually meet in a ridge (a step between two roofs), and
    pulling the boundary onto it would move it somewhere wrong.
    """
    fi, fj = faces[pair[0] - 1], faces[pair[1] - 1]

    if not (fi["has_height"] and fj["has_height"]):
        return None

    line = _plane_intersection_line(fi, fj)

    if line is None:
        return None

    point, direction = line
    normal = np.array([-direction[1], direction[0]])
    offset = float(normal @ point)

    if np.median(np.abs(raw_xy @ normal - offset)) > tolerance:
        return None

    return normal, offset

