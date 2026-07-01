# -*- coding: utf-8 -*-

from typing import List, Dict
import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import (sobel, label, laplace, gaussian_filter, generate_binary_structure, binary_dilation,
                           binary_closing, maximum_filter, binary_erosion)
from tqdm import tqdm
try:
    from .lidar import _read_laz
    from .dsm_dem_estimation import _aligned_grid, build_digital_model_from_lidar_tiled
    from .utils import _write_array_to_tiff, _filter_by_class_priority_grid
except ImportError:
    from greenshadow.lidar import _read_laz
    from greenshadow.dsm_dem_estimation import _aligned_grid, build_digital_model_from_lidar_tiled
    from greenshadow.utils import _write_array_to_tiff, _filter_by_class_priority_grid
from rasterio.features import shapes
from rasterio.transform import rowcol
from shapely.geometry import shape, MultiPoint, Point
from shapely.ops import unary_union
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from rasterio.warp import reproject
from rasterio.enums import Resampling
from shapely import contains_xy
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
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
    ridge_splitting_roof_planes=False
):
    # print(inspect.getsource(_read_laz))
    # =====================================================
    # Set paths from DSMs when detection is LIDAR refined
    # =====================================================

    lidar_dsm_path = None
    lidar_slope_path = None
    lidar_aspect_path = None
    lidar_classification_path = None

    if lidar_refinement:
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

    # =====================================================
    # Optional LIDAR refinement
    # =====================================================
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

        building_pts = np.column_stack([
            building_pts_struct["x"],
            building_pts_struct["y"],
            building_pts_struct["z"],
            building_pts_struct["classification"],
        ]).astype(np.float32)

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
        
        gpd.GeoDataFrame({"geometry":[building_geom]}, crs=f"EPSG:{crs.to_epsg()}").to_file("test.gpkg",driver="GPKG")
        if building_pts is not None and len(building_pts) > 0:

            # ---------------------------------------------
            # 2️⃣ Build refined DSM from in-memory points
            # ---------------------------------------------
            dsm_ref, class_ref, slope_ref, aspect_ref, transform_ref = \
                build_digital_model_from_lidar_tiled(
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
                    aggregation="smart_for_roofs",
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

            buildings_mask = np.isin(dsm_class, building_class_value)
            if dem_path:
                buildings_mask &= (dsm - dem > min_height_m)

    # Set the number of minimum pixels per roof, using the digital models resolution and the minimum roof plane area.
    min_pixels = np.ceil(min_roof_plane_area / (abs(transform[0]) ** 2))

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
    cluster_global_id = 1

    for part_id in tqdm(range(1, n_buildings + 1), desc="Processing building roof parts (only continuous)"):

        building_mask = labeled_buildings == part_id
        if not np.any(building_mask):
            continue

        # --------------------------------------------
        # Bounding box
        # --------------------------------------------
        rows, cols = np.where(building_mask)

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        row_slice = slice(min_row, max_row + 1)
        col_slice = slice(min_col, max_col + 1)

        local_mask = building_mask[row_slice, col_slice]
        local_dsm = dsm[row_slice, col_slice]
        local_nx = nx[row_slice, col_slice]
        local_ny = ny[row_slice, col_slice]
        local_nz = nz[row_slice, col_slice]
        local_transform = transform * rasterio.Affine.translation(min_col, min_row)

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

        ridge_mask[row_slice, col_slice] |= local_ridge_mask.astype(np.uint8)
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

                # write into global raster
                clustering_mask[
                    row_slice,
                    col_slice
                ][cluster_pixels] = cluster_global_id

                cluster_global_id += 1

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
            local_ridge_mask

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

        # Graph-based planes merging
        updated_planes, plane_id_raster = _merge_similar_adjacent_planes_graph(
            plane_id_raster,
            updated_planes,
            local_dsm,
            local_transform,
            normal_angle_threshold_deg=2.5,
            ridge_mask=local_ridge_mask

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

        # Roof planes vectorisation
        gdf_planes = _polygonize_and_compute_parameters(
            plane_id_raster,
            updated_planes,
            local_dsm,
            local_transform,
            part_id,
            crs,
            simplify_tolerance=None
        )

        planes_list.append(gdf_planes)

    gdf_all_planes = pd.concat(planes_list, ignore_index=True)

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

    if output_path:
        gdf_all_planes.to_file(output_path)

    return gdf_all_planes


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


def _segment_roof_by_surface_changes(
    dsm,
    slope,
    aspect,
    mask,
    slope_thresh=0.12,
    aspect_thresh_deg=18,
    height_thresh=0.30,
    smooth_sigma=1.0
):

    # -------------------------------
    # Smooth inputs
    # -------------------------------
    dsm_s = gaussian_filter(dsm, smooth_sigma)
    slope_s = gaussian_filter(slope, smooth_sigma)

    aspect_sin = np.sin(aspect)
    aspect_cos = np.cos(aspect)

    aspect_sin = gaussian_filter(aspect_sin, smooth_sigma)
    aspect_cos = gaussian_filter(aspect_cos, smooth_sigma)

    aspect_s = np.arctan2(aspect_sin, aspect_cos)

    # -------------------------------
    # height differences
    # -------------------------------
    dzx = np.abs(dsm_s[:, :-1] - dsm_s[:, 1:])
    dzy = np.abs(dsm_s[:-1, :] - dsm_s[1:, :])

    # -------------------------------
    # slope differences
    # -------------------------------
    dsx = np.abs(slope_s[:, :-1] - slope_s[:, 1:])
    dsy = np.abs(slope_s[:-1, :] - slope_s[1:, :])

    # -------------------------------
    # aspect differences
    # -------------------------------
    ax = np.abs(aspect_s[:, :-1] - aspect_s[:, 1:])
    ay = np.abs(aspect_s[:-1, :] - aspect_s[1:, :])

    ax = np.minimum(ax, 2*np.pi - ax)
    ay = np.minimum(ay, 2*np.pi - ay)

    ax = np.degrees(ax)
    ay = np.degrees(ay)

    # -------------------------------
    # slope-weighted aspect
    # -------------------------------
    slope_weight_x = np.sin(slope_s[:, :-1])
    slope_weight_y = np.sin(slope_s[:-1, :])

    ax *= slope_weight_x
    ay *= slope_weight_y

    # -------------------------------
    # ridge detection
    # -------------------------------
    ridge_mask = np.zeros(mask.shape, dtype=bool)

    ridge_mask[:, :-1] |= (
        (dzx > height_thresh) |
        (dsx > slope_thresh) |
        (ax > aspect_thresh_deg)
    )

    ridge_mask[:-1, :] |= (
        (dzy > height_thresh) |
        (dsy > slope_thresh) |
        (ay > aspect_thresh_deg)
    )

    ridge_mask &= mask

    # -------------------------------
    # segmentation
    # -------------------------------
    split_mask = mask & (~ridge_mask)

    structure = generate_binary_structure(2,2)
    segments, n_segments = label(split_mask, structure=structure)

    return segments, n_segments, ridge_mask

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

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    ).fit(normals)

    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = clustering.labels_

    return labels

def _normal_gradient(nx, ny, nz):

    dot_x = nx[:, :-1]*nx[:, 1:] + ny[:, :-1]*ny[:, 1:] + nz[:, :-1]*nz[:, 1:]
    dot_y = nx[:-1, :]*nx[1:, :] + ny[:-1, :]*ny[1:, :] + nz[:-1, :]*nz[1:, :]

    dot_x = np.clip(dot_x, -1, 1)
    dot_y = np.clip(dot_y, -1, 1)

    angle_x = np.degrees(np.arccos(dot_x))
    angle_y = np.degrees(np.arccos(dot_y))

    grad = np.zeros(nx.shape)

    grad[:, :-1] = np.maximum(grad[:, :-1], angle_x)
    grad[:-1, :] = np.maximum(grad[:-1, :], angle_y)

    return grad

def _segment_roof_by_normals(nx, ny, nz, mask, smooth_sigma=1.0):

    dot_x = nx[:, :-1]*nx[:, 1:] + ny[:, :-1]*ny[:, 1:] + nz[:, :-1]*nz[:, 1:]
    dot_y = nx[:-1, :]*nx[1:, :] + ny[:-1, :]*ny[1:, :] + nz[:-1, :]*nz[1:, :]

    dot_x = np.clip(dot_x, -1, 1)
    dot_y = np.clip(dot_y, -1, 1)

    angle_x = np.degrees(np.arccos(dot_x))
    angle_y = np.degrees(np.arccos(dot_y))

    grad = np.zeros(mask.shape)

    grad[:, :-1] = np.maximum(grad[:, :-1], angle_x)
    grad[:-1, :] = np.maximum(grad[:-1, :], angle_y)

    grad = gaussian_filter(grad, smooth_sigma)

    seeds = grad < np.percentile(grad[mask], 20)

    markers, _ = label(seeds & mask)

    segments = watershed(
        grad,
        markers,
        mask=mask
    )

    return segments

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

        if angle < threshold_rad:
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
            "mask": merge_mask,
            "a": a,
            "b": b,
            "c": c,
            "area_pixels": int(merge_mask.sum()),
        })

        next_plane_id += 1

    return merged_planes, new_plane_id_raster

def _split_roof_by_ridges(local_mask, ridge_mask):

    structure = generate_binary_structure(2,1)

    # ridges act as barriers
    split_mask = local_mask & (~ridge_mask)

    labeled, n = label(split_mask, structure)

    return labeled, n

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
    min_inliers: int = 3
):

    if len(points) < min_inliers:
        return None

    best_inliers = None
    best_params = None

    for _ in range(max_iterations):

        sample_indices = np.random.choice(len(points), 3, replace=False)
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

def _plane_slope(a, b):
    return np.degrees(np.arctan(np.sqrt(a*a + b*b)))

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
    planes = []
    plane_id = 0

    rows_all, cols_all = np.where(building_mask)

    for r, c in zip(rows_all, cols_all):

        if visited[r, c]:
            continue

        seed_normal = np.array([nx[r, c], ny[r, c], nz[r, c]])
        stack = [(r, c)]
        plane_mask = np.zeros_like(building_mask, dtype=bool)

        while stack:

            rr, cc = stack.pop()
            if visited[rr, cc]:
                continue

            visited[rr, cc] = True

            current = np.array([nx[rr, cc], ny[rr, cc], nz[rr, cc]])
            dot = np.clip(np.dot(seed_normal, current), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))

            if angle > normal_tolerance_deg:
                continue

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

        xs = transform.c + cols_p * transform.a
        ys = transform.f + rows_p * transform.e
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

def _merge_contiguous_flat_planes(
    planes,
    local_dsm,
    local_transform,
    tilt_threshold_deg=2.0
):

    if not planes:
        return planes

    structure = generate_binary_structure(2, 1)

    flat_planes = []
    non_flat_planes = []

    for p in planes:
        tilt = np.degrees(np.arctan(np.sqrt(p["a"]**2 + p["b"]**2)))
        if tilt < tilt_threshold_deg:
            flat_planes.append(p)
        else:
            non_flat_planes.append(p)

    if len(flat_planes) <= 1:
        return planes

    merged_planes = []
    used = set()

    for i, p in enumerate(flat_planes):

        if i in used:
            continue

        current_mask = p["mask"].copy()
        used.add(i)

        for j, q in enumerate(flat_planes):

            if j in used:
                continue

            # Check adjacency (4-connected)
            dilated = binary_dilation(current_mask, structure=structure)

            if np.any(dilated & q["mask"]):
                current_mask |= q["mask"]
                used.add(j)

        # Refit plane using merged mask
        rows, cols = np.where(current_mask)

        xs = local_transform.c + cols * local_transform.a
        ys = local_transform.f + rows * local_transform.e
        zs = local_dsm[current_mask]

        a, b, c = _fit_plane_ls(xs, ys, zs)

        merged_planes.append({
            "plane_id": p["plane_id"],
            "mask": current_mask,
            "a": a,
            "b": b,
            "c": c,
            "area_pixels": int(current_mask.sum()),
        })

    return non_flat_planes + merged_planes

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
    ridge_mask=None
):

    H, W = local_mask.shape
    plane_id_raster = np.zeros((H, W), dtype=np.int32)

    rows, cols = np.where(local_mask)

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    zs = local_dsm[rows, cols]

    # Precompute polygons and plane parameters
    polys = polygons_gdf.geometry.values
    plane_ids = polygons_gdf.plane_id.values

    for i in range(len(rows)):

        r, c = rows[i], cols[i]
        x, y, z = xs[i], ys[i], zs[i]

        pt = Point(x, y)

        best_plane = 0
        best_dist = np.inf

        for poly, pid in zip(polys, plane_ids):

            # Skip planes whose polygon does not contain the pixel
            if not poly.contains(pt):
                continue

            # Ridge constraint
            if ridge_mask is not None:
                if ridge_mask[r, c]:
                    continue

            plane = building_planes_dsm[pid]

            a = plane["a"]
            b = plane["b"]
            c_plane = plane["c"]

            dist = abs(a*x + b*y - z + c_plane) / np.sqrt(a*a + b*b + 1)

            if dist < best_dist:
                best_dist = dist
                best_plane = pid + 1

        # fallback if no polygon match
        if best_plane == 0:

            for plane in building_planes_dsm:

                pid = plane["plane_id"]

                if ridge_mask is not None and ridge_mask[r, c]:
                    continue

                a = plane["a"]
                b = plane["b"]
                c_plane = plane["c"]

                dist = abs(a*x + b*y - z + c_plane) / np.sqrt(a*a + b*b + 1)

                if dist < best_dist:
                    best_dist = dist
                    best_plane = pid + 1

        plane_id_raster[r, c] = best_plane

    return plane_id_raster

def _refit_planes_from_assignment(
    plane_id_raster,
    local_dsm,
    transform
):

    unique_ids = np.unique(plane_id_raster[plane_id_raster > 0])

    updated_planes = []

    for pid in unique_ids:

        mask = plane_id_raster == pid

        rows, cols = np.where(mask)

        xs = transform.c + cols * transform.a
        ys = transform.f + rows * transform.e
        zs = local_dsm[rows, cols]

        a, b, c = _fit_plane_ls(xs, ys, zs)
        #a, b, c = _fit_plane_ransac(xs, ys, zs)

        updated_planes.append({
            "plane_id": pid,
            "mask": mask,
            "a": a,
            "b": b,
            "c": c,
        })

    return updated_planes


def _polygonize_and_compute_parameters(
    plane_id_raster,
    updated_planes,
    local_dsm,
    transform,
    part_id,
    crs,
    simplify_tolerance=None
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

                footprint_area = poly.area
                roof_area = footprint_area / np.cos(np.radians(slope))

                # planarity metrics
                m = comp_mask

                x = X[m]
                y = Y[m]
                z = Z[m]

                dist = np.abs(a*x + b*y - z + c_plane) / np.sqrt(a*a + b*b + 1)

                rmse = np.sqrt(np.mean(dist**2))
                mad = np.median(np.abs(dist - np.median(dist)))

                polygons.append(poly)

                records.append({
                    "part_id": part_id,
                    "plane_id": pid,
                    "tilt_deg": slope,
                    "azimuth_deg": aspect,
                    "footprint_area": footprint_area,
                    "area": roof_area,
                    "planarity_rmse": rmse,
                    "planarity_mad": mad,
                    "roughness": mad / (rmse + 1e-6)
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


def _detect_ridge_mask(dsm, threshold=0.35):

    curv = laplace(dsm)

    ridge_mask = np.abs(curv) > threshold

    return ridge_mask

def _split_roof_by_normal_discontinuity(mask, ridge_mask):

    structure = generate_binary_structure(2,1)

    split_mask = mask & (~ridge_mask)

    segments, n_segments = label(split_mask, structure=structure)

    return segments, n_segments

def _detect_ridge_lines(nx, ny, nz, mask, angle_thresh_deg=18, min_length=2):

    nx_s, ny_s, nz_s = _smooth_normals_preserve_ridges(nx, ny, nz, mask, sigma=0.5)

    # ------------------------------------------------
    # Normal discontinuity
    # ------------------------------------------------
    dot_x = nx_s[:, :-1]*nx_s[:, 1:] + ny_s[:, :-1]*ny_s[:, 1:] + nz_s[:, :-1]*nz_s[:, 1:]
    dot_y = nx_s[:-1, :]*nx_s[1:, :] + ny_s[:-1, :]*ny_s[1:, :] + nz_s[:-1, :]*nz_s[1:, :]

    dot_x = np.clip(dot_x, -1, 1)
    dot_y = np.clip(dot_y, -1, 1)

    angle_x = np.degrees(np.arccos(dot_x))
    angle_y = np.degrees(np.arccos(dot_y))

    ridge_strength = np.zeros(mask.shape)

    ridge_strength[:, :-1] = np.maximum(ridge_strength[:, :-1], angle_x)
    ridge_strength[:-1, :] = np.maximum(ridge_strength[:-1, :], angle_y)

    ridge_mask = ridge_strength > angle_thresh_deg
    ridge_mask &= mask

    # ------------------------------------------------
    # Non-maximum suppression
    # ------------------------------------------------
    ridge_mask &= ridge_strength == maximum_filter(ridge_strength, size=3)

    # ------------------------------------------------
    # Skeletonize (1-pixel ridges)
    # ------------------------------------------------
    ridge_mask = skeletonize(ridge_mask)

    # ------------------------------------------------
    # Remove short fragments
    # ------------------------------------------------
    labeled, n = label(ridge_mask)

    for i in range(1, n+1):
        if np.sum(labeled == i) < min_length:
            ridge_mask[labeled == i] = False

    return ridge_mask


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

        pts = _read_laz(path=laz_path)

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

def _filter_points_above_dem(
    building_pts,
    dem,
    transform,
    min_height=1.5,
    nodata=-9999.0,
):

    if building_pts is None or len(building_pts) == 0:
        return building_pts

    x = building_pts["x"]
    y = building_pts["y"]
    z = building_pts["z"]

    # Map coordinates → DEM pixels
    inv = ~transform
    px, py = inv * (x, y)

    px = np.floor(px).astype(np.int64)
    py = np.floor(py).astype(np.int64)

    valid = (
        (px >= 0) & (px < dem.shape[1]) &
        (py >= 0) & (py < dem.shape[0])
    )

    if not np.any(valid):
        return None

    pts = building_pts[valid]
    px = px[valid]
    py = py[valid]

    dem_vals = dem[py, px]

    dem_valid = (dem_vals != nodata) & np.isfinite(dem_vals)

    pts = pts[dem_valid]
    dem_vals = dem_vals[dem_valid]

    height_diff = pts["z"] - dem_vals

    keep = height_diff >= min_height

    return pts[keep]
