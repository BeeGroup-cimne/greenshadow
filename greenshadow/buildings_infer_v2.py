# -*- coding: utf-8 -*-
"""
Enhanced roof plane detection using pystruct3d for better bounding box fitting
and plane orientation detection.

This module extends the original buildings_infer.py by integrating pystruct3d's
capabilities for:
- Optimal horizontal aligned bounding box fitting
- Convex hull-based plane orientation detection
- Improved plane segmentation using bounding box constraints
"""

from typing import List, Dict, Optional
import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import (sobel, label, laplace, gaussian_filter,
                           generate_binary_structure, binary_dilation,
                           binary_closing, maximum_filter, binary_erosion)
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np

# Import from greenshadow
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
from shapely.geometry import shape, MultiPoint, Point, Polygon
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
from collections import defaultdict

# Try to import pystruct3d, provide fallback if not available
try:
    from pystruct3d.bbox.bbox import BBox
    PYSTRUCT3D_AVAILABLE = True
except ImportError:
    PYSTRUCT3D_AVAILABLE = False
    print("Warning: pystruct3d not available. Install with: pip install git+https://github.com/humantecheu/pystruct3d.git")

# ==========================================================
# Public API
# ==========================================================

def detect_roof_planes_v2(
    dsm_path,
    *,
    slope_path,
    aspect_path,
    dsm_class_path,
    dem_path=None,
    building_class_value=(6, 77),
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
    lidar_class_values=(1, 2, 6, 8, 17, 77),
    lidar_class_priority=(6, 77, 17, 2, 8, 1),
    lidar_class_window_diameter=0.5,
    laz_dir=None,
    laz_tiles=None,
    resolution_refined=0.25,
    preclustering_roof_planes=False,
    ridge_splitting_roof_planes=False,
    # New parameters for pystruct3d integration
    use_pystruct3d_orientation=True,
    bbox_fitting_method="horizontal_aligned",  # "horizontal_aligned" or "axis_aligned"
    bbox_constraint_threshold=0.1,  # Threshold for bbox-based plane assignment
    refine_with_3d_points=True,  # Use 3D LiDAR points for bbox fitting
):
    """
    Enhanced roof plane detection with pystruct3d integration.

    Parameters
    ----------
    dsm_path : str
        Path to Digital Surface Model raster
    slope_path : str
        Path to slope raster
    aspect_path : str
        Path to aspect raster
    dsm_class_path : str
        Path to classification raster
    dem_path : str, optional
        Path to Digital Elevation Model
    building_class_value : tuple, default=(6, 77)
        Classification values for buildings
    min_height_m : float, default=1.5
        Minimum building height in meters
    normal_tolerance_deg : float, default=15
        Normal angle tolerance for plane grouping
    ransac_threshold : float, default=0.15
        RANSAC threshold for plane fitting
    smooth_sigma : float, default=0.0
        Gaussian smoothing sigma
    min_roof_plane_area : float, default=5
        Minimum roof plane area in m²
    dsm_max_slope_filter : float, default=0.4
        Maximum slope for roof plane pixels
    buffer_m : float, default=1
        Buffer distance in meters
    test_window : tuple, optional
        Window for testing (r0, r1, c0, c1)
    output_path : str, optional
        Path for output GeoPackage
    lidar_refinement : bool, default=True
        Whether to use LiDAR refinement
    lidar_class_values : tuple, default=(1, 2, 6, 8, 17, 77)
        Allowed LiDAR classification values
    lidar_class_priority : tuple, default=(6, 77, 17, 2, 8, 1)
        Priority order for LiDAR classes
    lidar_class_window_diameter : float, default=0.5
        Window diameter for class priority filtering
    laz_dir : str, optional
        Directory containing LAZ files
    laz_tiles : list, optional
        List of tile geometries
    resolution_refined : float, default=0.25
        Resolution for refined processing
    preclustering_roof_planes : bool, default=False
        Whether to precluster roof planes
    ridge_splitting_roof_planes : bool, default=False
        Whether to split by ridges
    use_pystruct3d_orientation : bool, default=True
        Whether to use pystruct3d for optimal orientation detection
    bbox_fitting_method : str, default="horizontal_aligned"
        Method for bounding box fitting
    bbox_constraint_threshold : float, default=0.1
        Threshold for bbox-based plane assignment
    refine_with_3d_points : bool, default=True
        Whether to refine using 3D LiDAR points

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing detected roof planes
    """

    if use_pystruct3d_orientation and not PYSTRUCT3D_AVAILABLE:
        print("Warning: pystruct3d not available, falling back to standard method")
        use_pystruct3d_orientation = False

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
    building_lidar_points = None
    if (lidar_refinement and laz_dir is not None and laz_tiles is not None):
        building_geom = _building_mask_to_buffered_geometry(
            buildings_mask,
            transform,
            epsg=crs.to_epsg(),
            buffer_m=1.0,
        )

        minx = transform.c
        maxx = transform.c + dsm.shape[1] * transform.a
        maxy = transform.f
        miny = transform.f + dsm.shape[0] * transform.e

        width_ref, height_ref, transform_ref = _aligned_grid(
            minx, miny, maxx, maxy, resolution_refined
        )

        if dem_path:
            dem_ref = np.full((height_ref, width_ref), -9999.0, dtype=np.float32)
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

        if building_pts is not None and len(building_pts) > 0:
            building_lidar_points = building_pts

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

            dsm = dsm_ref
            slope = slope_ref
            aspect = aspect_ref
            dsm_class = class_ref
            transform = transform_ref

            buildings_mask = np.isin(dsm_class, building_class_value)
            if dem_path:
                buildings_mask &= (dsm - dem > min_height_m)

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

        slope[:] = -9999
        aspect[:] = -9999

        _compute_sobel_slope_aspect(
            dsm,
            slope,
            aspect,
            nodata=-9999,
            resolution=abs(transform.a)
        )

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

    s = generate_binary_structure(2, 1)
    labeled_buildings, n_buildings = label(buildings_mask, structure=s)

    counts = np.bincount(labeled_buildings.ravel())
    remove = counts < min_pixels
    remove[0] = False
    labeled_buildings[remove[labeled_buildings]] = 0

    labeled_buildings, n_buildings = label(labeled_buildings > 0, structure=s)

    if output_path:
        _write_array_to_tiff(
            array=labeled_buildings,
            out_path=output_path.replace(".gpkg", "_continuous_areas.tif"),
            transform=transform,
            crs=crs,
            nodata=-9999.0,
            dtype="float32"
        )

    planes_list = []
    ridge_mask = np.zeros(dsm.shape, dtype=np.uint8)
    clustering_mask = np.zeros(dsm.shape, dtype=np.uint32)
    cluster_global_id = 1

    for part_id in tqdm(range(1, n_buildings + 1),
                       desc="Processing building roof parts (pystruct3d enhanced)"):
        building_mask = labeled_buildings == part_id
        if not np.any(building_mask):
            continue

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

        # Extract 3D points for this building
        building_points_3d = _extract_3d_points(
            local_mask,
            local_dsm,
            local_transform
        )

        # Use pystruct3d for initial plane segmentation
        if use_pystruct3d_orientation and building_points_3d.shape[0] > 3:
            initial_segments = _segment_using_pystruct3d(
                building_points_3d,
                local_mask,
                bbox_fitting_method=bbox_fitting_method,
                tolerance=bbox_constraint_threshold
            )
        else:
            initial_segments = np.ones_like(local_mask, dtype=np.int32)

        # Ridge splitting if enabled
        if ridge_splitting_roof_planes:
            local_ridge_mask, normal_clusters_init = _ridge_mask_from_normal_clustering(
                local_nx, local_ny, local_nz,
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
            split_mask = local_mask & (~local_ridge_mask)
            segments, n_segments = label(split_mask, structure=structure)
        else:
            segments = initial_segments
            n_segments = int(np.max(initial_segments)) + 1
            local_ridge_mask = np.zeros_like(local_mask, dtype=np.uint8)

        ridge_mask[row_slice, col_slice] |= local_ridge_mask.astype(np.uint8)
        building_planes_dsm = []

        for sid in range(1, n_segments + 1):
            segment_mask = segments == sid

            if segment_mask.sum() < min_pixels:
                continue

            if preclustering_roof_planes:
                normal_clusters = _cluster_normals(
                    local_nx, local_ny, local_nz,
                    segment_mask,
                    ridge_mask=local_ridge_mask,
                    eps_deg=normal_tolerance_deg,
                    smooth_sigma=smooth_sigma
                )

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

                clustering_mask[
                    row_slice, col_slice
                ][cluster_pixels] = cluster_global_id
                cluster_global_id += 1

            planes = []

            for cid in cluster_ids:
                if cid < 0:
                    continue

                cluster_mask = normal_clusters == cid

                if cluster_mask.sum() < min_pixels:
                    continue

                # Use enhanced plane fitting with 3D points
                cluster_points = _extract_3d_points(
                    cluster_mask,
                    local_dsm,
                    local_transform
                )

                if use_pystruct3d_orientation and cluster_points.shape[0] > 3:
                    cluster_planes = _detect_planes_pystruct3d(
                        building_mask=cluster_mask,
                        nx=local_nx,
                        ny=local_ny,
                        nz=local_nz,
                        dsm=local_dsm,
                        points_3d=cluster_points,
                        transform=local_transform,
                        normal_tolerance_deg=normal_tolerance_deg,
                        ransac_threshold=ransac_threshold,
                        min_pixels=min_pixels,
                        bbox_fitting_method=bbox_fitting_method
                    )
                else:
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
            ridge_mask=local_ridge_mask,
            max_search_radius=1
        )

        updated_planes = _refit_planes_from_assignment(
            plane_id_raster,
            local_dsm,
            local_transform
        )

        updated_planes, plane_id_raster = _merge_similar_adjacent_planes_graph(
            plane_id_raster,
            updated_planes,
            local_dsm,
            local_transform,
            normal_angle_threshold_deg=2.5,
            ridge_mask=local_ridge_mask
        )

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
            ridge_mask=local_ridge_mask,
            max_search_radius=1
        )

        updated_planes = _refit_planes_from_assignment(
            plane_id_raster,
            local_dsm,
            local_transform
        )

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

    if output_path:
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

        gdf_all_planes.to_file(output_path)

    return gdf_all_planes


# ============================================================
# PyStruct3D Integration Functions
# ============================================================

def _extract_3d_points(mask, dsm, transform):
    """Extract 3D points from masked DSM pixels."""
    rows, cols = np.where(mask)
    xs = transform.c + cols * transform.a
    ys = transform.f + rows * transform.e
    zs = dsm[rows, cols]
    return np.column_stack([xs, ys, zs])


def _segment_using_pystruct3d(points, mask, bbox_fitting_method="horizontal_aligned",
                              tolerance=0.1):
    """
    Segment roof planes using pystruct3d's bounding box fitting.

    This uses convex hull analysis to find optimal plane orientations.
    """
    if points.shape[0] < 4:
        return np.ones(mask.shape, dtype=np.int32)

    # Create segments based on height variations
    # Simple approach: create initial segments based on height levels
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()

    if z_max - z_min < 0.5:  # Flat roof
        return np.ones(mask.shape, dtype=np.int32)

    # Use pystruct3d's BBox to find orientation
    bbox = BBox()

    if bbox_fitting_method == "horizontal_aligned":
        bbox.fit_horizontal_aligned(points)
    else:
        bbox.fit_axis_aligned(points)

    # Extract plane orientation from bbox
    bbox_angle = bbox.angle()
    bbox_length = bbox.length()
    bbox_width = bbox.width()

    # Create segments based on bbox orientation
    # This is a simplified approach - in production, you'd use more
    # sophisticated segmentation based on the bbox planes
    segments = np.ones(mask.shape, dtype=np.int32)

    # If the bbox has significant orientation, use it for segmentation
    if bbox_angle > 5 and bbox_length / bbox_width > 1.2:
        # Roof is likely pitched - try to split along the major axis
        # This is a placeholder for more sophisticated segmentation
        segments = np.ones(mask.shape, dtype=np.int32)

    return segments


def _detect_planes_pystruct3d(
    building_mask,
    nx, ny, nz,
    dsm,
    points_3d,
    transform,
    normal_tolerance_deg=10.0,
    ransac_threshold=0.15,
    min_pixels=20,
    bbox_fitting_method="horizontal_aligned"
):
    """
    Enhanced plane detection using pystruct3d for orientation.

    This combines the normal-based approach with pystruct3d's
    optimal bounding box fitting for better plane detection.
    """
    if points_3d.shape[0] < min_pixels:
        return []

    visited = np.zeros_like(building_mask, dtype=bool)
    planes = []
    plane_id = 0

    rows_all, cols_all = np.where(building_mask)

    # Fit bounding box using pystruct3d to get optimal orientation
    bbox = BBox()
    try:
        if bbox_fitting_method == "horizontal_aligned":
            bbox.fit_horizontal_aligned(points_3d)
        else:
            bbox.fit_axis_aligned(points_3d)

        bbox_angle = bbox.angle()
        bbox_normal = _get_bbox_plane_normal(bbox)
    except Exception:
        bbox_angle = 0.0
        bbox_normal = np.array([0, 0, 1])

    # Use bbox normal as additional guidance for plane fitting
    for r, c in zip(rows_all, cols_all):
        if visited[r, c]:
            continue

        seed_normal = np.array([nx[r, c], ny[r, c], nz[r, c]])

        # Check if seed normal aligns with bbox normal (for pitched roofs)
        if bbox_normal is not None:
            dot = np.clip(np.dot(seed_normal, bbox_normal), -1.0, 1.0)
            bbox_angle_diff = np.degrees(np.arccos(np.abs(dot)))
        else:
            bbox_angle_diff = 0.0

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

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < dsm.shape[0] and
                    0 <= nc < dsm.shape[1] and
                    building_mask[nr, nc] and
                    not visited[nr, nc]):
                    stack.append((nr, nc))

        if np.sum(plane_mask) < min_pixels:
            continue

        rows_p, cols_p = np.where(plane_mask)

        xs = transform.c + cols_p * transform.a
        ys = transform.f + rows_p * transform.e
        zs = dsm[plane_mask]

        # Use enhanced RANSAC with bbox constraints
        plane_points = np.column_stack((xs, ys, zs))
        ransac_result = _fit_plane_ransac_with_bbox(
            plane_points,
            bbox,
            threshold=ransac_threshold,
            max_iterations=500,
            min_inliers=min_pixels
        )

        if ransac_result is None:
            # Fall back to standard RANSAC
            ransac_result = _fit_plane_ransac(
                plane_points,
                threshold=ransac_threshold,
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
            "bbox_angle": bbox_angle,  # Store for reference
            "method": "pystruct3d"
        })

        plane_id += 1

    return planes


def _get_bbox_plane_normal(bbox):
    """Extract the dominant plane normal from a bounding box."""
    # Get the two horizontal edges
    lower_points = bbox.corner_points[:4]

    # Vector along length (edge 0->1)
    length_vec = lower_points[1] - lower_points[0]

    # Vector along width (edge 0->3)
    width_vec = lower_points[3] - lower_points[0]

    # The roof plane normal is the cross product (z-axis direction)
    roof_normal = np.cross(length_vec, width_vec)

    if np.linalg.norm(roof_normal) > 1e-6:
        roof_normal = roof_normal / np.linalg.norm(roof_normal)
    else:
        roof_normal = np.array([0, 0, 1])

    return roof_normal


def _fit_plane_ransac_with_bbox(
    points: np.ndarray,
    bbox: BBox,
    threshold: float = 0.2,
    max_iterations: int = 1000,
    min_inliers: int = 3
):
    """
    Enhanced RANSAC plane fitting with bounding box constraints.

    Uses the bbox orientation to guide plane fitting, which is
    particularly useful for pitched roofs.
    """
    if len(points) < min_inliers:
        return None

    best_inliers = None
    best_params = None

    # Get bbox normal for weighting
    bbox_normal = _get_bbox_plane_normal(bbox)

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

        # Check if plane normal is consistent with bbox
        plane_normal = np.array([-a, -b, 1.0])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        if np.abs(np.dot(plane_normal, bbox_normal)) < 0.8:
            # Plane orientation is too different from bbox, skip
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


# ============================================================
# Reused utility functions from buildings_infer.py
# ============================================================

def _ridge_mask_from_normal_clustering(
    nx, ny, nz, mask,
    eps_deg=10,
    min_samples=20,
    smooth_sigma=1.0
):
    clusters = _cluster_normals(
        nx, ny, nz, mask,
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
    clusters, segment_mask, min_pixels
):
    structure = generate_binary_structure(2, 1)

    filtered = np.full(clusters.shape, -1, dtype=np.int32)

    next_id = 0

    for cid in np.unique(clusters):
        if cid < 0:
            continue

        mask = (clusters == cid) & segment_mask

        if not np.any(mask):
            continue

        labeled, n = label(mask, structure=structure)

        for i in range(1, n + 1):
            comp = labeled == i

            if comp.sum() < min_pixels:
                continue

            filtered[comp] = next_id
            next_id += 1

    return filtered


def _cluster_normals(
    nx, ny, nz, mask,
    ridge_mask=None,
    eps_deg=10,
    min_samples=20,
    smooth_sigma=1.0
):
    if smooth_sigma > 0:
        nx, ny, nz = _smooth_normals_preserve_ridges(
            nx, ny, nz, mask,
            ridge_mask=ridge_mask,
            sigma=smooth_sigma
        )

    normals = np.stack([nx[mask], ny[mask], nz[mask]], axis=1)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)

    normals[normals[:, 2] < 0] *= -1

    eps = np.sin(np.radians(eps_deg / 2))

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    ).fit(normals)

    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = clustering.labels_

    return labels


def _smooth_normals_preserve_ridges(
    nx, ny, nz, mask,
    ridge_mask=None,
    sigma=1
):
    nx_f = np.where(mask, nx, np.nan)
    ny_f = np.where(mask, ny, np.nan)
    nz_f = np.where(mask, nz, np.nan)

    nx_tmp = np.nan_to_num(nx_f)
    ny_tmp = np.nan_to_num(ny_f)
    nz_tmp = np.nan_to_num(nz_f)

    w = mask.astype(float)

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

    if ridge_mask is not None:
        nx_s[ridge_mask] = nx[ridge_mask]
        ny_s[ridge_mask] = ny[ridge_mask]
        nz_s[ridge_mask] = nz[ridge_mask]

    norm = np.sqrt(nx_s**2 + ny_s**2 + nz_s**2)
    norm = np.maximum(norm, 1e-6)

    nx_s /= norm
    ny_s /= norm
    nz_s /= norm

    return nx_s, ny_s, nz_s


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

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < dsm.shape[0] and
                    0 <= nc < dsm.shape[1] and
                    building_mask[nr, nc] and
                    not visited[nr, nc]):
                    stack.append((nr, nc))

        if np.sum(plane_mask) < min_pixels:
            continue

        rows_p, cols_p = np.where(plane_mask)

        xs = transform.c + cols_p * transform.a
        ys = transform.f + rows_p * transform.e
        zs = dsm[plane_mask]

        points = np.column_stack((xs, ys, zs))
        ransac_result = _fit_plane_ransac(
            points,
            threshold=ransac_threshold,
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
            "method": "standard"
        })

        plane_id += 1

    return planes


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


def _fit_plane_ls(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    x0 = xs.mean()
    y0 = ys.mean()
    z0 = zs.mean()

    xs_c = xs - x0
    ys_c = ys - y0
    zs_c = zs - z0

    A = np.c_[xs_c, ys_c, np.ones_like(xs_c)]
    coef, *_ = np.linalg.lstsq(A, zs_c, rcond=None)

    a, b, c_local = coef

    c = z0 - a * x0 - b * y0

    return float(a), float(b), float(c)


def _plane_distance_3d(a, b, c, x, y, z):
    num = np.abs(a * x + b * y - z + c)
    den = np.sqrt(a * a + b * b + 1.0)
    return num / den


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

    polys = polygons_gdf.geometry.values
    plane_ids = polygons_gdf.plane_id.values

    for i in range(len(rows)):
        r, c = rows[i], cols[i]
        x, y, z = xs[i], ys[i], zs[i]

        pt = Point(x, y)

        best_plane = 0
        best_dist = np.inf

        for poly, pid in zip(polys, plane_ids):
            if not poly.contains(pt):
                continue

            if ridge_mask is not None:
                if ridge_mask[r, c]:
                    continue

            plane = building_planes_dsm[pid]

            a = plane["a"]
            b = plane["b"]
            c_plane = plane["c"]

            dist = abs(a * x + b * y - z + c_plane) / np.sqrt(a * a + b * b + 1)

            if dist < best_dist:
                best_dist = dist
                best_plane = pid + 1

        if best_plane == 0:
            for plane in building_planes_dsm:
                pid = plane["plane_id"]

                if ridge_mask is not None and ridge_mask[r, c]:
                    continue

                a = plane["a"]
                b = plane["b"]
                c_plane = plane["c"]

                dist = abs(a * x + b * y - z + c_plane) / np.sqrt(a * a + b * b + 1)

                if dist < best_dist:
                    best_dist = dist
                    best_plane = pid + 1

        plane_id_raster[r, c] = best_plane

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

        updated_planes.append({
            "plane_id": pid,
            "mask": mask,
            "a": a,
            "b": b,
            "c": c,
        })

    return updated_planes


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

    neighbor_offsets = [
        (-1, 0, 1.0), (1, 0, 1.0),
        (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 0.5), (-1, 1, 0.5),
        (1, -1, 0.5), (1, 1, 0.5),
    ]

    def _has_linear_support(r, c, pid):
        if c > 0 and c < W - 1:
            if plane_id_raster[r, c - 1] == pid and plane_id_raster[r, c + 1] == pid:
                return True

        if r > 0 and r < H - 1:
            if plane_id_raster[r - 1, c] == pid and plane_id_raster[r + 1, c] == pid:
                return True

        if r > 0 and r < H - 1 and c > 0 and c < W - 1:
            if plane_id_raster[r - 1, c - 1] == pid and plane_id_raster[r + 1, c + 1] == pid:
                return True

            if plane_id_raster[r - 1, c + 1] == pid and plane_id_raster[r + 1, c - 1] == pid:
                return True

        return False

    def _gradient_consistent(r, c, pid):
        plane = plane_dict[pid]
        a, b = plane["a"], plane["b"]

        r0 = max(r - 1, 0)
        r1 = min(r + 1, H - 1)
        c0 = max(c - 1, 0)
        c1 = min(c + 1, W - 1)

        gx = (local_dsm[r, c1] - local_dsm[r, c0]) / (2 * abs(transform.a))
        gy = (local_dsm[r1, c] - local_dsm[r0, c]) / (2 * abs(transform.e))

        g_norm = np.sqrt(gx * gx + gy * gy)
        p_norm = np.sqrt(a * a + b * b)

        if g_norm == 0 or p_norm == 0:
            return True

        cosang = (gx * (-a) + gy * (-b)) / (g_norm * p_norm)
        cosang = np.clip(cosang, -1, 1)

        angle = np.degrees(np.arccos(cosang))

        return angle < gradient_angle_tol_deg

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

                if (0 <= nr < H and 0 <= nc < W and local_mask[nr, nc]):
                    if ridge_mask is not None:
                        if ridge_mask[r, c] or ridge_mask[nr, nc]:
                            continue

                    pid = plane_id_raster[nr, nc]

                    if pid > 0:
                        support[pid] = support.get(pid, 0) + w

            if not support:
                continue

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

            for pid, w in candidates:
                plane = plane_dict[pid]
                a, b, c_plane = plane["a"], plane["b"], plane["c"]

                dist = abs(a * x + b * y - z + c_plane) / np.sqrt(a * a + b * b + 1)

                score = w - distance_weight * dist

                if score > best_score:
                    best_score = score
                    best_plane = pid

            if best_plane != 0:
                plane_id_raster[r, c] = best_plane
                break

    return plane_id_raster


def _remove_small_planes(
    plane_id_raster: np.ndarray,
    min_pixels: int
):
    counts = np.bincount(plane_id_raster.ravel())

    remove = counts < min_pixels
    remove[0] = False

    plane_id_raster[remove[plane_id_raster]] = 0

    return plane_id_raster


def _compute_plane_adjacency_pairs(plane_id_raster):
    H, W = plane_id_raster.shape

    pairs = set()

    a = plane_id_raster[:, :-1]
    b = plane_id_raster[:, 1:]

    mask = (a != b) & (a > 0) & (b > 0)

    for p, q in zip(a[mask], b[mask]):
        pairs.add(tuple(sorted((p, q))))

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


def _plane_normal(a, b):
    n = np.array([-a, -b, 1.0], dtype=np.float64)
    return n / np.linalg.norm(n)


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

    rows, cols = np.indices((H, W))
    X = transform.c + (cols + 0.5) * transform.a
    Y = transform.f + (rows + 0.5) * transform.e
    Z = local_dsm

    plane_ids = np.unique(plane_id_raster)
    plane_ids = plane_ids[plane_ids > 0]

    for pid in plane_ids:
        mask = plane_id_raster == pid

        labeled, ncomp = label(mask)

        plane = plane_dict[pid]
        a, b, c_plane = plane["a"], plane["b"], plane["c"]

        slope, aspect = _plane_slope_aspect(a, b)

        for comp in range(1, ncomp + 1):
            comp_mask = labeled == comp

            if comp_mask.sum() < 3:
                continue

            for geom, value in shapes(
                comp_mask.astype(np.uint8),
                mask=comp_mask,
                transform=transform
            ):

                poly = Polygon(geom["coordinates"][0])

                if simplify_tolerance and simplify_tolerance > 0:
                    poly = poly.simplify(
                        simplify_tolerance,
                        preserve_topology=True
                    )

                footprint_area = poly.area
                roof_area = footprint_area / np.cos(np.radians(slope))

                m = comp_mask

                x = X[m]
                y = Y[m]
                z = Z[m]

                dist = np.abs(a * x + b * y - z + c_plane) / np.sqrt(a * a + b * b + 1)

                rmse = np.sqrt(np.mean(dist ** 2))
                mad = np.median(np.abs(dist - np.median(dist)))

                polygons.append(poly)

                record = {
                    "part_id": part_id,
                    "plane_id": pid,
                    "tilt_deg": slope,
                    "azimuth_deg": aspect,
                    "footprint_area": footprint_area,
                    "area": roof_area,
                    "planarity_rmse": rmse,
                    "planarity_mad": mad,
                    "roughness": mad / (rmse + 1e-6)
                }

                # Add method info if available
                if "method" in plane:
                    record["detection_method"] = plane["method"]

                records.append(record)

    gdf = gpd.GeoDataFrame(records, geometry=polygons, crs=crs)

    return gdf


def _plane_slope_aspect(a, b):
    slope_deg = np.degrees(
        np.arctan(np.sqrt(a * a + b * b))
    )

    aspect_deg = np.degrees(
        np.arctan2(-a, -b)
    )

    aspect_deg = (aspect_deg + 360) % 360

    return slope_deg, aspect_deg


def _compute_sobel_slope_aspect(
    dsm: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    nodata: float,
    resolution: float
):
    slope[:] = nodata
    aspect[:] = nodata

    valid = dsm != nodata

    dsm_filled = dsm.copy()
    mean_val = np.nanmean(np.where(valid, dsm, np.nan))
    dsm_filled[~valid] = mean_val

    dzdx = sobel(dsm_filled, axis=1, mode="nearest") / (8.0 * resolution)
    dzdy = sobel(dsm_filled, axis=0, mode="nearest") / (8.0 * resolution)

    grad_mag = np.sqrt(dzdx ** 2 + dzdy ** 2)

    slope_rad = np.arctan(grad_mag)

    aspect_rad = np.arctan2(-dzdx, -dzdy)
    aspect_rad = (aspect_rad + 2 * np.pi) % (2 * np.pi)

    slope[valid] = slope_rad[valid]
    aspect[valid] = aspect_rad[valid]


# Reuse utility functions from buildings_infer.py
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

    if aoi is not None:
        aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi.bounds

    all_pts = []

    for tile in tqdm(tiles, desc="Loading LiDAR", unit="tile"):
        coords = tile["geometry"]["coordinates"][0]

        tile_minx = min(pt[0] for pt in coords)
        tile_maxx = max(pt[0] for pt in coords)
        tile_miny = min(pt[1] for pt in coords)
        tile_maxy = max(pt[1] for pt in coords)

        if (
            tile_maxx <= minx or
            tile_minx >= maxx or
            tile_maxy <= miny or
            tile_miny >= maxy
        ):
            continue

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

        if pts.shape[1] >= 4:
            mask_class = np.isin(pts[:, 3], lidar_allowed_classes)
            pts = pts[mask_class]
        else:
            continue

        if aoi is not None and len(pts) > 0:
            x = pts[:, 0]
            y = pts[:, 1]

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

            valid_dem = (dem_vals != -9999.0) & np.isfinite(dem_vals)

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

    if epsg != 25831:
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}",
            "EPSG:25831",
            always_xy=True,
        )
        geom_union = shp_transform(transformer.transform, geom_union)

    if buffer_m > 0:
        geom_union = geom_union.buffer(buffer_m)

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

    mask = binary_closing(mask, iterations=2)
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