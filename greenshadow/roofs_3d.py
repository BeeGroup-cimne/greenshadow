# -*- coding: utf-8 -*-
"""
3D views of detected roof planes.

Renders the planes a detector already found - the same GeoDataFrame that is
written to the GeoPackage - rather than segmenting the points again. That
matters: this module used to run its own point-native segmentation, so the
picture showed planes that were not the ones in the output. What you see
here is the product.

Both detectors' output works, since both report `tilt_deg` and
`azimuth_deg`: `buildings_infer.detect_roof_planes`, before or after
`buildings_infer.regularize_roof_planes`.

The polygons are flat in plan view and carry no elevation, so each face is
lifted onto its own plane: `tilt_deg` and `azimuth_deg` give the plane's
orientation, and the DSM is sampled inside the polygon to anchor its height.
The result is the faceted model the detector believes in, not a drape over
the raw surface.
"""

import os
import warnings

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import triangulate
from shapely.strtree import STRtree
from tqdm import tqdm

try:
    from .buildings_infer import _plane_coefficients, _anchor_height
except ImportError:
    from greenshadow.buildings_infer import _plane_coefficients, _anchor_height


# Categorical slots, in fixed order, from the validated reference palette.
# Faces are coloured so that no two touching faces share a slot, which keeps
# every on-screen comparison an adjacent-pair one - the pairlist this order
# was validated against (worst CVD delta-E 9.1 light / 8.4 dark).
_FACE_COLORS_LIGHT = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

# Single hue, light to dark, for tilt. Magnitude gets a ramp, never hues.
_TILT_SCALE = [
    [0.0, "#e8f0fb"], [0.25, "#a8c8ee"], [0.5, "#5f9ae0"],
    [0.75, "#2a78d6"], [1.0, "#14407a"],
]

_SURFACE_LIGHT = "#fcfcfb"
_TEXT_PRIMARY_LIGHT = "#0b0b0b"
_TEXT_SECONDARY_LIGHT = "#52514e"
_DROPPED_COLOR_LIGHT = "#c9c8c2"

# Past this many triangles a browser starts to struggle with the scene.
_HEAVY_TRIANGLE_COUNT = 400_000


def roof_planes_3d(
    roof_planes,
    *,
    dsm_path,
    output_path=None,
    color_by="plane",
    surface="none",
    laz_dir=None,
    laz_tiles=None,
    lidar_points=None,
    points_margin_m=5.0,
    points_classes=None,
    max_points=1_500_000,
    surface_max_triangles=300_000,
    title=None,
    tilt_column="tilt_deg",
    azimuth_column="azimuth_deg",
    part_column="part_id",
    plane_column="plane_id",
    progress=False
):
    """
    Build - and optionally write - a 3D view of already-detected roof planes.

    Parameters
    ----------
    roof_planes : gpd.GeoDataFrame or str
        The detector's output, or a path to the GeoPackage it wrote. Needs
        polygon geometry plus the tilt and azimuth columns; both
        ``detect_roof_planes`` provides them.
    dsm_path : str
        DSM to sample for each face's height. The polygons say where a face
        is and which way it points, but not how high it sits. Prefer the DSM
        the planes were detected on - for the raster detector with LiDAR
        refinement that is the ``*_roofs.tif`` it writes, which is on the
        refined grid.
    output_path : str, optional
        Where to write. The extension picks the writer: ``.html`` for an
        interactive WebGL page (three.js, loaded from a CDN when opened),
        ``.png`` for a still image,
        ``.obj`` or ``.ply`` for a mesh file any 3D viewer can open. Without
        it nothing is written and only the mesh is returned.
    color_by : {"plane", "tilt", "building"}, default="plane"
        What the colours mean. "plane" gives touching faces different
        colours so they can be told apart; "tilt" ramps a single hue by
        slope, which is the one to read for solar; "building" colours each
        building as a whole.
    surface : {"none", "dsm"}, default="none"
        Draw the measured roof surface underneath the detected planes, in a
        recessive grey. "dsm" triangulates ``dsm_path`` itself - the very
        surface the detector read - so a plane that disagrees with the roof,
        or a roof that got no plane at all, is visible as such. This is the
        layer to look at when the detection looks wrong: it separates "the
        detector mis-read a good surface" from "the surface was already
        wrong", which need completely different fixes.
    laz_dir : str, optional
        Folder with the LiDAR tiles. When given, every return in the area
        the roofs cover - ground, vegetation, walls and roofs alike, plus
        ``points_margin_m`` around it - is drawn as a point layer under the
        detected planes, coloured by LiDAR class. The points are shown as
        measured: no triangulation, binning or gap filling, so a plane can
        be checked against the returns themselves.
    laz_tiles : list of dict, optional
        Tile records (``icgc_tiles_from_geojson`` output) to restrict the
        search to. Without them every ``.laz`` in ``laz_dir`` whose header
        bounds overlap the area is read.
    lidar_points : (N, >=3) array_like, optional
        Points already in memory, as x, y, z and optionally classification.
        Used instead of reading ``laz_dir``, and cropped to the same area.
    points_margin_m : float, default=5.0
        How far beyond the roofs' bounding box to keep points, so the
        ground and trees around the buildings are in the scene too.
    points_classes : iterable of int, optional
        Only draw these LiDAR classes. All of them by default.
    max_points : int, default=1500000
        Point budget. Above it the cloud is thinned uniformly at random
        (seeded, so the same call draws the same points). The .html viewer
        stores a return in ~7 bytes and draws it on the GPU, so 1.5 million
        is a ~14 MB page that stays fluid; raise it on a good GPU, lower it
        for a laptop or a whole municipality.
    surface_max_triangles : int, default=300000
        Triangle budget for the ``surface="dsm"`` layer. The DSM is subsampled by
        whatever whole step keeps it under this, so a full municipality
        window still opens in a browser.
    title : str, optional
        Title for the rendered page or image.
    tilt_column, azimuth_column, part_column, plane_column : str
        Column names, for output that does not use the default schema.
        ``part_column`` and ``plane_column`` are optional in the data; row
        position is used when they are absent.
    progress : bool, default=False
        Show a progress bar over the faces.

    Returns
    -------
    dict
        ``vertices`` (V, 3), ``triangles`` (T, 3) indexing them, and
        per-triangle ``part_id``, ``plane_id``, ``tilt_deg``,
        ``azimuth_deg``, ``color_slot`` and ``dropped`` (True only for the
        DSM surface layer), plus ``points`` (P, 3) and ``point_class`` (P,)
        for the LiDAR layer - empty when no points were requested.
    """
    if color_by not in ("plane", "tilt", "building"):
        raise ValueError(f"Unknown color_by: {color_by!r}")

    planes = _load_planes(roof_planes)

    for column in (tilt_column, azimuth_column):
        if column not in planes.columns:
            raise ValueError(
                f"roof_planes has no {column!r} column; got "
                f"{sorted(planes.columns)}"
            )

    if len(planes) == 0:
        return _empty_mesh(getattr(planes, "crs", None))

    with rasterio.open(dsm_path) as source:
        dsm = source.read(1).astype(np.float64)
        dsm_transform = source.transform
        dsm_nodata = source.nodata

    slots = _color_polygons(list(planes.geometry))

    chunks = []

    rows = planes.itertuples()

    if progress:
        rows = tqdm(rows, total=len(planes), desc="Building 3D roof mesh")

    for position, row in enumerate(rows):
        chunk = _face_mesh(
            geometry=row.geometry,
            tilt_deg=float(getattr(row, tilt_column)),
            azimuth_deg=float(getattr(row, azimuth_column)),
            part_id=int(getattr(row, part_column, position + 1) or 0),
            plane_id=int(getattr(row, plane_column, position + 1) or 0),
            color_slot=slots[position],
            dsm=dsm,
            dsm_transform=dsm_transform,
            dsm_nodata=dsm_nodata
        )

        if chunk is not None:
            chunks.append(chunk)

    surface_chunk = None

    if surface == "dsm":
        surface_chunk = _dsm_surface_mesh(
            dsm,
            dsm_transform,
            dsm_nodata,
            max_triangles=surface_max_triangles
        )
    elif surface != "none":
        raise ValueError(f"Unknown surface: {surface!r}")

    if surface_chunk is not None:
        # First, so it renders under the planes.
        chunks.insert(0, surface_chunk)

    points = np.zeros((0, 3), dtype=np.float64)
    point_class = np.zeros(0, dtype=np.uint8)

    if lidar_points is not None or laz_dir:
        minx, miny, maxx, maxy = planes.total_bounds
        bounds = (
            minx - points_margin_m, miny - points_margin_m,
            maxx + points_margin_m, maxy + points_margin_m
        )

        if lidar_points is not None:
            cloud = _crop_points(np.asarray(lidar_points, dtype=np.float64), bounds)
        else:
            cloud = _read_area_points(
                laz_dir, laz_tiles, bounds, progress=progress
            )

        points, point_class = _prepare_points(
            cloud, classes=points_classes, max_points=max_points
        )

    if not chunks and points.shape[0] == 0:
        return _empty_mesh(planes.crs)

    mesh = (
        _concatenate_meshes(chunks, planes.crs) if chunks
        else _empty_mesh(planes.crs)
    )

    mesh["points"] = points
    mesh["point_class"] = point_class

    if output_path:
        _warn_if_heavy(mesh, output_path)

        _write_mesh(
            mesh,
            output_path,
            color_by=color_by,
            title=title or "Roof planes"
        )

    return mesh


def _load_planes(roof_planes):
    """Accept a GeoDataFrame or a path to one."""
    if isinstance(roof_planes, gpd.GeoDataFrame):
        return roof_planes

    if isinstance(roof_planes, str):
        return gpd.read_file(roof_planes)

    raise TypeError(
        "roof_planes must be a GeoDataFrame or a path to one, got "
        f"{type(roof_planes).__name__}"
    )


def _face_mesh(
    *,
    geometry,
    tilt_deg,
    azimuth_deg,
    part_id,
    plane_id,
    color_slot,
    dsm,
    dsm_transform,
    dsm_nodata
):
    """Triangulate one face and lift it onto its own plane."""
    parts = [p for p in _iter_polygons(geometry) if p.area > 0]

    if not parts:
        return None

    a, b = _plane_coefficients(tilt_deg, azimuth_deg)

    triangles_2d = []

    for part in parts:
        triangles_2d.extend(_triangulate_polygon(part))

    if not triangles_2d:
        return None

    corners = np.array(triangles_2d, dtype=np.float64).reshape(-1, 2)

    c = _anchor_height(
        geometry, a, b, dsm, dsm_transform, dsm_nodata
    )

    if c is None:
        return None

    z = a * corners[:, 0] + b * corners[:, 1] + c

    vertices = np.column_stack([corners, z])

    n_triangles = vertices.shape[0] // 3

    return {
        "vertices": vertices,
        "triangles": np.arange(n_triangles * 3, dtype=np.int64).reshape(-1, 3),
        "part_id": np.full(n_triangles, part_id, dtype=np.int32),
        "plane_id": np.full(n_triangles, plane_id, dtype=np.int32),
        "tilt_deg": np.full(n_triangles, tilt_deg, dtype=np.float64),
        "azimuth_deg": np.full(n_triangles, azimuth_deg, dtype=np.float64),
        "color_slot": np.full(n_triangles, color_slot, dtype=np.int32),
        "dropped": np.zeros(n_triangles, dtype=bool),
    }






def _triangulate_polygon(poly):
    """
    Fan-free triangulation of a polygon, holes included.

    shapely's Delaunay triangulates the convex hull of the vertices, so the
    triangles that fall outside a concave outline - or inside a hole - are
    dropped by testing their centroid against the polygon itself.
    """
    try:
        candidates = triangulate(poly)
    except Exception:
        return []

    kept = []

    for triangle in candidates:
        if not poly.contains(triangle.representative_point()):
            continue

        coords = list(triangle.exterior.coords)[:3]

        if len(coords) == 3:
            kept.append(coords)

    return kept


def _color_polygons(geometries, tolerance=0.05):
    """
    Give touching faces different colours, greedily.

    Roof faces form a planar adjacency graph, so a handful of hues always
    suffices, and colouring by adjacency means every side-by-side comparison
    a reader makes is between two slots that were validated as an adjacent
    pair. Cycling hues by row order would instead put two near-identical
    colours next to each other sooner or later.
    """
    count = len(geometries)

    slots = np.zeros(count, dtype=np.int32)

    if count < 2:
        return slots

    tree = STRtree(geometries)

    neighbours = {}

    for index, geometry in enumerate(geometries):
        found = tree.query(geometry.buffer(tolerance))

        neighbours[index] = {
            int(other) for other in np.atleast_1d(found)
            if int(other) != index
        }

    # Most-constrained face first, so the awkward ones pick before the easy
    # ones and the total number of hues stays low.
    order = sorted(range(count), key=lambda k: len(neighbours[k]), reverse=True)

    assigned = {}

    for index in order:
        taken = {
            assigned[other] for other in neighbours[index]
            if other in assigned
        }

        slot = 0

        while slot in taken:
            slot += 1

        assigned[index] = slot
        slots[index] = slot

    return slots


def _surface_chunk(vertices, triangles):
    """Wrap a surface mesh in the per-triangle attributes the writers expect."""
    count = triangles.shape[0]

    return {
        "vertices": vertices,
        "triangles": triangles,
        "part_id": np.zeros(count, dtype=np.int32),
        "plane_id": np.zeros(count, dtype=np.int32),
        "tilt_deg": np.full(count, np.nan),
        "azimuth_deg": np.full(count, np.nan),
        "color_slot": np.full(count, -1, dtype=np.int32),
        # Flagged so the writers put it in the recessive layer and keep it
        # out of the face table - it is context, not a detected face.
        "dropped": np.ones(count, dtype=bool),
    }


def _dsm_surface_mesh(dsm, transform, nodata, *, max_triangles):
    """
    Triangulate the DSM itself, as the reference surface.

    Two triangles per grid cell whose four corners all carry data. On a
    regular grid this *is* the Delaunay triangulation of the pixel centres,
    so it is the same construction as the point-cloud version, just on
    samples that have already been binned and gap-filled.

    The grid is subsampled by whatever whole step keeps the result within
    ``max_triangles``. Subsampling thins the mesh but does not move it: the
    surface keeps its shape, which is all this layer is for.
    """
    valid = np.isfinite(dsm) & (dsm > -9000)

    if nodata is not None:
        valid &= dsm != nodata

    if not np.any(valid):
        return None

    height, width = dsm.shape

    step = int(max(1, np.ceil(np.sqrt(2.0 * height * width / max(max_triangles, 1)))))

    sub = dsm[::step, ::step]
    sub_valid = valid[::step, ::step]

    rows, cols = sub.shape

    if rows < 2 or cols < 2:
        return None

    # A cell becomes two triangles only when all four of its corners have
    # data, so the mesh stops at the edge of a roof instead of ramping down
    # to whatever lies beyond it.
    cell = (
        sub_valid[:-1, :-1] & sub_valid[:-1, 1:] &
        sub_valid[1:, :-1] & sub_valid[1:, 1:]
    )

    if not np.any(cell):
        return None

    col_index = np.arange(cols)
    row_index = np.arange(rows)

    xs = transform.c + (col_index * step + 0.5) * transform.a
    ys = transform.f + (row_index * step + 0.5) * transform.e

    grid_x, grid_y = np.meshgrid(xs, ys)

    vertices = np.column_stack([
        grid_x.ravel(), grid_y.ravel(), sub.ravel()
    ]).astype(np.float64)

    flat = np.arange(rows * cols).reshape(rows, cols)

    top_left = flat[:-1, :-1][cell]
    top_right = flat[:-1, 1:][cell]
    bottom_left = flat[1:, :-1][cell]
    bottom_right = flat[1:, 1:][cell]

    triangles = np.vstack([
        np.column_stack([top_left, top_right, bottom_right]),
        np.column_stack([top_left, bottom_right, bottom_left]),
    ]).astype(np.int64)

    # Keep only the vertices the triangles actually use. The full grid still
    # holds a nodata elevation at every empty pixel, and carrying those
    # along would stretch the scene's z range down to -9999 even though
    # nothing is drawn there.
    used, compact = np.unique(triangles.ravel(), return_inverse=True)

    return _surface_chunk(
        vertices[used],
        compact.reshape(-1, 3).astype(np.int64)
    )


# LiDAR classes (ASPRS / ICGC), in muted tones so the detected faces stay
# the foreground. Anything not listed falls back to the unclassified grey.
_POINT_CLASS_STYLE = {
    1: ("unclassified", "#b3b1a9"),
    2: ("ground", "#a88b62"),
    3: ("low vegetation", "#9cc08a"),
    4: ("medium vegetation", "#6fa35c"),
    5: ("high vegetation", "#4c8a3f"),
    6: ("building", "#7d7c77"),
    7: ("noise", "#d7a1a1"),
    8: ("model key point", "#b3b1a9"),
    9: ("water", "#77a3cc"),
    17: ("bridge", "#9d8fb5"),
}
_POINT_OTHER_STYLE = ("other", "#b3b1a9")

# Tile file name the ICGC download writes, as read by the detectors.
_LAZ_NAME = "lidar-territorial-v3r1-full1km{tile_id}-2021-2023.laz"


def _crop_points(points, bounds):
    """Keep the points inside an (minx, miny, maxx, maxy) box."""
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        return np.zeros((0, 4))

    minx, miny, maxx, maxy = bounds

    inside = (
        (points[:, 0] >= minx) & (points[:, 0] <= maxx) &
        (points[:, 1] >= miny) & (points[:, 1] <= maxy)
    )

    return points[inside]


def _read_area_points(laz_dir, laz_tiles, bounds, *, progress=False,
                      chunk_size=2_000_000):
    """
    Every LiDAR return inside ``bounds``, read tile by tile.

    Tiles are skipped on their header bounds before any point is decoded,
    and the ones that overlap are streamed in chunks and cropped as they
    go: a 1 km tile holds tens of millions of returns and only a small
    corner of it is usually wanted.
    """
    import laspy

    minx, miny, maxx, maxy = bounds

    if laz_tiles:
        paths = [
            os.path.join(
                laz_dir, _LAZ_NAME.format(tile_id=tile["properties"]["ID1K"])
            )
            for tile in laz_tiles
        ]
    else:
        paths = [
            os.path.join(laz_dir, name) for name in sorted(os.listdir(laz_dir))
            if name.lower().endswith((".laz", ".las"))
        ]

    selected = []

    for path in paths:
        if not os.path.exists(path):
            continue
        with laspy.open(path) as reader:
            low, high = reader.header.mins, reader.header.maxs
        if high[0] < minx or low[0] > maxx or high[1] < miny or low[1] > maxy:
            continue
        selected.append(path)

    if not selected:
        warnings.warn(
            "No LiDAR tile in %s overlaps the roofs' area; the point layer "
            "is empty." % laz_dir,
            stacklevel=3
        )
        return np.zeros((0, 4))

    if progress:
        selected = tqdm(selected, desc="Reading LiDAR points", unit="tile")

    parts = []

    for path in selected:
        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)

                inside = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)

                if not np.any(inside):
                    continue

                z = np.asarray(chunk.z, dtype=np.float64)
                classification = np.asarray(chunk.classification, dtype=np.float64)

                parts.append(np.column_stack((
                    x[inside], y[inside], z[inside], classification[inside]
                )))

    if not parts:
        return np.zeros((0, 4))

    return np.vstack(parts)


def _prepare_points(cloud, *, classes, max_points):
    """Filter by class and thin to the budget; returns (xyz, class)."""
    if cloud.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.uint8)

    if cloud.shape[1] >= 4:
        point_class = cloud[:, 3].astype(np.uint8)
    else:
        point_class = np.ones(cloud.shape[0], dtype=np.uint8)

    xyz = cloud[:, :3]

    if classes is not None:
        keep = np.isin(point_class, np.asarray(list(classes)))
        xyz, point_class = xyz[keep], point_class[keep]

    if max_points and xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(xyz.shape[0], max_points, replace=False))
        xyz, point_class = xyz[keep], point_class[keep]

    return xyz, point_class


def _point_style(code):
    return _POINT_CLASS_STYLE.get(int(code), _POINT_OTHER_STYLE)


def _scene_extent(mesh):
    """Every coordinate in the scene, faces and points together."""
    stacks = [mesh["vertices"]]

    if mesh.get("points") is not None and mesh["points"].shape[0]:
        stacks.append(mesh["points"])

    stacks = [s for s in stacks if s.size]

    if not stacks:
        return np.zeros((0, 3))

    return np.vstack(stacks)


def _empty_mesh(crs):
    return {
        "vertices": np.zeros((0, 3), dtype=np.float64),
        "triangles": np.zeros((0, 3), dtype=np.int64),
        "part_id": np.zeros(0, dtype=np.int32),
        "plane_id": np.zeros(0, dtype=np.int32),
        "tilt_deg": np.zeros(0, dtype=np.float64),
        "azimuth_deg": np.zeros(0, dtype=np.float64),
        "color_slot": np.zeros(0, dtype=np.int32),
        "dropped": np.zeros(0, dtype=bool),
        "points": np.zeros((0, 3), dtype=np.float64),
        "point_class": np.zeros(0, dtype=np.uint8),
        "crs": crs,
    }


def _warn_if_heavy(mesh, output_path):
    """Say so before writing a scene the viewer will choke on."""
    count = mesh["triangles"].shape[0]

    if count < _HEAVY_TRIANGLE_COUNT:
        return

    warnings.warn(
        "Roof mesh has %s triangles, which will be slow to open (especially "
        "as .html). Narrow the GeoDataFrame you pass in - by building, or "
        "with a spatial filter - before rendering."
        % f"{count:,}",
        stacklevel=3
    )


def _concatenate_meshes(chunks, crs):
    """Stack per-face meshes, re-basing the triangle indices."""
    vertices = []
    triangles = []

    offset = 0

    for chunk in chunks:
        vertices.append(chunk["vertices"])
        triangles.append(chunk["triangles"] + offset)

        offset += chunk["vertices"].shape[0]

    mesh = {
        "vertices": np.vstack(vertices),
        "triangles": np.vstack(triangles),
        "crs": crs,
    }

    for key in (
        "part_id", "plane_id", "tilt_deg",
        "azimuth_deg", "color_slot", "dropped"
    ):
        mesh[key] = np.concatenate([chunk[key] for chunk in chunks])

    return mesh


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


# ============================================================
# Writers - the extension of output_path picks one
# ============================================================

def _write_mesh(mesh, output_path, *, color_by, title):
    extension = os.path.splitext(output_path)[1].lower()

    writers = {
        ".html": _write_html,
        ".png": _write_png,
        ".obj": _write_obj,
        ".ply": _write_ply,
    }

    if extension not in writers:
        raise ValueError(
            f"Cannot write {extension!r}. Use one of: "
            + ", ".join(sorted(writers))
        )

    writers[extension](mesh, output_path, color_by=color_by, title=title)


def _triangle_colors(mesh, color_by, palette):
    """One hex colour per triangle, by whichever job the colour is doing."""
    dropped = mesh["dropped"]

    colors = np.full(mesh["triangles"].shape[0], _DROPPED_COLOR_LIGHT, dtype=object)

    if color_by == "plane":
        slots = mesh["color_slot"]

        for index in np.where(~dropped)[0]:
            colors[index] = palette[int(slots[index]) % len(palette)]

    elif color_by == "building":
        parts = mesh["part_id"]

        for index in np.where(~dropped)[0]:
            colors[index] = palette[int(parts[index]) % len(palette)]

    else:
        tilt = mesh["tilt_deg"]

        low, high = _tilt_domain(tilt)

        span = max(high - low, 1e-6)

        for index in np.where(~dropped)[0]:
            colors[index] = _sample_scale(
                _TILT_SCALE,
                (float(tilt[index]) - low) / span
            )

    return colors


def _tilt_domain(tilt):
    """
    Fixed 0-to-45-degree domain for the tilt ramp, not the observed range.

    Stretching the ramp over whatever happens to be in the scene would
    paint a roof where every face sits within a twentieth of a degree in
    the full light-to-dark sweep, as if the faces differed sharply. Tilt
    has a natural absolute scale, so the ramp uses it, and two runs stay
    comparable. The top end only moves for a roof steeper than 45 degrees.
    """
    finite = tilt[np.isfinite(tilt)]

    high = 45.0

    if finite.size and float(finite.max()) > high:
        high = float(np.ceil(finite.max() / 5.0) * 5.0)

    return 0.0, high


def _sample_scale(scale, position):
    """Interpolate a hex colour out of a [(stop, hex), ...] ramp."""
    position = min(max(position, 0.0), 1.0)

    for k in range(len(scale) - 1):
        low_stop, low_hex = scale[k]
        high_stop, high_hex = scale[k + 1]

        if position <= high_stop:
            width = high_stop - low_stop

            ratio = 0.0 if width <= 0 else (position - low_stop) / width

            low_rgb = _hex_to_rgb(low_hex)
            high_rgb = _hex_to_rgb(high_hex)

            blended = [
                int(round(low_rgb[c] + ratio * (high_rgb[c] - low_rgb[c])))
                for c in range(3)
            ]

            return "#%02x%02x%02x" % tuple(blended)

    return scale[-1][1]


def _hex_to_rgb(value):
    value = value.lstrip("#")

    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _face_table_rows(mesh):
    """
    Collapse triangles back to faces, for the table view.

    Three of the palette's light slots sit below 3:1 against the chart
    surface, so the colours need relief: the reader must be able to get the
    same information without them. This table is that relief, and it is
    also just a faster way to find a specific face than hunting in 3D.
    """
    rows = {}

    for index in range(mesh["triangles"].shape[0]):
        if mesh["dropped"][index]:
            continue

        key = (int(mesh["part_id"][index]), int(mesh["plane_id"][index]))

        entry = rows.setdefault(key, {
            "part_id": key[0],
            "plane_id": key[1],
            "tilt_deg": float(mesh["tilt_deg"][index]),
            "azimuth_deg": float(mesh["azimuth_deg"][index]),
            "color_slot": int(mesh["color_slot"][index]),
            "triangles": 0,
            "area": 0.0,
        })

        entry["triangles"] += 1

        entry["area"] += _triangle_area(mesh, index)

    return sorted(
        rows.values(),
        key=lambda row: (row["part_id"], row["plane_id"])
    )


def _triangle_area(mesh, index):
    """True (tilted) area of one triangle."""
    corners = mesh["vertices"][mesh["triangles"][index]]

    cross = np.cross(corners[1] - corners[0], corners[2] - corners[0])

    return float(np.linalg.norm(cross) / 2.0)


def _write_html(mesh, output_path, *, color_by, title):
    """
    Interactive WebGL page (three.js): orbit, hover, fly to a face.

    The scene is embedded as binary arrays rather than JSON, so the page
    stays small and opens fast even with a million-plus LiDAR returns; see
    ``roofs_3d_viewer`` for the page itself. No Python dependency beyond
    numpy - three.js is fetched from a CDN when the page is opened.
    """
    try:
        from .roofs_3d_viewer import build_viewer_html
    except ImportError:
        from greenshadow.roofs_3d_viewer import build_viewer_html

    extent = _scene_extent(mesh)
    origin = extent.min(axis=0) if extent.size else np.zeros(3)

    vertices = mesh["vertices"] - origin
    triangles = mesh["triangles"]
    dropped = mesh["dropped"]

    rows = _face_table_rows(mesh)
    face_index = {(row["part_id"], row["plane_id"]): k for k, row in enumerate(rows)}

    face_triangles = np.where(~dropped)[0]
    triangle_face = np.array([
        face_index[(int(mesh["part_id"][t]), int(mesh["plane_id"][t]))]
        for t in face_triangles
    ], dtype=np.int64)

    # Each face's triangles contiguous, so the page can address a face as a
    # (start, count) range.
    order = np.argsort(triangle_face, kind="stable")
    face_triangles = face_triangles[order]
    triangle_face = triangle_face[order]

    starts = np.searchsorted(triangle_face, np.arange(len(rows)), side="left")
    ends = np.searchsorted(triangle_face, np.arange(len(rows)), side="right")

    faces = [
        {
            "part": row["part_id"],
            "plane": row["plane_id"],
            "tilt": round(row["tilt_deg"], 2),
            "azimuth": round(row["azimuth_deg"], 2),
            "area": round(row["area"], 2),
            "slot": row["color_slot"],
            "start": int(starts[k]),
            "count": int(ends[k] - starts[k]),
        }
        for k, row in enumerate(rows)
    ]

    face_positions = vertices[triangles[face_triangles]]
    surface_positions = vertices[triangles[dropped]]

    points = mesh.get("points")
    if points is None:
        points = np.zeros((0, 3))

    point_class = mesh.get("point_class", np.zeros(0, dtype=np.uint8))

    codes, class_index, counts = np.unique(
        point_class, return_inverse=True, return_counts=True
    )

    point_classes = []
    for code, count in zip(codes, counts):
        name, color = _point_style(code)
        point_classes.append({
            "code": int(code), "name": name, "color": color, "count": int(count),
        })

    tilt = np.array([row["tilt_deg"] for row in rows]) if rows else np.zeros(0)

    page = build_viewer_html(
        title=title,
        crs=mesh.get("crs"),
        origin=origin,
        faces=faces,
        face_positions=face_positions,
        surface_positions=surface_positions,
        points=points - origin if points.shape[0] else points,
        point_class_index=class_index,
        point_classes=point_classes,
        color_by=color_by,
        face_palette=_FACE_COLORS_LIGHT,
        tilt_scale=_TILT_SCALE,
        tilt_high=_tilt_domain(tilt)[1],
        surface_color=_DROPPED_COLOR_LIGHT,
    )

    with open(output_path, "w") as handle:
        handle.write(page)


def _write_png(mesh, output_path, *, color_by, title):
    """Still image, for when a file on disk beats a browser tab."""
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.ticker import MaxNLocator

    vertices = mesh["vertices"]
    triangles = mesh["triangles"]

    colors = _triangle_colors(mesh, color_by, _FACE_COLORS_LIGHT)

    figure = plt.figure(figsize=(12, 8), facecolor=_SURFACE_LIGHT)

    axes = figure.add_subplot(111, projection="3d")
    axes.set_facecolor(_SURFACE_LIGHT)

    polygons = Poly3DCollection(
        vertices[triangles],
        facecolors=list(colors),
        edgecolors="none",
        linewidths=0
    )

    points = mesh.get("points")

    if points is not None and points.shape[0]:
        point_class = mesh["point_class"]
        # matplotlib draws every marker on the CPU; past a couple of
        # hundred thousand it only slows the render without adding detail
        # at this image size.
        if points.shape[0] > 200_000:
            keep = np.random.default_rng(0).choice(points.shape[0], 200_000, replace=False)
            points, point_class = points[keep], point_class[keep]
        colors_points = [_point_style(code)[1] for code in point_class]
        axes.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=0.3, c=colors_points, alpha=0.6, linewidths=0, depthshade=False
        )

    axes.add_collection3d(polygons)

    _set_true_aspect(axes, _scene_extent(mesh))

    # A flat scene pushes the axis labels into their own tick labels, so
    # both get pushed back out.
    axes.set_xlabel("x (m)", labelpad=18)
    axes.set_ylabel("y (m)", labelpad=18)
    axes.set_zlabel("z (m)", labelpad=10)

    axes.tick_params(axis="both", which="major", labelsize=9, pad=4)

    axes.zaxis.set_major_locator(MaxNLocator(nbins=5))

    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.pane.set_alpha(0.0)

    axes.set_title(title, color=_TEXT_PRIMARY_LIGHT, pad=16)

    axes.view_init(elev=42, azim=-58)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=_SURFACE_LIGHT)

    plt.close(figure)


def _set_true_aspect(axes, vertices):
    """
    True metre-per-unit on all three axes, without padding the scene.

    Giving every axis the same *range* would be the easy way to keep tilts
    honest, but a 50 m wide roof only 10 m tall would then sit in a 50 m
    tall box and read as flat. Each axis gets its own data range instead,
    and the box aspect carries the real proportions.
    """
    if vertices.size == 0:
        return

    low = vertices.min(axis=0)
    high = vertices.max(axis=0)

    span = np.maximum(high - low, 1e-6)

    # A little headroom so faces do not touch the panes.
    pad = span * 0.03

    axes.set_xlim(low[0] - pad[0], high[0] + pad[0])
    axes.set_ylim(low[1] - pad[1], high[1] + pad[1])
    axes.set_zlim(low[2] - pad[2], high[2] + pad[2])

    try:
        axes.set_box_aspect(tuple(span / span.max()))
    except AttributeError:
        pass


def _write_obj(mesh, output_path, *, color_by, title):
    """
    Wavefront OBJ plus a matching MTL, so faces keep their colours.

    Opens in MeshLab, Blender, CloudCompare and most GIS 3D viewers, which
    is the way to get this into a scene that already has other geometry.
    """
    material_path = os.path.splitext(output_path)[0] + ".mtl"

    colors = _triangle_colors(mesh, color_by, _FACE_COLORS_LIGHT)

    unique = sorted(set(colors))

    names = {color: "face_%02d" % k for k, color in enumerate(unique)}

    with open(material_path, "w") as handle:
        for color, name in names.items():
            red, green, blue = [c / 255.0 for c in _hex_to_rgb(color)]

            handle.write(
                "newmtl %s\nKd %.4f %.4f %.4f\nKa 0 0 0\nd 1.0\nillum 1\n\n"
                % (name, red, green, blue)
            )

    # OBJ has no georeferencing, and full UTM coordinates overflow the
    # precision most viewers keep, so the mesh is shifted to a local origin
    # and the offset is recorded in the header.
    extent = _scene_extent(mesh)
    origin = extent.min(axis=0) if extent.size else np.zeros(3)

    points = mesh.get("points")
    if points is None:
        points = np.zeros((0, 3))

    with open(output_path, "w") as handle:
        handle.write("# %s\n" % title)
        handle.write("# crs: %s\n" % mesh.get("crs"))
        handle.write(
            "# local origin (add back for map coordinates): %.3f %.3f %.3f\n"
            % tuple(origin)
        )
        handle.write("mtllib %s\n" % os.path.basename(material_path))

        for vertex in mesh["vertices"] - origin:
            handle.write("v %.4f %.4f %.4f\n" % tuple(vertex))

        current = None

        for index, triangle in enumerate(mesh["triangles"]):
            material = names[colors[index]]

            if material != current:
                handle.write("usemtl %s\n" % material)
                current = material

            handle.write("f %d %d %d\n" % tuple(triangle + 1))

        # LiDAR returns as point elements after the faces, so the face
        # indices above stay valid. OBJ has no per-point colour; use .ply
        # to keep the classes.
        if points.shape[0]:
            handle.write("g lidar_points\n")
            base = mesh["vertices"].shape[0]
            for point in points - origin:
                handle.write("v %.4f %.4f %.4f\n" % tuple(point))
            for index in range(points.shape[0]):
                handle.write("p %d\n" % (base + index + 1))


def _write_ply(mesh, output_path, *, color_by, title):
    """
    ASCII PLY with per-vertex and per-face colour, for CloudCompare.

    The LiDAR returns are appended as extra vertices that no face uses,
    each coloured by its class, which CloudCompare and MeshLab show as a
    point cloud alongside the faces.
    """
    colors = _triangle_colors(mesh, color_by, _FACE_COLORS_LIGHT)

    extent = _scene_extent(mesh)
    origin = extent.min(axis=0) if extent.size else np.zeros(3)

    points = mesh.get("points")
    if points is None:
        points = np.zeros((0, 3))

    face_vertex_rgb = _hex_to_rgb(_DROPPED_COLOR_LIGHT)

    with open(output_path, "w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment %s\n" % title)
        handle.write("comment crs: %s\n" % mesh.get("crs"))
        handle.write(
            "comment local origin: %.3f %.3f %.3f\n" % tuple(origin)
        )
        handle.write(
            "element vertex %d\n"
            % (mesh["vertices"].shape[0] + points.shape[0])
        )
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        handle.write("element face %d\n" % mesh["triangles"].shape[0])
        handle.write("property list uchar int vertex_index\n")
        handle.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        handle.write("end_header\n")

        for vertex in mesh["vertices"] - origin:
            handle.write(
                "%.4f %.4f %.4f %d %d %d\n" % (tuple(vertex) + face_vertex_rgb)
            )

        for point, code in zip(points - origin, mesh.get("point_class", [])):
            handle.write(
                "%.4f %.4f %.4f %d %d %d\n"
                % (tuple(point) + _hex_to_rgb(_point_style(code)[1]))
            )

        for index, triangle in enumerate(mesh["triangles"]):
            red, green, blue = _hex_to_rgb(colors[index])

            handle.write(
                "3 %d %d %d %d %d %d\n"
                % (triangle[0], triangle[1], triangle[2], red, green, blue)
            )
