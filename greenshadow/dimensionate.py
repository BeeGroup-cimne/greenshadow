# -*- coding: utf-8 -*-
"""
greenshadow.dimensionate

PV array dimensioning: module layout, string grouping and yield.

``plan_pv_strings_in_buildings`` packs detected roof planes. PV fields and
agrivoltaics over SIGPAC parcels live in ``greenshadow.parcel_pv``
(``plan_parcel_pv_scenarios``), which lays out only the agricultural parcels
(``rural_parcels.filter_agricultural_parcels``), so roads, buildings, water,
forest and urban zones never get a layout.

Packs as many modules as will fit onto each roof plane produced by
``detect_roof_planes``, evaluates each module
against the annual solar rasters, groups the survivors into strings, and
aggregates energy and peak power per building. The survivors are whatever
clears a POA threshold set by the best-lit module of the building, so shaded
roof faces and the aprons around chimneys drop out while everything that still
earns its keep is wired, however short the last string ends up.

Two mounting regimes, split by the plane's own tilt:

- **Coplanar** on an inclined plane. The module takes the roof's orientation,
  so the plane-of-array irradiation is what the ``mode="total"`` annual raster
  already measured on that surface. Rows never self-shade, so they are packed
  edge to edge; only the map footprint is foreshortened by cos(tilt).
- **Racks** on a flat plane. Modules face South at a tilt chosen by sweeping
  candidates, with the row pitch set so that no row shades the next at the
  winter-solstice sun altitude. Because the panel orientation no longer matches
  the surface, POA is reconstructed from the ``mode="fourier_poa"`` basis.

The flat-roof sweep deliberately includes 0 deg: laying modules flat packs the
most of them, and on a small or awkward roof that can beat a steeper tilt that
yields more per module but fits fewer.
"""

import os
import sys
from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import shapely
from rasterio.features import shapes
from rasterio.transform import Affine, rowcol
from shapely.affinity import rotate
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.prepared import prep
try:
    from .logging_utils import tqdm
except ImportError:
    from greenshadow.logging_utils import tqdm

try:
    from .buildings_infer import _plane_coefficients, _anchor_height, _WindowedBand
    from .shader import reconstruct_poa_from_basis
except ImportError:
    from greenshadow.buildings_infer import _plane_coefficients, _anchor_height, _WindowedBand
    from greenshadow.shader import reconstruct_poa_from_basis

try:
    from shapely import contains_xy as _contains_xy
except ImportError:  # shapely < 2.0
    _contains_xy = None


# ============================================================
# MODULE SPECIFICATIONS
# ============================================================

@dataclass
class PVModuleSpec:
    """
    One commercial module size.

    ``peak_power_kw`` is optional because catalogues quote it per model rather
    than per size; when it is None the engine derives it from the area at STC
    (1 kW/m2) and the ``module_efficiency`` passed to the planner, so a
    caller can change technology without editing every spec.
    """

    name: str
    length_m: float
    width_m: float
    peak_power_kw: float = None

    @property
    def area_m2(self):
        return self.length_m * self.width_m

    def peak_kw(self, module_efficiency):
        if self.peak_power_kw is not None:
            return self.peak_power_kw
        return self.area_m2 * module_efficiency


DEFAULT_MODULES = (
    PVModuleSpec("2.30x1.14", 2.30, 1.14),
    PVModuleSpec("2.20x1.14", 2.20, 1.14),
    PVModuleSpec("1.78x1.14", 1.78, 1.14),
)


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def solar_min_altitude(latitude_deg):
    """
    Solar altitude at solar noon on the winter solstice, in degrees.

    This is the worst case a fixed rack has to clear, so it sets the inter-row
    spacing: any shallower sun angle would shade the next row, but it only does
    so around the shortest days of the year, when there is little to lose.
    """
    return 90.0 - abs(latitude_deg) - 23.45


# ============================================================
# GRID PACKING
# ============================================================

def _grid_in_polygon(
    polygon,
    cell_u,
    cell_v,
    pitch_u,
    pitch_v,
    rotation_deg,
    phase_steps=3,
):
    """
    Lay a rotated, non-overlapping grid of cells fully inside a polygon.

    ``rotation_deg`` is the compass azimuth the grid's local +v axis should
    point along - the roof's aspect for coplanar mounting, due South for racks.
    The polygon is rotated so that axis lands on +y, an axis-aligned grid is
    generated there, and the surviving cells are rotated back. Rotating the
    polygon rather than each candidate cell is what keeps the grid coherent;
    rotating individual boxes about a fixed origin, as an earlier draft did,
    scatters them away from their intended cell.

    The grid origin is swept over ``phase_steps`` x ``phase_steps`` sub-pitch
    offsets and the arrangement with the most cells wins. A half-cell shift can
    easily be worth an extra row on a roof whose edge does not happen to align
    with the first candidate origin, and the sweep is cheap next to the
    per-module raster sampling that follows.

    Returns a list of cells in serpentine order - left to right, then right to
    left on the next row - so that consecutive modules in a string are physical
    neighbours rather than jumping the width of the roof at every row break.
    """
    if polygon.is_empty or cell_u <= 0 or cell_v <= 0:
        return []

    origin = polygon.centroid
    local = rotate(polygon, rotation_deg, origin=origin)

    if local.is_empty:
        return []

    minx, miny, maxx, maxy = local.bounds

    if maxx - minx < cell_u or maxy - miny < cell_v:
        return []

    # Every candidate cell of every phase is tested in one vectorised call:
    # a shapely box per cell was most of the planner's time. Positions are
    # accumulated exactly as a cell-by-cell walk would, so the cell edges do
    # not depend on how the loop is written.
    def walk(start, cell, pitch, stop):
        steps = []
        while start + cell <= stop + 1e-9:
            steps.append(start)
            start += pitch
        return np.asarray(steps)

    phases = []
    for step_u in range(phase_steps):
        xs = walk(minx + step_u * pitch_u / phase_steps, cell_u, pitch_u, maxx)
        for step_v in range(phase_steps):
            ys = walk(miny + step_v * pitch_v / phase_steps, cell_v, pitch_v, maxy)
            if xs.size and ys.size:
                phases.append(np.meshgrid(xs, ys))

    if not phases:
        return []

    flat_x = np.concatenate([grid_x.ravel() for grid_x, _ in phases])
    flat_y = np.concatenate([grid_y.ravel() for _, grid_y in phases])
    shapely.prepare(local)
    covered = shapely.covers(
        local, shapely.box(flat_x, flat_y, flat_x + cell_u, flat_y + cell_v)
    )

    # The first phase with the most cells wins.
    best_x = best_y = np.empty(0)
    offset = 0
    for grid_x, grid_y in phases:
        inside = covered[offset:offset + grid_x.size].reshape(grid_x.shape)
        offset += grid_x.size

        if inside.sum() <= best_x.size:
            continue

        # Serpentine: odd rows run right to left.
        inside[1::2] = inside[1::2, ::-1]
        grid_x[1::2] = grid_x[1::2, ::-1]
        best_x = grid_x[inside]
        best_y = grid_y[inside]

    if best_x.size == 0:
        return []

    # Corners as shapely.geometry.box lays them, turned back with the
    # arithmetic of shapely.affinity.rotate, so the cells are the same to
    # the last bit as rotating each box would give.
    x0, x1 = best_x, best_x + cell_u
    y0, y1 = best_y, best_y + cell_v
    corners_x = np.stack([x1, x1, x0, x0, x1], axis=1)
    corners_y = np.stack([y0, y1, y1, y0, y0], axis=1)

    angle = -rotation_deg * pi / 180.0
    cosp, sinp = cos(angle), sin(angle)
    if abs(cosp) < 2.5e-16:
        cosp = 0.0
    if abs(sinp) < 2.5e-16:
        sinp = 0.0
    ox, oy = origin.x, origin.y
    xoff = ox - ox * cosp + oy * sinp
    yoff = oy - ox * sinp - oy * cosp

    turned = np.stack([
        cosp * corners_x + -sinp * corners_y + xoff,
        sinp * corners_x + cosp * corners_y + yoff,
    ], axis=-1)

    return list(shapely.polygons(turned))


# ============================================================
# RASTER SAMPLING
# ============================================================

def _read_window(src, geometry, bands):
    """
    Read the raster block under a polygon, plus the grid it sits on.

    Returns ``(values, transform, valid)`` with ``values`` shaped (len(bands),
    h, w), or None when the polygon falls outside the raster. Reading one block
    per roof plane and sampling modules out of memory keeps the per-module cost
    to a local array reduction, rather than a masked read per module.
    """
    minx, miny, maxx, maxy = geometry.bounds

    top, left = rowcol(src.transform, minx, maxy, op=np.floor)
    bottom, right = rowcol(src.transform, maxx, miny, op=np.ceil)

    top = max(int(top) - 1, 0)
    left = max(int(left) - 1, 0)
    bottom = min(int(bottom) + 1, src.height)
    right = min(int(right) + 1, src.width)

    if bottom <= top or right <= left:
        return None

    window = rasterio.windows.Window(left, top, right - left, bottom - top)
    values = src.read(bands, window=window).astype(np.float32)

    transform = src.transform * Affine.translation(left, top)

    valid = np.ones(values.shape[1:], dtype=bool)
    if src.nodata is not None:
        valid &= ~np.any(np.isclose(values, src.nodata), axis=0)
    valid &= ~np.any(~np.isfinite(values), axis=0)

    return values, transform, valid


def _pixel_centres(transform, shape_hw):
    height, width = shape_hw
    xs = transform.c + (np.arange(width) + 0.5) * transform.a
    ys = transform.f + (np.arange(height) + 0.5) * transform.e
    return np.meshgrid(xs, ys)


def _polygon_pixel_mask(geometry, grid_x, grid_y):
    """
    Boolean mask of pixel centres falling inside a polygon.
    """
    if _contains_xy is not None:
        return _contains_xy(geometry, grid_x, grid_y)

    prepared = prep(geometry)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    from shapely.geometry import Point
    inside = np.fromiter(
        (prepared.contains(Point(x, y)) for x, y in zip(flat_x, flat_y)),
        dtype=bool,
        count=flat_x.size,
    )
    return inside.reshape(grid_x.shape)


def _sample_cells(cells, values, transform, valid):
    """
    Per-cell mean of every band, shaped (len(cells), n_bands).

    A cell whose pixel centres are all nodata - or which is smaller than a
    pixel and catches no centre at all - falls back to the nearest pixel, and
    reports NaN only when that is invalid too.
    """
    n_bands = values.shape[0]
    height, width = values.shape[1:]

    grid_x, grid_y = _pixel_centres(transform, (height, width))

    out = np.full((len(cells), n_bands), np.nan, dtype=np.float64)

    for index, cell in enumerate(cells):
        minx, miny, maxx, maxy = cell.bounds

        top, left = rowcol(transform, minx, maxy, op=np.floor)
        bottom, right = rowcol(transform, maxx, miny, op=np.ceil)

        top = max(int(top), 0)
        left = max(int(left), 0)
        bottom = min(int(bottom) + 1, height)
        right = min(int(right) + 1, width)

        if bottom <= top or right <= left:
            continue

        sub_x = grid_x[top:bottom, left:right]
        sub_y = grid_y[top:bottom, left:right]

        inside = _polygon_pixel_mask(cell, sub_x, sub_y)
        inside &= valid[top:bottom, left:right]

        if inside.any():
            block = values[:, top:bottom, left:right]
            out[index] = [np.mean(band[inside]) for band in block]
            continue

        row, col = rowcol(transform, cell.centroid.x, cell.centroid.y)
        row = min(max(int(row), 0), height - 1)
        col = min(max(int(col), 0), width - 1)
        if valid[row, col]:
            out[index] = values[:, row, col]

    return out


# ============================================================
# OBSTACLES
# ============================================================

def _obstacle_geometry(
    geometry,
    tilt_deg,
    azimuth_deg,
    dsm,
    dsm_transform,
    dsm_nodata,
    residual_m,
):
    """
    Polygons of everything sticking out of (or sunk into) the fitted roof plane.

    Chimneys, aerials, air-conditioning units, skylights and dormers all show up
    as DSM pixels that sit further than ``residual_m`` from the plane the
    detector fitted. The plane's orientation comes from the tilt and azimuth the
    detector already reported, and its height from the same median-vote anchor
    ``roofs_3d`` uses, so this agrees with the 3D view by construction.

    Returns None when the plane cannot be anchored, which means no obstacle
    information rather than no obstacles.
    """
    a, b = _plane_coefficients(tilt_deg, azimuth_deg)
    c = _anchor_height(geometry, a, b, dsm, dsm_transform, dsm_nodata)

    if c is None:
        return None

    minx, miny, maxx, maxy = geometry.bounds

    top, left = rowcol(dsm_transform, minx, maxy, op=np.floor)
    bottom, right = rowcol(dsm_transform, maxx, miny, op=np.ceil)

    top = max(int(top), 0)
    left = max(int(left), 0)
    bottom = min(int(bottom) + 1, dsm.shape[0])
    right = min(int(right) + 1, dsm.shape[1])

    if bottom <= top or right <= left:
        return None

    block = dsm[top:bottom, left:right]
    transform = dsm_transform * Affine.translation(left, top)

    grid_x, grid_y = _pixel_centres(transform, block.shape)

    finite = np.isfinite(block)
    if dsm_nodata is not None:
        finite &= ~np.isclose(block, dsm_nodata)

    inside = _polygon_pixel_mask(geometry, grid_x, grid_y) & finite
    if not inside.any():
        return None

    residual = np.abs(block - (a * grid_x + b * grid_y + c))
    flagged = inside & (residual > residual_m)

    if not flagged.any():
        return None

    pixel_size = max(abs(transform.a), abs(transform.e))

    parts = [
        shape(geom).buffer(pixel_size * 0.5)
        for geom, _ in shapes(
            flagged.astype(np.uint8),
            mask=flagged,
            transform=transform,
        )
    ]

    if not parts:
        return None

    return unary_union(parts)


# ============================================================
# PER-PLANE LAYOUT
# ============================================================

_COMPASS_POINTS = (
    (0.0, "north"),
    (45.0, "northeast"),
    (90.0, "east"),
    (135.0, "southeast"),
    (180.0, "south"),
    (225.0, "southwest"),
    (270.0, "west"),
    (315.0, "northwest"),
)


def _compass_name(azimuth_deg):
    """
    Short name for an azimuth, or ``az###`` when it sits between the points.

    Only used to build scenario keys, so that a layer reads ``poa_south_10``
    rather than ``poa_180_10``.
    """
    azimuth = float(azimuth_deg) % 360.0

    for reference, name in _COMPASS_POINTS:
        if min(abs(azimuth - reference), 360.0 - abs(azimuth - reference)) < 1.0:
            return name

    return f"az{azimuth:03.0f}"


def _scenario_label(mount, tilt_deg, azimuth_deg):
    """
    Key identifying one mounting geometry across the whole municipality.

    ``"coplanar"`` covers every module laid flush with its roof plane, whatever
    that plane's own tilt and azimuth: the defining property is that the panel
    and the surface agree, so the ``mode="total"`` raster already measures its
    POA. Racks are keyed by the orientation they impose instead -
    ``"south_10"`` - because that orientation is the same on every flat roof in
    the run and is therefore comparable across buildings.
    """
    if mount == "coplanar":
        return "coplanar"

    return f"{_compass_name(azimuth_deg)}_{float(tilt_deg):02.0f}"


def _scenario_sort_key(scenario):
    """
    Order scenarios for output: coplanar first, then racks by tilt, best last.
    """
    if scenario == "coplanar":
        return (0, 0.0, "")
    if scenario == "best":
        return (2, 0.0, "")

    name, _, tilt = scenario.rpartition("_")

    try:
        return (1, float(tilt), name)
    except ValueError:
        return (1, 0.0, scenario)


BEST_SCENARIO = "best"


def write_scenario_layers(modules_gdf, output_path, *, layer_prefix="poa_"):
    """
    Write one GeoPackage layer per scenario.

    The planner keeps every mounting geometry it costed, so the natural shape
    of the output is a stack of comparable layers - ``poa_coplanar``,
    ``poa_south_05``, ``poa_south_10`` ... - rather than one table a reader has
    to filter. ``poa_best`` holds the per-plane optimum, which is the layer to
    open when the question is how much this municipality could actually build.

    The file is rewritten from scratch, so a scenario that disappears between
    runs does not survive as a stale layer.

    Returns the list of layer names written.
    """
    import os

    if os.path.exists(output_path):
        os.remove(output_path)

    if modules_gdf.empty:
        layer = f"{layer_prefix}{BEST_SCENARIO}"
        modules_gdf.to_file(output_path, driver="GPKG", layer=layer)
        return [layer]

    written = []

    for scenario in sorted(modules_gdf["scenario"].unique(), key=_scenario_sort_key):

        subset = modules_gdf[modules_gdf["scenario"] == scenario]

        if subset.empty:
            continue

        layer = f"{layer_prefix}{scenario}"

        subset.to_file(
            output_path,
            driver="GPKG",
            layer=layer,
            mode="a" if written else "w",
        )
        written.append(layer)

    return written



def _layout_variants(plane, module, config):
    """
    Enumerate the candidate arrangements for one plane and one module size.

    Yields ``(mount, tilt_deg, azimuth_deg, orientation, cell_u, cell_v,
    pitch_u, pitch_v)``. An inclined plane has two candidates, one per module
    orientation; a flat plane has two per candidate tilt.
    """
    orientations = (
        ("portrait", module.width_m, module.length_m),
        ("landscape", module.length_m, module.width_m),
    )

    if plane["mount"] == "coplanar":
        tilt = plane["tilt_deg"]
        cos_tilt = np.cos(np.radians(tilt))

        for name, dim_u, dim_v in orientations:
            yield (
                "coplanar",
                tilt,
                plane["azimuth_deg"],
                name,
                dim_u,
                dim_v * cos_tilt,
                dim_u + config["module_gap_m"],
                dim_v * cos_tilt + config["row_gap_m"],
            )
        return

    tan_alpha = np.tan(np.radians(max(config["row_shading_altitude_deg"], 5.0)))

    for tilt in config["flat_tilts"]:
        cos_tilt = np.cos(np.radians(tilt))
        sin_tilt = np.sin(np.radians(tilt))

        for name, dim_u, dim_v in orientations:
            footprint_v = dim_v * cos_tilt
            shadow_v = dim_v * sin_tilt / tan_alpha

            yield (
                "rack",
                float(tilt),
                config["flat_azimuth_deg"],
                name,
                dim_u,
                footprint_v,
                dim_u + config["module_gap_m"],
                footprint_v + shadow_v + config["row_gap_m"],
            )


def _plan_plane(plane, module, config):
    """
    Every arrangement of one module size on one roof plane, keyed by scenario.

    Returns ``{scenario: [variant, ...]}``, where a variant is one packing -
    portrait or landscape - carrying its unfiltered ``records`` and the best
    ``poa_kwh_m2`` it reached. Nothing is discarded and no variant is chosen
    here: both the shading filter and the portrait/landscape decision need the
    building-wide POA maximum, which only the caller knows once every plane of
    the building has been laid out. ``_select_variant`` applies them.
    """
    buildable = plane["buildable"]

    if buildable.is_empty:
        return {}

    scenarios = {}

    for (
        mount,
        tilt_deg,
        azimuth_deg,
        orientation,
        cell_u,
        cell_v,
        pitch_u,
        pitch_v,
    ) in _layout_variants(plane, module, config):

        cells = _grid_in_polygon(
            buildable,
            cell_u,
            cell_v,
            pitch_u,
            pitch_v,
            azimuth_deg,
            phase_steps=config["phase_steps"],
        )

        if not cells:
            continue

        means = _sample_cells(
            cells,
            plane["values"],
            plane["transform"],
            plane["valid"],
        )

        if mount == "coplanar":
            poa_wh_m2 = means[:, 0]
        else:
            poa_wh_m2 = reconstruct_poa_from_basis(
                means[:, 0],
                means[:, 1],
                means[:, 2],
                means[:, 3],
                means[:, 4],
                tilt_deg=tilt_deg,
                azimuth_deg=azimuth_deg,
                albedo=config["albedo"],
            )

        keep = np.isfinite(poa_wh_m2) & (poa_wh_m2 > 0)
        if not keep.any():
            continue

        area_m2 = module.area_m2
        peak_kw = module.peak_kw(config["module_efficiency"])

        energy = (
            poa_wh_m2
            * area_m2
            * config["module_efficiency"]
            * config["performance_ratio"]
            / 1000.0
        )

        scenario = _scenario_label(mount, tilt_deg, azimuth_deg)

        records = []
        for index, cell in enumerate(cells):
            if not keep[index]:
                continue

            records.append({
                "scenario": scenario,
                "plane_id": plane["plane_id"],
                "module_name": module.name,
                "module_length_m": module.length_m,
                "module_width_m": module.width_m,
                "orientation": orientation,
                "mount": mount,
                "tilt_deg": float(tilt_deg),
                "azimuth_deg": float(azimuth_deg),
                "row_pitch_m": float(pitch_v),
                "area_m2": area_m2,
                "poa_kwh_m2": float(poa_wh_m2[index]) / 1000.0,
                "energy_kwh": float(energy[index]),
                "peak_kw": peak_kw,
                "geometry": cell,
            })

        if not records:
            continue

        scenarios.setdefault(scenario, []).append({
            "records": records,
            "poa_max": float(np.nanmax(poa_wh_m2[keep])) / 1000.0,
        })

    return scenarios


def _select_variant(variants, threshold, config):
    """
    Drop the shaded modules of each packing, then keep the packing that wins.

    ``threshold`` is an absolute POA in kWh/m2, derived by the caller from the
    best module anywhere on the building, so a plane the neighbours overshadow
    all winter - or the skirt of roof a chimney darkens - is measured against
    the roof's own best spot rather than against its local best, which on a bad
    plane would be nearly as bad. Filtering before the portrait/landscape
    comparison matters: the denser packing is only the better one if its extra
    modules survive the filter.
    """
    best = None

    for variant in variants:

        records = [
            record for record in variant["records"]
            if record["poa_kwh_m2"] >= threshold
        ]

        if not records:
            continue

        total = float(sum(record["energy_kwh"] for record in records))
        peak_kw = records[0]["peak_kw"]

        if config["objective"] == "specific_yield":
            # Per-kWp yield ignores how many modules fit, so it picks the tilt
            # that makes each module work hardest - the right objective when the
            # modules and inverter, not the roof, are the scarce resource.
            score = total / (peak_kw * len(records))
        else:
            score = total

        if best is None or score > best["score"]:
            best = {
                "records": records,
                "energy_kwh": total,
                "score": score,
            }

    return best


def _assign_strings(records, string_size, min_string_size, next_string_id):
    """
    Chunk a plane's modules into strings.

    The records arrive in the serpentine order ``_grid_in_polygon`` produced, so
    each chunk is a run of physical neighbours. Strings are unlimited in number
    and, at the default ``min_string_size`` of 1, the trailing remainder is
    wired as its own short string instead of being thrown away: the modules
    that reach this point have already passed the POA filter, so every one of
    them earns its energy and none is worth discarding to keep the string
    lengths uniform. Raising ``min_string_size`` brings the old behaviour back.

    ``string_size=None`` switches the electrical grouping off: the plane's
    modules become a single string, which is the right shape when the question
    is how much a roof can hold rather than how it would be wired.
    """
    if string_size is None:
        for record in records:
            record["string_id"] = next_string_id
        return records, next_string_id + 1

    kept = []

    for start in range(0, len(records), string_size):
        chunk = records[start:start + string_size]

        if len(chunk) < min_string_size:
            break

        for record in chunk:
            record["string_id"] = next_string_id
            kept.append(record)

        next_string_id += 1

    return kept, next_string_id


# ============================================================
# MAIN ENGINE
# ============================================================

def plan_pv_strings_in_buildings(
    roof_planes,
    *,
    poa_basis_path=None,
    annual_energy_path=None,
    dsm_path=None,
    building_id_column="part_id",
    plane_id_column="plane_id",
    tilt_column="tilt_deg",
    azimuth_column="azimuth_deg",
    modules=DEFAULT_MODULES,
    module_efficiency=0.21,
    performance_ratio=0.80,
    flat_slope_threshold_deg=5.0,
    flat_tilts=(0, 5, 10, 15, 20, 25, 30, 35),
    flat_azimuth_deg=180.0,
    allowed_azimuth_range=(90.0, 270.0),
    latitude_deg=None,
    row_shading_altitude_deg=None,
    setback_m=0.3,
    module_gap_m=0.02,
    row_gap_m=0.02,
    obstacle_residual_m=0.15,
    phase_steps=3,
    string_size=18,
    min_string_size=1,
    min_poa_ratio=0.7,
    albedo=0.2,
    objective="energy",
    min_plane_area_m2=5.0,
    output_path=None,
    output_summary_path=None,
    layer_prefix="poa_",
    progress=False,
    n_workers=None,
):
    """
    Plan PV strings over detected roof planes and aggregate them per building.

    Every mounting geometry that was costed survives into the output as its own
    **scenario**, rather than being optimised away: ``"coplanar"`` for modules
    laid flush with an inclined roof, ``"south_05"``, ``"south_10"`` ... for
    racks on a flat one, and ``"best"`` for the per-plane optimum. They are
    written as one GeoPackage layer each, so the scenarios can be compared in
    GIS side by side.

    A scenario only appears on the planes that can host it. ``"coplanar"``
    covers the inclined planes and the rack scenarios cover the flat ones, so
    on a building with both, no single rack layer accounts for the whole roof -
    ``"best"`` is the only layer that does.

    Parameters
    ----------
    roof_planes : GeoDataFrame or str
        Roof planes from ``detect_roof_planes``,
        ideally after ``regularize_roof_planes``. A path to the GeoPackage is
        accepted. The per-plane ``tilt_deg`` / ``azimuth_deg`` columns are read
        directly and are in **degrees**; do not substitute the slope/aspect
        GeoTIFFs, which the digital-model builder writes in **radians**.
    poa_basis_path : str, optional
        Raster written by ``compute_surface_solar_energy(mode="fourier_poa")``.
        Required only when the input contains flat planes, because racks are the
        only case where the panel orientation differs from the surface.
    annual_energy_path : str, optional
        Raster written by ``compute_surface_solar_energy(mode="total")``. Bands
        1 and 2 already measure direct and diffuse energy *on the roof surface*,
        which is exactly the plane-of-array irradiation a coplanar module sees.
        Required only when the input contains inclined planes.
    dsm_path : str, optional
        Refined roof DSM the planes were fitted to. Used to find roof clutter;
        when omitted, only the edge setback constrains the layout.
    building_id_column : str, default="part_id"
        Column grouping planes into buildings. The module size is chosen once
        per building, so that a single building is not fitted with three
        different module models.
    modules : sequence of PVModuleSpec, default=DEFAULT_MODULES
        Candidate sizes. Each is evaluated across the whole building and the
        best-yielding one is kept.
    module_efficiency : float, default=0.21
        Module efficiency at STC, used both for energy and - where a spec does
        not quote one - for peak power.
    performance_ratio : float, default=0.80
        System losses between the array and the meter: inverter, wiring,
        soiling, temperature.
    flat_slope_threshold_deg : float, default=5.0
        Below this tilt a plane counts as flat and gets racks. Above it the
        plane is steep enough that a rack would fight the roof.
    flat_tilts : sequence of float, default=(0, 5, ..., 35)
        Rack tilts to sweep on flat planes. 0 is included on purpose: flat-laid
        modules need no inter-row gap, so on a cramped roof the extra count can
        outweigh the lower yield per module.
    flat_azimuth_deg : float, default=180.0
        Rack azimuth, degrees from North clockwise. 180 is due South.
    allowed_azimuth_range : tuple of float or None, default=(90.0, 270.0)
        Inclined planes facing outside this window - roughly East through South
        to West - are skipped. North-facing roof faces are not worth equipping.
        Pass None to keep every orientation, North included, and let the POA
        each module reports carry that judgement instead.
    latitude_deg : float, optional
        Site latitude, used for the inter-row spacing. Taken from the raster
        centre when omitted.
    row_shading_altitude_deg : float, optional
        Sun altitude the rows must clear. Defaults to ``solar_min_altitude``,
        i.e. winter-solstice noon, the standard no-shading design case.
    setback_m : float, default=0.3
        Margin kept clear inside every roof edge, for access and wind uplift.
    obstacle_residual_m : float, default=0.15
        A DSM pixel this far from the fitted plane is treated as clutter and
        excluded. Loose enough to tolerate the detector's own planarity error,
        tight enough to catch a chimney.
    string_size : int or None, default=18
        Modules per string. The number of strings is not capped: the roof takes
        as many as its surviving modules fill. None wires each plane as a
        single string, which drops the electrical grouping altogether: no
        module is ever left out for failing to fill a chunk.
    min_string_size : int, default=1
        Shortest acceptable string. At 1 nothing is discarded and a plane's
        trailing modules become a short string of their own, which is what
        maximises the energy taken off the roof. Raise it - to half
        ``string_size``, say - to refuse stunted strings instead.
    min_poa_ratio : float or None, default=0.7
        A module is dropped when its annual POA is below this fraction of the
        best module **on the whole building**, not merely of the best on its own
        plane. One threshold per building is what makes the filter discriminate:
        it removes the skirt of roof a chimney or an antenna darkens, and the
        faces a neighbouring block overshadows all winter, rather than keeping
        the least bad modules of a hopeless plane just because its neighbours
        are equally shaded. Within a scenario layer the reference is that
        scenario's best module on the building; ``"best"``, which lets each
        plane choose its own mounting, is measured against the best module the
        building reaches under any mounting. 0 or None keeps every module that
        receives any light at all.
    objective : {"energy", "specific_yield"}, default="energy"
        What the tilt and orientation sweep maximises. ``"energy"`` takes the
        most annual kWh off the whole roof, which on a flat roof usually means
        laying the modules flat: tilting lifts the yield per module by roughly a
        fifth at mid latitudes but the no-shading row pitch costs far more than
        that in module count. ``"specific_yield"`` maximises kWh per kWp
        instead, which is what to use when the modules and the inverter rather
        than the roof area are the binding constraint.
    output_path : str, optional
        GeoPackage for the module geometries, written one layer per scenario.
    output_summary_path : str, optional
        CSV for the per-building, per-scenario summary.
    layer_prefix : str, default="poa_"
        Prefix for the GeoPackage layer names, so the scenarios sort together
        in the layer list: ``poa_coplanar``, ``poa_south_05``, ``poa_best``.
    progress : bool, default=False
        Show a progress bar over the buildings.
    n_workers : int, optional
        Buildings planned at once, in forked processes (default: one per
        CPU). The output does not depend on it.

    Returns
    -------
    tuple
        ``(modules_gdf, building_summary)``, both **long** tables carrying a
        ``scenario`` column: a building appears once per scenario it supports.
        Summing energy or peak power across the whole frame therefore counts
        the same roof several times - filter to one scenario, usually
        ``scenario == "best"``, before aggregating. Files are written only when
        the corresponding output path is given. All areas in the summary are
        measured on the roof surface rather than in map projection, so a
        pitched roof is not under-reported and ``coverage_ratio`` never exceeds
        1.
    """
    if objective not in {"energy", "specific_yield"}:
        raise ValueError("objective must be 'energy' or 'specific_yield'")

    if isinstance(roof_planes, str):
        roof_planes = gpd.read_file(roof_planes)

    planes = roof_planes.copy()

    if min_string_size is None:
        min_string_size = 1

    columns = [
        "scenario",
        building_id_column,
        plane_id_column,
        "module_name",
        "module_length_m",
        "module_width_m",
        "orientation",
        "mount",
        "tilt_deg",
        "azimuth_deg",
        "row_pitch_m",
        "string_id",
        "area_m2",
        "poa_kwh_m2",
        "energy_kwh",
        "peak_kw",
    ]

    summary_columns = [
        "scenario",
        building_id_column,
        "module_name",
        "n_modules",
        "n_strings",
        "peak_kw",
        "energy_kwh",
        "specific_yield_kwh_kwp",
        "roof_area_m2",
        "buildable_area_m2",
        "module_area_m2",
        "coverage_ratio",
    ]

    if planes.empty:
        empty = gpd.GeoDataFrame(
            {column: [] for column in columns},
            geometry=[],
            crs=planes.crs,
        )
        return empty, pd.DataFrame(columns=summary_columns)

    planes["mount"] = np.where(
        planes[tilt_column] < flat_slope_threshold_deg, "rack", "coplanar"
    )

    if allowed_azimuth_range is None:
        # No orientation is refused: even a North face is reported, and the
        # POA it earns says for itself whether it is worth equipping.
        usable = pd.Series(True, index=planes.index)
    else:
        low, high = allowed_azimuth_range
        usable = (planes["mount"] == "rack") | (
            (planes[azimuth_column] >= low) & (planes[azimuth_column] <= high)
        )
    usable &= planes.geometry.area >= min_plane_area_m2
    planes = planes[usable]

    needs_basis = (planes["mount"] == "rack").any()
    needs_total = (planes["mount"] == "coplanar").any()

    if needs_basis and poa_basis_path is None:
        raise ValueError(
            "poa_basis_path is required: the input contains flat planes, whose "
            "racks need the fourier_poa basis to be costed at a tilt that "
            "differs from the roof surface"
        )
    if needs_total and annual_energy_path is None:
        raise ValueError(
            "annual_energy_path is required: the input contains inclined "
            "planes, whose coplanar modules are costed from the mode='total' "
            "annual raster"
        )

    basis_src = rasterio.open(poa_basis_path) if needs_basis else None
    total_src = rasterio.open(annual_energy_path) if needs_total else None

    reference = basis_src or total_src

    if latitude_deg is None:
        import pyproj

        bounds = reference.bounds
        transformer = pyproj.Transformer.from_crs(
            reference.crs, "EPSG:4326", always_xy=True
        )
        _, latitude_deg = transformer.transform(
            (bounds.left + bounds.right) / 2.0,
            (bounds.bottom + bounds.top) / 2.0,
        )

    if row_shading_altitude_deg is None:
        row_shading_altitude_deg = solar_min_altitude(latitude_deg)

    config = {
        "module_efficiency": module_efficiency,
        "performance_ratio": performance_ratio,
        "flat_tilts": tuple(flat_tilts),
        "flat_azimuth_deg": float(flat_azimuth_deg),
        "row_shading_altitude_deg": float(row_shading_altitude_deg),
        "module_gap_m": module_gap_m,
        "row_gap_m": row_gap_m,
        "min_poa_ratio": 0.0 if min_poa_ratio is None else float(min_poa_ratio),
        "albedo": albedo,
        "phase_steps": phase_steps,
        "objective": objective,
    }

    dsm_transform = None
    dsm_nodata = None
    if dsm_path is not None:
        # Only its grid here: each process reads the windows under the planes
        # it plans, rather than the whole raster - 4 GB at 0.2 m on a
        # municipality - being loaded up front.
        with rasterio.open(dsm_path) as src:
            dsm_transform = src.transform
            dsm_nodata = src.nodata

    total_pixel_area = (
        abs(total_src.transform.a * total_src.transform.e) if total_src else None
    )

    all_records = []
    summary_rows = []

    global _PLAN_JOB
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    groups = list(planes.groupby(building_id_column, sort=False))

    _PLAN_JOB = {
        "groups": groups,
        "basis_path": poa_basis_path if needs_basis else None,
        "total_path": annual_energy_path if needs_total else None,
        "tilt_column": tilt_column,
        "azimuth_column": azimuth_column,
        "plane_id_column": plane_id_column,
        "building_id_column": building_id_column,
        "setback_m": setback_m,
        "obstacle_residual_m": obstacle_residual_m,
        "dsm_path": dsm_path,
        "dsm_transform": dsm_transform,
        "dsm_nodata": dsm_nodata,
        "total_pixel_area": total_pixel_area,
        "modules": modules,
        "config": config,
        "string_size": string_size,
        "min_string_size": min_string_size,
        "objective": objective,
    }

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(groups)))
    can_fork = sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods()

    # Buildings are independent, so they are spread over forked processes
    # and come back in order: the output does not depend on n_workers.
    pool = None
    try:
        if n_workers == 1 or not can_fork:
            results = map(_plan_building, range(len(groups)))
        else:
            pool = ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork"))
            results = pool.map(
                _plan_building, range(len(groups)),
                chunksize=max(1, min(8, len(groups) // (n_workers * 8)))
            )
        if progress:
            results = tqdm(results, total=len(groups), desc="Planning PV strings", unit="building")
        for building_records, building_summary in results:
            all_records.extend(building_records)
            summary_rows.extend(building_summary)
    finally:
        if pool is not None:
            pool.shutdown()
        _close_plan_sources()
        _PLAN_JOB = None

    if basis_src is not None:
        basis_src.close()
    if total_src is not None:
        total_src.close()

    if all_records:
        geometries = [record.pop("geometry") for record in all_records]
        modules_gdf = gpd.GeoDataFrame(
            all_records, geometry=geometries, crs=roof_planes.crs
        )[columns + ["geometry"]]
        building_summary = pd.DataFrame(summary_rows)[summary_columns]
    else:
        modules_gdf = gpd.GeoDataFrame(
            {column: [] for column in columns},
            geometry=[],
            crs=roof_planes.crs,
        )
        building_summary = pd.DataFrame(columns=summary_columns)

    if output_path is not None:
        write_scenario_layers(modules_gdf, output_path, layer_prefix=layer_prefix)

    if output_summary_path is not None:
        building_summary.to_csv(output_summary_path, index=False)

    return modules_gdf, building_summary



def _plan_building(k):
    """
    One building's modules and summary rows, every scenario, for
    ``plan_pv_strings_in_buildings``: ``(records, summary_rows)``.

    Everything else comes from ``_PLAN_JOB``, which forked workers inherit
    instead of having it pickled per building. The raster sources are opened
    once per process: a GDAL handle is not shared across ``fork``.
    """
    job = _PLAN_JOB
    building_id, building_planes = job["groups"][k]
    basis_src, total_src, dsm = _plan_sources(job)

    tilt_column = job["tilt_column"]
    azimuth_column = job["azimuth_column"]
    plane_id_column = job["plane_id_column"]
    building_id_column = job["building_id_column"]
    setback_m = job["setback_m"]
    obstacle_residual_m = job["obstacle_residual_m"]
    dsm_transform, dsm_nodata = job["dsm_transform"], job["dsm_nodata"]
    total_pixel_area = job["total_pixel_area"]
    modules = job["modules"]
    config = job["config"]
    string_size = job["string_size"]
    min_string_size = job["min_string_size"]
    objective = job["objective"]

    building_records = []
    building_summary = []


    contexts = []

    for _, plane in building_planes.iterrows():
        context = _plane_context(
            plane,
            tilt_column=tilt_column,
            azimuth_column=azimuth_column,
            plane_id_column=plane_id_column,
            setback_m=setback_m,
            obstacle_residual_m=obstacle_residual_m,
            dsm=dsm,
            dsm_transform=dsm_transform,
            dsm_nodata=dsm_nodata,
            basis_src=basis_src,
            total_src=total_src,
            total_pixel_area=total_pixel_area,
        )
        if context is not None:
            contexts.append(context)

    if not contexts:
        return [], []

    # scenario -> the module size that wins it, and what it places.
    winners = {}

    for module in modules:

        plane_plans = [_plan_plane(context, module, config) for context in contexts]

        # The POA filter is building-wide, so the whole roof has to be laid
        # out before a single module can be judged. Each scenario layer is
        # referenced to its own best module - a layer is one mounting
        # geometry and comparing across tilts inside it would be comparing
        # different designs - while "best", which mixes mountings freely,
        # answers to the best module the building reaches at all.
        scenario_reference = {}

        for plans in plane_plans:
            for scenario, variants in plans.items():
                top = max(variant["poa_max"] for variant in variants)
                scenario_reference[scenario] = max(
                    scenario_reference.get(scenario, 0.0), top
                )

        scenarios = sorted(scenario_reference)

        if not scenarios:
            continue

        ratio = config["min_poa_ratio"]
        best_threshold = ratio * max(scenario_reference.values())

        # scenario -> the plan each plane takes, aligned with contexts.
        selected = {
            scenario: [
                _select_variant(
                    plans.get(scenario, []),
                    ratio * scenario_reference[scenario],
                    config,
                )
                for plans in plane_plans
            ]
            for scenario in scenarios
        }

        # The per-plane optimum, which is what a designer would actually
        # build: each plane takes its own best mounting rather than being
        # forced into one municipality-wide orientation the way a scenario
        # layer is.
        selected[BEST_SCENARIO] = [
            max(
                (
                    plan
                    for plan in (
                        _select_variant(variants, best_threshold, config)
                        for variants in plans.values()
                    )
                    if plan is not None
                ),
                key=lambda item: item["score"],
                default=None,
            )
            for plans in plane_plans
        ]

        for scenario in scenarios + [BEST_SCENARIO]:

            records = []
            roof_area = 0.0
            buildable_area = 0.0
            next_string_id = 0

            for context, plan in zip(contexts, selected[scenario]):

                if plan is None:
                    continue

                # Copied because the same plan feeds both its own scenario
                # and "best", and _assign_strings writes into the records.
                strung, next_string_id = _assign_strings(
                    [dict(record) for record in plan["records"]],
                    string_size,
                    min_string_size,
                    next_string_id,
                )

                if not strung:
                    continue

                records.extend(strung)

                # Only the planes that actually took modules count towards
                # the area, so coverage_ratio stays meaningful on a scenario
                # that no pitched - or no flat - plane can host.
                roof_area += context["roof_area_m2"]
                buildable_area += context["buildable_area_m2"]

            if not records:
                continue

            energy = float(sum(record["energy_kwh"] for record in records))
            peak_kw = float(sum(record["peak_kw"] for record in records))

            if objective == "specific_yield":
                score = energy / peak_kw if peak_kw else 0.0
            else:
                score = energy

            previous = winners.get(scenario)

            if previous is None or score > previous["score"]:
                winners[scenario] = {
                    "score": score,
                    "energy_kwh": energy,
                    "peak_kw": peak_kw,
                    "records": records,
                    "module_name": module.name,
                    "roof_area_m2": roof_area,
                    "buildable_area_m2": buildable_area,
                }

    for scenario in sorted(winners, key=_scenario_sort_key):

        winner = winners[scenario]
        records = winner["records"]

        for record in records:
            record["scenario"] = scenario
            record[building_id_column] = building_id

        building_records.extend(records)

        module_area = float(sum(record["area_m2"] for record in records))
        buildable_area = winner["buildable_area_m2"]
        peak_kw = winner["peak_kw"]

        building_summary.append({
            "scenario": scenario,
            building_id_column: building_id,
            "module_name": winner["module_name"],
            "n_modules": len(records),
            "n_strings": len({record["string_id"] for record in records}),
            "peak_kw": peak_kw,
            "energy_kwh": winner["energy_kwh"],
            "specific_yield_kwh_kwp": (
                winner["energy_kwh"] / peak_kw if peak_kw else np.nan
            ),
            "roof_area_m2": winner["roof_area_m2"],
            "buildable_area_m2": buildable_area,
            "module_area_m2": module_area,
            "coverage_ratio": (
                module_area / buildable_area if buildable_area else np.nan
            ),
        })

    return building_records, building_summary


# What the workers of plan_pv_strings_in_buildings read their inputs from:
# inherited with the forked process. _PLAN_SOURCES holds the raster sources
# this process opened, with the pid that opened them.
_PLAN_JOB = None
_PLAN_SOURCES = None


def _plan_sources(job):
    """
    The (basis, total) raster sources and the DSM, read window by window,
    opened by this very process.
    """
    global _PLAN_SOURCES
    if _PLAN_SOURCES is None or _PLAN_SOURCES[0] != os.getpid():
        sources = [
            rasterio.open(path) if path is not None else None
            for path in (job["basis_path"], job["total_path"], job["dsm_path"])
        ]
        _PLAN_SOURCES = (os.getpid(), sources)
    basis_src, total_src, dsm_src = _PLAN_SOURCES[1]
    dsm = None if dsm_src is None else _WindowedBand(dsm_src, dtype=dsm_src.dtypes[0])
    return basis_src, total_src, dsm


def _close_plan_sources():
    global _PLAN_SOURCES
    if _PLAN_SOURCES is not None and _PLAN_SOURCES[0] == os.getpid():
        for src in _PLAN_SOURCES[1]:
            if src is not None:
                src.close()
    _PLAN_SOURCES = None


def _plane_context(
    plane,
    *,
    tilt_column,
    azimuth_column,
    plane_id_column,
    setback_m,
    obstacle_residual_m,
    dsm,
    dsm_transform,
    dsm_nodata,
    basis_src,
    total_src,
    total_pixel_area,
):
    """
    Everything about one plane that does not depend on the module size.

    The setback, the obstacle subtraction and the raster block read are all
    module-independent, so they are done once per plane and shared across the
    three candidate sizes rather than repeated for each.
    """
    geometry = plane.geometry

    if geometry is None or geometry.is_empty:
        return None

    buildable = geometry.buffer(-setback_m)

    if buildable.is_empty:
        return None

    if dsm is not None:
        obstacles = _obstacle_geometry(
            geometry,
            plane[tilt_column],
            plane[azimuth_column],
            dsm,
            dsm_transform,
            dsm_nodata,
            obstacle_residual_m,
        )
        if obstacles is not None:
            buildable = buildable.difference(obstacles)

    if buildable.is_empty:
        return None

    mount = plane["mount"]

    if mount == "rack":
        block = _read_window(basis_src, geometry, [1, 2, 3, 4, 5])
    else:
        block = _read_window(total_src, geometry, [1, 2])

    if block is None:
        return None

    values, transform, valid = block

    if mount == "coplanar":
        # Bands 1 and 2 are Wh per pixel; the layout works in Wh/m2 so that a
        # module's energy is a plain multiplication by its area.
        values = (values[0] + values[1])[None, :, :] / total_pixel_area

    tilt_deg = float(plane[tilt_column])

    return {
        "plane_id": plane[plane_id_column],
        "mount": mount,
        "tilt_deg": tilt_deg,
        "azimuth_deg": float(plane[azimuth_column]),
        "buildable": buildable,
        "buildable_area_m2": buildable.area / np.cos(np.radians(tilt_deg)),
        "roof_area_m2": geometry.area / np.cos(np.radians(tilt_deg)),
        "values": values,
        "transform": transform,
        "valid": valid,
    }
