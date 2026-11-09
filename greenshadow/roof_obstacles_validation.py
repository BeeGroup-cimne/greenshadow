# -*- coding: utf-8 -*-
"""
greenshadow.roof_obstacles_validation

How well the roof obstacles are found, against obstacles labelled by hand,
and what they change in the usable area and the PV yield; per-roof debug
figures.

    sample = sample_roofs_for_labelling(planes, per_type=10)
    write_labelling_template("labels.gpkg", sample)       # label in QGIS
    truth = gpd.read_file("labels.gpkg", layer="obstacles")
    table = validate_obstacles(planes, detections, truth, sample)
    write_debug_pngs("debug/", sample.part_id, planes, ortho=ortho,
                     coreg_log=log, detections=detections, usable=usable)

**Labels.** A label is a polygon of an obstacle on a roof of the sample
(``part_id``, ``class``), drawn where the LiDAR has the roof - on the
orthophoto registered with ``read_orthophoto(..., shift=(dx, dy))``, as the
debug figures show it. Every obstacle of a sampled roof is labelled, so that
what is not labelled is not an obstacle.

**Metrics** (``obstacle_metrics``), restricted to the sampled roofs:
object precision and recall (one-to-one matches at
``ObstacleConfig.validation_match_iou`` or within
``validation_match_distance_m``; and at the IoU alone), the same at any
overlap ("hit"),
and area precision, recall and IoU of the union of the obstacles.
"""

import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import unary_union

try:
    from .roof_obstacles import ObstacleConfig, _roof_outline, fuse_obstacles
except ImportError:
    from greenshadow.roof_obstacles import ObstacleConfig, _roof_outline, fuse_obstacles


# The detectors a validation compares: name -> sources fused.
DEFAULT_VARIANTS = {
    "lidar": ("lidar",),
    "svf": ("svf",),
    "sam3": ("sam3",),
    "shadow": ("shadow",),
    "lidar+svf": ("lidar", "svf"),
    "fused": ("lidar", "svf", "sam3", "shadow"),
}

LABEL_CLASSES = ("chimney", "vent", "skylight", "roof window", "solar panel",
                 "solar water heater", "air conditioning unit", "antenna", "dormer",
                 "parapet", "light well", "water tank", "other")


# ============================================================
# SAMPLE AND LABELS
# ============================================================

def roof_types(planes, *, flat_tilt_deg=5.0, large_m2=800.0, terraced_radius_m=12.0,
               terraced_neighbours=2, close_m=0.25):
    """
    One row per roof (``part_id``, outline): ``roof_area_m2`` (plan),
    ``mean_tilt_deg`` (area weighted), ``n_planes`` and ``roof_type`` -
    ``large`` (industrial, agricultural, public), ``flat`` (terraces),
    ``terraced`` (at least ``terraced_neighbours`` other roofs within
    ``terraced_radius_m``: a row of town houses) or ``detached``.
    """
    rows = []
    for part_id, frame in planes.groupby("part_id", sort=False):
        outline = _roof_outline(frame.geometry.values, close_m)
        weights = frame.geometry.area.to_numpy()
        tilt = (np.average(frame.tilt_deg, weights=weights)
                if "tilt_deg" in frame and weights.sum() > 0 else np.nan)
        rows.append({"part_id": part_id, "roof_area_m2": outline.area,
                     "mean_tilt_deg": tilt, "n_planes": len(frame), "geometry": outline})
    roofs = gpd.GeoDataFrame(rows, geometry="geometry", crs=planes.crs)
    tree = shapely.STRtree(roofs.geometry.values)
    near = tree.query(roofs.geometry.values, predicate="dwithin", distance=terraced_radius_m)
    neighbours = np.bincount(near[0], minlength=len(roofs)) - 1
    roofs["n_neighbours"] = neighbours
    roofs["roof_type"] = np.select(
        [roofs.roof_area_m2 >= large_m2, roofs.mean_tilt_deg < flat_tilt_deg,
         neighbours >= terraced_neighbours],
        ["large", "flat", "terraced"], "detached")
    return roofs


def sample_roofs_for_labelling(planes, per_type=10, *, seed=0, min_area_m2=20.0,
                               max_area_m2=2500.0, exclude=None, **types):
    """
    A stratified random sample of roofs to label: ``per_type`` of each
    ``roof_types`` class (fewer when a class has fewer), between
    ``min_area_m2`` and ``max_area_m2`` of plan area, not in ``exclude``
    (part ids: bridges, false roofs). Same seed, same sample.
    """
    roofs = roof_types(planes, **types)
    roofs = roofs[roofs.roof_area_m2.between(min_area_m2, max_area_m2)]
    if exclude is not None:
        roofs = roofs[~roofs.part_id.isin(list(exclude))]
    rng = np.random.default_rng(seed)
    picked = []
    for kind in ("terraced", "detached", "flat", "large"):
        pool = roofs[roofs.roof_type == kind]
        take = min(per_type, len(pool))
        if take:
            picked.append(pool.iloc[np.sort(rng.choice(len(pool), take, replace=False))])
    if not picked:
        return roofs.iloc[:0]
    return gpd.GeoDataFrame(pd.concat(picked, ignore_index=True), geometry="geometry",
                            crs=planes.crs)


def write_labelling_template(path, sample, truth=None):
    """
    A GeoPackage to label in: layer ``sample`` (the roofs) and layer
    ``obstacles`` (``part_id``, ``class``, ``note``; ``truth`` when given,
    else empty).
    """
    if os.path.exists(path):
        os.remove(path)
    sample.to_file(path, layer="sample", driver="GPKG")
    if truth is None or not len(truth):
        truth = gpd.GeoDataFrame({"part_id": pd.Series([], dtype="int64"),
                                  "class": pd.Series([], dtype="str"),
                                  "note": pd.Series([], dtype="str")},
                                 geometry=gpd.GeoSeries([], crs=sample.crs), crs=sample.crs)
    truth.to_file(path, layer="obstacles", driver="GPKG")


# ============================================================
# METRICS
# ============================================================

def obstacle_metrics(predicted, truth, sample, config: Optional[ObstacleConfig] = None):
    """
    Detected against labelled obstacles on the sampled roofs.

    Args:
        predicted, truth: obstacle polygons (any columns).
        sample: the labelled roofs (their outlines); both sets are clipped
            to them.

    Returns:
        dict: ``n_predicted``, ``n_truth``; ``precision``, ``recall``,
        ``f1`` (one-to-one, at ``validation_match_iou`` or within
        ``validation_match_distance_m``); ``iou_precision``, ``iou_recall``
        (one-to-one at the IoU alone); ``hit_precision``
        (detections touching a label), ``hit_recall`` (labels touched by a
        detection); ``area_precision``, ``area_recall``, ``area_iou`` (m2
        of the unions); ``mean_matched_iou``.
    """
    cfg = config or ObstacleConfig()
    region = unary_union(list(sample.geometry.values))
    pred = _clip(predicted, region)
    true = _clip(truth, region)

    # Candidate pairs: overlapping, or within validation_match_distance_m.
    pairs = []
    if pred and true:
        tree = shapely.STRtree(true)
        for i, p in enumerate(pred):
            for j in tree.query(p, predicate="dwithin", distance=cfg.validation_match_distance_m):
                inter = p.intersection(true[j]).area
                union = p.area + true[j].area - inter
                pairs.append((inter / union if union > 0 else 0.0, -p.distance(true[j]), i, int(j)))

    def one_to_one(eligible):
        # Best pairs first (IoU, then nearness); each object matched once.
        used_p, used_t, ious = set(), set(), []
        for iou, near, i, j in sorted(pairs, reverse=True):
            if eligible(iou, -near) and i not in used_p and j not in used_t:
                used_p.add(i)
                used_t.add(j)
                ious.append(iou)
        return ious

    matched = one_to_one(lambda iou, d: iou >= cfg.validation_match_iou
                         or d <= cfg.validation_match_distance_m)
    strict = one_to_one(lambda iou, d: iou >= cfg.validation_match_iou)
    touched_p = {i for iou, _, i, _ in pairs if iou > 0}
    touched_t = {j for iou, _, _, j in pairs if iou > 0}

    def ratio(a, b):
        return a / b if b else np.nan

    precision, recall = ratio(len(matched), len(pred)), ratio(len(matched), len(true))
    pred_area = unary_union(pred) if pred else shapely.Polygon()
    true_area = unary_union(true) if true else shapely.Polygon()
    inter = pred_area.intersection(true_area).area
    return {
        "n_predicted": len(pred), "n_truth": len(true),
        "precision": precision, "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)
               if precision and recall and np.isfinite(precision + recall) else np.nan),
        "iou_precision": ratio(len(strict), len(pred)),
        "iou_recall": ratio(len(strict), len(true)),
        "hit_precision": ratio(len(touched_p), len(pred)),
        "hit_recall": ratio(len(touched_t), len(true)),
        "area_precision": ratio(inter, pred_area.area),
        "area_recall": ratio(inter, true_area.area),
        "area_iou": ratio(inter, pred_area.union(true_area).area),
        "mean_matched_iou": float(np.mean(strict)) if strict else np.nan,
        "predicted_m2": pred_area.area, "truth_m2": true_area.area,
    }


def _clip(frame, region):
    if frame is None or not len(frame):
        return []
    out = []
    for g in frame.geometry.values:
        if g is None or g.is_empty or not g.intersects(region):
            continue
        piece = shapely.make_valid(g.intersection(region))
        piece = unary_union([p for p in getattr(piece, "geoms", [piece])
                             if p.geom_type in ("Polygon", "MultiPolygon")])
        if not piece.is_empty and piece.area > 0:
            out.append(piece)
    return out


def validate_obstacles(planes, detections, truth, sample, config: Optional[ObstacleConfig] = None,
                       *, variants: Optional[Dict[str, Sequence[str]]] = None, planes_qc=None,
                       modules=None, by_type=True):
    """
    Each detector, alone and fused, against the labels, and the usable area
    each leaves on the sampled roofs.

    Args:
        planes: all roof planes.
        detections: the ``detections`` layer (every source, before fusion).
        truth: labelled obstacles (``part_id``, polygons).
        sample: the labelled roofs (``sample_roofs_for_labelling``; a
            ``roof_type`` column adds per-type rows).
        variants: name -> sources fused (``DEFAULT_VARIANTS``); variants
            whose sources are all absent are left out.

    Returns:
        DataFrame, one row per variant and roof type ("all" first), the
        ``obstacle_metrics`` columns plus ``usable_area_m2`` and
        ``usable_vs_labels`` (its ratio to the usable area the labels
        leave); variant "labels" is the labels themselves.
    """
    cfg = config or ObstacleConfig()
    variants = DEFAULT_VARIANTS if variants is None else variants
    ids = set(sample.part_id)
    sample_planes = planes[planes.part_id.isin(ids)]
    present = set(detections.source) if len(detections) else set()
    truth = truth[truth.part_id.isin(ids)] if "part_id" in truth else truth

    labels_as_detections = truth.assign(source="labels", max_height_above_plane=np.nan,
                                        n_points=0)
    fused = {"labels": fuse_obstacles(sample_planes, [labels_as_detections], cfg,
                                      planes_qc=planes_qc, modules=modules)}
    for name, sources in variants.items():
        if not present.intersection(sources):
            continue
        chosen = detections[detections.source.isin(sources) & detections.part_id.isin(ids)]
        fused[name] = fuse_obstacles(sample_planes, [chosen], cfg, planes_qc=planes_qc,
                                     modules=modules)

    groups = [("all", sample)]
    if by_type and "roof_type" in sample:
        groups += [(kind, frame) for kind, frame in sample.groupby("roof_type")]
    rows = []
    for kind, roofs in groups:
        part_ids = set(roofs.part_id)
        label_usable = fused["labels"][1]
        label_m2 = label_usable[label_usable.part_id.isin(part_ids)].usable_area_m2.sum()
        for name, (obstacles, usable) in fused.items():
            metrics = obstacle_metrics(obstacles[obstacles.part_id.isin(part_ids)],
                                       truth[truth.part_id.isin(part_ids)], roofs, cfg)
            usable_m2 = usable[usable.part_id.isin(part_ids)].usable_area_m2.sum()
            rows.append({"variant": name, "roof_type": kind, "n_roofs": len(roofs), **metrics,
                         "usable_area_m2": usable_m2,
                         "usable_vs_labels": usable_m2 / label_m2 if label_m2 else np.nan})
    return pd.DataFrame(rows)


def pv_yield_by_variant(planes, usable_by_variant: Dict[str, gpd.GeoDataFrame], *,
                        scenario="best", **planner):
    """
    The PV layout and its annual yield on each variant's usable area
    (``plan_pv_strings_in_buildings`` with ``buildable_areas``; ``planner``:
    its other arguments - irradiance rasters, modules, setbacks).

    Returns:
        DataFrame per variant: ``n_modules``, ``peak_kw``, ``energy_kwh``,
        ``buildable_area_m2`` of the ``scenario`` layout.
    """
    try:
        from .dimensionate import plan_pv_strings_in_buildings
    except ImportError:
        from greenshadow.dimensionate import plan_pv_strings_in_buildings
    rows = []
    for name, usable in usable_by_variant.items():
        _, summary = plan_pv_strings_in_buildings(planes, buildable_areas=usable, **planner)
        best = summary[summary.scenario == scenario]
        rows.append({"variant": name, "n_modules": int(best.n_modules.sum()),
                     "peak_kw": best.peak_kw.sum(), "energy_kwh": best.energy_kwh.sum(),
                     "buildable_area_m2": best.buildable_area_m2.sum()})
    return pd.DataFrame(rows)


# ============================================================
# DEBUG FIGURES
# ============================================================

SOURCE_COLOURS = {"lidar": "#ff3b30", "svf": "#ff9500", "sam3": "#00c7ff", "terrace": "#d900ff",
                  "shadow": "#ffcc00", "labels": "#ffffff"}


def plot_roof_obstacles(part_id, planes, *, ortho=None, coreg_log=None, detections=None,
                        obstacles=None, usable=None, truth=None,
                        config: Optional[ObstacleConfig] = None, path=None, dpi=110):
    """
    One roof in three panels: the orthophoto as published with the LiDAR
    planes (their offset is what the registration measures); the registered
    orthophoto with every detection by source (LiDAR red, SVF orange, SAM 3
    cyan with its class, shadows yellow, terraces magenta) and the labels
    (white, dashed);
    the usable area (green) and the fused obstacles. Without an orthophoto the panels have
    no background. Saved to ``path`` when given; returns the figure.
    """
    # A bare Figure, not pyplot: no GUI backend, nothing left open in the
    # pyplot registry (or for forked workers to tear down).
    from matplotlib.figure import Figure
    try:
        from .orthophoto import read_orthophoto
    except ImportError:
        from greenshadow.orthophoto import read_orthophoto

    cfg = config or ObstacleConfig()
    crs = planes.crs
    part = planes[planes.part_id == part_id]
    outline = _roof_outline(part.geometry.values, cfg.outline_close_m)
    minx, miny, maxx, maxy = outline.buffer(cfg.debug_pad_m).bounds
    shift, title = (0.0, 0.0), f"roof {part_id}"
    if coreg_log is not None and len(coreg_log):
        row = coreg_log[coreg_log.part_id == part_id]
        if len(row):
            row = row.iloc[0]
            if bool(row.use_imagery):
                shift = (float(row.dx), float(row.dy))
            title += (f" | shift ({row.dx:+.2f}, {row.dy:+.2f}) m, score {row.score:.2f}, "
                      f"{row.status}")

    def background(ax, s):
        if ortho is None:
            return
        image, transform = read_orthophoto(ortho, (minx, miny, maxx, maxy), crs=crs, shift=s)
        rgb = np.clip(np.nan_to_num(image[:3].transpose(1, 2, 0)) / 255, 0, 1)
        h, w = rgb.shape[:2]
        ax.imshow(rgb, extent=(transform.c, transform.c + w * transform.a,
                               transform.f + h * transform.e, transform.f))

    def draw(ax, geometries, colour, *, fill=False, alpha=0.35, ls="-", lw=1.4):
        for g in geometries:
            for p in getattr(g, "geoms", [g]):
                if p.is_empty or p.geom_type != "Polygon":
                    continue
                x, y = p.exterior.xy
                if fill:
                    ax.fill(x, y, color=colour, alpha=alpha, lw=0)
                ax.plot(x, y, color=colour, lw=lw, ls=ls)

    def of_part(frame):
        # None for anything missing - a layer never written comes back empty,
        # without its columns.
        if frame is None or not len(frame) or "part_id" not in frame:
            return None
        frame = frame[frame.part_id == part_id]
        return frame if len(frame) else None

    fig = Figure(figsize=(15, 5.4))
    axes = fig.subplots(1, 3)
    background(axes[0], (0.0, 0.0))
    draw(axes[0], part.geometry.values, "#ff2d55", lw=1.0)
    axes[0].set_title("orthophoto as published, LiDAR planes", fontsize=9)

    background(axes[1], shift)
    draw(axes[1], part.geometry.values, "#ffffff", lw=0.6)
    found = of_part(detections)
    if found is not None and "source" in found:
        for source, frame in found.groupby("source"):
            draw(axes[1], frame.geometry.values, SOURCE_COLOURS.get(source, "#ffcc00"), fill=True)
            if source == "sam3" and "class" in frame:
                for g, name, score in zip(frame.geometry, frame["class"], frame["score"]):
                    c = g.representative_point()
                    axes[1].annotate(f"{name} {score:.2f}", (c.x, c.y), color="#00c7ff",
                                     fontsize=6, ha="center", va="bottom")
    labelled = of_part(truth)
    if labelled is not None:
        draw(axes[1], labelled.geometry.values, SOURCE_COLOURS["labels"], ls="--", lw=1.2)
    axes[1].set_title("registered: LiDAR (red), SVF (orange), SAM 3 (cyan),\n"
                      "shadows (yellow), terraces (magenta), labels (dashed)", fontsize=9)

    background(axes[2], shift)
    left = of_part(usable)
    if left is not None:
        draw(axes[2], left.geometry.values, "#34c759", fill=True, alpha=0.45, lw=0.8)
    fused = of_part(obstacles)
    if fused is not None:
        draw(axes[2], fused.geometry.values, "#ffcc00", lw=1.0)
    m2 = left.usable_area_m2.sum() if left is not None else 0.0
    axes[2].set_title(f"usable area {m2:.0f} m2 (green), obstacles (yellow)", fontsize=9)

    for ax in axes:
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(useOffset=False, style="plain")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return fig


def write_debug_pngs(out_dir, part_ids, planes, **layers):
    """``plot_roof_obstacles`` for each of ``part_ids`` into
    ``out_dir/roof_<part_id>.png``; ``layers``: its keyword arguments.
    Returns the paths."""
    paths = []
    for part_id in part_ids:
        path = os.path.join(out_dir, f"roof_{part_id}.png")
        plot_roof_obstacles(part_id, planes, path=path, **layers)
        paths.append(path)
    return paths


def debug_part_ids(planes, config: Optional[ObstacleConfig] = None):
    """The roofs the pipeline draws: ``debug_part_ids`` plus a fixed random
    sample of ``debug_pngs``."""
    cfg = config or ObstacleConfig()
    chosen = list(cfg.debug_part_ids or [])
    if cfg.debug_pngs:
        pool = [p for p in pd.unique(planes.part_id) if p not in chosen]
        rng = np.random.default_rng(0)
        take = min(cfg.debug_pngs, len(pool))
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), take, replace=False))]
    return chosen
