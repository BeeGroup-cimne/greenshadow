# -*- coding: utf-8 -*-
# --- Build a QGIS project that animates hillshade rasters from a folder ---
# Usage (inside QGIS Python Console):
#   build_temporal_hillshade_project("/path/to/hillshades", "/path/to/project.qgz", crs_epsg=25831)
wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES/lidar/"
wd += "25023/20250621"
folders_by_group = {
    "Hillshades at 1.5m above DEM": f"{wd}/hillshades_dem1.5",
    "Hillshades over DSM":          f"{wd}/hillshades",
}
# Expected filenames include an ISO-ish Z time like   *_2025-08-31T10-00-00Z.tif
# If not present, the code tries GeoTIFF metadata keys: DATETIME / ACQUISITION_DATETIME / TIFFTAG_DATETIME.

import os
import re
import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple

from qgis.core import (
    QgsProject, QgsRasterLayer, QgsCoordinateReferenceSystem,
    QgsDateTimeRange, QgsInterval, QgsRasterBandStats, QgsSingleBandGrayRenderer,
    QgsContrastEnhancement, QgsRasterBandStats,
    Qgis
)
from qgis.PyQt.QtCore import QDateTime, QTimeZone

# -------- utilities --------

_RX_TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}[:-]?\d{2}[:-]?\d{2}Z)")

def _round_to_step(t: dt.datetime, anchor: dt.datetime, step: dt.timedelta) -> dt.datetime:
    """Round timestamp t to nearest frame edge anchored at 'anchor'."""
    delta = (t - anchor).total_seconds()
    s = step.total_seconds()
    k = int(round(delta / s))
    return anchor + dt.timedelta(seconds=k * s)

def _floor_to_step(t: dt.datetime, anchor: dt.datetime, step: dt.timedelta) -> dt.datetime:
    """Floor timestamp t to frame edge anchored at 'anchor'."""
    delta = (t - anchor).total_seconds()
    s = step.total_seconds()
    k = int(delta // s)
    return anchor + dt.timedelta(seconds=k * s)

def _qdt_utc(py_dt: dt.datetime) -> QDateTime:
    if py_dt.tzinfo is None:
        py_dt = py_dt.replace(tzinfo=dt.timezone.utc)
    msec = int(py_dt.timestamp() * 1000)
    return QDateTime.fromMSecsSinceEpoch(msec, QTimeZone.utc())

def _parse_iso_from_name(p: Path) -> Optional[dt.datetime]:
    m = _RX_TS.search(p.name)
    if not m:
        return None
    s = m.group(1)
    # normalize T10-00-00Z → T10:00:00Z
    if "T" in s and "-" in s.split("T", 1)[1]:
        d, tz = s.split("T", 1)
        s = f"{d}T{tz.replace('-', ':')}"
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _read_dt_from_metadata(path: Path) -> Optional[dt.datetime]:
    try:
        from osgeo import gdal
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if not ds:
            return None
        for dom in (None, "TIFF"):  # default domain first
            md = ds.GetMetadata(dom)
            if not md:
                continue
            for key in ("DATETIME", "ACQUISITION_DATETIME", "START_DATETIME", "TIFFTAG_DATETIME"):
                v = md.get(key)
                if not v:
                    continue
                # TIFFTAG_DATETIME = "YYYY:MM:DD HH:MM:SS" → convert to ISO
                if key == "TIFFTAG_DATETIME" and ":" in v[:10]:
                    try:
                        y, mo, d = v[:10].split(":")
                        hhmmss = v[11:19]
                        v = f"{y}-{mo}-{d}T{hhmmss}Z"
                    except Exception:
                        continue
                try:
                    return dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
                except Exception:
                    continue
        return None
    except Exception:
        return None

def _detect_granularity_seconds(times: List[dt.datetime]) -> int:
    times = sorted(times)
    diffs = [ (b - a).total_seconds() for a,b in zip(times, times[1:]) if (b - a).total_seconds() > 0 ]
    if not diffs:
        return 3600  # fallback 1h
    # robust mode of diffs with small tolerance
    import statistics
    med = statistics.median(diffs)
    tol = max(med * 0.01, 0.001)  # 1% or 1 ms
    buckets = {}
    for d in diffs:
        key = round(d / tol) * tol
        buckets[key] = buckets.get(key, 0) + 1
    step = max(buckets.items(), key=lambda kv: kv[1])[0]
    return int(round(step))


def _add_temporal_hillshade_group(
    proj: QgsProject,
    folder: str,
    group_name: str,
    *,
    crs_epsg: int,
    step_seconds: Optional[int],
    recursive: bool = True,
) -> Tuple[List[dt.datetime], int]:
    """
    Adds a group of hillshade rasters as a temporally-enabled layer stack to an EXISTING project.
    Returns (snapped_frame_starts, step_seconds_used).
    """
    root = Path(folder)
    files = list(root.rglob("*.tif") if recursive else root.glob("*.tif"))
    items: List[Tuple[Path, dt.datetime]] = []

    for p in sorted(files):
        when = _parse_iso_from_name(p) or _read_dt_from_metadata(p)
        if when is None:
            continue
        when = when.replace(tzinfo=dt.timezone.utc) if when.tzinfo is None else when.astimezone(dt.timezone.utc)
        items.append((p, when))
    if not items:
        raise RuntimeError(f"No GeoTIFFs with parsable timestamps in {folder}")

    # detect or use provided step
    times = sorted(t for _, t in items)
    if step_seconds is None:
        step_seconds = _detect_granularity_seconds(times)
    step_td = dt.timedelta(seconds=step_seconds)

    first_day_utc = times[0].date()  # times already UTC
    anchor = dt.datetime.combine(first_day_utc, dt.time(0, 0, tzinfo=dt.timezone.utc))

    snapped: dict[dt.datetime, Tuple[Path, dt.datetime]] = {}
    for p, when in items:
        t_snap = _round_to_step(when, anchor, step_td)
        prev = snapped.get(t_snap)
        if prev is None or abs((when - t_snap).total_seconds()) < abs((prev[1] - t_snap).total_seconds()):
            snapped[t_snap] = (p, when)

    items_snapped = sorted([(p, t_snap) for t_snap, (p, _orig) in snapped.items()], key=lambda x: x[1])
    snapped_starts = [t for _, t in items_snapped]

    # add group and layers (no proj.clear() here!)
    root_lt = proj.layerTreeRoot()
    group = root_lt.addGroup(group_name)

    for p, t_start in items_snapped:
        lyr = QgsRasterLayer(str(p), p.stem)
        if not lyr.isValid():
            continue
        provider = lyr.dataProvider()
        try:
            if provider.sourceHasNoDataValue(1) and provider.sourceNoDataValue(1) == 0:
                provider.setNoDataValue(1, -9999)
        except Exception:
            pass

        renderer = QgsSingleBandGrayRenderer(provider, 1)
        ce = QgsContrastEnhancement(provider.dataType(1))
        ce.setMinimumValue(0.0)
        ce.setMaximumValue(255.0)
        ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum, True)
        renderer.setContrastEnhancement(ce)
        lyr.setRenderer(renderer)

        proj.addMapLayer(lyr, False)
        group.addLayer(lyr)

        # temporal range per layer: [start, start + step)
        tp = lyr.temporalProperties()
        tp.setIsActive(True)
        tp.setMode(Qgis.RasterTemporalMode.FixedTemporalRange)
        tp.setFixedTemporalRange(QgsDateTimeRange(_qdt_utc(t_start), _qdt_utc(t_start + step_td), True, False))

    return snapped_starts, step_seconds


def build_temporal_hillshade_project_multi(
    folders_by_group: dict,
    project_path: str,
    *,
    crs_epsg: int = 25831,
    step_seconds: Optional[int] = None,   # force a common step (recommended). If None, taken from the first group.
    recursive: bool = True,
    fps: float = 2.0,
) -> str:
    """
    Build ONE project containing multiple hillshade groups that animate together.
    folders_by_group: { "Group name": "/path/to/folder", ... }
    """
    proj = QgsProject.instance()
    proj.clear()  # only ONCE
    proj.setCrs(QgsCoordinateReferenceSystem.fromEpsgId(crs_epsg))

    # If no common step given, take it from the first group we process
    common_step = step_seconds
    all_frame_starts: List[dt.datetime] = []

    for group_name, folder in folders_by_group.items():
        snapped_starts, used_step = _add_temporal_hillshade_group(
            proj, folder, group_name,
            crs_epsg=crs_epsg,
            step_seconds=common_step,
            recursive=recursive,
        )
        if common_step is None:
            common_step = used_step
        elif used_step != common_step:
            raise ValueError(f"Time step mismatch for '{group_name}': {used_step}s != {common_step}s. "
                             "Provide a common step_seconds to resample/snap consistently.")
        all_frame_starts.extend(snapped_starts)

    if not all_frame_starts:
        raise RuntimeError("No temporal frames were added.")

    step_td = dt.timedelta(seconds=common_step)
    # Build a global temporal range that exactly covers all frames
    all_frame_starts = sorted(set(all_frame_starts))
    global_start = all_frame_starts[0]
    global_end = all_frame_starts[-1] + step_td
    # small pad and inclusive end to avoid precision issues
    global_end_padded = global_end + dt.timedelta(milliseconds=1)

    ts = proj.timeSettings()
    ts.setTemporalRange(QgsDateTimeRange(_qdt_utc(global_start), _qdt_utc(global_end_padded), True, True))
    ts.setTimeStep(float(common_step))
    ts.setTimeStepUnit(Qgis.TemporalUnit.TemporalSeconds)
    ts.setFramesPerSecond(float(fps))

    try:
        tc = iface.mapCanvas().temporalController()
        tc.setTemporalRange(QgsDateTimeRange(_qdt_utc(global_start), _qdt_utc(global_end_padded), True, True))
        tc.setFrameDuration(QgsInterval(common_step))
        tc.setCurrentTimePosition(_qdt_utc(global_start))
    except Exception:
        pass

    os.makedirs(os.path.dirname(os.path.abspath(project_path)), exist_ok=True)
    if not proj.write(project_path):
        raise RuntimeError(f"Failed to write project to {project_path}")
    return project_path
    
def build_temporal_hillshade_project(
    folder: str,
    project_path: str,
    group_name: str = "hillshade",
    *,
    crs_epsg: int = 25831,
    step_seconds: Optional[int] = None,    # if None, auto-detect
    recursive: bool = True,
) -> Tuple[List[Tuple[Path, dt.datetime]], str]:
    """
    Scan folder for *.tif hillshades, derive timestamps, set per-layer temporal ranges,
    and save a QGIS project ready for the Temporal Controller.

    Returns ([(path, datetime_utc), ...], project_path)
    """
    root = Path(folder)
    files = list(root.rglob("*.tif") if recursive else root.glob("*.tif"))
    items: List[Tuple[Path, dt.datetime]] = []

    # 1) collect timestamps
    for p in sorted(files):
        when = _parse_iso_from_name(p)
        if when is None:
            when = _read_dt_from_metadata(p)
        if when is None:
            continue
        # normalize to UTC
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        else:
            when = when.astimezone(dt.timezone.utc)
        items.append((p, when))

    if not items:
        raise RuntimeError("No GeoTIFFs with parsable timestamps found (filename or metadata).")

    # 2) detect granularity if not provided
    times = sorted([t for _, t in items])
    if step_seconds is None:
        step_seconds = _detect_granularity_seconds(times)
    step_td = dt.timedelta(seconds=step_seconds)

    # Choose a stable anchor: midnight UTC of the first day (avoids DST/local drift)
    first_day_utc = times[0].astimezone(dt.timezone.utc).date()
    anchor = dt.datetime.combine(first_day_utc, dt.time(0, 0, tzinfo=dt.timezone.utc))

    # SNAP each timestamp to the frame grid (nearest edge) and deduplicate
    # If two rasters snap to the same frame start, keep the one whose original time is closer.
    snapped: dict[dt.datetime, Tuple[Path, dt.datetime]] = {}
    for p, when in items:
        t_snap = _round_to_step(when, anchor, step_td)
        prev = snapped.get(t_snap)
        if prev is None:
            snapped[t_snap] = (p, when)
        else:
            # keep the closest to the snapped edge
            if abs((when - t_snap).total_seconds()) < abs((prev[1] - t_snap).total_seconds()):
                snapped[t_snap] = (p, when)

    # Rebuild items from snapped times, sorted by snapped key
    items_snapped = sorted([(p, t_snap) for t_snap, (p, _orig) in snapped.items()], key=lambda x: x[1])

    # 3) set up project (same as before)
    proj = QgsProject.instance()
    proj.clear()
    proj.setCrs(QgsCoordinateReferenceSystem.fromEpsgId(crs_epsg))
    root_lt = proj.layerTreeRoot()
    group = root_lt.addGroup(group_name)

    # 4) add layers with temporal properties & renderer
    for p, t_start in items_snapped:
        lyr = QgsRasterLayer(str(p), p.stem)
        if not lyr.isValid():
            continue

        provider = lyr.dataProvider()
        try:
            if provider.sourceHasNoDataValue(1) and provider.sourceNoDataValue(1) == 0:
                provider.setNoDataValue(1, -9999)
        except Exception:
            pass

        renderer = QgsSingleBandGrayRenderer(provider, 1)
        ce = QgsContrastEnhancement(provider.dataType(1))
        ce.setMinimumValue(0.0)
        ce.setMaximumValue(255.0)
        ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum, True)
        renderer.setContrastEnhancement(ce)
        lyr.setRenderer(renderer)
        proj.addMapLayer(lyr, False)
        group.addLayer(lyr)

        # HALF-OPEN range: [t_start, t_start + step)
        tp = lyr.temporalProperties()
        tp.setIsActive(True)
        tp.setMode(Qgis.RasterTemporalMode.FixedTemporalRange)
        start_q = _qdt_utc(t_start)
        end_q   = _qdt_utc(t_start + step_td)
        tp.setFixedTemporalRange(QgsDateTimeRange(start_q, end_q, True, False))

    # 5) project temporal range aligned to grid (covers all frames exactly)
    snapped_starts = [t for _, t in items_snapped]
    start_all = _floor_to_step(min(snapped_starts), anchor, step_td)

    # use frame count to build the end; last frame is [last_start, last_start+step)
    frame_count = len(snapped_starts)
    end_all = start_all + frame_count * step_td

    # tiny pad + inclusive end to avoid precision/off-by-one drops
    end_all_padded = end_all + dt.timedelta(milliseconds=1)

    ts = proj.timeSettings()
    ts.setTemporalRange(QgsDateTimeRange(_qdt_utc(start_all), _qdt_utc(end_all_padded), 
        includeBeginning=True, includeEnd=True))
    ts.setTimeStep(float(step_seconds))
    ts.setTimeStepUnit(Qgis.TemporalUnit.TemporalSeconds)
    ts.setFramesPerSecond(2.0)

    # If running inside an open QGIS session, also update the UI controller now:
    try:
        tc = iface.mapCanvas().temporalController()
        tc.setTemporalRange(QgsDateTimeRange(_qdt_utc(start_all), _qdt_utc(end_all_padded), True, True))
        # older APIs use setFrameDuration(QgsInterval(...)), newer use:
        tc.setFrameDuration(QgsInterval(step_seconds))
        tc.setCurrentTimePosition(_qdt_utc(start_all))
    except Exception:
        pass

    # 6) save project
    os.makedirs(os.path.dirname(os.path.abspath(project_path)), exist_ok=True)
    if not proj.write(project_path):
        raise RuntimeError(f"Failed to write project to {project_path}")

    return items, project_path
    
project = build_temporal_hillshade_project_multi(
    folders_by_group,
    project_path=f"{wd}/Hillshades_both.qgz",
    crs_epsg=25831,
    step_seconds=None,   # or e.g. 3600 to force hourly frames across both groups
    fps=2.0,
)
print("Project written:", project)