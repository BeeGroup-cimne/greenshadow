#!/usr/bin/env python3
"""
Score detected crop rows against hand-drawn ones, parcel by parcel.

    python eval_crop_rows.py --truth rows.gpkg --parcels sigpac.gpkg \
        --detected crop_rows.gpkg

A hand-drawn file is normally a *sample* of the real rows rather than an
exhaustive tracing - the gaps between consecutive drawn rows run anywhere from
a few centimetres to a dozen metres - so "how many rows are there" and "what is
the spacing" cannot be read off it, and a detected row with no drawn
counterpart is not a false positive. What a sample does support is the two
things that matter most:

- **cover**: of all the length that was drawn, how much has a detected row
  running along it, within ``--tolerance``. Sampling-robust, indifferent to
  fragmentation, and it penalises both a missing row and one that stops short.
- **offset** and **azimuth**: how far, and at what angle, the nearest detected
  row sits from each drawn one.

Spacing is compared against the drawn gaps only where the drawing is dense and
regular enough to carry it, which the ``regular`` column reports; elsewhere it
is shown for reference and should not be read as an error. The detected spacing
is the value the detector itself reports, length-weighted - pooling the row
offsets would be wrong on any parcel holding more than one block, since two
blocks sit on two lattices in two places and their pooled gaps are not a
spacing at all.
"""
import argparse

import numpy as np, pandas as pd, geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point

STEP = 2.0            # metres between samples along a drawn row


def bearing(geom):
    c = np.asarray(geom.coords)
    d = c[-1] - c[0]
    return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)


def circular_mean_180(angles, weights=None):
    a = np.radians(np.asarray(angles, float) * 2.0)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    return float((np.degrees(np.arctan2((w * np.sin(a)).sum(),
                                        (w * np.cos(a)).sum())) / 2.0) % 180.0)


def angle_error(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def offsets_of(geoms, azimuth_deg):
    theta = np.radians(-azimuth_deg)
    mids = np.array([np.asarray(g.coords).mean(axis=0) for g in geoms])
    return mids[:, 0] * np.cos(theta) + mids[:, 1] * np.sin(theta)


def clip(geoms, polygon):
    out = []
    for geom in geoms:
        if not geom.intersects(polygon):
            continue
        piece = geom.intersection(polygon)
        if piece.geom_type == "LineString" and not piece.is_empty:
            out.append(piece)
        elif piece.geom_type == "MultiLineString":
            out.extend(p for p in piece.geoms if not p.is_empty)
    return out


def drawn_spacing(geoms, azimuth_deg):
    """
    The drawn spacing, and whether the drawing is regular enough to mean it.

    A complete tracing has gaps clustered about one value. A sample has gaps
    all over, so the spread of the gaps is the test.
    """
    if len(geoms) < 6:
        return np.nan, False
    gaps = np.diff(np.sort(offsets_of(geoms, azimuth_deg)))
    gaps = gaps[gaps > 0.2]
    if gaps.size < 5:
        return np.nan, False
    median = float(np.median(gaps))
    inside = np.mean((gaps > 0.6 * median) & (gaps < 1.6 * median))
    return median, bool(inside >= 0.85)


def load_truth(TRUTH, PARCELS, truth_layer, parcel_layer):
    truth = gpd.read_file(TRUTH, layer=truth_layer).to_crs(25831)
    parcels = gpd.read_file(PARCELS, layer=parcel_layer).to_crs(25831).reset_index(drop=True)
    parcels["pid"] = parcels.index

    pieces = gpd.overlay(
        truth.assign(rid=range(len(truth))),
        parcels[["pid", "geometry"]], how="identity", keep_geom_type=True)
    pieces["seg_len"] = pieces.length
    best = pieces.sort_values("seg_len").groupby("rid").tail(1)

    truth = truth.assign(rid=range(len(truth))).merge(best[["rid", "pid"]], on="rid", how="left")
    return gpd.GeoDataFrame(truth, geometry="geometry", crs=25831), parcels


def reported_spacing(detected, polygon):
    """
    The spacing the detector itself claims, length-weighted over the parcel.

    Pooling the row offsets instead would be wrong wherever a parcel holds more
    than one block: rows of two blocks sit on two different lattices in two
    different places, and their pooled gaps are not a spacing at all.
    """
    inside = detected[detected.intersects(polygon)]
    if not len(inside) or "row_spacing_m" not in inside:
        return np.nan
    order = inside.sort_values("row_spacing_m")
    weight = order.geometry.length.to_numpy()
    if weight.sum() <= 0:
        return np.nan
    share = np.cumsum(weight) / weight.sum()
    return float(order.row_spacing_m.to_numpy()[int(np.searchsorted(share, 0.5))])


def score(detected, truth, parcels, keep_pids, tolerance=1.0):
    rows = []
    for pid in keep_pids:
        parcel = parcels.loc[parcels.pid == pid].iloc[0]
        want = clip(list(truth.loc[truth.pid == pid].geometry), parcel.geometry)
        got = clip(list(detected.geometry.values), parcel.geometry)

        az_true = circular_mean_180([bearing(g) for g in want], [g.length for g in want])
        sp_true, regular = drawn_spacing(want, az_true)

        record = dict(pid=pid, uso=parcel.uso_sigpac,
                      ha=round(parcel.geometry.area / 1e4, 2),
                      n_drawn=len(want), n_det=len(got),
                      sp_drawn=round(sp_true, 2), regular=regular)

        if not got:
            record.update(cover=0.0, off_p50=np.nan, off_p90=np.nan,
                          az_p50=np.nan, sp_det=np.nan, sp_ratio=np.nan)
            rows.append(record)
            continue

        union = unary_union(got)

        covered, distances, angles = [], [], []
        for line in want:
            n = max(2, int(np.ceil(line.length / STEP)))
            for point in (line.interpolate(i / n, normalized=True) for i in range(n + 1)):
                distances.append(union.distance(point))
            covered.append(np.mean([union.distance(
                line.interpolate(i / n, normalized=True)) <= tolerance
                for i in range(n + 1)]) * line.length)
            # bearing of the detected row nearest this drawn row's midpoint
            mid = line.interpolate(0.5, normalized=True)
            nearest = min(got, key=lambda g: g.distance(mid))
            angles.append(angle_error(bearing(line), bearing(nearest)))

        total = sum(line.length for line in want)
        sp_det = reported_spacing(detected, parcel.geometry)

        record.update(
            cover=round(sum(covered) / total, 2),
            off_p50=round(float(np.percentile(distances, 50)), 2),
            off_p90=round(float(np.percentile(distances, 90)), 2),
            az_p50=round(float(np.nanmedian(angles)), 1),
            sp_det=round(sp_det, 2),
            sp_ratio=round(sp_det / sp_true, 2)
                if np.isfinite(sp_true) and np.isfinite(sp_det) else np.nan,
        )
        rows.append(record)
    return pd.DataFrame(rows)


def summarise(table, label=""):
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))
    regular = table[table.regular]
    print(f"\n{label}  parcels={len(table)}  "
          f"cover mean={table.cover.mean():.2f}  "
          f"offset p50 median={table.off_p50.median():.2f} m  "
          f"|az| median={table.az_p50.median():.1f} deg  "
          f"blank={int((table.cover == 0).sum())}")
    if len(regular):
        print(f"  on the {len(regular)} regularly-drawn parcels: "
              f"spacing ratio {list(regular.sp_ratio)}")

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", required=True, help="GeoPackage of hand-drawn rows")
    parser.add_argument("--truth-layer", default="r")
    parser.add_argument("--parcels", required=True, help="SIGPAC recintos")
    parser.add_argument("--parcel-layer", default="recinto")
    parser.add_argument("--detected", required=True,
                        help="GeoPackage written by detect_crop_rows")
    parser.add_argument("--detected-layer", default="crop_rows")
    parser.add_argument("--min-drawn", type=int, default=5,
                        help="parcels with fewer drawn rows than this are skipped")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="metres a detected row may sit from a drawn one")
    args = parser.parse_args()

    truth, parcels = load_truth(args.truth, args.parcels,
                                args.truth_layer, args.parcel_layer)
    counts = truth.groupby("pid").size()
    keep = list(counts[counts >= args.min_drawn].index)

    detected = gpd.read_file(args.detected, layer=args.detected_layer).to_crs(parcels.crs)
    summarise(score(detected, truth, parcels, keep, args.tolerance),
              args.detected.split("/")[-1])


if __name__ == "__main__":
    main()
