#!/usr/bin/env python3
"""
Test script for crop-row detection.

Everything runs on synthetic canopy models written to a temporary directory,
so there is no municipality to download and nothing to configure. The cases
are the ones that break a naive detector: a sharp comb whose harmonics pull a
spectral score off the true angle, a parcel whose long axis competes with its
planting direction, an amplitude-modulated grove that invites a doubled
spacing, and buildings that look exactly like trees until the LiDAR class is
consulted.

    python test_crop_rows.py
"""

import os
import sys
import tempfile

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import greenshadow as gs
from greenshadow.rural_parcels import sigpac_parcel_id, tag_parcels_with_preserved_areas


RESOLUTION = 0.25
CRS = "EPSG:25831"
X0, Y0 = 400000.0, 4600000.0
SHAPE = (400, 400)                       # 100 x 100 m
FULL = box(X0 + 2, Y0 - 98, X0 + 98, Y0 - 2)


# ------------------------------------------------------------------
# SYNTHETIC SCENES
# ------------------------------------------------------------------

def grid_xy(shape=SHAPE, resolution=RESOLUTION):
    height, width = shape
    xs = X0 + (np.arange(width) + 0.5) * resolution
    ys = Y0 - (np.arange(height) + 0.5) * resolution
    return np.meshgrid(xs, ys)


def write_scene(directory, name, chm, classes=None, resolution=RESOLUTION):
    """
    Write the DSM / bare-DSM pair whose difference is ``chm``.
    """
    height, width = chm.shape
    transform = from_origin(X0, Y0, resolution, resolution)
    ground = np.full(chm.shape, 100.0, np.float32)

    paths = {}
    for key, array in (("dsm", ground + chm), ("nv", ground)):
        path = f"{directory}/{name}_{key}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                           count=1, dtype="float32", crs=CRS, transform=transform,
                           nodata=-9999.0) as dataset:
            dataset.write(array.astype("float32"), 1)
        paths[key] = path

    if classes is not None:
        path = f"{directory}/{name}_cls.tif"
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                           count=1, dtype="uint8", crs=CRS, transform=transform,
                           nodata=255) as dataset:
            dataset.write(classes.astype("uint8"), 1)
        paths["cls"] = path

    return paths


def orchard(shape=SHAPE, *, spacing, azimuth_deg, tree_radius_m=1.0, height_m=3.5,
            along_m=4.0, missing=0.10, seed=0, resolution=RESOLUTION):
    """
    Gaussian canopy bumps on a lattice whose rows run at ``azimuth_deg``.
    """
    rng = np.random.default_rng(seed)
    grid_x, grid_y = grid_xy(shape, resolution)
    centre = np.array([grid_x.mean(), grid_y.mean()])

    bearing = np.radians(azimuth_deg)
    along_row = np.array([np.sin(bearing), np.cos(bearing)])
    across_row = np.array([np.cos(bearing), -np.sin(bearing)])

    chm = np.zeros(shape, np.float32)
    extent = max(shape) * resolution

    for i in range(-int(extent / spacing) - 2, int(extent / spacing) + 3):
        for j in range(-int(extent / along_m) - 2, int(extent / along_m) + 3):
            if rng.random() < missing:
                continue
            tree = centre + across_row * (i * spacing) + along_row * (j * along_m)
            squared = (grid_x - tree[0]) ** 2 + (grid_y - tree[1]) ** 2
            chm = np.maximum(chm, height_m * np.exp(-squared / (2 * tree_radius_m ** 2)))

    chm += rng.normal(0, 0.05, shape).astype(np.float32)

    return np.clip(chm, 0, None)


def parcels_gdf(polygons, usos):
    n = len(polygons)
    return gpd.GeoDataFrame({
        "provincia": [25] * n, "municipio": [23] * n, "agregado": [0] * n,
        "zona": [0] * n, "poligono": list(range(1, n + 1)),
        "parcela": list(range(1, n + 1)), "recinto": [1] * n,
        "uso_sigpac": list(usos),
    }, geometry=list(polygons), crs=CRS)


# ------------------------------------------------------------------
# HARNESS
# ------------------------------------------------------------------

class Report:
    def __init__(self):
        self.failures = 0

    def check(self, label, condition, detail=""):
        if not condition:
            self.failures += 1
        suffix = f"  -- {detail}" if detail else ""
        print(f"  {'PASS' if condition else 'FAIL'}  {label}{suffix}")
        return condition


def lattice_offsets(rows, azimuth_deg):
    """
    Each row's perpendicular offset along the shared row normal, sorted.

    A block's rows are meant to be parallel and evenly spaced, so the gaps
    between consecutive offsets are the thing to look at.
    """
    theta = np.radians(-azimuth_deg)
    midpoints = np.array([np.asarray(geometry.coords).mean(axis=0)
                          for geometry in rows.geometry])
    return np.sort(midpoints[:, 0] * np.cos(theta) + midpoints[:, 1] * np.sin(theta))


def bearing_error(measured, expected):
    if not np.isfinite(measured):
        return np.inf
    difference = abs(measured - expected) % 180.0
    return min(difference, 180.0 - difference)


def bend(geometry):
    """
    How far a row strays from the straight line between its own endpoints.
    """
    coords = np.asarray(geometry.coords)

    if coords.shape[0] < 3:
        return 0.0

    start, end = coords[0], coords[-1]
    span = end - start
    length = np.hypot(*span)

    if length <= 0:
        return 0.0

    normal = np.array([-span[1], span[0]]) / length

    return float(np.abs((coords - start) @ normal).max())


def detect(directory, name, chm, *, polygon=FULL, uso="OV", classes=None,
           resolution=RESOLUTION, **kwargs):
    paths = write_scene(directory, name, chm, classes, resolution=resolution)
    rows, parcels = gs.detect_crop_rows(
        parcels_gdf([polygon], [uso]),
        dsm_path=paths["dsm"],
        dsm_no_vegetation_path=paths["nv"],
        dsm_class_path=paths.get("cls"),
        **kwargs,
    )
    return rows, parcels.iloc[0]


# ------------------------------------------------------------------
# CASES
# ------------------------------------------------------------------

def main():
    report = Report()

    with tempfile.TemporaryDirectory() as directory:

        print("1. olive grove, 7 m rows at azimuth 37")
        chm = orchard(spacing=7.0, azimuth_deg=37.0, tree_radius_m=1.2, along_m=5.0)
        rows, parcel = detect(directory, "orchard", chm)
        error = bearing_error(parcel.row_azimuth_deg, 37.0)
        print(f"     pattern={parcel.row_pattern} az={parcel.row_azimuth_deg:.2f} "
              f"spacing={parcel.row_spacing_m:.2f} n_rows={parcel.n_rows} "
              f"periodicity={parcel.row_periodicity:.3f} "
              f"width={parcel.canopy_width_m:.2f} alley={parcel.alley_width_m:.2f}")
        report.check("classified as rows", parcel.row_pattern == "rows")
        report.check("azimuth within 1.5 deg", error < 1.5, f"error={error:.2f}")
        report.check("spacing within 0.3 m", abs(parcel.row_spacing_m - 7.0) < 0.3)
        report.check("confidence above 0.7", parcel.row_confidence > 0.7)
        report.check("a line per row", len(rows) == parcel.n_rows >= 10)
        report.check("alley plus canopy is the spacing",
                     abs(parcel.alley_width_m + parcel.canopy_width_m
                         - parcel.row_spacing_m) < 1e-6)

        print("2. vineyard, 3 m rows running East-West")
        chm = orchard(spacing=3.0, azimuth_deg=90.0, tree_radius_m=0.5, along_m=1.5,
                      height_m=2.0, missing=0.05, seed=3)
        _, parcel = detect(directory, "vineyard", chm, uso="VI")
        error = bearing_error(parcel.row_azimuth_deg, 90.0)
        print(f"     pattern={parcel.row_pattern} az={parcel.row_azimuth_deg:.2f} "
              f"spacing={parcel.row_spacing_m:.2f} n_rows={parcel.n_rows}")
        report.check("classified as rows", parcel.row_pattern == "rows")
        report.check("azimuth within 1.5 deg", error < 1.5, f"error={error:.2f}")
        report.check("spacing within 0.3 m", abs(parcel.row_spacing_m - 3.0) < 0.3)

        print("3. harmonic trap: 5 m rows, every second row weaker")
        chm = orchard(spacing=5.0, azimuth_deg=20.0, along_m=4.0, seed=7)
        grid_x, grid_y = grid_xy()
        bearing = np.radians(20.0)
        across = ((grid_x - grid_x.mean()) * np.cos(bearing)
                  - (grid_y - grid_y.mean()) * np.sin(bearing))
        chm = np.where((np.floor(across / 5.0).astype(int) % 2) == 0, chm * 0.55, chm)
        _, parcel = detect(directory, "harmonic", chm)
        print(f"     pattern={parcel.row_pattern} az={parcel.row_azimuth_deg:.2f} "
              f"spacing={parcel.row_spacing_m:.2f}")
        report.check("reports 5 m, not the 10 m sub-harmonic",
                     abs(parcel.row_spacing_m - 5.0) < 0.5,
                     f"got {parcel.row_spacing_m:.2f}")

        print("4. long thin parcel, rows across its long axis")
        thin = box(X0 + 2, Y0 - 98, X0 + 30, Y0 - 2)        # 28 m by 96 m
        chm = orchard(spacing=6.0, azimuth_deg=90.0, along_m=4.0, seed=11)
        _, parcel = detect(directory, "thin", chm, polygon=thin)
        error = bearing_error(parcel.row_azimuth_deg, 90.0)
        print(f"     pattern={parcel.row_pattern} az={parcel.row_azimuth_deg:.2f} "
              f"spacing={parcel.row_spacing_m:.2f} "
              f"working_axis={parcel.working_axis_deg:.2f}")
        report.check("the chord-length envelope does not win",
                     parcel.row_pattern == "rows" and error < 3.0, f"error={error:.2f}")

        print("5. no prior: a wide search band must not lock onto the parcel length")
        chm = orchard(spacing=7.0, azimuth_deg=37.0, tree_radius_m=1.2, along_m=5.0)
        for uso, band in (("TA", (1.5, 14.0)), ("OV", (1.5, 40.0))):
            _, parcel = detect(directory, "noprior", chm, uso=uso,
                               row_spacing_range_m=band)
            acceptable = (parcel.row_pattern != "rows"
                          or abs(parcel.row_spacing_m - 7.0) < 0.5)
            print(f"     uso={uso} band={band} -> pattern={parcel.row_pattern} "
                  f"spacing={parcel.row_spacing_m:.2f}")
            report.check(f"no degenerate period (uso={uso}, band={band})", acceptable)

        print("6. uniform herbaceous cover")
        rng = np.random.default_rng(1)
        chm = np.abs(rng.normal(0.25, 0.08, SHAPE)).astype(np.float32)
        rows, parcel = detect(directory, "herbaceous", chm, uso="TA")
        print(f"     pattern={parcel.row_pattern} cover={parcel.canopy_cover_frac:.4f} "
              f"working_axis={parcel.working_axis_deg:.2f}")
        report.check("not called rows", parcel.row_pattern in ("herbaceous", "empty"))
        report.check("no row azimuth invented", not np.isfinite(parcel.row_azimuth_deg))
        report.check("working axis still reported",
                     np.isfinite(parcel.working_axis_deg))
        report.check("no lines", len(rows) == 0)

        print("7. two isolated sheds")
        chm = np.zeros(SHAPE, np.float32)
        chm[80:110, 80:130] = 4.5
        chm[250:280, 250:300] = 5.0
        _, parcel = detect(directory, "sheds", chm, uso="TA")
        print(f"     pattern={parcel.row_pattern} cover={parcel.canopy_cover_frac:.4f}")
        report.check("not called rows", parcel.row_pattern != "rows")

        print("8. building stripes rejected by the LiDAR class filter")
        chm = np.zeros(SHAPE, np.float32)
        for i in range(6):
            chm[:, 40 + i * 50: 52 + i * 50] = 4.0
        classes = np.full(SHAPE, 6, np.uint8)               # 6 = building
        _, parcel = detect(directory, "buildings", chm, classes=classes)
        print(f"     pattern={parcel.row_pattern} cover={parcel.canopy_cover_frac:.4f}")
        report.check("buildings are not crop rows", parcel.row_pattern != "rows")

        print("9. non-agricultural parcels are dropped, the rest are flagged")
        chm = orchard(spacing=7.0, azimuth_deg=37.0, tree_radius_m=1.2, along_m=5.0)
        paths = write_scene(directory, "flags", chm)
        polygons = [box(X0 + 2, Y0 - 48, X0 + 48, Y0 - 2),
                    box(X0 + 52, Y0 - 98, X0 + 98, Y0 - 52),
                    box(X0 + 52, Y0 - 48, X0 + 98, Y0 - 2)]
        parcels = parcels_gdf(polygons, ["OV", "OV", "ZU"])
        protected = gpd.GeoDataFrame({"resource": ["Red Natura 2000"]},
                                     geometry=[box(X0, Y0 - 60, X0 + 70, Y0)], crs=CRS)
        rows, out = gs.detect_crop_rows(
            parcels,
            dsm_path=paths["dsm"], dsm_no_vegetation_path=paths["nv"],
            exclusions=protected,
            output_path=f"{directory}/crop_rows.gpkg",
        )
        report.check("the urban parcel is dropped", len(out) == 2
                     and "ZU" not in set(out.uso_sigpac))
        report.check("protected parcels are kept, flagged",
                     bool(out.is_preserved.iloc[0]))
        report.check("parcel id is the SIGPAC 7-tuple",
                     list(out.parcel_id) == list(sigpac_parcel_id(parcels.iloc[:2])))
        report.check("no row belongs to a dropped parcel",
                     set(rows.parcel_id) <= set(out.parcel_id))

        # The FEGA classification itself: 22 agricultural uses kept, the
        # non-agricultural and "other" ones and a missing use all dropped.
        codes = (list(gs.SIGPAC_AGRICULTURAL_USES)
                 + list(gs.SIGPAC_NON_AGRICULTURAL_USES) + ["ZC", "ZV", None])
        kept = gs.filter_agricultural_parcels(
            parcels_gdf([FULL] * len(codes), codes))
        report.check("only FEGA agricultural uses survive the filter",
                     sorted(kept.uso_sigpac) == sorted(gs.SIGPAC_AGRICULTURAL_USES)
                     and len(kept) == 22)

        import fiona
        layers = set(fiona.listlayers(f"{directory}/crop_rows.gpkg"))
        report.check("two output layers", layers == {"crop_row_parcels", "crop_rows"},
                     str(sorted(layers)))
        written = gpd.read_file(f"{directory}/crop_rows.gpkg", layer="crop_rows")
        report.check("rows round-trip through the GeoPackage",
                     len(written) == len(rows) and written.crs == parcels.crs)

        print("10. preserved-area ratios are per parcel")
        expected = [geometry.intersection(protected.geometry.iloc[0]).area / geometry.area
                    for geometry in parcels.geometry]
        tagged = tag_parcels_with_preserved_areas(
            parcels.assign(parcel_id=sigpac_parcel_id(parcels)), protected)
        print(f"     expected {[round(v, 4) for v in expected]}")
        print(f"          got {[round(v, 4) for v in tagged.preserved_area_ratio]}")
        report.check("ratios are not smeared across parcels",
                     np.allclose(expected, tagged.preserved_area_ratio))
        report.check("resource name attached",
                     tagged.preserved_resource.iloc[0] == "Red Natura 2000")

        far = gpd.GeoDataFrame({"resource": ["ENP"]},
                               geometry=[box(X0 + 500, Y0 - 500, X0 + 600, Y0 - 400)],
                               crs=CRS)
        untouched = tag_parcels_with_preserved_areas(
            parcels.assign(parcel_id=sigpac_parcel_id(parcels)), far)
        report.check("a parcel touching nothing does not raise",
                     not untouched.is_preserved.any()
                     and untouched.preserved_resource.isna().all())

        print("11. rows are estimated where the crop is woody, not where the code says so")
        # SIGPAC carries an orchard as arable land for years after planting, so
        # a canopy that looks like a plantation is enough on its own.
        chm = orchard(spacing=7.0, azimuth_deg=37.0, tree_radius_m=1.2, along_m=5.0)
        rows, parcel = detect(directory, "woodygate", chm, uso="TA")
        print(f"     orchard on uso=TA -> pattern={parcel.row_pattern} "
              f"cover={parcel.canopy_cover_frac:.3f} h50={parcel.canopy_height_p50_m:.1f} "
              f"n_rows={parcel.n_rows}")
        report.check("an orchard coded as arable is still found",
                     parcel.row_pattern == "rows" and len(rows) > 0)

        # Sparse low scrub on the same code: nothing a plantation would show.
        thin = orchard(spacing=7.0, azimuth_deg=37.0, tree_radius_m=0.5,
                       along_m=14.0, height_m=1.4, missing=0.75, seed=71)
        rows, parcel = detect(directory, "scrub", thin, uso="TA")
        print(f"     scrub on uso=TA -> pattern={parcel.row_pattern} "
              f"cover={parcel.canopy_cover_frac:.3f}")
        report.check("thin scrub on a herbaceous use is refused",
                     parcel.row_pattern != "rows" and len(rows) == 0)
        report.check("working axis still reported",
                     np.isfinite(parcel.working_axis_deg))

        # Woodland: more cover than any orchard, but far too tall to be a crop.
        # Flat-topped crowns, so the median height is the canopy's own height
        # rather than the skirt of a Gaussian bump.
        tall = np.where(
            orchard(spacing=7.0, azimuth_deg=37.0, tree_radius_m=1.6,
                    along_m=5.0, height_m=9.0, seed=72) > 1.0,
            9.0, 0.0).astype(np.float32)
        rows, parcel = detect(directory, "woodland", tall, uso="PA")
        print(f"     woodland on uso=PA -> pattern={parcel.row_pattern} "
              f"cover={parcel.canopy_cover_frac:.3f} h50={parcel.canopy_height_p50_m:.1f}")
        report.check("woodland is not a plantation, however dense",
                     parcel.row_pattern != "rows" and len(rows) == 0)
        _, forced = detect(directory, "woodland", tall, uso="PA", woody_only=False)
        report.check("woody_only=False lets even that through",
                     forced.row_pattern == "rows")

        print("12. the canopy width is reported, not enforced")
        chm = orchard(spacing=7.0, azimuth_deg=0.0, tree_radius_m=1.2, along_m=5.0,
                      seed=17)
        rows, parcel = detect(directory, "normal", chm)
        print(f"     pattern={parcel.row_pattern} canopy={parcel.canopy_width_m:.2f} "
              f"alley={parcel.alley_width_m:.2f} n_rows={parcel.n_rows}")
        report.check("the planted strip is measured",
                     parcel.row_pattern == "rows"
                     and np.isfinite(parcel.canopy_width_m))
        report.check("and the alley is the rest of the spacing",
                     abs(parcel.alley_width_m + parcel.canopy_width_m
                         - parcel.row_spacing_m) < 1e-6)
        report.check("every row carries the block's width",
                     len(rows) > 0
                     and np.allclose(rows.canopy_width_m, parcel.canopy_width_m))

        # Rows whose width marches from a hedgerow's worth down to a single
        # tree. They are all still rows: a weed, a dead stretch or a wider
        # crown changes what a row looks like, not whether it was planted, and
        # holding every row of a block to one width only threw such rows away.
        grid_x, grid_y = grid_xy()
        chm = np.zeros(SHAPE, np.float32)
        across = grid_x - X0
        for index in range(13):
            centre = 4.0 + index * 7.0
            half = 0.3 + index * 0.22
            chm[np.abs(across - centre) <= half] = 3.5
        rows, parcel = detect(directory, "ragged", chm)
        print(f"     marching widths -> pattern={parcel.row_pattern} "
              f"n_rows={parcel.n_rows} spacing={parcel.row_spacing_m:.2f}")
        report.check("rows of differing width are all kept",
                     parcel.row_pattern == "rows" and parcel.n_rows >= 11,
                     f"{parcel.n_rows} of 13")
        report.check("at the spacing they were planted at",
                     abs(parcel.row_spacing_m - 7.0) < 0.4)

        print("13. rows are near-straight, parallel and evenly spaced")
        chm = orchard(spacing=6.0, azimuth_deg=25.0, tree_radius_m=1.0, along_m=4.0,
                      seed=31)
        rows, parcel = detect(directory, "straight", chm)
        offsets = lattice_offsets(rows, parcel.row_azimuth_deg)
        gaps = np.diff(offsets)
        bends = np.array([bend(g) for g in rows.geometry])
        segments = np.concatenate([
            np.hypot(*np.diff(np.asarray(g.coords), axis=0).T) for g in rows.geometry])
        print(f"     n_rows={parcel.n_rows} max bend={bends.max():.2f} m "
              f"shortest segment={segments.min():.1f} m "
              f"gap min/max={gaps.min():.2f}/{gaps.max():.2f} "
              f"spacing={parcel.row_spacing_m:.2f}")
        report.check("rows bend only gently",
                     bends.max() <= parcel.row_spacing_m / 3.0,
                     f"{bends.max():.2f} m of {parcel.row_spacing_m:.2f}")
        report.check("no segment is shorter than the regularisation length",
                     segments.min() >= 9.0, f"{segments.min():.1f} m")
        report.check("one geometry per row", len(rows) == parcel.n_rows)
        report.check("consecutive rows are about one spacing apart",
                     np.allclose(gaps, parcel.row_spacing_m,
                                 atol=0.15 * parcel.row_spacing_m),
                     f"{gaps.min():.2f}..{gaps.max():.2f}")
        report.check("one block, so no azimuth spread",
                     parcel.n_blocks == 1 and parcel.row_azimuth_spread_deg < 1e-9)
        report.check("every row carries its block azimuth",
                     np.allclose(rows.row_azimuth_deg, parcel.row_azimuth_deg))

        print("14. two plantations, different directions, corridor between")
        grid_x, grid_y = grid_xy()
        north = orchard(spacing=6.0, azimuth_deg=20.0, along_m=4.0, seed=41)
        south = orchard(spacing=6.0, azimuth_deg=80.0, along_m=4.0, seed=42)
        chm = np.where(grid_y > Y0 - 44.0, north,
                       np.where(grid_y < Y0 - 56.0, south, 0.0)).astype(np.float32)
        rows, parcel = detect(directory, "twoblocks", chm)
        found = sorted(rows.groupby("block_index").row_azimuth_deg.first())
        print(f"     pattern={parcel.row_pattern} n_blocks={parcel.n_blocks} "
              f"azimuths={[round(a, 2) for a in found]} "
              f"spread={parcel.row_azimuth_spread_deg:.2f} n_rows={parcel.n_rows}")
        report.check("two blocks, not one compromise", parcel.n_blocks == 2)
        report.check("both directions recovered",
                     len(found) == 2
                     and bearing_error(found[0], 20.0) < 2.0
                     and bearing_error(found[1], 80.0) < 2.0)
        report.check("the spread says the parcel has no single direction",
                     abs(parcel.row_azimuth_spread_deg - 60.0) < 3.0)
        # Each block's rows must sit on its own side of the corridor. They are
        # allowed past the last tree by the edge margin - a plantation is
        # planted up to its headland - but never over the middle of the gap
        # that made these two separate blocks.
        for block_index, group in rows.groupby("block_index"):
            ys = np.concatenate([np.asarray(g.coords)[:, 1] for g in group.geometry])
            report.check(f"block {block_index} never reaches the other planting",
                         (ys > Y0 - 56.0).all() or (ys < Y0 - 44.0).all(),
                         f"y range {ys.min() - Y0:.1f}..{ys.max() - Y0:.1f}")

        print("15. a bare stretch inside a row does not break it")
        chm = orchard(spacing=6.0, azimuth_deg=90.0, along_m=4.0, seed=51)
        intact_rows, intact = detect(directory, "intact", chm)
        gapped = chm.copy()
        gapped[:, 180:212] = 0.0                         # an 8 m bare band
        rows, parcel = detect(directory, "gapped", gapped)
        print(f"     intact n_rows={intact.n_rows} -> gapped n_rows={parcel.n_rows} "
              f"lines={len(rows)}")
        report.check("the row is bridged, not split in two",
                     len(rows) == parcel.n_rows <= intact.n_rows)
        report.check("and it still spans the gap",
                     rows.row_length_m.median() > 0.8 * intact_rows.row_length_m.median(),
                     f"{rows.row_length_m.median():.1f} m vs "
                     f"{intact_rows.row_length_m.median():.1f} m")

        print("16. a written GeoPackage is not recomputed")
        chm = orchard(spacing=7.0, azimuth_deg=37.0, tree_radius_m=1.2, along_m=5.0)
        paths = write_scene(directory, "cached", chm)
        cached = f"{directory}/cached.gpkg"
        arguments = dict(dsm_path=paths["dsm"],
                         dsm_no_vegetation_path=paths["nv"],
                         output_path=cached)
        first_rows, first_parcels = gs.detect_crop_rows(
            parcels_gdf([FULL], ["OV"]), **arguments)

        # A second call with the rasters pointed at nothing: only a run that
        # never opens them can succeed.
        missing = dict(arguments, dsm_path=f"{directory}/gone.tif",
                       dsm_no_vegetation_path=f"{directory}/gone.tif")
        again_rows, again_parcels = gs.detect_crop_rows(
            parcels_gdf([FULL], ["OV"]), **missing)
        print(f"     first {len(first_rows)} rows -> reused {len(again_rows)} rows")
        report.check("the stored rows come back untouched",
                     len(again_rows) == len(first_rows)
                     and np.allclose(again_rows.row_azimuth_deg,
                                     first_rows.row_azimuth_deg))
        report.check("and so do the parcels",
                     len(again_parcels) == len(first_parcels)
                     and again_parcels.row_pattern.iloc[0] == "rows")

        stamp = os.path.getmtime(cached)
        gs.detect_crop_rows(parcels_gdf([FULL], ["OV"]), **arguments)
        report.check("nothing is rewritten either",
                     os.path.getmtime(cached) == stamp)

        gs.detect_crop_rows(parcels_gdf([FULL], ["OV"]), update=True, **arguments)
        report.check("update=True recomputes and rewrites",
                     os.path.getmtime(cached) > stamp)

        # A cereal municipality writes no row layer at all, and reading it back
        # must give the same empty frame rather than an error.
        rng = np.random.default_rng(9)
        bare = np.abs(rng.normal(0.2, 0.05, SHAPE)).astype(np.float32)
        bare_paths = write_scene(directory, "bare", bare)
        empty_path = f"{directory}/bare.gpkg"
        empty_arguments = dict(dsm_path=bare_paths["dsm"],
                               dsm_no_vegetation_path=bare_paths["nv"],
                               output_path=empty_path)
        written, _ = gs.detect_crop_rows(
            parcels_gdf([FULL], ["TA"]), **empty_arguments)
        reread, _ = gs.detect_crop_rows(
            parcels_gdf([FULL], ["TA"]), **empty_arguments)
        report.check("a parcel set with no rows round-trips as an empty frame",
                     len(written) == len(reread) == 0
                     and list(reread.columns) == list(written.columns))

        print("17. on a half-metre grid the rows still run the parcel end to end")
        coarse, shape = 0.5, (200, 200)              # the same 100 x 100 m
        chm = orchard(shape, spacing=6.0, azimuth_deg=90.0, tree_radius_m=1.0,
                      along_m=4.0, seed=61, resolution=coarse)
        rows, parcel = detect(directory, "coarse", chm, resolution=coarse)
        lengths = rows.row_length_m
        expected = 96.0                              # the parcel is 96 m across
        print(f"     n_rows={parcel.n_rows} spacing={parcel.row_spacing_m:.2f} "
              f"length min/median/max="
              f"{lengths.min():.1f}/{lengths.median():.1f}/{lengths.max():.1f} m")
        report.check("rows are found at all on a coarse grid",
                     parcel.row_pattern == "rows")
        report.check("about one row per spacing across the parcel",
                     abs(parcel.n_rows - expected / 6.0) <= 2,
                     f"{parcel.n_rows} rows")
        report.check("every row spans the parcel, not a fragment of it",
                     lengths.min() > 0.9 * (expected - 2 * 2.0),
                     f"shortest {lengths.min():.1f} m of {expected:.0f} m")

        # The rows must actually reach the boundary, not stop at the outermost
        # tree a few metres inside it.
        boundary = FULL.exterior
        ends = np.array([
            boundary.distance(geom.__class__(np.asarray(geom.coords)[[0, -1]]).boundary)
            for geom in rows.geometry
        ]) if False else np.array([
            max(boundary.distance(Point(np.asarray(g.coords)[0])),
                boundary.distance(Point(np.asarray(g.coords)[-1])))
            for g in rows.geometry
        ])
        report.check("row ends sit on the parcel boundary, set in by the margin",
                     abs(ends.max() - 2.0) < 1.0,
                     f"worst end {ends.max():.2f} m from the edge")

        print("18. separate trees on a grid follow one direction, not a patchwork")
        # Rows 6 m apart and trees 5.5 m apart down the row, but the west half is
        # planted with its rows at 30 and the east half square to it. Every
        # tree stands alone, so both halves are in rows both ways, and which
        # way a sweep prefers flips across the middle of the parcel.
        grid_x, grid_y = grid_xy()
        west = orchard(spacing=6.0, azimuth_deg=30.0, along_m=5.5,
                       tree_radius_m=1.2, seed=81)
        east = orchard(spacing=6.0, azimuth_deg=120.0, along_m=5.5,
                       tree_radius_m=1.2, seed=82)
        chm = np.where(grid_x < X0 + 50.0, west, east).astype(np.float32)
        _, crossed = detect(directory, "grid", chm, uso="FY",
                            hedgerow_max_periodicity=2.0)
        rows, parcel = detect(directory, "grid", chm, uso="FY")
        print(f"     never follow -> n_blocks={crossed.n_blocks} "
              f"spread={crossed.row_azimuth_spread_deg:.1f}; "
              f"follow -> n_blocks={parcel.n_blocks} "
              f"spread={parcel.row_azimuth_spread_deg:.1f} n_rows={parcel.n_rows}")
        report.check("the scene does split two ways when nothing is followed",
                     crossed.n_blocks >= 2 and crossed.row_azimuth_spread_deg > 60.0)
        report.check("followed, every row runs one way",
                     parcel.row_pattern == "rows"
                     and parcel.row_azimuth_spread_deg < 3.0
                     and rows.row_azimuth_deg.max() - rows.row_azimuth_deg.min() < 3.0)

        print("19. rows closer than 2 m: one is deleted, a touch is trimmed")
        from shapely.geometry import LineString
        from greenshadow.crop_rows import _drop_crowded_rows
        config = {"min_row_separation_m": 2.0, "row_segment_m": 10.0}
        block = {"spacing_m": 6.0}
        # Canopy along y = 0 and y = 6; the row at y = 1.5 stands on nothing.
        xs = np.arange(0.0, 100.0, 0.25)
        x_canopy = np.concatenate([xs, xs])
        y_canopy = np.concatenate([np.zeros_like(xs), np.full_like(xs, 6.0)])
        laid = [(0, block, LineString([(0, 0), (100, 0)])),
                (0, block, LineString([(0, 1.5), (100, 1.5)])),
                (1, block, LineString([(0, 6), (100, 6)])),
                # Square to the others, ending a metre short of y = 6.
                (1, block, LineString([(50, 40), (50, 7)]))]
        kept = _drop_crowded_rows(laid, x_canopy, y_canopy, config)
        ys = sorted(round(line.coords[0][1], 1) for _, _, line in kept
                    if line.coords[0][1] == line.coords[-1][1])
        cross = [line for _, _, line in kept if line.coords[0][0] == line.coords[-1][0]]
        print(f"     parallel rows kept at y={ys}; "
              f"cross row {[round(c.length, 1) for c in cross]} m of 33")
        report.check("the row squeezed between two others is deleted", ys == [0.0, 6.0])
        report.check("the row that only touches another is cut back, not deleted",
                     len(cross) == 1 and 29.0 <= cross[0].length < 33.0)
        separated = min(a.distance(b) for i, (_, _, a) in enumerate(kept)
                        for _, _, b in kept[i + 1:])
        report.check("no two survivors closer than 2 m", separated >= 2.0 - 1e-6,
                     f"closest {separated:.2f} m")

        vineyard = {"spacing_m": 1.8}
        rows_18 = [(0, vineyard, LineString([(0, 1.8 * k), (100, 1.8 * k)]))
                   for k in range(6)]
        kept = _drop_crowded_rows(rows_18, x_canopy, y_canopy, config)
        report.check("a vineyard planted at 1.8 m keeps every row",
                     len(kept) == len(rows_18))

        print("20. superintensive olive: hedges 4 m apart, under the OV band")
        # Trees 1.5 m apart down the row merge into one hedge; the rows are
        # 4 m apart, which the traditional OV band (5-12 m) refuses outright.
        chm = orchard(spacing=4.0, azimuth_deg=30.0, tree_radius_m=0.8,
                      along_m=1.5, height_m=2.5, missing=0.03, seed=91)
        rows, parcel = detect(directory, "superintensive", chm, uso="OV")
        error = bearing_error(parcel.row_azimuth_deg, 30.0)
        print(f"     pattern={parcel.row_pattern} az={parcel.row_azimuth_deg:.2f} "
              f"spacing={parcel.row_spacing_m:.2f} n_rows={parcel.n_rows}")
        report.check("a hedge planting on an olive parcel is found",
                     parcel.row_pattern == "rows" and error < 1.5,
                     f"error={error:.2f}")
        report.check("at its own 4 m spacing",
                     abs(parcel.row_spacing_m - 4.0) < 0.3)

        # Separate trees 8 m apart both ways: the 4 m harmonic is in the hedge
        # band too, and must not be taken for a hedge planting.
        chm = orchard(spacing=8.0, azimuth_deg=30.0, tree_radius_m=1.3,
                      along_m=8.0, seed=92)
        _, parcel = detect(directory, "traditional", chm, uso="OV")
        print(f"     traditional grid -> pattern={parcel.row_pattern} "
              f"spacing={parcel.row_spacing_m:.2f}")
        report.check("a traditional grove keeps its 8 m spacing",
                     parcel.row_pattern != "rows"
                     or abs(parcel.row_spacing_m - 8.0) < 0.5,
                     f"got {parcel.row_spacing_m:.2f}")

    print()
    if report.failures:
        print(f"{report.failures} FAILURE(S)")
        return 1

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
