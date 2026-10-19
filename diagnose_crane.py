# -*- coding: utf-8 -*-
"""
Say why each thin tall object in a DSM was removed or kept.

The crane filter makes one decision per connected object, and the decision
turns on five numbers that the picture alone cannot show you. This prints
them, so that when a crane survives - or a chimney does not - you can see
which test was responsible instead of guessing at thresholds.

The second half is the before/after summary. The number to watch there is
the restored/interpolated split: a real crane is a lattice, so most of its
pixels should come back from a LiDAR return that reached the roof through
it. A run that interpolates nearly everything is a run whose
`crane_point_gap_m` is too large for the cloud, and its output is a guess
rather than a measurement.

    python diagnose_crane.py
"""

import numpy as np
import rasterio

import greenshadow.dsm_dem_estimation as m


# ---- edit to match your build_digital_model_from_lidar_* call -------
CONFIG = dict(
    dsm_path="/mnt/data/greenshadow/Alpicat/dsm.tif",
    resolution=0.5,
    # Stage B, the detector.
    width_m=4.0,
    height_m=4.0,
    min_length_m=12.0,
    min_elongation=4.0,
    max_area_m2=600.0,
    min_clearance_m=4.0,
    group_gap_m=1.5,
    absorb_max_area_m2=150.0,
    opening_shape="square",
)
# Rows/cols of the DSM to look at. The Alpicat crane sits here.
TEST_WINDOW = (9550, 9920, 6500, 6880)
# Set to None to see every component, kept and rejected alike.
MAX_COMPONENTS = 30
# ---------------------------------------------------------------------


def _read_window(path, window):
    r0, r1, c0, c1 = window
    with rasterio.open(path) as src:
        dm = src.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
        return dm, src.nodata, src.window_transform(
            rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        )


def _print_components(stats, transform, limit):
    if not stats:
        print("  no candidates at all - nothing was thin and tall enough")
        return

    stats = sorted(stats, key=lambda s: -s["clearance_m"])
    shown = stats if limit is None else stats[:limit]

    print(f"{'area_m2':>8} {'len_m':>7} {'r_in_m':>7} {'elong':>6} {'clear_m':>8} "
          f"{'resid_m':>8} {'seeds':>6} {'seed%':>6}  decision")
    print("-" * 86)

    for s in shown:
        print(f"{s['area_m2']:>8.1f} {s['length_m']:>7.1f} {s['r_inscribed_m']:>7.2f} "
              f"{s['elongation']:>6.1f} {s['clearance_m']:>8.1f} {s['resid_max']:>8.1f} "
              f"{s['seeds']:>6d} {100 * s['seed_fraction']:>5.0f}%  {s['decision']}")

    if limit is not None and len(stats) > limit:
        print(f"... and {len(stats) - limit} more")


def diagnose(config, window, limit):
    dm, nodata, transform = _read_window(config["dsm_path"], window)
    res = config["resolution"]

    print(f"window {window}  ->  {dm.shape[0]}x{dm.shape[1]} px at {res} m")
    valid = dm[dm != nodata]
    print(f"elevation {valid.min():.1f} - {valid.max():.1f} m "
          f"({100 * valid.size / dm.size:.1f}% valid)\n")

    detect_kw = {k: v for k, v in config.items()
                 if k not in ("dsm_path", "resolution")}

    # No seed mask here: this script reads a finished GeoTIFF, and the Stage A
    # bands only exist while the DSM is being built. Every component therefore
    # reports zero seeds, and the decisions below are the shape and clearance
    # tests alone - which is what you want when tuning those. Run the builder
    # itself to see the seed vote applied.
    crane_mask, stats, context = m._detect_thin_tall_structures(
        dm, nodata, res, seed_mask=None, min_seeds=0, min_seed_fraction=0.0,
        **detect_kw,
    )

    print("COMPONENTS (shape and clearance only - no seed vote)")
    _print_components(stats, transform, limit)

    removed = [s for s in stats if s["decision"] == "REMOVE"]
    print(f"\n{len(stats)} candidates, {len(removed)} would be removed on shape, "
          f"{int(crane_mask.sum())} px after absorption and skirt")

    if not crane_mask.any():
        print("\nNothing to remove in this window.")
        return

    # ---- before / after ------------------------------------------------
    dm_out, _, report = m._remove_thin_tall_outliers(
        dm.copy(), None,
        nodata=nodata, CLASS_NODATA=255, resolution=res,
        crane_bands=None, min_seeds=0, min_seed_fraction=0.0,
        **detect_kw,
    )

    changed = (dm != nodata) & (dm_out != nodata) & (np.abs(dm_out - dm) > 1e-3)
    drops = dm[changed] - dm_out[changed]

    print("\nBEFORE / AFTER")
    print(f"  pixels changed      {int(changed.sum())} "
          f"({changed.sum() * res * res:.0f} m2, "
          f"{100 * changed.sum() / (dm != nodata).sum():.4f}% of the window)")
    if drops.size:
        print(f"  height removed      mean {drops.mean():.1f}  "
              f"median {np.median(drops):.1f}  p95 {np.percentile(drops, 95):.1f}  "
              f"max {drops.max():.1f} m")
    print(f"  restored from LiDAR {report['restored_px']} px")
    print(f"  interpolated        {report['interpolated_px']} px")

    print("  (restored is always 0 here: the Stage A bands live only inside "
          "the builder,\n   so this script can only interpolate. Read the "
          "restored share off the\n   builder's own 'crane removal' log line.)")

    rises = dm_out[changed] - dm[changed]
    print(f"  max rise            {max(0.0, float(rises.max())):.3f} m "
          "(removal must only ever lower the surface)")

    print("\nHOW TO READ THIS")
    print("  clear_m is the discriminator. A crane stands 10-60 m clear of "
          "everything around it;")
    print("  a roof rim or a cliff edge reads near 0 no matter how thin it "
          "looks in the raster.")
    print("  If a crane is being kept, look at which column fails: r_in_m over "
          "width_m/2 means")
    print("  raise crane_width_m; a short len_m means the object arrived "
          "fragmented, so raise")
    print("  crane_group_gap_m. If a chimney or spire is being removed, its "
          "seeds should be 0 -")
    print("  if it has seeds, it is see-through and crane_max_upper_fraction "
          "is too generous.")


if __name__ == "__main__":
    diagnose(CONFIG, TEST_WINDOW, MAX_COMPONENTS)
