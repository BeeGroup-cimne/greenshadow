"""
Synthetic regression test for the crane / thin-tall-outlier filter.

The point of a synthetic scene is that we know the right answer. A DSM built
with ``aggregation="max"`` must come back with

  - the crane jib AND its solid mast gone, even though the mast never lets a
    return through to the roof (it is removed by association with the seeded
    jib it joins - this is the single assertion most likely to break);
  - the chimney and the spire still standing, because they are solid: their
    shadows are real and a PV yield computed without them is wrong;
  - the roof under the crane restored to its measured height rather than
    interpolated, because the lattice let us see it.

Run directly:  python test_crane_removal.py
"""

import numpy as np

import greenshadow.dsm_dem_estimation as m


RES = 0.5
NODATA = -9999.0

# Scene layout, in metres, on a local grid whose origin is (0, 0).
SCENE = 80.0
ROOF_X = (10.0, 70.0)
ROOF_Y = (10.0, 70.0)
ROOF_Z = 20.0
GROUND_Z = 0.0

JIB_Z = 55.0
JIB_HALF_WIDTH = 1.0          # 2 m wide lattice
JIB_Y = 40.0
JIB_X = (5.0, 65.0)           # crosses roof and ground

MAST_C = (35.0, 40.0)
MAST_HALF = 1.0               # 2 m square solid column

CHIMNEY_C = (20.0, 20.0)
CHIMNEY_HALF = 0.5            # 1 m square, 3 m above the roof
CHIMNEY_Z = 23.0

SPIRE_C = (58.0, 22.0)
SPIRE_BASE_R = 1.5
SPIRE_TOP_Z = 40.0


def _grid(x0, x1, y0, y1, step, rng, jitter=0.0):
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel().astype(np.float64)
    gy = gy.ravel().astype(np.float64)
    if jitter:
        gx += rng.uniform(-jitter, jitter, gx.size)
        gy += rng.uniform(-jitter, jitter, gy.size)
    return gx, gy


def build_scene(seed=0):
    """Return an (N, 4) array of x, y, z, classification."""
    rng = np.random.default_rng(seed)
    parts = []

    # Ground everywhere, ~4 pts/m2.
    gx, gy = _grid(0, SCENE, 0, SCENE, 0.5, rng, jitter=0.15)
    parts.append((gx, gy, np.full(gx.size, GROUND_Z), np.full(gx.size, 2)))

    # Flat roof, ~10 pts/m2. Ground is NOT removed underneath - the roof simply
    # wins the max, exactly as in a real cloud.
    rx, ry = _grid(ROOF_X[0], ROOF_X[1], ROOF_Y[0], ROOF_Y[1], 0.3, rng, jitter=0.1)
    parts.append((rx, ry, np.full(rx.size, ROOF_Z), np.full(rx.size, 6)))

    # Lattice jib: sparse, ~1 return per pixel, so the surface below stays
    # visible through it. Class 6 on purpose - real ICGC clouds label most
    # crane returns as building, which is why dropping class 1 does not help.
    jx = np.arange(JIB_X[0], JIB_X[1], 0.5)
    jy = rng.uniform(JIB_Y - JIB_HALF_WIDTH, JIB_Y + JIB_HALF_WIDTH, jx.size)
    parts.append((jx, jy, np.full(jx.size, JIB_Z), np.full(jx.size, 6)))

    # Solid mast: dense, opaque - nothing below it is ever seen.
    mx, my = _grid(MAST_C[0] - MAST_HALF, MAST_C[0] + MAST_HALF,
                   MAST_C[1] - MAST_HALF, MAST_C[1] + MAST_HALF, 0.2, rng)
    mz = rng.uniform(JIB_Z - 2.0, JIB_Z, mx.size)
    parts.append((mx, my, mz, np.full(mx.size, 6)))

    # Chimney: compact, solid, 3 m proud of the roof.
    cx, cy = _grid(CHIMNEY_C[0] - CHIMNEY_HALF, CHIMNEY_C[0] + CHIMNEY_HALF,
                   CHIMNEY_C[1] - CHIMNEY_HALF, CHIMNEY_C[1] + CHIMNEY_HALF,
                   0.15, rng)
    parts.append((cx, cy, np.full(cx.size, CHIMNEY_Z), np.full(cx.size, 6)))

    # Spire: a solid cone from the roof up to 40 m. Thin near the top and
    # 20 m above its surroundings - the shape a top-hat cannot tell from a jib.
    sxs, sys, szs = [], [], []
    for z in np.arange(ROOF_Z, SPIRE_TOP_Z, 0.25):
        frac = 1.0 - (z - ROOF_Z) / (SPIRE_TOP_Z - ROOF_Z)
        r = max(SPIRE_BASE_R * frac, 0.05)
        n = max(int(2 * np.pi * r / 0.12), 6)
        a = rng.uniform(0, 2 * np.pi, n)
        sxs.append(SPIRE_C[0] + r * np.cos(a))
        sys.append(SPIRE_C[1] + r * np.sin(a))
        szs.append(np.full(n, z))
    sx, sy, sz = np.concatenate(sxs), np.concatenate(sys), np.concatenate(szs)
    parts.append((sx, sy, sz, np.full(sx.size, 6)))

    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    z = np.concatenate([p[2] for p in parts])
    c = np.concatenate([p[3] for p in parts])
    return np.column_stack((x, y, z, c)).astype(np.float64)


def _build(pts, **kw):
    params = dict(
        laz_points=pts,
        out_path="/tmp/_crane_test.tif",
        return_arrays=True,
        cache_rasterization=False,
        resolution=RES,
        bounds=(0.0, 0.0, SCENE, SCENE),
        epsg=25831,
        classification=(2, 6),
        building_classes=(6,),
        terrain_classes=(2,),
        vegetation_classes=(3, 4, 5),
        aggregation="max",
        nodata=NODATA,
        fill_gaps="nearest",
    )
    params.update(kw)
    dm, class_dm, slope, aspect, transform = m.build_digital_model_from_lidar_pure(**params)
    return dm, transform


def _px(transform, x, y):
    inv = ~transform
    c, r = inv * (x, y)
    return int(r), int(c)


def _box(dm, transform, x0, x1, y0, y1):
    r0, c0 = _px(transform, x0, y1)
    r1, c1 = _px(transform, x1, y0)
    return dm[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1]


def main():
    pts = build_scene()
    print(f"scene: {len(pts)} points")

    dm_off, tf = _build(pts, crane_removal=False)
    dm_on, _ = _build(pts, crane_removal=True)

    failures = []

    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print("\n-- baseline (crane_removal=False) --")
    jib_off = _box(dm_off, tf, 45.0, 60.0, JIB_Y - 1.0, JIB_Y + 1.0)
    check("crane present without the filter", jib_off.max() > 50.0,
          f"jib max {jib_off.max():.1f} m (expect ~{JIB_Z})")

    print("\n-- with the filter --")

    # 1. The jib itself.
    jib_on = _box(dm_on, tf, 45.0, 60.0, JIB_Y - 1.0, JIB_Y + 1.0)
    check("jib removed", jib_on.max() < 25.0,
          f"max {jib_on.max():.1f} m (expect < 25)")

    # 2. The mast - the assertion that matters most. It is opaque, so Stage A
    #    never seeds it; it can only go by association with the jib.
    mast_on = _box(dm_on, tf, MAST_C[0] - 1.0, MAST_C[0] + 1.0,
                   MAST_C[1] - 1.0, MAST_C[1] + 1.0)
    check("solid mast removed by association", mast_on.max() < 25.0,
          f"max {mast_on.max():.1f} m (expect < 25)")

    # 3. The chimney survives.
    ch_on = _box(dm_on, tf, CHIMNEY_C[0] - 0.5, CHIMNEY_C[0] + 0.5,
                 CHIMNEY_C[1] - 0.5, CHIMNEY_C[1] + 0.5)
    check("chimney preserved", ch_on.max() > CHIMNEY_Z - 0.5,
          f"max {ch_on.max():.1f} m (expect ~{CHIMNEY_Z})")

    # 4. The spire survives.
    sp_on = _box(dm_on, tf, SPIRE_C[0] - 2.0, SPIRE_C[0] + 2.0,
                 SPIRE_C[1] - 2.0, SPIRE_C[1] + 2.0)
    check("spire preserved", sp_on.max() > SPIRE_TOP_Z - 2.0,
          f"max {sp_on.max():.1f} m (expect ~{SPIRE_TOP_Z})")

    # 5. The roof under the jib is the measured roof, not an interpolation.
    roof_under = _box(dm_on, tf, 45.0, 60.0, JIB_Y - 1.0, JIB_Y + 1.0)
    valid = roof_under[roof_under != NODATA]
    check("roof under the crane restored",
          abs(np.median(valid) - ROOF_Z) < 0.5,
          f"median {np.median(valid):.2f} m (expect ~{ROOF_Z})")

    # 6. Removal only ever lowers the surface.
    both = (dm_on != NODATA) & (dm_off != NODATA)
    raised = both & (dm_on > dm_off + 1e-3)
    check("surface never raised", not raised.any(),
          f"{int(raised.sum())} px raised")

    # 7. Untouched elsewhere: the filter must be a no-op away from the crane.
    changed = both & (np.abs(dm_on - dm_off) > 1e-3)
    frac = changed.sum() / both.sum()
    check("change confined to the crane", frac < 0.05,
          f"{frac * 100:.2f}% of pixels changed")

    # 8. nodata never raises a false candidate (the +1e6 opening guard).
    holed = dm_off.copy()
    holed[10:40, 10:40] = NODATA
    mask, _, _ = m._detect_thin_tall_structures(holed, NODATA, RES, width_m=4.0)
    r0, r1, c0, c1 = 5, 45, 5, 45
    check("no candidates raised beside a hole", not mask[r0:r1, c0:c1].any(),
          f"{int(mask[r0:r1, c0:c1].sum())} px flagged around the hole")

    print(f"\n{'ALL PASS' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
