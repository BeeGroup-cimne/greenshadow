#!/usr/bin/env python3
"""
Check the vertical and tracking irradiance bases against pvlib.

An open, flat site: every hillshade is the sine of the sun's altitude (a
horizontal surface, nothing casting a shadow) and the sky view factor is 1.
The three bases ``compute_surface_solar_energy`` writes from it - fourier_poa,
vertical_basis and tracking_basis - are read back with their reconstruction
functions and compared, panel by panel, with pvlib run over the same sun
positions and the same irradiance: a clear-sky year at Lleida, so nothing has
to be downloaded.

    python test_tracking_basis.py
"""

import sys
import tempfile
import datetime as dt

import numpy as np
import pandas as pd
import pvlib
import rasterio
from rasterio.transform import from_origin

import greenshadow as gs
from greenshadow.shader import (
    reconstruct_poa_from_basis, reconstruct_tracking_poa, reconstruct_vertical_poa,
    read_sun_table,
)

LAT, LON = 41.66, 0.55
CRS = "EPSG:25831"
SHAPE = (8, 8)
ALBEDO = 0.2


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, label, condition, detail=""):
        if not condition:
            self.failures += 1
        suffix = f"  -- {detail}" if detail else ""
        print(f"  {'PASS' if condition else 'FAIL'}  {label}{suffix}")
        return condition


def clear_sky_table():
    """The annual solar table, with a clear-sky Ineichen year as its weather."""
    table = gs.generate_azimuth_altitude_pairs_by_year(
        year=2023, lat=LAT, lon=LON, tz_name="UTC", granularity="PT1H",
        solar_degrees_discretization=3, only_daylight=True)
    df = table["df"]
    times = pd.DatetimeIndex(pd.to_datetime(df.datetime_utc, utc=True))
    sky = pvlib.location.Location(LAT, LON).get_clearsky(times, model="ineichen")
    df["GHI"], df["DNI"], df["DHI"] = sky.ghi.to_numpy(), sky.dni.to_numpy(), sky.dhi.to_numpy()
    return table


def write_scene(directory, pairs):
    """Flat open ground: hillshade = sin(altitude), SVF = 1."""
    transform = from_origin(300000.0, 4615000.0, 1.0, 1.0)
    profile = dict(driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1,
                   dtype="float32", crs=CRS, transform=transform, nodata=-9999.0)
    with rasterio.open(f"{directory}/svf.tif", "w", **profile) as out:
        out.write(np.ones(SHAPE, np.float32), 1)
    hillshades = f"{directory}/hillshades"
    import os
    os.makedirs(hillshades, exist_ok=True)
    for azimuth, altitude in pairs:
        name = f"{hillshades}/{int(azimuth):03d}_{int(altitude):02d}.tif"
        if os.path.exists(name):
            continue
        with rasterio.open(name, "w", **profile) as out:
            out.write(np.full(SHAPE, np.sin(np.radians(int(altitude))), np.float32), 1)
    return hillshades, f"{directory}/svf.tif"


def reference(df):
    """
    pvlib over the same discretised sun positions and energies the bases use.

    Returns annual POA in Wh/m2 for each panel, isotropic sky, plus the
    single-axis rotation pvlib finds, for the record.
    """
    table = df[df.elevation > 0]
    zenith = 90.0 - table.elevation.astype(int).to_numpy(dtype=float)
    azimuth = table.azimuth.astype(int).to_numpy(dtype=float)
    dni, dhi = table.DNI.to_numpy(), table.DHI.to_numpy()
    ghi = dni * np.cos(np.radians(zenith)) + dhi

    def total(tilt, surface_azimuth):
        poa = pvlib.irradiance.get_total_irradiance(
            tilt, surface_azimuth, zenith, azimuth, dni, ghi, dhi,
            albedo=ALBEDO, model="isotropic")
        return float(np.nansum(poa["poa_global"]))

    tracked = pvlib.tracking.singleaxis(zenith, azimuth, axis_tilt=0, axis_azimuth=180,
                                        max_angle=60, backtrack=False)
    single = total(np.nan_to_num(np.asarray(tracked["surface_tilt"], dtype=float)),
                   np.nan_to_num(np.asarray(tracked["surface_azimuth"], dtype=float)))

    return {
        "fixed_south_30": total(30.0, 180.0),
        "dual_axis": total(zenith, azimuth),
        "single_axis": single,
        "vertical_east": total(90.0, 90.0),
        "vertical_west": total(90.0, 270.0),
        "vertical_south": total(90.0, 180.0),
        "vertical_north": total(90.0, 0.0),
    }


def main():
    report = Report()
    table = clear_sky_table()
    df = table["df"]

    with tempfile.TemporaryDirectory() as directory:
        hillshades, svf = write_scene(directory, table["azimuth_altitude_pairs"])
        paths = {
            mode: gs.compute_surface_solar_energy(
                df, hillshade_dir=hillshades, svf_path=svf,
                out_path=f"{directory}/basis.tif", mode=mode)
            for mode in ("fourier_poa", "vertical_basis", "tracking_basis")
        }

        def centre(path):
            with rasterio.open(path) as src:
                return src.read()[:, SHAPE[0] // 2, SHAPE[1] // 2].astype(float)

        poa_bands = centre(paths["fourier_poa"])
        vertical = centre(paths["vertical_basis"])
        tracking = centre(paths["tracking_basis"])
        sun = read_sun_table(paths["tracking_basis"])

    expected = reference(df)

    ours = {
        "fixed_south_30": reconstruct_poa_from_basis(
            *poa_bands, tilt_deg=30.0, azimuth_deg=180.0, albedo=ALBEDO),
        "dual_axis": sum(reconstruct_tracking_poa(tracking, kind="dual_axis", albedo=ALBEDO)),
        "single_axis": sum(reconstruct_tracking_poa(tracking, kind="single_axis", albedo=ALBEDO)),
    }
    for name, facing in (("vertical_east", 90.0), ("vertical_west", 270.0),
                         ("vertical_south", 180.0), ("vertical_north", 0.0)):
        ours[name] = sum(reconstruct_vertical_poa(
            vertical, poa_bands[3], poa_bands[4], facing_azimuth_deg=facing, albedo=ALBEDO))

    print("1. the bases reproduce pvlib on open flat ground (clear-sky year, Lleida)")
    for name in expected:
        error = ours[name] / expected[name] - 1.0
        print(f"     {name:15s} basis {ours[name] / 1000:7.1f}  pvlib "
              f"{expected[name] / 1000:7.1f} kWh/m2  ({error:+.2%})")
        report.check(f"{name} within 1 % of pvlib", abs(error) < 0.01, f"{error:+.2%}")

    print("2. what the year's beam is worth on each, against a fixed South 30")
    fixed = ours["fixed_south_30"]
    for name in ("single_axis", "dual_axis"):
        print(f"     {name}: {ours[name] / fixed - 1:+.1%}")
    report.check("single axis gains over fixed", ours["single_axis"] > fixed)
    report.check("dual axis gains over single axis", ours["dual_axis"] > ours["single_axis"])
    report.check("an East face takes the morning, a West face the afternoon, "
                 "and they differ from North",
                 ours["vertical_east"] > ours["vertical_north"]
                 and ours["vertical_west"] > ours["vertical_north"])

    print("3. the sun table written with the basis")
    beam = (df.DNI * 1.0)[df.elevation > 0].sum()
    report.check("it carries the year's beam", np.isclose(sun.dni_wh_m2.sum(), beam, rtol=1e-6),
                 f"{sun.dni_wh_m2.sum() / 1000:.1f} vs {beam / 1000:.1f} kWh/m2")
    report.check("one row per sun-position bin",
                 len(sun) == len(set(map(tuple, sun[["azimuth_deg", "altitude_deg"]].values))))

    print()
    if report.failures:
        print(f"{report.failures} FAILURE(S)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
