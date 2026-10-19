#!/usr/bin/env python3
"""
Test script for the parcel PV and agrivoltaic scenarios.

Runs on a synthetic plane-of-array basis written to a temporary directory: a
uniform sky, a South-leaning sun, nothing to download.

    python test_parcel_pv.py
"""

import sys
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import greenshadow as gs
from greenshadow.parcel_pv import (
    _no_shading_pitch, DEFAULT_SYSTEMS, project_cashflows, Economics,
)
from greenshadow.dimensionate import DEFAULT_MODULES


RESOLUTION = 0.5
CRS = "EPSG:25831"
X0, Y0 = 300000.0, 4615000.0            # Lleida plain, latitude about 41.7
SHAPE = (300, 300)                       # 150 x 150 m

# Annual Wh/m2 of an open site there: the beam leans South (B_north < 0) and
# is symmetric East-West, diffuse about a third of global.
BASIS = (1.20e6, 0.0, -0.60e6, 0.60e6, 1.80e6)


def write_basis(directory, *, hole=None):
    """
    The five fourier_poa bands, uniform, optionally with a nodata hole.
    """
    path = f"{directory}/basis_fourier_poa.tif"
    bands = np.stack([np.full(SHAPE, value, np.float32) for value in BASIS])
    if hole is not None:
        bands[:, hole[0], hole[1]] = -9999.0
    with rasterio.open(path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1],
                       count=5, dtype="float32", crs=CRS, nodata=-9999.0,
                       transform=from_origin(X0, Y0, RESOLUTION, RESOLUTION)) as out:
        out.write(bands)
    return path


def parcels(polygons, usos):
    n = len(polygons)
    return gpd.GeoDataFrame({
        "provincia": [25] * n, "municipio": [23] * n, "agregado": [0] * n,
        "zona": [0] * n, "poligono": list(range(1, n + 1)),
        "parcela": list(range(1, n + 1)), "recinto": [1] * n,
        "uso_sigpac": list(usos),
    }, geometry=list(polygons), crs=CRS)


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, label, condition, detail=""):
        if not condition:
            self.failures += 1
        suffix = f"  -- {detail}" if detail else ""
        print(f"  {'PASS' if condition else 'FAIL'}  {label}{suffix}")
        return condition


def main():
    report = Report()
    field = box(X0 + 10, Y0 - 140, X0 + 140, Y0 - 10)      # 130 x 130 m

    with tempfile.TemporaryDirectory() as directory:
        basis = write_basis(directory)

        print("1. rows run the way they say, inside the setback, at their pitch")
        scenarios, best, rows = gs.plan_parcel_pv_scenarios(
            parcels([field], ["TA"]), poa_basis_path=basis, rows="all")
        ns = rows[rows.scenario == "apv_interspace_NS_10.0m_cereal"]
        ew = rows[rows.scenario == "apv_interspace_EW_10.0m_cereal"]
        ns_x = np.array([np.asarray(g.coords)[:, 0] for g in ns.geometry])
        ew_y = np.array([np.asarray(g.coords)[:, 1] for g in ew.geometry])
        print(f"     {len(scenarios)} scenarios; NS rows {len(ns)}, EW rows {len(ew)}")
        report.check("North-South rows have constant x",
                     len(ns) > 0 and np.allclose(ns_x[:, 0], ns_x[:, 1]))
        report.check("East-West rows have constant y",
                     len(ew) > 0 and np.allclose(ew_y[:, 0], ew_y[:, 1]))
        report.check("rows are 10 m apart",
                     np.allclose(np.diff(np.sort(ns_x[:, 0])), 10.0, atol=RESOLUTION))
        inner = field.buffer(-5.0 + RESOLUTION)
        report.check("every row stays inside the 5 m setback",
                     all(inner.contains(g) for g in rows.geometry))
        report.check("rows run the length of the parcel, less the setbacks",
                     abs(ns.length_m.median() - 120.0) <= 2 * RESOLUTION,
                     f"{ns.length_m.median():.1f} m")

        print("2. geometry and energy of the PV field")
        fixed = DEFAULT_SYSTEMS[0]
        pitch = _no_shading_pitch(fixed, "EW", DEFAULT_MODULES[0], 41.7)
        south = scenarios.set_index("scenario").loc["pv_fixed_EW_8.9m"]
        tent = scenarios[(scenarios.system == "pv_fixed")
                         & (scenarios.direction == "NS")].iloc[0]
        print(f"     no-shading pitch {pitch:.2f} m; specific yield South "
              f"{south.specific_yield_kwh_kwp:.0f}, tent {tent.specific_yield_kwh_kwp:.0f}"
              f" kWh/kWp; cover South {south.ground_cover_ratio:.2f}, tent "
              f"{tent.ground_cover_ratio:.2f}")
        report.check("no-shading pitch clears the solstice shadow",
                     8.5 < pitch < 9.5, f"{pitch:.2f} m")
        report.check("South-facing rows yield more per kWp than the tent",
                     south.specific_yield_kwh_kwp > tent.specific_yield_kwh_kwp)
        report.check("the tent packs the ground tighter",
                     tent.ground_cover_ratio > south.ground_cover_ratio)
        expected = (BASIS[0] * np.cos(np.radians(30))
                    - BASIS[2] * np.sin(np.radians(30))
                    + BASIS[3] * (1 + np.cos(np.radians(30))) / 2
                    + BASIS[4] * 0.2 * (1 - np.cos(np.radians(30))) / 2) / 1000.0
        module = DEFAULT_MODULES[0]
        per_kwp = expected * module.area_m2 * 0.21 * 0.80 / (module.area_m2 * 0.21)
        report.check("specific yield is POA times the performance ratio",
                     abs(south.specific_yield_kwh_kwp - per_kwp) < 1.0,
                     f"{south.specific_yield_kwh_kwp:.1f} vs {per_kwp:.1f}")

        print("3. the crop answers to the shade")
        overhead = scenarios[(scenarios.system == "apv_overhead")
                             & (scenarios.crop == "forage")
                             & (scenarios.direction == "EW")].set_index("pitch_m")
        print(f"     overhead forage relative yield: "
              f"{overhead.relative_crop_yield.round(2).to_dict()}")
        report.check("denser overhead rows cost the crop more",
                     overhead.relative_crop_yield.loc[6.0]
                     < overhead.relative_crop_yield.loc[8.0])
        fields = scenarios[scenarios.kind == "pv_field"]
        report.check("a PV field keeps no crop",
                     (fields.agri_margin_eur_year == 0).all()
                     and (fields.land_loss_frac == 1.0).all())

        print("4. cash flows")
        flat = Economics(discount_rate=0.0, lifetime_years=10,
                         degradation_per_year=0.0, electricity_price_eur_mwh=100.0)
        flows = project_cashflows(peak_kwp=1.0, energy_mwh=1.0, capex_eur=1000.0,
                                  opex_eur_year=0.0, agri_delta_eur_year=0.0,
                                  economics=flat)
        print(f"     npv={flows['npv_eur']:.2f} irr={flows['irr']:.4f} "
              f"roi={flows['roi']:.3f} payback={flows['payback_years']:.2f}")
        report.check("break-even at zero discount", abs(flows["npv_eur"]) < 1e-6
                     and abs(flows["irr"]) < 1e-4 and abs(flows["roi"]) < 1e-9
                     and abs(flows["payback_years"] - 10.0) < 1e-9)
        annuity = Economics(discount_rate=0.05, lifetime_years=20,
                            degradation_per_year=0.0, electricity_price_eur_mwh=100.0)
        flows = project_cashflows(peak_kwp=1.0, energy_mwh=1.0, capex_eur=1000.0,
                                  opex_eur_year=20.0, agri_delta_eur_year=0.0,
                                  economics=annuity)
        factor = (1 - 1.05 ** -20) / 0.05
        report.check("NPV is the annuity of the net flow",
                     abs(flows["npv_eur"] - (80.0 * factor - 1000.0)) < 1e-6)
        report.check("LCOE is cost over discounted energy",
                     abs(flows["lcoe_eur_mwh"] - (1000.0 + 20.0 * factor) / factor) < 1e-6)
        report.check("IRR zeroes the NPV",
                     abs(sum(flows["cashflows"] / (1 + flows["irr"]) ** np.arange(21))) < 1e-3)

        print("5. what the parcel earns today is paid for")
        today = scenarios.set_index("scenario").loc["pv_fixed_EW_8.9m"]
        nothing, _, _ = gs.plan_parcel_pv_scenarios(
            parcels([field], ["TA"]), poa_basis_path=basis,
            baseline_margin_eur_ha={}, rows=None)
        free = nothing.set_index("scenario").loc["pv_fixed_EW_8.9m"]
        economics = Economics()
        factor = (1 - (1 + economics.discount_rate) ** -economics.lifetime_years) \
            / economics.discount_rate
        lost = 300.0 * field.area / 10_000.0 * factor
        print(f"     NPV over cereal {today.npv_eur:.0f}, over nothing "
              f"{free.npv_eur:.0f}, difference {free.npv_eur - today.npv_eur:.0f}")
        report.check("the cereal margin given up is in the NPV",
                     abs((free.npv_eur - today.npv_eur) - lost) < 1.0)

        print("6. the criterion picks the winner")
        for criterion, column, bigger in (("npv", "npv_eur", True),
                                          ("lcoe", "lcoe_eur_mwh", False),
                                          ("energy", "energy_mwh", True),
                                          ("ler", "ler", True)):
            _, chosen, _ = gs.plan_parcel_pv_scenarios(
                parcels([field], ["TA"]), poa_basis_path=basis,
                criterion=criterion, rows=None)
            winner = chosen.best_scenario.iloc[0]
            values = scenarios.set_index("scenario")[column]
            target = values.idxmax() if bigger else values.idxmin()
            report.check(f"{criterion} -> {winner}",
                         np.isclose(values[winner], values[target]))
        report.check("viability says whether the best beats the parcel as it is",
                     bool(best.best_viable.iloc[0])
                     == bool(best.best_npv_eur.iloc[0] > 0))
        report.check("best PV field and best agrivoltaic both reported",
                     isinstance(best.pv_field_scenario.iloc[0], str)
                     and isinstance(best.agrivoltaic_scenario.iloc[0], str))

        print("7. which parcels are laid out")
        small = box(X0 + 10, Y0 - 40, X0 + 40, Y0 - 10)        # 900 m2
        _, chosen, _ = gs.plan_parcel_pv_scenarios(
            parcels([field, small, field], ["TA", "TA", "ZU"]),
            poa_basis_path=basis, rows=None)
        report.check("small and non-agricultural parcels are skipped",
                     len(chosen) == 1)

        print("8. rows are cut where the raster has no data")
        holed = write_basis(directory, hole=(slice(None), slice(140, 150)))
        _, _, cut = gs.plan_parcel_pv_scenarios(
            parcels([field], ["TA"]), poa_basis_path=holed, rows="all")
        across = cut[cut.scenario == "apv_interspace_EW_10.0m_cereal"]
        gap_x = (X0 + 140 * RESOLUTION, X0 + 150 * RESOLUTION)
        crossing = [g for g in across.geometry
                    if min(g.coords[0][0], g.coords[-1][0]) < gap_x[1]
                    and max(g.coords[0][0], g.coords[-1][0]) > gap_x[0]]
        report.check("no East-West row crosses the nodata band",
                     len(across) > 0 and not crossing,
                     f"{len(crossing)} of {len(across)}")

        print("9. agrivoltaics built into the rows already planted")
        from shapely.geometry import LineString

        def planted(azimuth_deg, spacing=7.0, width=2.0, height=3.0):
            """Crop rows across the field, as detect_crop_rows reports them."""
            theta = np.radians(azimuth_deg)
            along = np.array([np.sin(theta), np.cos(theta)])
            across = np.array([np.cos(theta), -np.sin(theta)])
            centre = np.array(field.centroid.coords[0])
            lines = [
                LineString([centre + across * k * spacing - along * 80.0,
                            centre + across * k * spacing + along * 80.0])
                .intersection(field.buffer(-2.0))
                for k in range(-12, 13)
            ]
            lines = [line for line in lines if line.length > 10.0]
            return gpd.GeoDataFrame({
                "parcel_id": list(sigpac_ids)[0], "block_index": 0,
                "row_azimuth_deg": azimuth_deg, "row_spacing_m": spacing,
                "canopy_width_m": width, "canopy_height_p90_m": height,
            }, index=range(len(lines)), geometry=lines, crs=CRS), lines

        from greenshadow.rural_parcels import sigpac_parcel_id
        sigpac_ids = sigpac_parcel_id(parcels([field], ["FY"]))
        crop_rows, crop_lines = planted(80.0)

        scenarios, chosen, laid = gs.plan_parcel_pv_scenarios(
            parcels([field], ["FY"]), poa_basis_path=basis,
            crop_rows=crop_rows, rows="all")
        mine = scenarios[scenarios.direction == "rows"].set_index("scenario")
        print(f"     without a vegetated basis: {sorted(mine.index)}")
        report.check("overhead scenarios over the rows, one per step",
                     {"apv_rows_overhead_k1", "apv_rows_overhead_k2",
                      "apv_rows_overhead_k3"} <= set(mine.index))
        report.check("no alley scenario without the vegetated basis",
                     not any(name.startswith("apv_rows_alley") for name in mine.index))
        report.check("always agrivoltaic, keeping the crop already there",
                     (mine.kind == "agrivoltaic").all()
                     and (mine.crop == "existing_FY").all())

        over = laid[laid.scenario == "apv_rows_overhead_k1"]
        bearings = [np.degrees(np.arctan2(*np.diff(np.asarray(g.coords)[[0, -1]], axis=0)[0])) % 180
                    for g in over.geometry]
        distance = max(min(g.distance(row) for row in crop_lines) for g in over.geometry)
        print(f"     overhead k1: {len(over)} PV rows over {len(crop_lines)} crop rows, "
              f"bearing {np.mean(bearings):.1f}, furthest {distance:.2f} m from a crop row")
        report.check("overhead rows lie on the crop rows, parallel to them",
                     len(over) > 0 and distance < 0.01
                     and np.allclose(bearings, 80.0, atol=0.5))
        report.check("every second row at k=2",
                     abs(mine.at["apv_rows_overhead_k2", "n_rows"]
                         - mine.at["apv_rows_overhead_k1", "n_rows"] / 2) <= 1)
        report.check("the table faces the side of the rows nearer South",
                     mine.at["apv_rows_overhead_k1", "tilt_deg"] == 20.0)
        report.check("clearance needed is the canopy plus a metre",
                     np.isclose(mine.at["apv_rows_overhead_k1", "required_clearance_m"], 4.0)
                     and bool(mine.at["apv_rows_overhead_k1", "clearance_ok"]))
        report.check("denser overhead rows shade the crop more",
                     mine.at["apv_rows_overhead_k1", "relative_crop_yield"]
                     <= mine.at["apv_rows_overhead_k3", "relative_crop_yield"])
        report.check("the best built into the rows is reported",
                     str(chosen.existing_rows_scenario.iloc[0]).startswith("apv_rows_"))

        scenarios, _, laid = gs.plan_parcel_pv_scenarios(
            parcels([field], ["FY"]), poa_basis_path=basis,
            crop_rows=crop_rows, vegetated_basis_path=basis, rows="all")
        alley = laid[laid.scenario == "apv_rows_alley_k2"]
        gaps = [min(g.distance(row) for row in crop_lines) for g in alley.geometry]
        print(f"     alley k2: {len(alley)} PV rows, {np.min(gaps):.2f}-{np.max(gaps):.2f} m "
              f"from the nearest crop row")
        report.check("alley rows run down the middle of the alleys",
                     len(alley) > 0 and np.allclose(gaps, 3.5, atol=0.05))

        narrow, _ = planted(80.0, spacing=4.0, width=2.5)
        scenarios, _, _ = gs.plan_parcel_pv_scenarios(
            parcels([field], ["FY"]), poa_basis_path=basis,
            crop_rows=narrow, vegetated_basis_path=basis, rows=None)
        report.check("an alley too narrow for the table takes none",
                     not scenarios.scenario.str.startswith("apv_rows_alley").any())

        north_south, _ = planted(5.0)
        scenarios, _, _ = gs.plan_parcel_pv_scenarios(
            parcels([field], ["FY"]), poa_basis_path=basis,
            crop_rows=north_south, rows=None)
        tilt = scenarios.set_index("scenario").at["apv_rows_overhead_k1", "tilt_deg"]
        report.check("North-South rows get the East-West tent", tilt == 10.0,
                     f"tilt {tilt}")

        print("10. trackers and vertical bifacial fences, on bases built from a clear sky")
        import os
        import pvlib
        table = gs.generate_azimuth_altitude_pairs_by_year(
            year=2023, lat=41.66, lon=0.55, tz_name="UTC", granularity="PT1H",
            solar_degrees_discretization=3, only_daylight=True)
        df = table["df"]
        times = pd.DatetimeIndex(pd.to_datetime(df.datetime_utc, utc=True))
        sky = pvlib.location.Location(41.66, 0.55).get_clearsky(times, model="ineichen")
        df["GHI"], df["DNI"], df["DHI"] = sky.ghi.to_numpy(), sky.dni.to_numpy(), sky.dhi.to_numpy()

        profile = dict(driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1,
                       dtype="float32", crs=CRS, nodata=-9999.0,
                       transform=from_origin(X0, Y0, RESOLUTION, RESOLUTION))
        with rasterio.open(f"{directory}/svf.tif", "w", **profile) as out:
            out.write(np.ones(SHAPE, np.float32), 1)
        os.makedirs(f"{directory}/hillshades", exist_ok=True)
        for azimuth, altitude in table["azimuth_altitude_pairs"]:
            name = f"{directory}/hillshades/{int(azimuth):03d}_{int(altitude):02d}.tif"
            if not os.path.exists(name):
                with rasterio.open(name, "w", **profile) as out:
                    out.write(np.full(SHAPE, np.sin(np.radians(int(altitude))), np.float32), 1)
        bases = {
            mode: gs.compute_surface_solar_energy(
                df, hillshade_dir=f"{directory}/hillshades", svf_path=f"{directory}/svf.tif",
                out_path=f"{directory}/clear.tif", mode=mode)
            for mode in ("fourier_poa", "vertical_basis", "tracking_basis")
        }

        scenarios, _, _ = gs.plan_parcel_pv_scenarios(
            parcels([field], ["TA"]), poa_basis_path=bases["fourier_poa"],
            vertical_basis_path=bases["vertical_basis"],
            tracking_basis_path=bases["tracking_basis"], rows=None)
        by = scenarios.set_index("scenario")
        fixed = by.loc["pv_fixed_EW_8.9m"]
        single = scenarios[scenarios.system == "pv_single_axis"].iloc[0]
        dual = scenarios[scenarios.system == "pv_dual_axis"].iloc[0]
        print(f"     specific yield kWh/kWp: fixed {fixed.specific_yield_kwh_kwp:.0f}, "
              f"single axis {single.specific_yield_kwh_kwp:.0f} "
              f"(backtracking {single.row_shading_factor:.3f}), dual axis "
              f"{dual.specific_yield_kwh_kwp:.0f} (shading {dual.row_shading_factor:.3f})")
        report.check("trackers turn on North-South rows only",
                     set(scenarios[scenarios.mounting.isin(["single_axis", "dual_axis"])]
                         .direction) == {"NS"})
        report.check("single axis beats fixed per kWp, dual axis beats both",
                     fixed.specific_yield_kwh_kwp < single.specific_yield_kwh_kwp
                     < dual.specific_yield_kwh_kwp)
        report.check("the gains are in the published range (+10-30 %, +25-45 %)",
                     1.10 < single.specific_yield_kwh_kwp / fixed.specific_yield_kwh_kwp < 1.30
                     and 1.25 < dual.specific_yield_kwh_kwp / fixed.specific_yield_kwh_kwp < 1.45)
        report.check("backtracking and mutual shading cost something, not everything",
                     0.8 < single.row_shading_factor < 1.0 and 0.8 < dual.row_shading_factor < 1.0)

        wide = scenarios[(scenarios.system == "apv_single_axis") & (scenarios.pitch_m == 12.0)]
        report.check("a wider agrivoltaic pitch backtracks less than a PV field",
                     len(wide) and wide.row_shading_factor.iloc[0] > single.row_shading_factor)

        fences = scenarios[scenarios.system == "apv_vertical_bifacial"]
        ns = fences[(fences.direction == "NS") & (fences.crop == "forage")].set_index("pitch_m")
        ew = fences[(fences.direction == "EW") & (fences.crop == "forage")].set_index("pitch_m")
        print(f"     vertical forage, 8 / 10 m: East-West faces "
              f"{ns.specific_yield_kwh_kwp.loc[8.0]:.0f} / {ns.specific_yield_kwh_kwp.loc[10.0]:.0f}, "
              f"South-North faces {ew.specific_yield_kwh_kwp.loc[8.0]:.0f} / "
              f"{ew.specific_yield_kwh_kwp.loc[10.0]:.0f} kWh/kWp; light kept by the crop "
              f"{1 - ns.radiation_reduction.loc[8.0]:.2f} / {1 - ns.radiation_reduction.loc[10.0]:.2f}")
        report.check("fences closer together shade each other more",
                     ns.row_shading_factor.loc[8.0] < ns.row_shading_factor.loc[10.0])
        report.check("the crop keeps about 75-80 % of its light at 8-10 m",
                     0.72 < 1 - ns.radiation_reduction.loc[8.0] < 0.80
                     and 0.78 < 1 - ns.radiation_reduction.loc[10.0] < 0.84)
        report.check("every fence layout yields energy",
                     (fences.energy_mwh > 0).all())

    print()
    if report.failures:
        print(f"{report.failures} FAILURE(S)")
        return 1

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
