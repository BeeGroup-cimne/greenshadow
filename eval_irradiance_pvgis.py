#!/usr/bin/env python3
"""
Check the beam/sky split of the weather against PVGIS-SARAH3, for one point
and one year.

    python eval_irradiance_pvgis.py --lat 41.66 --lon 0.55 --year 2023 \
        --weather-dir /mnt/data/greenshadow/weather

ERA5-Land only publishes the global irradiance, so the direct and diffuse
parts the solar tables use are either modelled from it (``dirint``, ``disc``,
``erbs``) or taken from ERA5's own direct radiation (``era5``). PVGIS-SARAH3
is satellite-derived and publishes the split directly, which makes it the
reference here: its horizontal series gives the beam and diffuse on the
ground, and its two-axis-tracker series - a plane that always faces the sun -
gives the DNI.

The weather is loaded with ``load_ERA5Land_with_direct_radiation`` (cached
downloads are reused) and every split is evaluated at the middle of each
ERA5 hour, exactly as ``generate_azimuth_altitude_pairs_by_year`` does,
without the sun-position discretisation. PVGIS is asked to ignore the terrain
horizon, so both sides describe an open sky.

Prints annual and monthly totals and hourly error statistics; ``--out``
writes them as CSV.
"""
import argparse
import datetime as dt
import json
import urllib.request
import warnings

import numpy as np
import pandas as pd
import pvlib

import greenshadow as gs
from greenshadow.shader import _attach_weather

PVGIS = ("https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat={lat}&lon={lon}"
         "&raddatabase=PVGIS-SARAH3&startyear={year}&endyear={year}"
         "&components=1&usehorizon=0&outputformat=json&{plane}")

SPLITS = ("era5", "dirint", "disc", "erbs")


def pvgis_series(lat, lon, year, plane):
    with urllib.request.urlopen(PVGIS.format(lat=lat, lon=lon, year=year, plane=plane),
                                timeout=120) as response:
        hourly = pd.DataFrame(json.load(response)["outputs"]["hourly"])
    hourly.index = pd.to_datetime(hourly.time, format="%Y%m%d:%H%M").dt.tz_localize("UTC")
    return hourly


def sarah3(lat, lon, year):
    """GHI, beam on the horizontal, DHI and DNI in W/m2, as PVGIS stamps them."""
    flat = pvgis_series(lat, lon, year, "angle=0&aspect=0")
    tracked = pvgis_series(lat, lon, year, "trackingtype=2")
    return pd.DataFrame({
        "GHI": flat["Gb(i)"] + flat["Gd(i)"],
        "beam_h": flat["Gb(i)"],
        "DHI": flat["Gd(i)"],
        "DNI": tracked["Gb(i)"],
    })


def era5_splits(lat, lon, year, weather_dir):
    """The same four quantities from ERA5-Land, under every split."""
    weather = gs.load_ERA5Land_with_direct_radiation(
        data_dir=weather_dir, lat_range=[lat, lat], lon_range=[lon, lon],
        ym_range=[dt.datetime(year, 1, 1), dt.datetime(year, 12, 31)],
        tz_local="UTC", update=False, use_point_location=True,
        mode="instantaneous").to_pandas()

    stamps = pd.DatetimeIndex(pd.to_datetime(weather.time)).tz_convert("UTC")
    middle = stamps - pd.Timedelta(minutes=30)
    middle = middle[middle.year == year]

    elevation = pvlib.solarposition.get_solarposition(middle, lat, lon).apparent_elevation
    sine = np.clip(np.sin(np.radians(elevation.to_numpy())), 0.0, None)

    out = {}
    for split in SPLITS:
        table = pd.DataFrame({"datetime_utc": middle})
        _attach_weather(table, weather, lat=lat, lon=lon,
                        decomposition=split, time_offset_minutes=-30.0)
        out[split] = pd.DataFrame({
            "GHI": table.GHI.to_numpy(),
            "beam_h": table.DNI.to_numpy() * sine,
            "DHI": table.DHI.to_numpy(),
            "DNI": table.DNI.to_numpy(),
        }, index=middle).fillna(0.0)
    return out


def hourly_errors(reference, estimate):
    """
    MBE, RMSE, relative RMSE and correlation over daylight hours.

    SARAH3 is an instant a few minutes past each hour, ERA5 an hourly mean, so
    both are filed under the hour they fall in; part of the scatter is that
    mismatch and the clouds moving within the hour, not the split.
    """
    a = reference.copy()
    b = estimate.copy()
    a.index = a.index.floor("h")
    b.index = b.index.floor("h")
    joined = a.join(b, lsuffix="_ref", rsuffix="_est", how="inner").dropna()
    joined = joined[(joined.GHI_ref > 20.0) | (joined.GHI_est > 20.0)]

    rows = {}
    for component in ("GHI", "DNI", "DHI"):
        diff = joined[f"{component}_est"] - joined[f"{component}_ref"]
        rows[component] = {
            "MBE_W_m2": diff.mean(),
            "RMSE_W_m2": float(np.sqrt((diff ** 2).mean())),
            "rRMSE": float(np.sqrt((diff ** 2).mean())) / joined[f"{component}_ref"].mean(),
            "r": float(np.corrcoef(joined[f"{component}_est"], joined[f"{component}_ref"])[0, 1]),
        }
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--weather-dir", required=True)
    parser.add_argument("--out", help="prefix for the CSV outputs")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    reference = sarah3(args.lat, args.lon, args.year)
    splits = era5_splits(args.lat, args.lon, args.year, args.weather_dir)

    annual = pd.DataFrame(
        {"PVGIS-SARAH3": reference.sum() / 1000.0,
         **{f"ERA5-Land + {name}": frame.sum() / 1000.0 for name, frame in splits.items()}}
    ).T[["GHI", "beam_h", "DHI", "DNI"]]
    annual["diffuse_fraction"] = annual.DHI / annual.GHI
    annual["DNI_vs_SARAH3"] = annual.DNI / annual.DNI.iloc[0] - 1.0
    annual["DHI_vs_SARAH3"] = annual.DHI / annual.DHI.iloc[0] - 1.0

    monthly = pd.DataFrame(
        {"SARAH3": reference.DNI.resample("ME").sum() / 1000.0,
         **{name: frame.DNI.resample("ME").sum() / 1000.0 for name, frame in splits.items()}})
    monthly.index = monthly.index.strftime("%b")

    errors = pd.concat({name: hourly_errors(reference, frame)
                        for name, frame in splits.items()})

    print(f"Annual totals, kWh/m2 - {args.lat}, {args.lon}, {args.year}")
    print(annual.round(3).to_string())
    print("\nMonthly DNI, kWh/m2")
    print(monthly.round(0).to_string())
    print("\nHourly agreement with SARAH3, daylight hours")
    print(errors.round(3).to_string())

    if args.out:
        annual.to_csv(f"{args.out}_annual.csv")
        monthly.to_csv(f"{args.out}_monthly_dni.csv")
        errors.to_csv(f"{args.out}_hourly.csv")


if __name__ == "__main__":
    main()
