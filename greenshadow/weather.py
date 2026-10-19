# -*- coding: utf-8 -*-
"""ERA5-Land weather gathering, reused from ``hypercadaster_ES.weather``.

greenshadow no longer maintains its own copy of the ERA5-Land downloader: the
hypercadaster_ES implementation is a superset of it (ARCO analysis-ready Zarr
priority path with cdsapi fallback, GRIB de-accumulation of GHI/precipitation,
province-scale cache tiles, surface-pressure backfill) and returns the same
canonical hourly polars frame consumed by :mod:`greenshadow.shader`.

What it does not carry is how the sunshine splits between beam and sky.
ERA5-Land only publishes the global ``ssrd``; the direct part has to be
modelled from it, and any model has to guess the clouds from the GHI alone.
ERA5 itself - the 0.25 degree reanalysis ERA5-Land is downscaled from - also
publishes ``fdir``, the all-sky direct radiation its own clouds let through,
and ``tcc``, the cloud cover. This module adds them, as the point time series
the Climate Data Store serves for ERA5 single levels, in the same shape as
hypercadaster's ERA5-Land time series:

* :func:`download_ERA5_radiation` fetches ``ssrd``, ``fdir`` and ``tcc`` for
  one point and a date range into a CSV cache.
* :func:`load_ERA5_radiation` reads it back as hourly W/m2.
* :func:`load_ERA5Land_with_direct_radiation` is ``load_ERA5Land`` plus those
  columns, ready for ``irradiance_decomposition="era5"`` in the solar tables.

Docs: https://github.com/BeeGroup-cimne/hypercadaster_ES
"""
import datetime as dt
import os
import shutil
import sys
import zipfile

import pandas as pd
import polars as pl

from hypercadaster_ES.weather import (
    cleaning_pipe,
    transformation_pipe,
    load_ERA5Land,
    download_ERA5Land,
    ERA5Land_variables,
    closest_station_with_data,
    station_timeseries,
    weather_statistics,
    get_municipality_weather,
    write_epw,
    municipality_epw,
)

__all__ = [
    "cleaning_pipe",
    "transformation_pipe",
    "load_ERA5Land",
    "download_ERA5Land",
    "ERA5Land_variables",
    "closest_station_with_data",
    "station_timeseries",
    "weather_statistics",
    "get_municipality_weather",
    "write_epw",
    "municipality_epw",
    "ERA5_RADIATION_VARIABLES",
    "download_ERA5_radiation",
    "load_ERA5_radiation",
    "load_ERA5Land_with_direct_radiation",
]


#: ERA5 single-level variables fetched for the beam/sky split, and the column
#: each becomes. ``ssrd`` is fetched alongside ``fdir`` so the beam fraction
#: is taken between two fields of the same model on the same cell.
ERA5_RADIATION_VARIABLES = (
    {"CDSLongName": "surface_solar_radiation_downwards",
     "CDSShortName": "ssrd", "name": "GHI_ERA5"},
    {"CDSLongName": "total_sky_direct_solar_radiation_at_surface",
     "CDSShortName": "fdir", "name": "directHorizontal"},
    {"CDSLongName": "total_cloud_cover",
     "CDSShortName": "tcc", "name": "totalCloudCover"},
)

_ERA5_TIMESERIES = "reanalysis-era5-single-levels-timeseries"

#: ERA5 is published about five days behind real time.
_ERA5_LAG_DAYS = 5


def _era5_cell(value):
    """Snap a coordinate to the 0.25 degree ERA5 grid."""
    return round(round(float(value) * 4.0) / 4.0, 2)


def _era5_filename(data_dir, start, end, lat, lon):
    return f"{data_dir}/era5_radiation_{start:%Y-%m-%d}_{end:%Y-%m-%d}_{lat:.2f}_{lon:.2f}.csv"


def _cached_era5(data_dir, start, end, lat, lon):
    """An existing cache file for this cell that covers ``start``..``end``."""
    if not os.path.isdir(data_dir):
        return None

    for name in sorted(os.listdir(data_dir)):
        if not (name.startswith("era5_radiation_") and name.endswith(".csv")):
            continue
        try:
            _, _, first, last, cell_lat, cell_lon = name[:-4].split("_")
            first = dt.date.fromisoformat(first)
            last = dt.date.fromisoformat(last)
        except ValueError:
            continue
        if (abs(float(cell_lat) - lat) < 1e-6 and abs(float(cell_lon) - lon) < 1e-6
                and first <= start and last >= end):
            return os.path.join(data_dir, name)

    return None


def _requested_dates(ym_range):
    """First and last day of ``ym_range``, the last held back to ERA5's lag."""
    start = min(ym_range)
    end = max(ym_range)
    start = start.date() if isinstance(start, dt.datetime) else start
    end = end.date() if isinstance(end, dt.datetime) else end
    latest = dt.date.today() - dt.timedelta(days=_ERA5_LAG_DAYS)
    return start, min(end, latest)


def download_ERA5_radiation(data_dir, ym_range, lat, lon, *, update=False):
    """
    Download ERA5 ``ssrd``, ``fdir`` and ``tcc`` for one point as a CSV.

    Uses the Climate Data Store's ``reanalysis-era5-single-levels-timeseries``
    point product - one request, one small CSV - and needs the same
    ``~/.cdsapirc`` as the ERA5-Land download. The point is snapped to the
    0.25 degree ERA5 grid, so municipalities on the same cell share a file.
    A cached file covering the period is reused unless ``update``.

    Parameters
    ----------
    data_dir : str
        Cache directory; may be the ERA5-Land one, the names do not collide.
    ym_range : sequence of date or datetime
        Period to cover, from its earliest to its latest day.
    lat, lon : float
        Point of interest, degrees.

    Returns
    -------
    str
        Path of the CSV: ``valid_time`` plus the raw ``ssrd`` and ``fdir`` in
        J/m2 accumulated over the hour ending at ``valid_time``, and ``tcc``
        as a fraction.
    """
    import cdsapi

    start, end = _requested_dates(ym_range)
    lat, lon = _era5_cell(lat), _era5_cell(lon)

    cached = None if update else _cached_era5(data_dir, start, end, lat, lon)
    if cached is not None:
        return cached

    os.makedirs(data_dir, exist_ok=True)
    filename = _era5_filename(data_dir, start, end, lat, lon)
    archive = filename.replace(".csv", ".zip")

    sys.stderr.write(f"\tDownloading ERA5 direct radiation for {lat}, {lon}, "
                     f"{start} to {end}\n")

    client = cdsapi.Client()
    client.retrieve(_ERA5_TIMESERIES, {
        "variable": [item["CDSLongName"] for item in ERA5_RADIATION_VARIABLES],
        "location": {"longitude": lon, "latitude": lat},
        "date": [f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"],
        "data_format": "csv",
    }).download(archive)

    scratch = f"{data_dir}/tmp_era5_radiation"
    os.makedirs(scratch, exist_ok=True)

    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.endswith(".csv")]
            bundle.extractall(scratch)

        # The service may split the variables across files; they share the
        # time and the cell, and are joined on them.
        combined = None
        for member in members:
            frame = pd.read_csv(os.path.join(scratch, member))
            if combined is None:
                combined = frame
            else:
                keys = [key for key in ("valid_time", "latitude", "longitude")
                        if key in frame.columns and key in combined.columns]
                combined = combined.merge(frame, on=keys, how="outer")

        if combined is None:
            raise RuntimeError(f"{archive} holds no CSV")

        combined.to_csv(filename, index=False)

    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if os.path.exists(archive):
            os.remove(archive)

    return filename


def load_ERA5_radiation(data_dir, lat, lon, ym_range, *, update=False):
    """
    ERA5 global, direct and cloud cover for one point, hourly.

    Downloads first when nothing cached covers the period (see
    :func:`download_ERA5_radiation`).

    Returns
    -------
    pandas.DataFrame
        ``time`` (UTC, the ERA5 stamp: the **end** of the hour the radiation
        was accumulated over), ``GHI_ERA5`` and ``directHorizontal`` in W/m2
        averaged over that hour, ``totalCloudCover`` from 0 to 1, and the
        cell's ``latitude`` and ``longitude``.
    """
    path = download_ERA5_radiation(data_dir, ym_range, lat, lon, update=update)
    frame = pd.read_csv(path)

    frame["time"] = pd.to_datetime(frame["valid_time"]).dt.tz_localize("UTC")
    start, end = _requested_dates(ym_range)
    frame = frame[(frame["time"].dt.date >= start) & (frame["time"].dt.date <= end)]

    out = pd.DataFrame({"time": frame["time"].to_numpy()})
    for item in ERA5_RADIATION_VARIABLES:
        values = frame[item["CDSShortName"]].to_numpy(dtype=float)
        if item["CDSShortName"] in ("ssrd", "fdir"):
            values = values / 3600.0          # J/m2 over an hour -> mean W/m2
        out[item["name"]] = values.clip(min=0.0)

    for column in ("latitude", "longitude"):
        if column in frame:
            out[column + "_ERA5"] = frame[column].to_numpy()

    return out.reset_index(drop=True)


def load_ERA5Land_with_direct_radiation(data_dir, lat_range, lon_range, ym_range,
                                        tz_local, *, era5_data_dir=None,
                                        update=False, **kwargs):
    """
    ``load_ERA5Land``, with ERA5's own direct radiation and cloud cover joined.

    Everything ``load_ERA5Land`` returns is kept as it is - its 0.1 degree GHI
    stays the GHI. What is added, per hour, is ERA5's ``directHorizontal``,
    the ``GHI_ERA5`` it is a share of, and ``totalCloudCover``, taken at the
    centre of ``lat_range`` / ``lon_range``. With
    ``irradiance_decomposition="era5"`` the solar tables then split the
    ERA5-Land GHI by ERA5's own beam fraction instead of modelling it.

    ``era5_data_dir`` defaults to ``data_dir``; ``kwargs`` go to
    ``load_ERA5Land``. If the ERA5 download fails the ERA5-Land frame is
    returned without the extra columns and a warning is written, so the solar
    tables fall back to a modelled split rather than stopping the run.
    """
    weather = load_ERA5Land(
        data_dir=data_dir, lat_range=lat_range, lon_range=lon_range,
        ym_range=ym_range, tz_local=tz_local, update=update, **kwargs)

    lat = (min(lat_range) + max(lat_range)) / 2.0
    lon = (min(lon_range) + max(lon_range)) / 2.0

    try:
        radiation = load_ERA5_radiation(
            era5_data_dir or data_dir, lat, lon, ym_range, update=update)
    except Exception as error:  # the CDS is a network service; degrade, do not die
        sys.stderr.write(f"\tWarning: ERA5 direct radiation unavailable ({error}); "
                         f"the beam/sky split will be modelled from GHI\n")
        return weather

    radiation = pl.from_pandas(
        radiation[["time", "GHI_ERA5", "directHorizontal", "totalCloudCover"]]
    ).with_columns(pl.col("time").dt.cast_time_unit("us").dt.convert_time_zone("UTC"))

    time_dtype = weather.schema["time"]
    radiation = radiation.with_columns(pl.col("time").cast(time_dtype))

    return weather.join(radiation, on="time", how="left")
