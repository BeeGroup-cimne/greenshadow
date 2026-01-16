# -*- coding: utf-8 -*-
import polars as pl
import sys
import pandas as pd
import numpy as np
import warnings
import pvlib
import os
from tqdm import tqdm
import random
import functools as ft
import cdsapi

def cleaning_pipe(df, lat = None, lon = None):
    df = df.drop_nulls()

    if lat is not None or lon is not None:
        df = df.filter(
            (pl.col("latitude").round(1) == round(lat, 1)) &
            (pl.col("longitude").round(1) == round(lon, 1))
        )
    elif lon is not None:
        df = df.filter(pl.col("longitude").round(1) == round(lon, 1))
    elif lat is not None:
        df = df.filter(pl.col("latitude").round(1) == round(lat, 1))

    df = df.drop(["time", "step", "number", "surface"], strict=False)  # Ignore errors

    renamed_columns = {**{'valid_time': 'time'},
                       **{x['CDSShortName']: x['name'] for x in ERA5Land_variables()}}
    renamed_columns_filtered = {k: v for k, v in renamed_columns.items() if k in df.columns}

    df = df.rename(renamed_columns_filtered)

    return df


def transformation_pipe(df):

    df = df.sort(["latitude", "longitude", "time"])

    # Transform time
    df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))  # Localize to UTC

    # Transform temperature, wind variables and humidity
    df = df.with_columns([
        np.sqrt(pl.col("windSpeedEast") ** 2 + pl.col("windSpeedNorth") ** 2)
        .alias("windSpeed"),
        ((np.degrees(np.arctan2(pl.col("windSpeedEast"), pl.col("windSpeedNorth")))) % 360)
        .alias("windDirection"),
        (pl.col("soilTemperature") - 273.15).alias("soilTemperature"),
        (pl.col("dewAirTemperature") - 273.15).alias("dewAirTemperature"),
        (pl.col("airTemperature") - 273.15).alias("airTemperature")
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = df.with_columns(
            pl.when(pl.col("airTemperature").is_null() | pl.col("dewAirTemperature").is_null())
            .then(None)
            .otherwise(
                (100 * (np.exp(17.67 * pl.col("dewAirTemperature") / (pl.col("dewAirTemperature") + 243.5)) /
                        np.exp(17.67 * pl.col("airTemperature") / (pl.col("airTemperature") + 243.5))))
            ).alias("relativeHumidity")
        )

    for var in ["windSpeed", "soilTemperature", "dewAirTemperature", "airTemperature", "relativeHumidity"]:
        df = df.with_columns(
            pl.when(pl.col("time") < df.select(pl.col("time").max()).item())
            .then((pl.col(var).shift(-1) + pl.col(var)) / 2)
            .otherwise(pl.col(var))
            .alias(f"{var}_avg")
        )

    # Transform GHI and total precipitation
    df = df.with_columns(
        pl.when(pl.col("time") == df.select(pl.col("time").min()).item())
        .then(0)
        .when(pl.col("time") == df.select(pl.col("time").max()).item())
        .then(0)
        .otherwise((pl.col("GHI").shift(-1) - pl.col("GHI")) / 3600)
        .alias("GHI_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("time") == df.select(pl.col("time").min()).item())
        .then(0)
        .when(pl.col("time") == df.select(pl.col("time").max()).item())
        .then(pl.col("totalPrecipitation"))
        .otherwise((pl.col("totalPrecipitation").shift(-1) - pl.col("totalPrecipitation")) * 1000)
        .alias("totalPrecipitation_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("GHI_avg") > 0)
        .then(pl.col("GHI_avg"))
        .otherwise(0)
        .alias("GHI_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("totalPrecipitation_avg") > 0)
        .then(pl.col("totalPrecipitation_avg"))
        .otherwise("totalPrecipitation")
        .alias("totalPrecipitation_avg")
    )

    # Add the weatherStation column
    df = df.with_columns(
        pl.format("{},{}", pl.col("latitude").round(1), pl.col("longitude").round(1)).alias("weatherStation")
    )

    # Drop some columns
    df = df.drop(["windSpeed", "soilTemperature", "dewAirTemperature", "airTemperature", "relativeHumidity", "GHI",
                  "totalPrecipitation"])
    rename_map = {col: col.replace("_avg", "") for col in df.columns if col.endswith("_avg")}
    df = df.rename(rename_map)

    # Add solar position and solar radiation components
    def join_solar_data(df):

        dfp = df.to_pandas()
        location = pvlib.location.Location(
            latitude=df.select("latitude").unique().item(),
            longitude=df.select("longitude").unique().item())
        solar_df = location.get_solarposition(
            dfp['time'] + pd.Timedelta(minutes=30),
            pressure=dfp['surfacePressure'],
            temperature=dfp['airTemperature']).reset_index()
        dni = pvlib.irradiance.disc(
            ghi=dfp["GHI"],
            solar_zenith=solar_df['apparent_zenith'],
            datetime_or_doy=solar_df['time'].dt.dayofyear,
            pressure=dfp["surfacePressure"])
        rad_df = pvlib.irradiance.complete_irradiance(
            solar_zenith=solar_df['apparent_zenith'],
            ghi=dfp["GHI"],
            dni=dni["dni"],
            dhi=None).rename(columns={'ghi':'GHI','dni':'DNI','dhi':'DHI'})
        solar_df = solar_df.drop(['apparent_zenith', 'zenith', 'apparent_elevation', 'equation_of_time'], axis=1)
        solar_df = solar_df.rename(columns={'elevation': 'sunElevation', 'azimuth': 'sunAzimuth'})
        solar_df['time'] = solar_df['time'] - pd.Timedelta(minutes=30)
        solar_df = pl.from_pandas(pd.concat([solar_df, rad_df], axis=1))

        return solar_df.join(df.drop("GHI"), on="time", how="inner")

    # Loop through each group and apply the join_solar_data function
    result_list = []
    for group_name, group_df in df.group_by("weatherStation"):
        result = join_solar_data(group_df)
        result_list.append(result)
    df = pl.concat(result_list)

    return df


def load_ERA5Land(data_dir, lat_range, lon_range, ym_range, update = True):
    sys.stderr.write(f"Loading weather data from ERA5Land dataset\n")
    os.makedirs(data_dir, exist_ok=True)
    grib_contains = f"{max(lat_range)}_{min(lon_range)}_{min(lat_range)}_{max(lon_range)}"
    file = f"{data_dir}/weather_{'~'.join([str(i) for i in ym_range])}_{grib_contains}.parquet"

    if (not os.path.exists(file)) or update:

        download_ERA5Land(
            data_dir=data_dir,
            ym_range=ym_range,
            lat_range=lat_range,
            lon_range=lon_range)

        list_grib_files = sorted([f'{data_dir}/{fn}' for fn in os.listdir(data_dir) if
                                  fn.endswith(".grib") and
                                  (ym_range is None or
                                   ((int(os.path.splitext(fn)[0].split('_')[0]) >= min(ym_range)) and
                                    (int(os.path.splitext(fn)[0].split('_')[0]) <= max(ym_range)))) and
                                  (grib_contains is None or grib_contains in fn)
                                  ])
        df = None
        l = len(list_grib_files)

        for i in tqdm(range(l), desc="Loading..."):

            fn = list_grib_files[i]

            hash = random.getrandbits(128)

            import cfgrib

            # Step 1: Convert GRIB data to Polars DataFrame
            dg = cfgrib.open_datasets(fn, backend_kwargs={'indexpath': f'{fn}.{hash}.idx'})
            os.remove(f'{fn}.{hash}.idx')
            df_ = [pl.from_pandas(dg[j].to_dataframe().reset_index()) for j in range(len(dg))]

            # Step 2: Apply cleaning and transformation pipelines
            df_ = [dfi.pipe(cleaning_pipe) for dfi in df_]
            df_ = ft.reduce(lambda x, y: x.join(y, on=["latitude", "longitude", "time"], how="left"), df_)
            df_ = df_.pipe(transformation_pipe)


            if df is not None:
                df = pl.concat([df, df_])
            else:
                df = df_

        df.write_parquet(file)

    else:
        df = pl.read_parquet(file)

    # Discretise sun position
    df = df.with_columns(
        ((((pl.col("sunAzimuth") + 2) / 4).floor() * 4).cast(pl.Int64).cast(str)).alias("sunAzimuthDiscrete"),
        ((((pl.col("sunElevation") + 1) / 2).floor() * 2).cast(pl.Int64).cast(str)).alias("sunElevationDiscrete")
    )
    df = df.with_columns(
        pl.when(pl.col("sunElevation") > 1)
        .then(
            pl.format("{},{}",
                      pl.col("sunAzimuthDiscrete").str.zfill(3),
                      pl.col("sunElevationDiscrete").str.zfill(2))
        )
        .otherwise(None)
        .alias("sunPosition")
    )

    return df


# Define the variables that should be obtained from the ERA5-Land dataset.
# Check the units and description: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
# Check the long and short names used in CDS API: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation (table 2)
def ERA5Land_variables():
    return [
        {
            'CDSLongName': '10m_u_component_of_wind',
            'CDSShortName': 'u10',
            'name': 'windSpeedEast'
        },
        {
            'CDSLongName': '10m_v_component_of_wind',
            'CDSShortName': 'v10',
            'name': 'windSpeedNorth'
        },
        {
            'CDSLongName': '2m_dewpoint_temperature',
            'CDSShortName': 'd2m',
            'name': 'dewAirTemperature'
        },
        {
            'CDSLongName': '2m_temperature',
            'CDSShortName': 't2m',
            'name': 'airTemperature'
        },
        {
            'CDSLongName': 'leaf_area_index_high_vegetation',
            'CDSShortName': 'lai_hv',
            'name': 'highVegetationRatio'
        },
        {
            'CDSLongName': 'leaf_area_index_low_vegetation',
            'CDSShortName': 'lai_lv',
            'name': 'lowVegetationRatio'
        },
        {
            'CDSLongName': 'total_precipitation',
            'CDSShortName': 'tp',
            'name': 'totalPrecipitation'
        },
        {
            'CDSLongName': 'surface_solar_radiation_downwards',
            'CDSShortName': 'ssrd',
            'name': 'GHI'
        },
        {
            'CDSLongName': 'forecast_albedo',
            'CDSShortName': 'fal',
            'name': 'albedo'
        },
        {
            'CDSLongName': 'soil_temperature_level_4',
            'CDSShortName': 'stl4',
            'name': 'soilTemperature'
        },
        {
            'CDSLongName': 'volumetric_soil_water_layer_4',
            'CDSShortName': 'swvl4',
            'name': 'soilWaterRatio'
        },
        {
            'CDSLongName': 'surface_pressure',
            'CDSShortName': 'sp',
            'name': 'surfacePressure'
        }
        ]


# Function to gather weather data from ERA5-Land
def download_ERA5Land(data_dir, ym_range, lat_range=[90, -90], lon_range=[-180, 180]):

    sys.stderr.write("\tDownloading the ERA5Land from Climate Data Store\n")
    # Connect to the Copernicus Climate Date Store
    c = None
    def load_copernicus():
        try:
            c = cdsapi.Client()  # You'll need a CDS credentials file in ~/.cdsapirc
        except:
            raise sys.stderr.write(
                "\tObtain credentials to Climate Data Store and set them in ~/.cdsapirc file. " \
                "\n\tPlease, follow instructions in: " \
                "\n\thttps://cds.climate.copernicus.eu/how-to-api")
        return c

    # Gather the GRIB file for each year-month.
    ym = min(ym_range)
    os.makedirs(data_dir, exist_ok=True)
    while ym <= max(ym_range):
        year = int(str(ym)[0:4])
        month = int(str(ym)[4:6])
        filename = f'{data_dir}/{year}{month:02}_{max(lat_range)}_{min(lon_range)}_{min(lat_range)}_{max(lon_range)}.grib'
        sys.stderr.write(f'\t\t--> Obtaining {year}{month:02} in bounding box: {max(lat_range)},{min(lon_range)} ' \
                         f'and {min(lat_range)},{max(lon_range)}\n')
        if not os.path.exists(filename):
            if c is None:
                c = load_copernicus()
            dataset = "reanalysis-era5-land"
            request = {
                    'variable': [x['CDSLongName'] for x in ERA5Land_variables()],
                    'year': f'{year}',
                    'month': f'{month:02}',
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [  # Bounding box
                        max(lat_range), min(lon_range),  # Upper left point (Lat-lon)
                        min(lat_range), max(lon_range)  # Lower right point (Lat-lon)
                    ],
                    'data_format': 'grib',
                    'download_format': "unarchived"
                }
            c.retrieve(dataset, request).download(filename)
        sys.stderr.write('\tDone!\n')
        month += 1
        if month > 12:
            year += 1
            month = 1
        ym = int(f'{year}{month:02}')

    sys.stderr.write('\t\tSuccess! All data has been downloaded.\n')
