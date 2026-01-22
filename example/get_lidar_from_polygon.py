# -*- coding: utf-8 -*-
"""
greenshadow Example: LiDAR processing for environmental shading analysis

This example demonstrates how to:
1. Find and download LiDAR tiles for specific locations in Catalonia
2. Generate Digital Surface Models (DSM) and Digital Elevation Models (DEM)
3. Filter building geometries from cadastral data and remove them from DSM
4. Generate building-filtered DEM from the processed DSM
5. Calculate solar positions for specific dates
6. Generate hillshades with cast shadows for solar analysis
7. Create temporal solar exposure analysis

The building filtering step removes building structures from the DSM before DEM 
generation, resulting in a more accurate ground elevation model that represents
the bare earth surface rather than including building heights.

Example locations: Alpicat and Barcelona
"""

import greenshadow as gs
import hypercadaster_ES as hc
import os
import datetime as dt
from hypercadaster_ES.mergers import get_census_gdf
from hypercadaster_ES.utils import census_tracts_dir_
import polars as pl
from shapely.ops import unary_union

# Configuration
wd = "/home/gmor/Nextcloud2/Beegroup/data/greenshadow"
wd_hypercadaster = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
weather_data_dir = f"{wd}/weather"
municipal_codes = {
    "Alpicat": "25023",  # Alpicat (lon, lat)
    #"Barcelona": "08900"  # Barcelona (lon, lat)
}

# Choose one of the following variables and comment the other.
# When day_to_analyse is defined, the shades and solar radiation for that specific days will be calculated,
# when year_to_analyse is defined, the aggregated shades and solar radiation during that year will be calculated.
day_to_analyse = [dt.date(2023,6,21), dt.date(2023,12,22)]
# e.g. summer and winter solstices, it is recommended to define days within the same year
# year_to_analyse = 2023  # Weather analysis year

include_weather_analysis = True
dsm_resolution = 0.5  # meters
tz_local = "Europe/Madrid"
sim_time_granularity = "PT15M" if "day_to_analyse" in locals() else "PT1H"
sim_solar_degree_discretization = 0.1 if "day_to_analyse" in locals() else 3 # degrees
sim_dem_height = 1.5  # meters above ground for elevated analysis
recalculate_digital_models = False

# Get municipalities geometries
# hc.download(wd=wd_hypercadaster, cadaster_codes=list(municipal_codes.values()))
census_gdf = hc.mergers.get_census_gdf(
    census_tract_dir=hc.utils.census_tracts_dir_(wd_hypercadaster)
)
municipal_geometries = census_gdf[
    census_gdf.ine_municipality_code.isin(
        hc.utils.cadaster_to_ine_codes(
            cadaster_dir=hc.utils.cadaster_dir_(wd_hypercadaster),
            cadaster_codes=municipal_codes.values()
        )
    )
].groupby('ine_municipality_code').agg({
      'municipality_name': 'first',
      'province_name': 'first',
      'autonomous_community_name': 'first',
      'geometry': lambda x: unary_union(x.tolist())
  }).reset_index()
municipal_geometries["cadaster_municipality_code"] = (
    hc.utils.ine_to_cadaster_codes(
            cadaster_dir=hc.utils.cadaster_dir_(wd_hypercadaster),
            ine_codes=municipal_geometries["ine_municipality_code"]
    ))

# Process each municipality
for municipality_name, cadastral_code in municipal_codes.items():

    print(f"\n=== Processing {municipality_name.title()} ===")
    os.makedirs(f"{wd}/lidar/{municipality_name}", exist_ok=True)
    dem_path = f"{wd}/{municipality_name}/dem.tif"
    dsm_path = f"{wd}/{municipality_name}/dsm.tif"
    svf_path = f"{wd}/{municipality_name}/svf.tif"
    
    # 1) Define Area of Interest (AOI)
    aoi = municipal_geometries.geometry[municipal_geometries["cadaster_municipality_code"]==cadastral_code].iloc[0]
    
    # 2) Weather data loading for this municipality
    weather_df = None
    if include_weather_analysis:
        print(f"\n=== Loading Weather Data for {municipality_name.title()} ===")
        try:
            # Get centroid of this municipality
            centroid = aoi.centroid
            municipality_bounds = aoi.bounds  # (minx, miny, maxx, maxy)
            
            # Create year-month range for the analysis year
            years_weather = list(set([date.year for date in day_to_analyse]))
            start_ym = dt.datetime(min(years_weather), 1, 1)  # January 1st
            end_ym = dt.datetime(max(years_weather), 12, 31)   # December 31st
            ym_range = [start_ym, end_ym]
            
            # Define lat/lon range based on municipality centroid
            lat_range = [centroid.y, centroid.y]  # Point location
            lon_range = [centroid.x, centroid.x]  # Point location
            
            # Load ERA5-Land weather data using point location for efficiency
            weather_df = gs.load_ERA5Land(
                data_dir=weather_data_dir,
                lat_range=lat_range,
                lon_range=lon_range,
                ym_range=ym_range,
                update=False,  # Set to True to force re-download
                use_point_location=True,  # Use municipality centroid for efficient downloads
                tz_local=tz_local,
                mode="instant"
            )
            
        except Exception as e:
            print(f"Warning: Could not load weather data for {municipality_name}: {e}")
            print("Continuing without weather analysis...")
            weather_df = None
    aoi_buffer_in_meters = 500

    # 3) Find matching LiDAR tiles
    print("Finding ICGC LiDAR tiles...")
    tiles = gs.icgc_tiles_from_geojson(
        aoi=aoi,
        aoi_epsg=4326,
        aoi_buffer_in_meters=aoi_buffer_in_meters,
        pdf_plot_path=f"{wd}/{municipality_name}/tiles.pdf"
    )

    # 4) Download LAZ files
    print("Downloading LAZ files...")
    paths = gs.icgc_download_from_geojson(
        out_dir=f"{wd}/icgc",
        tiles=tiles,
        aoi=aoi,
        aoi_epsg=4326,
        aoi_buffer_in_meters=aoi_buffer_in_meters,
        max_workers=8
    )

    # 5) Generate DSM, DEM and SVF
    if not os.path.exists(dsm_path) or recalculate_digital_models:
        print("Generating Digital Surface Model (DSM)...")
        dsm_path = gs.build_digital_model_from_lidar_pure(
            laz_paths=paths,
            out_path=f"{wd}/{municipality_name}/dsm.tif",
            out_classification_path=f"{wd}/{municipality_name}/dsm_class.tif",
            resolution=dsm_resolution,
            classification=(2,3,4,5,6,7,8,9,17),  # https://datacloud.ide.cat/especificacions/lidar-territorial-v3r1-20250905.pdf
            epsg=25831,  # UTM Zone 31N for Catalonia
            aoi=aoi,
            aoi_epsg=4326,
            aoi_buffer_in_meters=aoi_buffer_in_meters,
            fill_gaps="nearest",
            aggregation="mean"
        )

    if not os.path.exists(dem_path) or recalculate_digital_models:
        print("Generating Digital Elevation Model (DEM)...")
        dem_path = gs.build_digital_model_from_lidar_pure(
            laz_paths=paths,
            out_path=dem_path,
            resolution=dsm_resolution,
            classification=(2, 8),
            # https://datacloud.ide.cat/especificacions/lidar-territorial-v3r1-20250905.pdf
            epsg=25831,  # UTM Zone 31N for Catalonia
            aoi=aoi,
            aoi_epsg=4326,
            aoi_buffer_in_meters=aoi_buffer_in_meters,
            fill_gaps="nearest",
            aggregation="mean"
        )
    
    if not os.path.exists(svf_path) or recalculate_digital_models:
        gs.calculate_sky_view_factor(
            dsm_path=dsm_path,
            out_path=svf_path,
            search_radius=250.0,  # 500m search radius
            angular_resolution=3.0  # 2° angular steps
        )
    
    # 6) Solar shading analysis
    if "day_to_analyse" in locals():
        if not isinstance(day_to_analyse, list):
            day_to_analyse = [day_to_analyse]
        
        for day in day_to_analyse:
            
            print(f"Processing solar analysis for {day}...")
            
            # Get coordinates from the municipality centroid
            aoi_centroid = aoi.centroid
            lat, lon = aoi_centroid.y, aoi_centroid.x
            
            # Generate solar positions for this date
            solar_data = gs.generate_azimuth_altitude_pairs_by_date(
                date=day,
                lat=lat,
                lon=lon,
                tz_name=tz_local,
                granularity=sim_time_granularity,
                solar_degrees_discretization=sim_solar_degree_discretization,
                only_daylight=True,
                weather_df=weather_df.to_pandas()
            )
            
            print(f"Generated {len(solar_data['azimuth_altitude_pairs'])} solar positions")
            
            # Generate hillshades with cast shadows
            print("Generating hillshades with cast shadows...")
            hillshade_paths = gs.generate_multiple_hillshades_gdal(
                raster_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                dsm_path=dsm_path,
                azimuth_altitude=solar_data["azimuth_altitude_pairs"],
                related_datetimes=solar_data["df"]["datetime_local"].tolist(),
                include_datetime_info=True,
                qgis_temporal=True  # Add timestamps for QGIS temporal controller
            )
            
            # Apply height-based shadows (e.g., for rooftop or elevated analysis)
            print(f"Applying height-based shadows at {sim_dem_height}m elevation...")
            gs.apply_height_shadow_to_hillshades(
                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                dem_path=dem_path,
                dsm_path=dsm_path,
                out_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m",
                height_m=sim_dem_height,
                pattern="*.tif",
                recursive=True,
                mask_value=0  # 0% direct illumination in shadow areas
            )
            
            # Create daily averages
            print("Creating daily average solar exposure...")
            gs.average_hillshades(
                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                out_path=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/daily_average.tif",
                aggregation="mean"
            )
            
            gs.average_hillshades(
                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m",
                out_path=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/daily_average_dem{sim_dem_height}m.tif",
                aggregation="mean"
            )

    # Optional: Annual analysis (commented out for speed)
    elif "year_to_analyse" in locals():
        print(f"Processing annual solar analysis for {year_to_analyse}...")
        
        # Get coordinates from the municipality centroid
        aoi_centroid = aoi.centroid
        lat, lon = aoi_centroid.y, aoi_centroid.x
        
        # Filter weather data for this municipality (if available)
        municipality_weather_df = None
        if weather_df is not None:
            # Filter weather data to the nearest weather station
            municipality_weather_df = weather_df.filter(
                (pl.col("latitude").round(1) == round(lat, 1)) &
                (pl.col("longitude").round(1) == round(lon, 1))
            )
            
            # If no exact match, find closest weather station
            if len(municipality_weather_df) == 0:
                # Calculate distances to all weather stations
                distances = weather_df.with_columns([
                    ((pl.col("latitude") - lat) ** 2 + (pl.col("longitude") - lon) ** 2).sqrt().alias("distance")
                ])
                closest_station = distances.group_by("weatherStation").agg(pl.col("distance").min()).sort("distance").row(0)[0]
                municipality_weather_df = weather_df.filter(pl.col("weatherStation") == closest_station)
                
                if len(municipality_weather_df) > 0:
                    station_lat = municipality_weather_df.select("latitude").unique().item()
                    station_lon = municipality_weather_df.select("longitude").unique().item()
                    print(f"Using weather station at ({station_lat:.1f}, {station_lon:.1f}) for {municipality_name}")
            
            # Save municipality-specific weather data
            if len(municipality_weather_df) > 0:
                muni_weather_path = f"{wd}/{municipality_name}/weather_{year_to_analyse}.parquet"
                municipality_weather_df.write_parquet(muni_weather_path)
                print(f"Municipality weather data saved: {len(municipality_weather_df)} records")
        
        annual_solar = gs.generate_azimuth_altitude_pairs_by_year(
            year=year_to_analyse,
            lat=lat,
            lon=lon,
            tz_name=tz_local,
            granularity="PT1H",  # Hourly for annual analysis
            solar_degrees_discretization=2.0,  # Coarser for speed
            only_daylight=True
        )
        
        print(f"Generated {len(annual_solar['azimuth_altitude_pairs'])} unique solar positions for the year")
        
        gs.generate_multiple_hillshades_gdal(
            raster_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
            dsm_path=dsm_path,
            azimuth_altitude=annual_solar["azimuth_altitude_pairs"],
            shadow_floor=0.0
        )
        
        # Create annual average
        gs.average_hillshades(
            hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
            out_path=f"{wd}/{municipality_name}/annual_solar_exposure.tif",
            aggregation="mean"
        )
    
    else:
        print("Warning: No analysis period defined. Set either 'day_to_analyse' or 'year_to_analyse'")

    # 6) Clean up temporary files
    print("Cleaning up temporary files...")
    deleted, errors = gs.delete_matching_files(
        f"{wd}",
        "*.xml", 
        mode="glob", 
        dry_run=False
    )
    if deleted:
        print(f"Cleaned up {len(deleted)} temporary files")
    if errors:
        print(f"Errors during cleanup: {len(errors)}")
    
    print(f"Completed processing for {municipality_name.title()}!")

print("\\n=== Solar shading analysis complete ===")
print(f"Results saved in: {wd}/")
print("\\nGenerated outputs:")
print("- DSM (Digital Surface Model) and DEM (Digital Elevation Model) rasters")
print("- Sky View Factor (SVF) analysis")
print("- Individual hillshades with timestamps for temporal analysis")
print("- Daily/annual average solar exposure maps")
print("- Height-based shadow analysis for elevated surfaces")
if include_weather_analysis and weather_df is not None:
    print("- Weather data (ERA5-Land) with solar radiation and meteorological variables")
    print("- Municipality-specific weather datasets for environmental correlation")
print("\\nTips:")
print("1. Load TIF files in QGIS with temporal controller for time-based visualization")
print("2. Use the PDF tile plots to verify LiDAR coverage")
print("3. Sky View Factor (SVF) helps identify urban canyon effects")
print("4. Weather data includes GHI, DHI, DNI solar components plus meteorology")
print("5. Combine hillshades with weather data for energy potential analysis")
print("6. Adjust resolution and time granularity based on computational resources")
print("7. For large areas, consider processing in smaller tiles")