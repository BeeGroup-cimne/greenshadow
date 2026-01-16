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
from shapely.ops import unary_union

# Configuration
wd = "/home/gmor/Nextcloud2/Beegroup/data/greenshadow"
wd_hypercadaster = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
municipal_codes = {
    "Alpicat": "25023",  # Alpicat (lon, lat)
    #"Barcelona": "08900"  # Barcelona (lon, lat)
}
hc.download(wd=wd_hypercadaster, cadaster_codes=list(municipal_codes.values()))

day_to_analyse = [dt.date(2025,6,21), dt.date(2025,12,21)]  # Summer and winter solstices
# year_to_analyse = 2025  # Uncomment for annual analysis

dsm_resolution = 0.5  # meters
tz_local = "Europe/Madrid"
sim_time_granularity = "PT15M"  # 15-minute intervals
sim_solar_degree_discretization = 0.1  # degrees
sim_dem_height = 1.5  # meters above ground for elevated analysis
recalculate_digital_models = False

# Get municipalities geometries
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
    aoi_buffer_in_meters = 500

    # 2) Find matching LiDAR tiles
    print("Finding ICGC LiDAR tiles...")
    tiles = gs.icgc_tiles_from_geojson(
        aoi=aoi,
        aoi_epsg=4326,
        aoi_buffer_in_meters=aoi_buffer_in_meters,
        pdf_plot_path=f"{wd}/{municipality_name}/tiles.pdf"
    )

    # 3) Download LAZ files
    print("Downloading LAZ files...")
    paths = gs.icgc_download_from_geojson(
        out_dir=f"{wd}/icgc",
        tiles=tiles,
        aoi=aoi,
        aoi_epsg=4326,
        aoi_buffer_in_meters=aoi_buffer_in_meters,
        max_workers=8
    )

    # 4) Generate DSM and DEM
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
            search_radius=500.0,  # 500m search radius
            angular_resolution=2.0  # 2° angular steps
        )
    
    # 5) Solar shading analysis
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
                only_daylight=True
            )
            
            print(f"Generated {len(solar_data['azimuth_altitude_pairs'])} solar positions")
            
            # Generate hillshades with cast shadows
            print("Generating hillshades with cast shadows...")
            hillshade_paths = gs.generate_multiple_hillshades_gdal(
                raster_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                dsm_path=dsm_path,
                azimuth_altitude=solar_data["azimuth_altitude_pairs"],
                related_datetimes=solar_data["df"]["datetime_local"].tolist(),
                shadow_floor=0.0,  # Full shadows
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
print("- DSM (original) and DSM without buildings rasters")
print("- Building geometries and debug masks")
print("- DEM rasters generated from building-filtered DSM")
print("- Individual hillshades with timestamps")
print("- Daily/annual average solar exposure")
print("- Height-based shadow analysis")
print("\\nTips:")
print("1. Load TIF files in QGIS with temporal controller for time-based visualization")
print("2. Use the PDF tile plots to verify coverage")
print("3. Compare dsm.tif vs dsm_no_buildings.tif to see building removal effect")
print("4. Check buildings_mask_debug.tif to verify building detection accuracy")
print("5. Adjust resolution and time granularity based on your needs")
print("6. For large areas, consider processing in smaller tiles")