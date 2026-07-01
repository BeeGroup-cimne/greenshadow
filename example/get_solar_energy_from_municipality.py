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
from shapely.ops import unary_union
from greenshadow.logging_utils import section, subsection, info, success, warning, error, indent, section_context, summary

# Configuration
wd = "/media/gmor/data/greenshadow"
wd_hypercadaster = "/media/gmor/data/greenshadow/hypercadaster_ES"
weather_data_dir = f"{wd}/weather"
preserved_areas_path = f"{wd}/preserved_areas"
laz_dir = f"{wd}/icgc"
municipal_codes = {
    "Alpicat": "25023",  # Alpicat (lon, lat)
    #"Barcelona": "08900"  # Barcelona (lon, lat)
}

# Choose one, or both, of the following variables depending on the type of analysis you prefer.
# Comment the variable when 
# When days_to_analyse is defined, the shades and solar radiation for that specific days will be calculated,
# e.g. summer and winter solstices, it is recommended to define days within the same year
days_to_analyse = [dt.date(2023,12,22), dt.date(2023,6,21)]#,
# when year_to_analyse is defined, the aggregated shades and solar radiation during that year will be calculated.
year_to_analyse = 2023

rural_parcels_year = 2025
include_weather_analysis = True
dsm_resolution = 0.5  # meters
tz_local = "Europe/Madrid"
sim_time_granularity_when_daily_analysis = "PT15M"
sim_time_granularity_when_yearly_analysis = "PT1H"
sim_solar_degree_discretization_when_daily_analysis = 0.1
sim_solar_degree_discretization_when_yearly_analysis = 3 # degrees
sim_dem_height = 1.5  # meters above ground for elevated analysis
recalculate_digital_models = False
aoi_buffer_in_meters = 500

# Get municipalities geometries
hc.download(wd=wd_hypercadaster, cadaster_codes=list(municipal_codes.values()))
census_gdf = hc.mergers.get_census_gdf(
    census_tract_dir=hc.utils.census_tracts_dir_(wd_hypercadaster)
)
municipal_geometries = census_gdf[
    census_gdf.ine_municipality_code.isin(
        hc.utils.cadaster_to_ine_codes(
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
            ine_codes=municipal_geometries["ine_municipality_code"]
    ))

# Get the protected areas
preserved_regions = gs.load_spanish_preserved_regions(
                data_dir=preserved_areas_path,
                datasets=["natura2000", "enp"],
                join_geometries_per_resource=True
            )

# Process each municipality
for municipality_name, cadastral_code in municipal_codes.items():
    
    with (section_context(f"Processing {municipality_name.title()}")):
        dem_path = f"{wd}/{municipality_name}/dem.tif"
        dsm_path = f"{wd}/{municipality_name}/dsm.tif"
        dsm_class_path = f"{wd}/{municipality_name}/dsm_class.tif"
        dsm_slope_path = f"{wd}/{municipality_name}/dsm_slope.tif"
        dsm_aspect_path = f"{wd}/{municipality_name}/dsm_aspect.tif"
        dsm_no_vegetation_path = f"{wd}/{municipality_name}/dsm_no_vegetation.tif"
        dem_slope_path = f"{wd}/{municipality_name}/dem_slope.tif"
        dem_aspect_path = f"{wd}/{municipality_name}/dem_aspect.tif"
        dem_class_path = f"{wd}/{municipality_name}/dem_class.tif"
        dsm_no_vegetation_slope_path = f"{wd}/{municipality_name}/dsm_no_vegetation_slope.tif"
        dsm_no_vegetation_aspect_path = f"{wd}/{municipality_name}/dsm_no_vegetation_aspect.tif"
        dsm_no_vegetation_class_path = f"{wd}/{municipality_name}/dsm_no_vegetation_class.tif"
        rural_parcels_path = f"{wd}/{municipality_name}/rural_parcels"
        svf_path = f"{wd}/{municipality_name}/svf.tif"
        pv_potential_raster_path = f"{wd}/{municipality_name}/pv_results.tif"
        roof_planes_path= f"{wd}/{municipality_name}/roofs.gpkg"
        
        # 1) Define Area of Interest (AOI)
        aoi = municipal_geometries.geometry[municipal_geometries["cadaster_municipality_code"]==cadastral_code].iloc[0]
        # Get coordinates from the municipality centroid
        aoi_centroid = aoi.centroid
        lat, lon = aoi_centroid.y, aoi_centroid.x
        info(f"Centroid: {lat:.3f}, {lon:.3f}")

        # 2) Weather data loading for this municipality
        weather_df = None
        if include_weather_analysis:
            with section_context("Loading Weather Data", level=2):
                try:
                    with indent("Preparing weather data request"):
                        # Create year-month range for the analysis year
                        years_weather = list(set([date.year for date in days_to_analyse]))
                        start_ym = dt.datetime(min(years_weather), 1, 1)  # January 1st
                        end_ym = dt.datetime(max(years_weather), 12, 31)   # December 31st
                        ym_range = [start_ym, end_ym]
                        
                        # Define lat/lon range based on municipality centroid
                        lat_range = [lat, lat]  # Point location
                        lon_range = [lon, lon]  # Point location
                    
                    with indent("Loading ERA5-Land data"):
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
                    
                    success(f"Loaded weather data: {len(weather_df)} records")
                    
                except Exception as e:
                    error(f"Could not load weather data: {e}")
                    warning("Continuing without weather analysis")
                    weather_df = None

        # 3) Find matching LiDAR tiles
        with section_context("Finding and downloading ICGC LiDAR tiles", level=2):
            tiles = gs.icgc_tiles_from_geojson(
                aoi=aoi,
                aoi_epsg=4326,
                aoi_buffer_in_meters=aoi_buffer_in_meters,
                pdf_plot_path=f"{wd}/{municipality_name}/tiles.pdf"
            )
            success(f"Found {len(tiles)} matching tiles")

            # 4) Download LAZ files
            paths = gs.icgc_download_from_geojson(
                out_dir=laz_dir,
                tiles=tiles,
                aoi=aoi,
                aoi_epsg=4326,
                aoi_buffer_in_meters=aoi_buffer_in_meters,
                max_workers=8
            )
            success(f"Downloaded {len(paths)} LAZ files")

        # 5) Get building geometries for classification correction
        building_geoms_gdf = None
        try:
            building_geoms_gdf = hc.mergers.join_cadaster_building(
                gdf=None,
                cadaster_dir=hc.utils.cadaster_dir_(wd_hypercadaster),
                cadaster_codes=[cadastral_code],
                results_dir=hc.utils.results_dir_(wd_hypercadaster),
                open_street_dir=hc.utils.open_street_dir_(wd_hypercadaster)
            )
            if not building_geoms_gdf.empty:
                building_geoms_gdf = building_geoms_gdf[["building_geometry"]].rename(columns={"building_geometry": "geometry"}).set_geometry("geometry").to_crs(epsg=25831)
                info(f"Found {len(building_geoms_gdf)} buildings for classification correction")
            else:
                building_geoms_gdf = None
                warning("No building geometries found for classification correction")
        except Exception as e:
            building_geoms_gdf = None
            warning(f"Could not get building geometries for classification correction: {e}")
        
        # 6) Get the rural parcels and enrich them
        with section_context("Get rural parcels", level=2):
            sigpac_parcels = gs.load_spanish_sigpac_parcels(
                municipality_codes=cadastral_code,
                year=rural_parcels_year,
                workdir=rural_parcels_path,
                target_crs=25831
            )
            # sigpac_parcels = gs.tag_parcels_with_preserved_areas(
            #     sigpac_parcels,
            #     preserved_regions,
            #     min_intersection_ratio=0.01  # 1% threshold
            # )

        # 7) Generate the DSMs, DEM and SVF
        with section_context("Generating Digital Models", level=2):
            if not os.path.exists(dsm_path) or recalculate_digital_models:
                with indent("Generating Digital Surface Model (DSM)"):
                    dsm_path = gs.build_digital_model_from_lidar_tiled(
                        laz_paths=paths,
                        out_path=dsm_path,
                        out_classification_path=dsm_class_path,
                        out_slope_path=dsm_slope_path,
                        out_aspect_path=dsm_aspect_path,
                        resolution=dsm_resolution,
                        classification=(1,2,3,4,5,6,8,9,17,77),  # https://datacloud.ide.cat/especificacions/lidar-territorial-v3r1-20250905.pdf
                        building_classes=(6, 17, 77),
                        epsg=25831,  # UTM Zone 31N for Catalonia
                        aoi=aoi,
                        aoi_epsg=4326,
                        aoi_buffer_in_meters=aoi_buffer_in_meters,
                        fill_gaps="smart_general",
                        aggregation="max",
                        # Classification correction parameters
                        classification_correction_buildings_gdf=building_geoms_gdf,
                        classification_correction_buildings_mode="strict",
                        classification_correction_building_class_value=6,
                        classification_correction_building_buffer_m=1,
                        # Smoothing
                        smooth_classes = (2,3,4,5,6,8,9,17),
                        smooth_mode = {
                            "terrain": "gaussian",
                            "buildings": None,
                            "vegetation": None
                        },
                        smooth_window_size = 11,
                        smooth_iterations = 1
                    )
                    success(f"DSM generated: {dsm_path}")
            else:
                info(f"Using existing DSM: {dsm_path}")

            # DSM without vegetation
            if not os.path.exists(dsm_no_vegetation_path) or recalculate_digital_models:
                with indent("Generating Digital Surface Model (DSM) without vegetation"):
                    dsm_no_vegetation_path = gs.build_digital_model_from_lidar_tiled(
                        laz_paths=paths,
                        out_path=dsm_no_vegetation_path,
                        out_slope_path=dsm_no_vegetation_slope_path,
                        out_aspect_path=dsm_no_vegetation_aspect_path,
                        out_classification_path=dsm_no_vegetation_class_path,
                        resolution=dsm_resolution,
                        classification=(1,2,6,8,17,9,77),  # https://datacloud.ide.cat/especificacions/lidar-territorial-v3r1-20250905.pdf
                        building_classes=(6, 17, 77),
                        epsg=25831,  # UTM Zone 31N for Catalonia
                        aoi=aoi,
                        aoi_epsg=4326,
                        aoi_buffer_in_meters=aoi_buffer_in_meters,
                        fill_gaps="smart_general",
                        aggregation="max",
                        # Classification correction parameters
                        classification_correction_buildings_gdf=building_geoms_gdf,
                        classification_correction_buildings_mode="strict",
                        classification_correction_building_class_value=6,
                        classification_correction_building_buffer_m=1,
                        # Smoothing
                        smooth_classes=(2,8,6,17,9,77),
                        smooth_mode={
                            "terrain": "gaussian",
                            "buildings": None
                        },
                        smooth_window_size=11,
                        smooth_iterations=1
                    )
                    success(f"DSM without vegetation generated: {dsm_no_vegetation_path}")
                    # dsm_no_vegetation_550_path = gs.force_raster_value_based_on_geometries(
                    #     raster_path=dsm_no_vegetation_path,
                    #     new_raster_path=dsm_no_vegetation_550_path,
                    #     geoms=sigpac_parcels,
                    #     target_value=5.5,
                    #     mode="add",  # "set" | "increment"
                    #     band=1  # band to modify (1-based)
                    # )
                    # success(f"DSM without vegetation (+5.5m) generated: {dsm_no_vegetation_550_path}")
                    # dsm_no_vegetation_300_path = gs.force_raster_value_based_on_geometries(
                    #     raster_path=dsm_no_vegetation_path,
                    #     new_raster_path=dsm_no_vegetation_300_path,
                    #     geoms=sigpac_parcels,
                    #     target_value=3,
                    #     mode="add",
                    #     band=1
                    # )
                    # success(f"DSM without vegetation (+3.0m) generated: {dsm_no_vegetation_300_path}")
            else:
                info(f"Using existing DSM without vegetation: {dsm_no_vegetation_path}")

            # DEM
            if not os.path.exists(dem_path) or recalculate_digital_models:
                with indent("Generating Digital Elevation Model (DEM)"):
                    dem_path = gs.build_digital_model_from_lidar_tiled(
                        laz_paths=paths,
                        out_path=dem_path,
                        out_slope_path=dem_slope_path,
                        out_aspect_path=dem_aspect_path,
                        resolution=dsm_resolution,
                        classification=(2,8,17,9),
                        epsg=25831,  # UTM Zone 31N for Catalonia
                        aoi=aoi,
                        aoi_epsg=4326,
                        aoi_buffer_in_meters=aoi_buffer_in_meters,
                        fill_gaps="dem",
                        aggregation="max",
                        # Filter points of the LIDAR
                        filter_raster_path=dsm_class_path,
                        filter_raster_band=1,
                        filter_points_from_class=[6, 17, 77],
                        filter_mode="exclude",
                        # Filter points that are isolated before the rasterisation
                        isolated_points_exclusion_ratio=0.9,
                        isolated_window_size=7,
                        # Classification correction parameters
                        classification_correction_buildings_gdf=building_geoms_gdf,
                        classification_correction_buildings_mode="strict",
                        classification_correction_building_class_value=6,
                        classification_correction_building_buffer_m=1,
                        # Smoothing
                        smooth_classes=(1, 2, 6, 8, 17, 9), # Class 6 was added because classification_correction_building_class_value=6
                        smooth_unary=True,
                        smooth_mode="gaussian",
                        smooth_window_size=11,
                        smooth_iterations=1
                    )
                    success(f"DEM generated: {dem_path}")
            else:
                info(f"Using existing DEM: {dem_path}")

            # Roof plane detector
            if not os.path.exists(roof_planes_path) or recalculate_digital_models:
                with indent("Detecting the roof planes"):
                    roof_gdf = gs.detect_roof_planes_v2(
                        dsm_path=dsm_no_vegetation_path,
                        slope_path=dsm_no_vegetation_slope_path,
                        aspect_path=dsm_no_vegetation_aspect_path,
                        dsm_class_path=dsm_no_vegetation_class_path,
                        dem_path=dem_path,
                        min_height_m=1.5,
                        building_class_value=(6, 17),
                        normal_tolerance_deg=10,
                        ransac_threshold=0.25,
                        buffer_m=0.0,
                        dsm_max_slope_filter=3,
                        min_roof_plane_area=6,
                        smooth_sigma=0.3,
                        test_window=(9000, 10000, 4000, 6000),
                        output_path=roof_planes_path,
                        lidar_refinement=True,
                        lidar_class_values=(2, 8, 6, 17),
                        resolution_refined=0.2,
                        laz_dir=laz_dir,
                        laz_tiles=tiles,
                        preclustering_roof_planes=True,
                        ridge_splitting_roof_planes=True
                    )
                    success(f"Roof planes generated: {roof_planes_path}")
            else:
                info(f"Using existing roof planes: {roof_planes_path}")

            if not os.path.exists(svf_path) or recalculate_digital_models:
                with indent("Calculating Sky View Factor"):
                    gs.calculate_sky_view_factor(
                        dsm_path=dsm_path,
                        out_path=svf_path,
                        search_radius=250.0,  # 250m search radius
                        angular_resolution=10.0,  # 5° angular steps,
                        cuda=True,
                        hybrid=True,
                        subsample_factor=3,
                        building_buffer_m=250,
                        class_path=dsm_class_path,
                        building_classes=(6,17,76,77),
                        vegetation_classes=(3,4,5),
                    )
                    success(f"SVF calculated: {svf_path}")
            else:
                info(f"Using existing SVF: {svf_path}")

        # 8) Solar potential at daily level
        if "days_to_analyse" in locals():
            with section_context("Solar Shading Analysis", level=2):
                if not isinstance(days_to_analyse, list):
                    days_to_analyse = [days_to_analyse]
                
                for day in days_to_analyse:
                    
                    with section_context(f"Processing {day}", level=1):
                        
                        with indent("Generating solar positions"):
                            # Generate solar positions for this date
                            solar_data = gs.generate_azimuth_altitude_pairs_by_date(
                                date=day,
                                lat=lat,
                                lon=lon,
                                tz_name=tz_local,
                                granularity=sim_time_granularity_when_daily_analysis,
                                solar_degrees_discretization=sim_solar_degree_discretization_when_daily_analysis,
                                only_daylight=True,
                                weather_df=weather_df.to_pandas() if weather_df is not None else None
                            )
                            success(f"Generated {len(solar_data['azimuth_altitude_pairs'])} solar positions")
            
                        # Generate hillshades with cast shadows
                        with indent("Generating hillshades with cast shadows"):
                            hillshade_paths = gs.generate_multiple_hillshades_gdal(
                                raster_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                                dsm_path=dsm_path,
                                aoi=aoi,
                                azimuth_altitude=solar_data["azimuth_altitude_pairs"],
                                related_datetimes=solar_data["df"]["datetime_local"].tolist(),
                                include_datetime_info=False,
                                qgis_temporal=False
                            )
                            success(f"Generated {len(hillshade_paths)} hillshades")
                            info(f"Generating the hillshades video...")
                            gs.generate_daily_shadow_mp4(
                                solar_df=solar_data["df"],
                                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                                out_mp4=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades.mp4",
                                fps=10,
                                duration_seconds=10,
                                zoom = 4,
                                frame_size = (1920, 1088),
                                overwrite=False
                            )

                        # Generate hillshades with cast shadows in a scenario of no vegetation
                        with indent("Generating hillshades with cast shadows in a scenario of no vegetation"):
                            hillshade_paths = gs.generate_multiple_hillshades_gdal(
                                raster_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_no_vegetation",
                                dsm_path=dsm_no_vegetation_path,
                                aoi=aoi,
                                azimuth_altitude=solar_data["azimuth_altitude_pairs"],
                                related_datetimes=solar_data["df"]["datetime_local"].tolist(),
                                include_datetime_info=False,
                                qgis_temporal=False
                            )
                            success(f"Generated {len(hillshade_paths)} hillshades")
                            info(f"Generating the hillshades video...")
                            gs.generate_daily_shadow_mp4(
                                solar_df=solar_data["df"],
                                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_no_vegetation",
                                out_mp4=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_no_vegetation.mp4",
                                fps=10,
                                duration_seconds=10,
                                zoom = 4,
                                frame_size = (1920, 1088),
                                overwrite=False
                            )


                        # # Apply height-based shadows (e.g., for rooftop or elevated analysis)
                        # with indent(f"Applying height-based shadows at {sim_dem_height}m elevation"):
                        #     elevated_paths = gs.apply_height_shadow_to_hillshades(
                        #         hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                        #         dem_path=dem_path,
                        #         dsm_path=dsm_path,
                        #         aoi=aoi,
                        #         out_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m",
                        #         height_m=sim_dem_height,
                        #         pattern="*.tif",
                        #         recursive=True
                        #     )
                        #     success(f"Applied elevation shadows to {len(elevated_paths)} files")
                        #     info(f"Generating the hillshades video...")
                        #     gs.generate_daily_shadow_mp4(
                        #         solar_df=solar_data["df"],
                        #         hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m",
                        #         out_mp4=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m.mp4",
                        #         fps=10,
                        #         duration_seconds=10,
                        #         zoom=3,
                        #         frame_size=(1920, 1088)
                        #     )

                        # Create daily averages
                        with indent("Computing daily solar energy exposure"):
                            avg_path = gs.compute_surface_solar_energy(
                                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                                solar_df=solar_data["df"],
                                svf_path=svf_path,
                                aoi=aoi,
                                out_path=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/daily_solar_energy.tif"
                            )
                            # avg_elevated_path = gs.compute_surface_solar_energy(
                            #     hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m",
                            #     solar_df=solar_data["df"],
                            #     aoi=aoi,
                            #     svf_path=svf_path,
                            #     out_path=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/daily_solar_energy_dem{sim_dem_height}m.tif",
                            # )

        # 8) Optional: Annual analysis
        if "year_to_analyse" in locals():
            with section_context(f"Annual Solar Analysis ({year_to_analyse})", level=2):

                with indent("Generating annual solar positions"):
                    annual_solar = gs.generate_azimuth_altitude_pairs_by_year(
                        year=year_to_analyse,
                        lat=lat,
                        lon=lon,
                        tz_name=tz_local,
                        granularity=sim_time_granularity_when_yearly_analysis,  # Hourly for annual analysis
                        solar_degrees_discretization=sim_solar_degree_discretization_when_yearly_analysis,  # Coarser for speed
                        only_daylight=True,
                        weather_df=weather_df.to_pandas() if weather_df is not None else None
                    )
                    success(f"Generated {len(annual_solar['azimuth_altitude_pairs'])} unique solar positions for the year")

                with indent("Generating annual hillshades"):
                    gs.generate_multiple_hillshades_gdal(
                        raster_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
                        dsm_path=dsm_path,
                        azimuth_altitude=annual_solar["azimuth_altitude_pairs"],
                        qgis_temporal=False,
                        include_datetime_info=False
                    )
                    success("Annual hillshades generation complete")

                with indent("Creating annual average solar exposure"):
                    annual_raster_path = gs.compute_surface_solar_energy(
                        hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
                        solar_df=annual_solar["df"],
                        aoi=aoi,
                        svf_path=svf_path,
                        mode="fourier_decomposition",
                        out_path=f"{wd}/{municipality_name}/{year_to_analyse}/annual_solar_exposure.tif"
                    )

                # with indent("Computing the PV potential"):
                #     # 1. Compute raster PV only where plants exist
                #     pv_raster = gs.pv_potential_from_fourier_raster(
                #         fourier_raster_path=annual_raster_path,
                #         out_raster_path=pv_potential_raster_path,
                #         polygons_gdf=plants_gdf,  # Polygon + MultiPolygon OK
                #         tilt=dsm_slope_path,
                #         azimuth=dsm_aspect_path,
                #         latitude_deg=aoi_centroid.y,
                #         technology="mono_Si",
                #         coverage_ratio=0.75,
                #     )
                #
                #     # 2. Aggregate per plant
                #     pv_plants = aggregate_pv_by_polygons(
                #         pv_raster="pv_monoSi.tif",
                #         polygons_gdf=plants_gdf,
                #         id_field="plant_id",
                #     )

        success(f"Completed processing for {municipality_name.title()}!")

    # panels, summary = run_pv_layout_engine_advanced(
    #     roof_gdf=roof_gdf[1307:1308],
    #     fourier_raster=f"{wd}/{municipality_name}/{year_to_analyse}/annual_solar_exposure_fourier_decomposition.tif",
    #     latitude=aoi_centroid.y,
    #     output_panel_path=f"{wd}/{municipality_name}/pv_panels.gpkg",
    #     output_building_summary_path=f"{wd}/{municipality_name}/pv_summary.csv",
    #     slope_raster_path=dsm_no_vegetation_slope_path,
    #     aspect_raster_path=dsm_no_vegetation_aspect_path,
    # )



# Clean up temporary files
with section_context("Cleanup", level=2):
    with indent("Removing temporary XML files"):
        deleted, errors = gs.delete_matching_files(
            f"{wd}",
            "*.xml",
            mode="glob",
            dry_run=False
        )
        if deleted:
            success(f"Cleaned up {len(deleted)} temporary files")
        if errors:
            error(f"Errors during cleanup: {len(errors)}")

# Final summary section
section("Analysis Complete")
with indent():
    summary({
        "Municipalities processed": f"{len(municipal_codes)} ({",".join(municipal_codes.keys())})",
        "Output directory": wd,
        "Daily analysis": f"✓ ({", ".join([d.strftime("%Y-%m-%d") for d in days_to_analyse])})" if "days_to_analyse" in locals() else "✗",
        "Yearly analysis": f"✓ ({year_to_analyse})" if "year_to_analyse" in locals() else "✗"
    })