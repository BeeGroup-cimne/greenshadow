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
import geopandas as gpd
from shapely.ops import unary_union
from greenshadow.logging_utils import section, subsection, info, success, warning, error, indent, section_context, summary

# Configuration
wd = "/mnt/data/greenshadow"
wd_hypercadaster = "/mnt/data/greenshadow/hypercadaster_ES"
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
days_to_analyse = [dt.date(2023,12,22)]#, dt.date(2023,6,21)]#,
# when year_to_analyse is defined, the aggregated shades and solar radiation during that year will be calculated.
year_to_analyse = 2023

rural_parcels_year = 2025
include_weather_analysis = True
# Treat trees as a semi-transparent medium instead of an opaque wall: high
# vegetation is resolved into individual crowns sized from H = DSM - DEM, and the
# hillshades gain three path-length bands so one raster serves every month.
# Set to None to keep the original opaque behaviour.
semitransparent_vegetation = True
vegetation_leaf_type = 0  # 0 = deciduous, 1 = evergreen
# Age the canopy from the LiDAR flight to the year being simulated. The ICGC
# publishes a per-tile flight date, so the elapsed time is read from the data.
grow_vegetation_to_simulated_year = True
vegetation_height_growth_m_per_year = 0.25
vegetation_crown_growth_m_per_year = 0.20
dsm_resolution = 0.5  # meters
tz_local = "Europe/Madrid"
sim_time_granularity_when_daily_analysis = "PT15M"
sim_time_granularity_when_yearly_analysis = "PT1H"
sim_solar_degree_discretization_when_daily_analysis = 0.1
sim_solar_degree_discretization_when_yearly_analysis = 3 # degrees
sim_dem_height = 1.5  # meters above ground for elevated analysis
recalculate_digital_models = False
# Rebuild the solar-energy rasters from the existing hillshades. The hillshades
# only depend on the terrain and the sun, the energy rasters on the weather as
# well, so a change in how the weather is read needs only this. Leave it True
# once after upgrading: rasters written before the daylight-gap fix in
# compute_surface_solar_energy overstate the annual energy by about a third.
recalculate_solar_exposure = True
# How ERA5-Land's all-sky GHI (ssrd, clouds included) is split into direct and
# diffuse at each sun position. Against PVGIS-SARAH3 at Lleida, 2023
# (docs/irradiance_era5.md): "disc" is closest on the annual DNI (-2 %) and on a
# South 30 deg plane (-0.5 %); "dirint" is within 1 % on vertical faces;
# "era5" - the beam share of ERA5's own fdir, downloaded with the weather -
# follows the hours best but puts 12 % too much in the beam and 16 % too little
# in the diffuse; "erbs" is 10 % short on DNI.
irradiance_decomposition = "disc"
aoi_buffer_in_meters = 500
# Write an interactive 3D page of the segmented roofs next to the GeoPackage.
# Useful for checking the detection parameters before trusting a full run.
render_roof_3d_view = False

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
        svf_no_vegetation_path = f"{wd}/{municipality_name}/svf_no_vegetation.tif"
        parcel_pv_path = f"{wd}/{municipality_name}/parcel_pv.gpkg"
        parcel_pv_summary_path = f"{wd}/{municipality_name}/parcel_pv_scenarios.csv"
        pv_potential_raster_path = f"{wd}/{municipality_name}/pv_results.tif"
        roof_planes_path= f"{wd}/{municipality_name}/roofs.gpkg"
        crop_rows_path = f"{wd}/{municipality_name}/crop_rows.gpkg"
        
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
                        # The annual analysis always runs on year_to_analyse,
                        # so that year is always downloaded; the daily runs add
                        # the years of their own days.
                        years_weather = set()
                        if "year_to_analyse" in locals():
                            years_weather.add(year_to_analyse)
                        if "days_to_analyse" in locals():
                            days = days_to_analyse if isinstance(days_to_analyse, list) else [days_to_analyse]
                            years_weather.update(day.year for day in days)
                        start_ym = dt.datetime(min(years_weather), 1, 1)  # January 1st
                        end_ym = dt.datetime(max(years_weather), 12, 31)   # December 31st
                        ym_range = [start_ym, end_ym]
                        
                        # Define lat/lon range based on municipality centroid
                        lat_range = [lat, lat]  # Point location
                        lon_range = [lon, lon]  # Point location
                    
                    with indent("Loading ERA5-Land data, with ERA5 direct radiation"):
                        # ERA5-Land at the municipality centroid, plus ERA5's
                        # own all-sky direct radiation (fdir) and cloud cover
                        # (tcc) for the same hours, so the beam/sky split comes
                        # from the reanalysis' clouds instead of a model.
                        weather_df = gs.load_ERA5Land_with_direct_radiation(
                            data_dir=weather_data_dir,
                            lat_range=lat_range,
                            lon_range=lon_range,
                            ym_range=ym_range,
                            update=False,  # Set to True to force re-download
                            use_point_location=True,  # Use municipality centroid for efficient downloads
                            tz_local=tz_local,
                            mode="instantaneous"
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
                census_tracts_dir=hc.utils.census_tracts_dir_(wd_hypercadaster),
                DEM_raster_dir=hc.utils.DEM_raster_dir_(wd_hypercadaster),
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
                        # Construction cranes: the jib returns sit tens of
                        # metres above the roofs and win the "max" aggregation,
                        # leaving a streak that casts a long false shadow.
                        # Removed only when thin AND elongated AND standing
                        # clear of everything around it AND seen through by the
                        # laser - so chimneys, lift shafts and spires stay.
                        crane_removal=True,
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
                        # Construction cranes: the jib returns sit tens of
                        # metres above the roofs and win the "max" aggregation,
                        # leaving a streak that casts a long false shadow.
                        # Removed only when thin AND elongated AND standing
                        # clear of everything around it AND seen through by the
                        # laser - so chimneys, lift shafts and spires stay.
                        crane_removal=True,
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

            # The sky a PV field on a rural parcel would see: the parcel is
            # cleared to build it, so the trees come out of the horizon too.
            if not os.path.exists(svf_no_vegetation_path) or recalculate_digital_models:
                with indent("Calculating Sky View Factor without vegetation"):
                    gs.calculate_sky_view_factor(
                        dsm_path=dsm_no_vegetation_path,
                        out_path=svf_no_vegetation_path,
                        search_radius=250.0,
                        angular_resolution=10.0,
                        cuda=True,
                        hybrid=True,
                        subsample_factor=3,
                        building_buffer_m=250,
                        class_path=dsm_no_vegetation_class_path,
                        building_classes=(6,17,76,77),
                        vegetation_classes=(3,4,5),
                    )
                    success(f"SVF without vegetation calculated: {svf_no_vegetation_path}")
            else:
                info(f"Using existing SVF without vegetation: {svf_no_vegetation_path}")

            # How each rural parcel is planted. On a woody crop the canopy
            # height model is a comb, and its direction and pitch are the
            # plantation geometry that any agrivoltaic row has to fit
            # between. On wheat or lettuce there is no canopy to read, so
            # the parcel reports its working axis instead - the way the
            # field is already driven, which is what the machinery needs
            # left clear. Only agricultural SIGPAC uses are kept - roads,
            # buildings, water, forest and urban zones are dropped - and a
            # protected parcel comes back flagged, not missing.
            # The rows are a function of the digital models above, so they
            # are recomputed exactly when those are - otherwise the GeoPackage
            # on disk is the answer and no raster is opened at all.
            with indent("Detecting the crop rows"):
                if os.path.exists(crop_rows_path) and not recalculate_digital_models:
                    info(f"Using existing crop rows: {crop_rows_path}")
                crop_row_lines, crop_row_parcels = gs.detect_crop_rows(
                    sigpac_parcels,
                    dsm_path=dsm_path,
                    dsm_no_vegetation_path=dsm_no_vegetation_path,
                    # Without the class raster a farm shed is a tree, and
                    # two sheds are a row.
                    dsm_class_path=dsm_class_path,
                    dem_slope_path=dem_slope_path,
                    dem_aspect_path=dem_aspect_path,
                    output_path=crop_rows_path,
                    update=recalculate_digital_models,
                    progress=True,
                )
                planted = crop_row_parcels[crop_row_parcels.row_pattern == "rows"]
                success(
                    f"Crop rows: {len(crop_row_lines)} rows over {len(planted)} "
                    f"planted parcels of {len(crop_row_parcels)}"
                )
                for pattern, group in crop_row_parcels.groupby("row_pattern"):
                    info(f"  {pattern}: {len(group)} parcels")
                if len(planted):
                    info(
                        f"  median spacing {planted.row_spacing_m.median():.2f} m, "
                        f"canopy {planted.canopy_height_p50_m.median():.2f} m tall and "
                        f"{planted.canopy_width_m.median():.2f} m wide, "
                        f"alley {planted.alley_width_m.median():.2f} m"
                    )
                    # A parcel can hold more than one plantation, each with its
                    # own direction. Where it does, the rows have to be read
                    # per block - the parcel's own azimuth is only the largest
                    # block's.
                    mixed = planted[planted.n_blocks > 1]
                    info(
                        f"  {len(mixed)} planted parcels hold more than one "
                        f"direction"
                        + (f", up to {mixed.row_azimuth_spread_deg.max():.0f} "
                           f"degrees apart" if len(mixed) else "")
                    )

            # Roof plane detector. The raster detector is used rather than
            # the point-native v2: its region growing stays continuous
            # across a roof, where v2 stops at every crease and thin patch.
            #
            # What it reads is improved instead of how it reads it. The
            # LiDAR returns are segmented into planes over their Delaunay
            # graph and each point is pulled toward the plane of its own
            # segment before they are rasterized into the refined roof DSM.
            # That drops the ranging noise the slope raster inherits - and
            # the normals with it - without rounding off the ridges, which
            # is why smooth_sigma (an isotropic gaussian on the finished
            # DSM) stays off.
            #
            # 0.0 disables it and restores the previous behaviour exactly;
            # 1.0 flattens every facet onto its fitted plane.
            roof_test_window = (9000, 9500, 4000, 5000)

            # The refined DSM the detector builds from the LiDAR is the
            # surface the planes were fitted to, so it is the one to sample
            # for their heights. It only exists once the detector has run.
            roof_dsm_path = dsm_no_vegetation_path.replace(".tif", "_roofs.tif")

            # The detector's own output, facets still on the pixel grid.
            # roofs.gpkg holds them once regularized into straight edges.
            roof_raster_planes_path = roof_planes_path.replace(".gpkg", "_rasterised.gpkg")

            if (not os.path.exists(roof_raster_planes_path)
                    or not os.path.exists(roof_dsm_path)
                    or recalculate_digital_models):
                with indent("Detecting the roof planes"):
                    # Tiled and in parallel: at 0.20 m the whole municipality
                    # at once is ~1.4 billion pixels per array and runs out of
                    # memory. Each tile is a process of its own, holding whole
                    # building clusters only, so no roof is cut and the planes
                    # are the ones a single call would find. Finished tiles are
                    # kept next to roofs_rasterised.gpkg until the end, so a run
                    # that dies part-way resumes where it stopped. The workers
                    # are sized to the free memory, and a tile that would run
                    # the machine out of it is stopped and retried on its own.
                    roof_gdf_ = gs.detect_roof_planes_tiled(
                        dsm_path=dsm_no_vegetation_path,
                        slope_path=dsm_no_vegetation_slope_path,
                        aspect_path=dsm_no_vegetation_aspect_path,
                        dsm_class_path=dsm_no_vegetation_class_path,
                        dem_path=dem_path,
                        laz_dir=laz_dir,
                        laz_tiles=tiles,
                        min_height_m=1.5,
                        building_class_value=(6, 7, 12, 17, 77),
                        lidar_class_values=(1, 2, 6, 7, 8, 12, 17, 77),
                        lidar_refinement=True,
                        # 0.20 m only pays off because refined_aggregation="tin"
                        # fills every pixel from the triangulation. Raise it to
                        # 0.25-0.30 if the returns are sparser than ~8 pts/m2:
                        # on a 6 pts/m2 test 0.20 cost 15 points of coverage.
                        resolution_refined=0.20,
                        # Sample the refined DSM off the Delaunay triangulation
                        # of the returns instead of binning them. At 0.25 m with
                        # this point density roughly half the pixels catch no
                        # return, and binning hands those to a gap filler that
                        # minimises curvature - rounding the facets being looked
                        # for. Interpolating on measured triangles leaves no
                        # pixel empty in the first place.
                        refined_aggregation="tin",
                        # 8, not 9: averaged over three RANSAC seeds, 9 cost
                        # 0.015 of mean IoU against the digitised Alpicat roofs
                        # and 7 cost 0.02.
                        normal_tolerance_deg=8,
                        ransac_threshold=0.05,   # measured inert 0.05-0.2
                        # Slope is in RADIANS: 40 filtered nothing at all.
                        # 1.2 rad (~69 deg) drops the near-vertical wall pixels.
                        dsm_max_slope_filter=1.9,
                        # 3 m2 keeps the small facets (dormers, hip ends): at 8
                        # their mean IoU was 0.30, at 3 it is 0.51. 2 and 4 were
                        # both worse overall.
                        min_roof_plane_area=3,
                        buffer_m=0.0,
                        # Off: an isotropic gaussian on the finished DSM
                        # rounds ridges as hard as it removes noise. The
                        # Delaunay smoothing below does the job crease-aware.
                        smooth_sigma=0.0,
                        # 0.2, tuned on the digitised Alpicat roofs. Higher
                        # values MERGE small facets: at 0.7 the mean IoU against
                        # the real planes fell from 0.64 to 0.53. The synthetic
                        # benchmark said 0.7; it was wrong.
                        delaunay_smooth_strength=0.2,
                        preclustering_roof_planes=True,
                        ridge_splitting_roof_planes=False,
                        # Region growing now releases the pixels it rejects, so
                        # small facets get their own plane instead of being eaten
                        # by a big neighbour. That over-segments large facets a
                        # little, which a looser merge (8 deg, was 2.5) undoes.
                        # The offset guard stops two stacked flat roofs - same
                        # orientation, different height - merging across the step.
                        merge_normal_angle_deg=6.0,
                        merge_max_offset_m=0.3,
                        # Distance-only assignment measured better than adding a
                        # normal term (IoU 0.636 vs 0.623); left off.
                        assignment_normal_weight=0.0,
                        # Give the roof pixels region growing left in no plane
                        # to the neighbour they fit, or to a plane of their own.
                        # 7% of the real roof area was in the building mask but
                        # in no plane; this lifts coverage from 90% to 94.5%.
                        # 0.15 m measured best of 0.10-0.40.
                        recover_distance_m=0.15,
                        # Last step: merge touching planes that are one surface
                        # split in two - normals within 8 deg and no step over
                        # 0.15 m along the seam they share. The graph merge above
                        # misses them (any ridge pixel on the seam vetoes it).
                        # Mean IoU 0.686 -> 0.698 on every seed tried; 12 deg
                        # already merges real facets (0.664), and a 0.3 m step
                        # guard lets parallel planes at different heights merge.
                        coplanar_merge_angle_deg=8.0,
                        coplanar_merge_step_m=0.15,
                        # Keep the skirt: along the eaves the refined DSM drops
                        # steeply over its last pixel or two (median 60 deg), so
                        # those pixels fit no plane and the outline came out
                        # ragged and short of the real eave. They are handed to
                        # the nearest plane at the very end, without refitting.
                        # Raw IoU 0.697 -> 0.703, coverage 91% -> 95%.
                        skirt_assign_distance_m=0.3,
                        # Roof clutter removal - chimneys, aerials, AC units
                        # cut out of the facets - is OFF. Uncomment svf_path to
                        # turn it on; the rest of the arguments are inert
                        # without it, and are the values tuned against the 48
                        # digitised obstacles in
                        # real_roof_obstacles_9000_9500_4000_5000_Alpicat.gpkg.
                        #
                        # Off by default because it is not accurate enough to
                        # run unsupervised: on that ground truth it recalls
                        # 16/48 obstacles at 52% precision, and the holes it
                        # does cut come out larger than the object and offset by
                        # about half a metre. The limit is the SVF raster's
                        # 0.5 m resolution - it cannot resolve the shadow ring
                        # of a ~1.2 m object - not the thresholds.
                        #
                        # Note the SVF is the SIGNAL, not a confirmation. The
                        # refined DSM barely knows these obstacles exist: the
                        # median one clears its own fitted plane by 0.09 m,
                        # because the TIN sampling and delaunay_smooth_strength
                        # above flatten a flue into the roof it stands on. The
                        # SVF is built from the unrefined dsm_path, which still
                        # has them - median depression 0.30 against a roof-wide
                        # p90 of 0.08. Hence min_height_m=0: asking the
                        # flattened DSM to confirm only deletes real obstacles.
                        # svf_path=svf_path,
                        # svf_obstruction_drop=0.12,
                        # svf_obstruction_min_height_m=0.0,
                        # # Recovers an object from the ring of shadow round its
                        # # base - on top of a chimney the sky is open again, so
                        # # the object itself is not depressed at all.
                        # svf_obstruction_close_m=0.6,
                        # svf_obstruction_min_area_m2=0.5,
                        # # A stairwell is not clutter: it should get a plane of
                        # # its own rather than a hole.
                        # svf_obstruction_max_area_m2=15.0,
                        # # Sky view cannot tell a small object from a long one -
                        # # a step between roof levels shadows the roof below it
                        # # exactly as a chimney does - so shape does that part.
                        # svf_obstruction_max_elongation=2.5,
                        # svf_obstruction_max_planes=1,
                        # svf_obstruction_buffer_m=0.0,
                        # test_window=roof_test_window,
                        # 1 km matches the ICGC LiDAR tiles, so a tile mostly
                        # reads one LAZ file. Peak memory per tile measured on
                        # Alpicat: 10 GB for the old town, 1-7 GB elsewhere.
                        tile_size_m=1000.0,
                        # Only the buildings touching the municipality itself:
                        # the rasters reach aoi_buffer_in_meters beyond it for
                        # the shadows, not for the roofs. A building on the
                        # boundary is kept whole.
                        aoi=aoi,
                        aoi_crs="EPSG:4326",
                        output_path=roof_raster_planes_path
                    )
                    success(
                        f"Roof planes detected: {len(roof_gdf_)} facets on "
                        f"{roof_gdf_.part_id.nunique()} roofs"
                    )
            else:
                # Trims a file detected before the AOI filter existed; a
                # no-op on one detected with it.
                roof_gdf_ = gs.filter_roof_planes_by_aoi(
                    roof_raster_planes_path, aoi, aoi_crs="EPSG:4326"
                )
                info(f"Using existing roof planes: {roof_raster_planes_path}")

            if not os.path.exists(roof_planes_path) or recalculate_digital_models:
                with indent("Regularizing the roof planes"):

                    # Straighten the facet boundaries. The raster detector
                    # polygonizes pixels, so each facet arrives with ~158
                    # staircase vertices and every shared ridge is drawn twice,
                    # once per side. The default "wireframe" method rebuilds
                    # each roof of touching facets as one wireframe (~8
                    # vertices per facet): a simplified outline, shared
                    # junction nodes, and one straight edge per ridge or hip on
                    # the plane-plane intersection. method="facet" is the
                    # older per-facet straightening (~29 vertices, no shared
                    # vertices); simplify_tolerance only applies to it.
                    # 
                    # Against the digitised Alpicat planes (112 in the test
                    # window) the detector alone scores mean IoU 0.703 and the
                    # wireframe 0.696 at ~12 vertices per facet, covering 95% of
                    # the real roof. The residual error is mostly which pixels
                    # went to which plane (130 planes for 112 real ones), not
                    # edge shape.
                    roof_gdf = gs.regularize_roof_planes(
                        roof_gdf_.copy(),
                        dsm_path=roof_dsm_path,
                        # "intersections": the same wireframe, but every
                        # boundary is made straight segments first (the ridge
                        # line, or a line fitted to the pixels) and each node
                        # is put where its segments collide. Scores within
                        # 0.003 IoU of "wireframe" on the digitised planes,
                        # with cleaner corners where steps meet; "wireframe"
                        # to go back.
                        method="intersections",
                        # The wireframe defaults (max_gap=0.3, edge_tolerance=
                        # 0.45, node_merge_distance=0.6) were tuned against the
                        # digitised Alpicat planes, on the coplanar-merged
                        # detection: mean IoU 0.691 at ~11 vertices per facet.
                        # edge_tolerance=0.3 scores 0.698 at ~15 vertices;
                        # 0.6 with node_merge_distance=1.0 gives the cleanest
                        # shapes (~7.5 vertices) at 0.679. The snap angle barely
                        # matters (0-25 within 0.001); 8 squares near-rectilinear
                        # roofs without bending the rest.
                        angle_snap_deg=8.0,
                        ridge_snap_tolerance=0.9,
                        # Facets whose planes are more than 0.3 m apart in height
                        # where they meet have a wall between them: they are
                        # regularized as separate roofs, so their contours are
                        # not pulled together. IoU 0.687 -> 0.696 on this
                        # detection (with skirts).
                        wall_step_m=0.2,
                        # Roofs are rebuilt independently, one process each;
                        # the DSM is read under each facet, never whole.
                        n_workers=None,
                        progress=True
                    )
                    roof_gdf.to_file(roof_planes_path, driver="GPKG")
                    success(f"Roof planes regularized: {len(roof_gdf)} facets")
            else:
                roof_gdf = gs.filter_roof_planes_by_aoi(
                    roof_planes_path, aoi, aoi_crs="EPSG:4326"
                )
                info(f"Using existing regularized roof planes: {roof_planes_path}")

            # Interactive 3D check of what was just segmented: orbit the
            # roofs, hover a face for its tilt and azimuth. Grey shows the
            # triangles no plane claimed - mostly ridges. It is a diagnostic
            # layer only: the faces in the GeoPackage have no holes in them.
            if render_roof_3d_view:
                with indent("Rendering the 3D roof view"):
                    roof_view_path = roof_planes_path.replace(
                        ".gpkg", "_3d.html"
                    )
                    # Renders the planes just detected - the same rows that
                    # went into the GeoPackage - rather than segmenting the
                    # points again, so the picture and the product cannot
                    # disagree.
                    #
                    # The DSM is only sampled to give each face its height.
                    # Prefer the refined one the detector built from the
                    # (smoothed) LiDAR, since that is the surface the planes
                    # were fitted to.
                    if not os.path.exists(roof_dsm_path):
                        roof_dsm_path = dsm_no_vegetation_path
                    gs.roof_planes_3d(
                        roof_gdf_,
                        dsm_path=roof_dsm_path,
                        output_path=roof_view_path,
                        color_by="plane",
                        # Every LiDAR return of the area around the roofs,
                        # as measured and coloured by class, under the
                        # detected planes. A plane floating off its roof's
                        # returns, or roof returns carrying no plane, show
                        # up at once - with nothing interpolated in between.
                        laz_dir=laz_dir,
                        laz_tiles=tiles,
                        title=f"Roof planes - {municipality_name}",
                        progress=True
                    )
                    success(f"3D roof view: {roof_view_path}")

        # 8) Semi-transparent vegetation, shared by the daily and annual runs
        canopy = None
        if semitransparent_vegetation:
            with section_context("Semi-transparent canopy", level=2):
                growth = gs.NO_GROWTH
                if grow_vegetation_to_simulated_year:
                    simulated_year = (
                        year_to_analyse if "year_to_analyse" in locals()
                        else days_to_analyse[0].year
                    )
                    # ICGC publishes a flight window per 1 km tile, so how stale
                    # the point cloud is comes from the data rather than a guess.
                    flown = gs.icgc_lidar_acquisition_date(aoi, cache_dir=laz_dir)
                    growth = gs.vegetation_growth(
                        flown,
                        simulated_year,
                        height_rate_m_per_year=vegetation_height_growth_m_per_year,
                        crown_rate_m_per_year=vegetation_crown_growth_m_per_year,
                    )
                    info(f"LiDAR flown {flown:%Y-%m-%d}, simulating {simulated_year}: "
                         f"{growth.years:+.1f} y -> {growth.height_gain_m:+.2f} m height, "
                         f"{growth.crown_gain_m:+.2f} m crown diameter")
                canopy = gs.build_canopy_model_from_rasters(
                    dsm_path=dsm_path,
                    dem_path=dem_path,
                    class_path=dsm_class_path,
                    under_canopy_path=dsm_no_vegetation_path,
                    growth=growth,
                )
                success(f"{canopy.trees['row'].size} tree crowns detected")

        # 9) Solar potential at daily level
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
                                weather_df=weather_df.to_pandas() if weather_df is not None else None,
                                irradiance_decomposition=irradiance_decomposition
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
                                qgis_temporal=False,
                                canopy=canopy
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
                                overwrite=False,
                                canopy_leaf_type=vegetation_leaf_type if canopy is not None else None
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


                        # Apply height-based shadows (e.g., for rooftop or elevated analysis)
                        with indent(f"Applying height-based shadows at {sim_dem_height}m elevation"):
                            elevated_paths = gs.apply_height_shadow_to_hillshades(
                                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                                dem_path=dem_path,
                                dsm_path=dsm_path,
                                aoi=aoi,
                                out_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m",
                                height_m=sim_dem_height,
                                pattern="*.tif",
                                recursive=True
                            )
                            success(f"Applied elevation shadows to {len(elevated_paths)} files")
                            info(f"Generating the hillshades video...")
                            gs.generate_daily_shadow_mp4(
                                solar_df=solar_data["df"],
                                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m",
                                out_mp4=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m.mp4",
                                fps=10,
                                duration_seconds=10,
                                zoom=3,
                                frame_size=(1920, 1088)
                            )

                        # Create daily averages
                        with indent("Computing daily solar energy exposure"):
                            avg_path = gs.compute_surface_solar_energy(
                                hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades",
                                solar_df=solar_data["df"],
                                svf_path=svf_path,
                                aoi=aoi,
                                out_path=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/daily_solar_energy.tif",
                                canopy_leaf_type=vegetation_leaf_type if canopy is not None else None,
                                overwrite=recalculate_solar_exposure
                            )
                            # avg_elevated_path = gs.compute_surface_solar_energy(
                            #     hillshade_dir=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/hillshades_dem{sim_dem_height}m",
                            #     solar_df=solar_data["df"],
                            #     aoi=aoi,
                            #     svf_path=svf_path,
                            #     out_path=f"{wd}/{municipality_name}/{day.strftime('%Y%m%d')}/daily_solar_energy_dem{sim_dem_height}m.tif",
                            # )

        # 10) Optional: Annual analysis
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
                        weather_df=weather_df.to_pandas() if weather_df is not None else None,
                        irradiance_decomposition=irradiance_decomposition
                    )
                    success(f"Generated {len(annual_solar['azimuth_altitude_pairs'])} unique solar positions for the year")

                with indent("Generating annual hillshades"):
                    gs.generate_multiple_hillshades_gdal(
                        raster_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
                        dsm_path=dsm_path,
                        azimuth_altitude=annual_solar["azimuth_altitude_pairs"],
                        qgis_temporal=False,
                        include_datetime_info=False,
                        canopy=canopy
                    )
                    success("Annual hillshades generation complete")

                with indent("Creating annual average solar exposure"):
                    # Three passes over the same hillshade stack, so the extra
                    # modes cost a read each rather than another shadow run.
                    annual_solar_exposure_path = f"{wd}/{municipality_name}/{year_to_analyse}/annual_solar_exposure.tif"

                    # Wh on the surface itself: the plane-of-array irradiation a
                    # module mounted flush with an inclined roof receives.
                    annual_total_path = gs.compute_surface_solar_energy(
                        hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
                        solar_df=annual_solar["df"],
                        aoi=aoi,
                        svf_path=svf_path,
                        mode="total",
                        out_path=annual_solar_exposure_path,
                        canopy_leaf_type=vegetation_leaf_type if canopy is not None else None,
                        overwrite=recalculate_solar_exposure
                    )

                    # Orientation-free basis: needed on flat roofs, where the
                    # rack tilt no longer matches the surface underneath it.
                    poa_basis_path = gs.compute_surface_solar_energy(
                        hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
                        solar_df=annual_solar["df"],
                        aoi=aoi,
                        svf_path=svf_path,
                        mode="fourier_poa",
                        out_path=annual_solar_exposure_path,
                        canopy_leaf_type=vegetation_leaf_type if canopy is not None else None,
                        overwrite=recalculate_solar_exposure
                    )

                    annual_raster_path = gs.compute_surface_solar_energy(
                        hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades",
                        solar_df=annual_solar["df"],
                        aoi=aoi,
                        svf_path=svf_path,
                        mode="fourier_decomposition",
                        out_path=annual_solar_exposure_path,
                        canopy_leaf_type=vegetation_leaf_type if canopy is not None else None,
                        overwrite=recalculate_solar_exposure
                    )
                    success("Annual solar exposure rasters written")

                with indent("Planning the PV strings"):
                    pv_modules_gdf, pv_summary = gs.plan_pv_strings_in_buildings(
                        roof_gdf_,
                        poa_basis_path=poa_basis_path,
                        annual_energy_path=annual_total_path,
                        dsm_path=roof_dsm_path,
                        latitude_deg=aoi_centroid.y,
                        # Roof capacity, not a wiring design: no electrical
                        # grouping, no shading filter, no orientation refused,
                        # and the array packed to 5 cm of the plane contour.
                        string_size=None,
                        min_poa_ratio=None,
                        allowed_azimuth_range=None,
                        setback_m=0.05,
                        phase_steps=5,
                        output_path=f"{wd}/{municipality_name}/pv_modules.gpkg",
                        output_summary_path=f"{wd}/{municipality_name}/pv_building_summary.csv",
                        progress=True,
                    )
                    # The planner returns every scenario it costed, one
                    # per mounting geometry, so the frame counts each building
                    # several times. "best" is the per-plane optimum and the
                    # only scenario that covers a whole roof.
                    if len(pv_summary):
                        best = pv_summary[pv_summary.scenario == "best"]
                        success(
                            f"Placed {(pv_modules_gdf.scenario == 'best').sum()} modules "
                            f"({best.peak_kw.sum() / 1000:.2f} MWp, "
                            f"{best.energy_kwh.sum() / 1e6:.2f} GWh/year) "
                            f"over {len(best)} buildings"
                        )
                        for scenario, rows in pv_summary.groupby("scenario"):
                            info(
                                f"  {scenario}: {rows.n_modules.sum()} modules, "
                                f"{rows.energy_kwh.sum() / 1e6:.2f} GWh/year "
                                f"over {len(rows)} buildings"
                            )
                    else:
                        warning("No PV modules could be placed on the detected roofs")

                # PV fields and agrivoltaics over the rural parcels. Building
                # either one clears what grows there today, so the irradiance
                # comes from hillshades of the DSM without vegetation; the
                # parcel's current crop enters as the margin the scenario
                # gives up, not as shade.
                with indent("Annual hillshades without vegetation"):
                    gs.generate_multiple_hillshades_gdal(
                        raster_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades_no_vegetation",
                        dsm_path=dsm_no_vegetation_path,
                        azimuth_altitude=annual_solar["azimuth_altitude_pairs"],
                        qgis_temporal=False,
                        include_datetime_info=False,
                    )
                    poa_basis_no_vegetation_path = gs.compute_surface_solar_energy(
                        hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades_no_vegetation",
                        solar_df=annual_solar["df"],
                        aoi=aoi,
                        svf_path=svf_no_vegetation_path,
                        mode="fourier_poa",
                        out_path=f"{wd}/{municipality_name}/{year_to_analyse}/annual_solar_exposure_no_vegetation.tif",
                        overwrite=recalculate_solar_exposure,
                    )
                    success(f"No-vegetation POA basis: {poa_basis_no_vegetation_path}")

                    # The same hillshades again, for the panels a fixed
                    # orientation basis cannot cost: vertical bifacial fences
                    # (the beam split by the side of the panel it reaches) and
                    # single- and dual-axis trackers (their orientation follows
                    # the sun). Each also writes a *_sun.csv the planner uses
                    # for the shading between rows.
                    vertical_basis_no_vegetation_path = gs.compute_surface_solar_energy(
                        hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades_no_vegetation",
                        solar_df=annual_solar["df"],
                        aoi=aoi,
                        svf_path=svf_no_vegetation_path,
                        mode="vertical_basis",
                        out_path=f"{wd}/{municipality_name}/{year_to_analyse}/annual_solar_exposure_no_vegetation.tif",
                        overwrite=recalculate_solar_exposure,
                    )
                    tracking_basis_no_vegetation_path = gs.compute_surface_solar_energy(
                        hillshade_dir=f"{wd}/{municipality_name}/{year_to_analyse}/hillshades_no_vegetation",
                        solar_df=annual_solar["df"],
                        aoi=aoi,
                        svf_path=svf_no_vegetation_path,
                        mode="tracking_basis",
                        out_path=f"{wd}/{municipality_name}/{year_to_analyse}/annual_solar_exposure_no_vegetation.tif",
                        overwrite=recalculate_solar_exposure,
                    )
                    success("Vertical and tracking bases written")

                with indent("Planning PV fields and agrivoltaics over the parcels"):
                    # Every scenario in the catalogue - fixed, single- and
                    # dual-axis PV fields, solar grazing, overhead, interspace,
                    # single-axis and vertical bifacial agrivoltaics, each crop
                    # and pitch, rows North-South and East-West - is laid out
                    # on every agricultural parcel and valued against what the
                    # parcel earns today. Pass economics=gs.Economics(...),
                    # systems=..., crops=... or baseline_margin_eur_ha=... to
                    # change the assumptions, and criterion= to rank by "irr",
                    # "roi", "lcoe", "payback", "energy" or "ler" instead.
                    # Parcels already planted in rows also get agrivoltaics
                    # built into that plantation: overhead over every k-th
                    # row, or tables down every k-th alley. Those alley
                    # tables stand among the trees, so they are costed on the
                    # basis with vegetation.
                    parcel_scenarios, parcel_best, parcel_rows = gs.plan_parcel_pv_scenarios(
                        sigpac_parcels,
                        poa_basis_path=poa_basis_no_vegetation_path,
                        crop_rows=crop_row_lines,
                        vegetated_basis_path=poa_basis_path,
                        vertical_basis_path=vertical_basis_no_vegetation_path,
                        tracking_basis_path=tracking_basis_no_vegetation_path,
                        economics=gs.Economics(),
                        criterion="npv",
                        latitude_deg=aoi_centroid.y,
                        output_path=parcel_pv_path,
                        output_summary_path=parcel_pv_summary_path,
                        progress=True,
                    )
                    if len(parcel_best):
                        viable = parcel_best[parcel_best.best_viable]
                        success(
                            f"{len(parcel_scenarios)} scenarios over "
                            f"{len(parcel_best)} parcels; {len(viable)} parcels "
                            f"have a scenario with positive NPV"
                        )
                        for kind, group in viable.groupby("best_kind"):
                            info(f"  best is {kind}: {len(group)} parcels, "
                                 f"{group.best_peak_kwp.sum() / 1000:.1f} MWp, "
                                 f"{group.best_energy_mwh.sum() / 1000:.1f} GWh/year")
                    else:
                        warning("No parcel could be laid out")

        success(f"Completed processing for {municipality_name.title()}!")



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