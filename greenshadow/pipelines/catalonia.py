# -*- coding: utf-8 -*-
"""
Solar pipeline for a municipality of Catalonia.

The area comes from the INE census tracts (hypercadaster_ES), the terrain from
the ICGC territorial LiDAR, the building footprints that correct its
classification from the Spanish cadastre, and the rural parcels from SIGPAC.

    import greenshadow as gs

    gs.pipelines.run_catalonia_municipalities(
        {"Alpicat": "25023"},
        wd="/mnt/data/greenshadow",
        year=2023,
        days=[dt.date(2023, 12, 22)],
    )

or from the shell::

    python -m greenshadow.pipelines catalonia 25023 --name Alpicat \\
        --wd /mnt/data/greenshadow --year 2023 --day 2023-12-22
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Mapping, Optional, Sequence, Union

from ..buildings_infer import detect_roof_planes_tiled, filter_roof_planes_by_aoi
from ..dsm_dem_estimation import build_digital_model_from_lidar_tiled
from ..lidar import (icgc_download_from_geojson, icgc_lidar_acquisition_date,
                     icgc_tiles_from_geojson)
from ..logging_utils import indent, info, section_context, success, warning
from ..roofs_3d import roof_planes_3d
from ..rural_parcels import (load_spanish_preserved_regions,
                             load_spanish_sigpac_parcels,
                             tag_parcels_with_preserved_areas)
from .base import (PipelineConfig, PipelineResult, RoofPlanes, SolarPipeline,
                   TerrainModels, run_pipelines)

# LiDAR classes of the ICGC territorial LiDAR v3r1:
# https://datacloud.ide.cat/especificacions/lidar-territorial-v3r1-20250905.pdf
BUILDING_CLASSES = (6, 17, 77)


@dataclass
class CataloniaConfig(PipelineConfig):
    """``PipelineConfig`` plus where the Catalan sources are kept."""
    hypercadaster_dir: Optional[str] = None     # default: {wd}/hypercadaster_ES
    laz_dir: Optional[str] = None               # default: {wd}/icgc, shared by municipalities
    preserved_areas_dir: Optional[str] = None   # default: {wd}/preserved_areas
    rural_parcels_year: int = 2025
    # Flag the parcels inside Natura 2000 / ENP areas (they stay in, flagged).
    tag_preserved_areas: bool = False
    epsg: int = 25831                           # UTM 31N
    download_workers: int = 8
    # Write an interactive 3D page of the segmented roofs next to the
    # GeoPackage. Useful for checking the detection before trusting a full run.
    render_roof_3d_view: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.hypercadaster_dir is None:
            self.hypercadaster_dir = os.path.join(self.wd, "hypercadaster_ES")
        if self.laz_dir is None:
            self.laz_dir = os.path.join(self.wd, "icgc")
        if self.preserved_areas_dir is None:
            self.preserved_areas_dir = os.path.join(self.wd, "preserved_areas")


class CataloniaMunicipalityPipeline(SolarPipeline):
    """
    One Catalan municipality, by its cadastral code.

    ``name`` names the output directory ``{wd}/{name}``; it defaults to the
    municipality's name in the census tracts.
    """

    def __init__(self, cadastral_code: str, config: CataloniaConfig,
                 name: Optional[str] = None):
        super().__init__(name, config)
        self.cadastral_code = str(cadastral_code)
        self.tiles = None
        self.laz_paths = None

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = self._municipality["municipality_name"]
        return self._name

    @cached_property
    def _municipality(self):
        import hypercadaster_ES as hc
        from shapely.ops import unary_union

        hc_dir = self.config.hypercadaster_dir
        hc.download(wd=hc_dir, cadaster_codes=[self.cadastral_code])
        census = hc.mergers.get_census_gdf(
            census_tract_dir=hc.utils.census_tracts_dir_(hc_dir))
        ine_codes = hc.utils.cadaster_to_ine_codes(cadaster_codes=[self.cadastral_code])
        tracts = census[census.ine_municipality_code.isin(ine_codes)]
        if tracts.empty:
            raise ValueError(f"No census tracts for cadastral code {self.cadastral_code}")
        return {
            "municipality_name": tracts.municipality_name.iloc[0],
            "geometry": unary_union(tracts.geometry.tolist()),
        }

    def area(self):
        return self._municipality["geometry"]

    # ------------------------------------------------------------- parcels

    def load_parcels(self, aoi):
        cfg = self.config
        parcels = load_spanish_sigpac_parcels(
            municipality_codes=self.cadastral_code,
            year=cfg.rural_parcels_year,
            workdir=self.path("rural_parcels"),
            target_crs=cfg.epsg,
        )
        if cfg.tag_preserved_areas:
            preserved = load_spanish_preserved_regions(
                data_dir=cfg.preserved_areas_dir,
                datasets=["natura2000", "enp"],
                join_geometries_per_resource=True,
            )
            parcels = tag_parcels_with_preserved_areas(
                parcels, preserved, min_intersection_ratio=0.01)
        return parcels

    # ------------------------------------------------------------- terrain

    def _download_lidar(self, aoi):
        cfg = self.config
        with section_context("Finding and downloading ICGC LiDAR tiles", level=2):
            self.tiles = icgc_tiles_from_geojson(
                aoi=aoi,
                aoi_epsg=4326,
                aoi_buffer_in_meters=cfg.aoi_buffer_m,
                pdf_plot_path=self.path("tiles.pdf"),
                cache_path=self.path("tiles.geojson"),
            )
            success(f"Found {len(self.tiles)} matching tiles")
            self.laz_paths = icgc_download_from_geojson(
                out_dir=cfg.laz_dir,
                tiles=self.tiles,
                aoi=aoi,
                aoi_epsg=4326,
                aoi_buffer_in_meters=cfg.aoi_buffer_m,
                max_workers=cfg.download_workers,
            )
            success(f"Downloaded {len(self.laz_paths)} LAZ files")

    def _cadastre_buildings(self):
        """Cadastral footprints, to correct the LiDAR building classification."""
        import hypercadaster_ES as hc

        hc_dir = self.config.hypercadaster_dir
        try:
            buildings = hc.mergers.join_cadaster_building(
                gdf=None,
                cadaster_dir=hc.utils.cadaster_dir_(hc_dir),
                census_tracts_dir=hc.utils.census_tracts_dir_(hc_dir),
                DEM_raster_dir=hc.utils.DEM_raster_dir_(hc_dir),
                cadaster_codes=[self.cadastral_code],
                results_dir=hc.utils.results_dir_(hc_dir),
                open_street_dir=hc.utils.open_street_dir_(hc_dir),
            )
        except Exception as e:
            warning(f"Could not get building geometries for classification correction: {e}")
            return None
        if buildings.empty:
            warning("No building geometries found for classification correction")
            return None
        buildings = (buildings[["building_geometry"]]
                     .rename(columns={"building_geometry": "geometry"})
                     .set_geometry("geometry")
                     .to_crs(epsg=self.config.epsg))
        info(f"Found {len(buildings)} buildings for classification correction")
        return buildings

    def build_terrain(self, aoi) -> TerrainModels:
        cfg = self.config
        t = TerrainModels(
            dsm=self.path("dsm.tif"),
            dsm_class=self.path("dsm_class.tif"),
            dsm_slope=self.path("dsm_slope.tif"),
            dsm_aspect=self.path("dsm_aspect.tif"),
            dsm_no_vegetation=self.path("dsm_no_vegetation.tif"),
            dsm_no_vegetation_class=self.path("dsm_no_vegetation_class.tif"),
            dsm_no_vegetation_slope=self.path("dsm_no_vegetation_slope.tif"),
            dsm_no_vegetation_aspect=self.path("dsm_no_vegetation_aspect.tif"),
            dem=self.path("dem.tif"),
            dem_slope=self.path("dem_slope.tif"),
            dem_aspect=self.path("dem_aspect.tif"),
            svf=self.path("svf.tif"),
            svf_no_vegetation=self.path("svf_no_vegetation.tif"),
        )
        # The tiles are needed by the roof detector even when every raster is
        # already on disk; finding them is cached and downloading skips the
        # files already there.
        self._download_lidar(aoi)

        rebuild = cfg.recalculate_digital_models
        missing = [p for p in (t.dsm, t.dsm_no_vegetation, t.dem) if not os.path.exists(p)]
        buildings = self._cadastre_buildings() if (missing or rebuild) else None

        common = dict(
            laz_paths=self.laz_paths,
            resolution=cfg.resolution,
            epsg=cfg.epsg,
            aoi=aoi,
            aoi_epsg=4326,
            aoi_buffer_in_meters=cfg.aoi_buffer_m,
            aggregation="max",
            # Classification correction from the cadastral footprints
            classification_correction_buildings_gdf=buildings,
            classification_correction_buildings_mode="strict",
            classification_correction_building_class_value=6,
            classification_correction_building_buffer_m=1,
            smooth_window_size=11,
            smooth_iterations=1,
        )
        # Construction cranes: the jib returns sit tens of metres above the
        # roofs and win the "max" aggregation, leaving a streak that casts a
        # long false shadow. Removed only when thin AND elongated AND standing
        # clear of everything around it AND seen through by the laser - so
        # chimneys, lift shafts and spires stay.
        surface = dict(common, building_classes=BUILDING_CLASSES,
                       fill_gaps="smart_general", crane_removal=True)

        if not os.path.exists(t.dsm) or rebuild:
            with indent("Generating Digital Surface Model (DSM)"):
                build_digital_model_from_lidar_tiled(
                    out_path=t.dsm,
                    out_classification_path=t.dsm_class,
                    out_slope_path=t.dsm_slope,
                    out_aspect_path=t.dsm_aspect,
                    classification=(1, 2, 3, 4, 5, 6, 8, 9, 17, 77),
                    smooth_classes=(2, 3, 4, 5, 6, 8, 9, 17),
                    smooth_mode={"terrain": "gaussian", "buildings": None,
                                 "vegetation": None},
                    **surface,
                )
                success(f"DSM generated: {t.dsm}")
        else:
            info(f"Using existing DSM: {t.dsm}")

        if not os.path.exists(t.dsm_no_vegetation) or rebuild:
            with indent("Generating Digital Surface Model (DSM) without vegetation"):
                build_digital_model_from_lidar_tiled(
                    out_path=t.dsm_no_vegetation,
                    out_classification_path=t.dsm_no_vegetation_class,
                    out_slope_path=t.dsm_no_vegetation_slope,
                    out_aspect_path=t.dsm_no_vegetation_aspect,
                    classification=(1, 2, 6, 8, 17, 9, 77),
                    smooth_classes=(2, 8, 6, 17, 9, 77),
                    smooth_mode={"terrain": "gaussian", "buildings": None},
                    **surface,
                )
                success(f"DSM without vegetation generated: {t.dsm_no_vegetation}")
        else:
            info(f"Using existing DSM without vegetation: {t.dsm_no_vegetation}")

        if not os.path.exists(t.dem) or rebuild:
            with indent("Generating Digital Elevation Model (DEM)"):
                build_digital_model_from_lidar_tiled(
                    out_path=t.dem,
                    out_slope_path=t.dem_slope,
                    out_aspect_path=t.dem_aspect,
                    classification=(2, 8, 17, 9),
                    fill_gaps="dem",
                    # Drop the points under the buildings of the DSM
                    filter_raster_path=t.dsm_class,
                    filter_raster_band=1,
                    filter_points_from_class=[6, 17, 77],
                    filter_mode="exclude",
                    # and the isolated ones, before rasterising
                    isolated_points_exclusion_ratio=0.9,
                    isolated_window_size=7,
                    # Class 6 because classification_correction_building_class_value=6
                    smooth_classes=(1, 2, 6, 8, 17, 9),
                    smooth_unary=True,
                    smooth_mode="gaussian",
                    **common,
                )
                success(f"DEM generated: {t.dem}")
        else:
            info(f"Using existing DEM: {t.dem}")
        return t

    def svf_options(self, class_path):
        return dict(
            hybrid=True,
            subsample_factor=3,
            building_buffer_m=250,
            building_classes=(6, 17, 76, 77),
            vegetation_classes=(3, 4, 5),
        )

    def acquisition_date(self, aoi):
        # ICGC publishes a flight window per 1 km tile, so how stale the point
        # cloud is comes from the data rather than a guess.
        return icgc_lidar_acquisition_date(aoi, cache_dir=self.config.laz_dir)

    # ------------------------------------------------------------- roofs

    def detect_roofs(self, aoi, terrain: TerrainModels) -> Optional[RoofPlanes]:
        cfg = self.config
        roof_planes_path = self.path("roofs.gpkg")
        # The detector's own output, facets still on the pixel grid.
        raster_planes_path = roof_planes_path.replace(".gpkg", "_rasterised.gpkg")
        # The refined DSM the detector builds from the LiDAR is the surface the
        # planes were fitted to, so it is the one to sample for their heights.
        roof_dsm_path = terrain.dsm_no_vegetation.replace(".tif", "_roofs.tif")

        if (not os.path.exists(raster_planes_path) or not os.path.exists(roof_dsm_path)
                or cfg.recalculate_digital_models):
            with indent("Detecting the roof planes"):
                planes = detect_roof_planes_tiled(**self.roof_detection_options(
                    aoi, terrain, raster_planes_path))
                success(f"Roof planes detected: {len(planes)} facets on "
                        f"{planes.part_id.nunique()} roofs")
        else:
            # Trims a file detected before the AOI filter existed; a no-op on
            # one detected with it.
            planes = filter_roof_planes_by_aoi(raster_planes_path, aoi, aoi_crs="EPSG:4326")
            info(f"Using existing roof planes: {raster_planes_path}")

        if cfg.render_roof_3d_view:
            self._render_roofs_3d(planes, roof_dsm_path if os.path.exists(roof_dsm_path)
                                  else terrain.dsm_no_vegetation,
                                  roof_planes_path.replace(".gpkg", "_3d.html"))
        return RoofPlanes(planes=planes, dsm_path=roof_dsm_path)

    def roof_detection_options(self, aoi, terrain: TerrainModels, output_path) -> dict:
        """
        Arguments of ``detect_roof_planes_tiled``, tuned on the digitised
        Alpicat roofs. Override in a subclass to experiment.

        The raster detector is used rather than the point-native v2: its region
        growing stays continuous across a roof, where v2 stops at every crease
        and thin patch. What it reads is improved instead: the LiDAR returns
        are segmented into planes over their Delaunay graph and each point is
        pulled toward its segment's plane before rasterising the refined roof
        DSM. That drops the ranging noise the slope raster inherits without
        rounding off the ridges, which is why smooth_sigma stays off.
        """
        return dict(
            dsm_path=terrain.dsm_no_vegetation,
            slope_path=terrain.dsm_no_vegetation_slope,
            aspect_path=terrain.dsm_no_vegetation_aspect,
            dsm_class_path=terrain.dsm_no_vegetation_class,
            dem_path=terrain.dem,
            laz_dir=self.config.laz_dir,
            laz_tiles=self.tiles,
            min_height_m=1.5,
            building_class_value=(6, 7, 12, 17, 77),
            lidar_class_values=(1, 2, 6, 7, 8, 12, 17, 77),
            lidar_refinement=True,
            # 0.20 m only pays off because refined_aggregation="tin" fills
            # every pixel from the triangulation. Raise it to 0.25-0.30 if the
            # returns are sparser than ~8 pts/m2: on a 6 pts/m2 test 0.20 cost
            # 15 points of coverage.
            resolution_refined=0.20,
            # Sample the refined DSM off the Delaunay triangulation of the
            # returns instead of binning them. Binning leaves about half the
            # pixels empty at this density, and the gap filler that takes them
            # minimises curvature - rounding the facets being looked for.
            refined_aggregation="tin",
            # 8, not 9: averaged over three RANSAC seeds, 9 cost 0.015 of mean
            # IoU against the digitised Alpicat roofs and 7 cost 0.02.
            normal_tolerance_deg=8,
            ransac_threshold=0.05,   # measured inert 0.05-0.2
            # Slope is in RADIANS; drops the near-vertical wall pixels.
            dsm_max_slope_filter=1.9,
            # 3 m2 keeps the small facets (dormers, hip ends): at 8 their mean
            # IoU was 0.30, at 3 it is 0.51. 2 and 4 were both worse overall.
            min_roof_plane_area=3,
            buffer_m=0.0,
            # Off: an isotropic gaussian on the finished DSM rounds ridges as
            # hard as it removes noise. The Delaunay smoothing does it
            # crease-aware.
            smooth_sigma=0.0,
            # 0.2, tuned on the digitised Alpicat roofs. Higher values MERGE
            # small facets: at 0.7 the mean IoU fell from 0.64 to 0.53.
            delaunay_smooth_strength=0.2,
            preclustering_roof_planes=True,
            ridge_splitting_roof_planes=False,
            # Region growing releases the pixels it rejects, so small facets get
            # their own plane; the slight over-segmentation of large facets is
            # undone by a looser merge. The offset guard stops two stacked flat
            # roofs - same orientation, different height - merging.
            merge_normal_angle_deg=6.0,
            merge_max_offset_m=0.3,
            # Distance-only assignment measured better than adding a normal
            # term (IoU 0.636 vs 0.623).
            assignment_normal_weight=0.0,
            # Give the roof pixels region growing left in no plane to the
            # neighbour they fit, or a plane of their own: coverage 90% ->
            # 94.5%. 0.15 m measured best of 0.10-0.40.
            recover_distance_m=0.15,
            # Merge touching planes that are one surface split in two: normals
            # within 8 deg and no step over 0.15 m along the shared seam. Mean
            # IoU 0.686 -> 0.698 on every seed; 12 deg already merges real
            # facets, and a 0.3 m step lets parallel planes merge.
            coplanar_merge_angle_deg=8.0,
            coplanar_merge_step_m=0.15,
            # Keep the skirt: along the eaves the refined DSM drops steeply over
            # its last pixel or two, so those pixels fit no plane and the
            # outline came out short of the real eave. They are handed to the
            # nearest plane at the very end. IoU 0.697 -> 0.703, coverage 91% ->
            # 95%.
            skirt_assign_distance_m=0.3,
            # Roof clutter removal (chimneys, aerials, AC units) is OFF: pass
            # svf_path=terrain.svf to turn it on, with svf_obstruction_* tuned
            # against the 48 digitised obstacles in
            # real_roof_obstacles_9000_9500_4000_5000_Alpicat.gpkg
            # (drop=0.12, min_height_m=0, close_m=0.6, min_area_m2=0.5,
            # max_area_m2=15, max_elongation=2.5, max_planes=1). It recalls
            # 16/48 obstacles at 52% precision - limited by the 0.5 m SVF, which
            # cannot resolve the shadow ring of a ~1.2 m object.
            #
            # 1 km matches the ICGC LiDAR tiles, so a tile mostly reads one LAZ
            # file. Peak memory per tile on Alpicat: 10 GB for the old town,
            # 1-7 GB elsewhere.
            tile_size_m=1000.0,
            # Only the buildings touching the municipality itself: the rasters
            # reach aoi_buffer_m beyond it for the shadows, not for the roofs.
            aoi=aoi,
            aoi_crs="EPSG:4326",
            output_path=output_path,
        )

    def _render_roofs_3d(self, planes, dsm_path, out_path):
        # Orbit the roofs, hover a face for its tilt and azimuth. The LiDAR
        # returns, coloured by class, sit under the planes, so a plane floating
        # off its roof's returns shows up at once.
        with indent("Rendering the 3D roof view"):
            roof_planes_3d(
                planes,
                dsm_path=dsm_path,
                output_path=out_path,
                color_by="plane",
                laz_dir=self.config.laz_dir,
                laz_tiles=self.tiles,
                title=f"Roof planes - {self.name}",
                progress=True,
            )
            success(f"3D roof view: {out_path}")


def run_catalonia_municipalities(
    municipalities: Union[Mapping[str, str], Sequence[str], str],
    config: Optional[CataloniaConfig] = None,
    **config_options,
) -> Dict[str, PipelineResult]:
    """
    Run the Catalan pipeline over one or more municipalities.

    ``municipalities`` is ``{name: cadastral_code}``, or cadastral codes alone
    (the names then come from the census tracts). Settings are a
    ``CataloniaConfig`` or its fields as keyword arguments::

        run_catalonia_municipalities({"Alpicat": "25023"},
                                     wd="/mnt/data/greenshadow", year=2023)
    """
    if config is None:
        config = CataloniaConfig(**config_options)
    elif config_options:
        raise TypeError("Pass either config or its fields as keyword arguments, not both")
    if isinstance(municipalities, str):
        municipalities = [municipalities]
    if isinstance(municipalities, Mapping):
        items = list(municipalities.items())
    else:
        items = [(None, code) for code in municipalities]
    return run_pipelines([CataloniaMunicipalityPipeline(code, config, name=name)
                          for name, code in items])


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Solar irradiation and PV potential of Catalan municipalities.")
    parser.add_argument("codes", nargs="+", help="cadastral municipality codes, e.g. 25023")
    parser.add_argument("--name", action="append",
                        help="output name per code, in the same order (default: census name)")
    parser.add_argument("--wd", required=True, help="working directory")
    parser.add_argument("--year", type=int, help="year of the annual analysis")
    parser.add_argument("--day", action="append", default=[], type=dt.date.fromisoformat,
                        help="day of a daily analysis, YYYY-MM-DD (repeatable)")
    parser.add_argument("--no-roofs", action="store_true")
    parser.add_argument("--no-parcels", action="store_true")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--opaque-vegetation", action="store_true")
    parser.add_argument("--recalculate-digital-models", action="store_true")
    parser.add_argument("--recalculate-solar-exposure", action="store_true")
    args = parser.parse_args(argv)

    if args.name and len(args.name) != len(args.codes):
        parser.error("give one --name per code, or none")
    municipalities = (dict(zip(args.name, args.codes)) if args.name else args.codes)
    run_catalonia_municipalities(
        municipalities,
        wd=args.wd,
        year=args.year,
        days=args.day,
        roofs=not args.no_roofs,
        parcels=not args.no_parcels,
        include_weather=not args.no_weather,
        daily_videos=not args.no_videos,
        semitransparent_vegetation=not args.opaque_vegetation,
        recalculate_digital_models=args.recalculate_digital_models,
        recalculate_solar_exposure=args.recalculate_solar_exposure,
    )
