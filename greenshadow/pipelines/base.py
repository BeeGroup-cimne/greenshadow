# -*- coding: utf-8 -*-
"""
Region-agnostic solar pipeline.

Everything that follows once there is a terrain model is the same wherever the
terrain came from: the weather at the site, the sky view factor, the sun
positions, the hillshades with cast shadows, the irradiation rasters and the PV
layouts on roofs and parcels. ``SolarPipeline`` runs that part. What depends on
the region - where the area of interest comes from, how the DSM and DEM are
built, which parcels and roofs there are - is left to a subclass:

==========================  ====================================================
hook                        returns
==========================  ====================================================
``area()``                  the AOI, a shapely geometry in EPSG:4326 (required)
``build_terrain(aoi)``      a ``TerrainModels`` with the rasters (required)
``load_parcels(aoi)``       parcels to lay PV fields and agrivoltaics on, or None
``detect_roofs(aoi, t)``    a ``RoofPlanes`` to lay PV on, or None
``acquisition_date(aoi)``   when the terrain was surveyed, to age the canopy
``svf_options(...)``        extra arguments for ``calculate_sky_view_factor``
``load_weather(aoi)``       hourly weather; ERA5-Land by default, which is global
==========================  ====================================================

A pipeline built on a bare DEM (no LiDAR) fills only ``dsm`` and ``dem`` - the
same raster in both - and the stages that need more simply do not run: no
canopy, no crop rows, no elevated shadows, no hillshades "without vegetation"
(the terrain has none), and no roofs.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from ..logging_utils import (error, indent, info, section, section_context,
                             success, summary, warning)
from ..parcel_pv import Economics, plan_parcel_pv_scenarios
from ..dimensionate import plan_pv_strings_in_buildings
from ..crop_rows import detect_crop_rows
from ..shader import (NO_GROWTH, apply_height_shadow_to_hillshades,
                      build_canopy_model_from_rasters,
                      calculate_sky_view_factor, compute_surface_solar_energies,
                      compute_surface_solar_energy, delete_matching_files,
                      generate_azimuth_altitude_pairs_by_date,
                      generate_azimuth_altitude_pairs_by_year,
                      generate_daily_shadow_mp4,
                      generate_multiple_hillshades_gdal, vegetation_growth)


@dataclass
class PipelineConfig:
    """
    Settings shared by every pipeline.

    At least one of ``year`` (annual irradiation and PV layouts) and ``days``
    (daily shadows, videos and irradiation) must be given.
    """
    wd: str
    # When year is given, the aggregated shades and irradiation over that year
    # are computed, and the PV layouts on roofs and parcels with them.
    year: Optional[int] = None
    # When days are given, the shades and irradiation of those specific days,
    # e.g. the solstices. Best kept within one year.
    days: Sequence[dt.date] = ()
    tz_local: str = "Europe/Madrid"

    # Weather: ERA5-Land at the AOI centroid, with ERA5's own direct radiation.
    include_weather: bool = True
    weather_data_dir: Optional[str] = None      # default: {wd}/weather
    # How ERA5-Land's all-sky GHI (ssrd, clouds included) is split into direct
    # and diffuse at each sun position. Against PVGIS-SARAH3 at Lleida, 2023
    # (docs/irradiance_era5.md): "disc" is closest on the annual DNI (-2 %) and
    # on a South 30 deg plane (-0.5 %); "dirint" is within 1 % on vertical
    # faces; "era5" - the beam share of ERA5's own fdir - follows the hours best
    # but puts 12 % too much in the beam and 16 % too little in the diffuse;
    # "erbs" is 10 % short on DNI.
    irradiance_decomposition: str = "disc"

    # Terrain
    resolution: float = 0.5                     # metres
    aoi_buffer_m: float = 500.0                 # shadows cast from beyond the AOI
    svf_search_radius_m: float = 250.0
    svf_angular_resolution_deg: float = 10.0

    # Sun sampling
    daily_granularity: str = "PT15M"
    annual_granularity: str = "PT1H"
    daily_solar_step_deg: float = 0.1
    annual_solar_step_deg: float = 3.0

    # Daily analysis
    elevated_height_m: Optional[float] = 1.5    # shadows at this height above ground; None skips
    daily_videos: bool = True

    # Treat trees as a semi-transparent medium instead of an opaque wall: high
    # vegetation is resolved into individual crowns sized from H = DSM - DEM,
    # and the hillshades gain three path-length bands so one raster serves
    # every month. Needs a classified terrain with and without vegetation.
    semitransparent_vegetation: bool = True
    vegetation_leaf_type: int = 0               # 0 = deciduous, 1 = evergreen
    # Age the canopy from the survey date to the simulated year.
    grow_vegetation_to_simulated_year: bool = True
    vegetation_height_growth_m_per_year: float = 0.25
    vegetation_crown_growth_m_per_year: float = 0.20

    # What to plan
    roofs: bool = True
    parcels: bool = True
    crop_rows: bool = True
    economics: Economics = field(default_factory=Economics)
    criterion: str = "npv"
    # Passed through to plan_pv_strings_in_buildings / plan_parcel_pv_scenarios,
    # over the defaults the pipeline sets.
    roof_pv_options: Dict[str, Any] = field(default_factory=dict)
    parcel_pv_options: Dict[str, Any] = field(default_factory=dict)

    # Caching: every product on disk is reused unless one of these says not to.
    recalculate_digital_models: bool = False
    # Rebuild the irradiation rasters from the existing hillshades. The
    # hillshades depend only on the terrain and the sun, the energy rasters on
    # the weather as well, so a change in how the weather is read needs only
    # this.
    recalculate_solar_exposure: bool = False
    cleanup_xml: bool = True

    def __post_init__(self):
        if self.year is None and not self.days:
            raise ValueError("Give a year, days, or both to analyse")
        if isinstance(self.days, dt.date):
            self.days = [self.days]
        self.days = list(self.days)
        if self.weather_data_dir is None:
            self.weather_data_dir = os.path.join(self.wd, "weather")

    @property
    def simulated_year(self) -> int:
        return self.year if self.year is not None else self.days[0].year


@dataclass
class TerrainModels:
    """
    The rasters a pipeline's terrain stage produces.

    Only ``dsm`` and ``dem`` are required. ``dsm_no_vegetation`` is the surface
    with the trees taken out - what a PV field built on a parcel would see once
    the parcel is cleared; when it is None the DSM has no vegetation to remove
    and is used in its place. The class rasters carry the LiDAR classification.
    """
    dsm: str
    dem: str
    dsm_class: Optional[str] = None
    dsm_slope: Optional[str] = None
    dsm_aspect: Optional[str] = None
    dsm_no_vegetation: Optional[str] = None
    dsm_no_vegetation_class: Optional[str] = None
    dsm_no_vegetation_slope: Optional[str] = None
    dsm_no_vegetation_aspect: Optional[str] = None
    dem_slope: Optional[str] = None
    dem_aspect: Optional[str] = None
    svf: Optional[str] = None
    svf_no_vegetation: Optional[str] = None

    @property
    def has_vegetation(self) -> bool:
        """Whether there is a separate surface without vegetation."""
        return self.dsm_no_vegetation is not None and self.dsm_no_vegetation != self.dsm

    @property
    def bare_dsm(self) -> str:
        return self.dsm_no_vegetation if self.has_vegetation else self.dsm

    @property
    def bare_svf(self) -> Optional[str]:
        return self.svf_no_vegetation if self.has_vegetation else self.svf


@dataclass
class RoofPlanes:
    """Roof facets to lay PV on, and the DSM their heights are read from."""
    planes: Any                 # GeoDataFrame from detect_roof_planes(_tiled)
    dsm_path: str


@dataclass
class PipelineResult:
    """Everything a run produced. Paths are on disk; frames are in memory."""
    name: str
    out_dir: str
    aoi: Any = None
    terrain: Optional[TerrainModels] = None
    weather: Any = None
    parcels: Any = None
    crop_row_lines: Any = None
    crop_row_parcels: Any = None
    roofs: Optional[RoofPlanes] = None
    canopy: Any = None
    # {date: {"hillshades": dir, "hillshades_no_vegetation": dir,
    #         "hillshades_elevated": dir, "daily_solar_energy": path}}
    daily: Dict[dt.date, Dict[str, str]] = field(default_factory=dict)
    # {mode: path} of the annual irradiation rasters, with vegetation, and
    # the same without it
    annual: Dict[str, str] = field(default_factory=dict)
    annual_no_vegetation: Dict[str, str] = field(default_factory=dict)
    roof_pv_modules: Any = None
    roof_pv_summary: Any = None
    parcel_pv_scenarios: Any = None
    parcel_pv_best: Any = None
    parcel_pv_tables: Any = None


class SolarPipeline:
    """
    Base class of the solar pipelines; see the module docstring for the hooks.

    ``run()`` executes every stage and returns a ``PipelineResult``. Every
    product lands under ``out_dir`` (``{wd}/{name}``) and is reused on the next
    run, so a run that stops part-way resumes where it stopped.
    """

    def __init__(self, name: str, config: PipelineConfig):
        self._name = name
        self.config = config

    # ------------------------------------------------------------------ hooks

    @property
    def name(self) -> str:
        return self._name

    @property
    def out_dir(self) -> str:
        return os.path.join(self.config.wd, self.name)

    def path(self, *parts) -> str:
        """A path under ``out_dir``."""
        return os.path.join(self.out_dir, *parts)

    def area(self):
        """The area of interest, a shapely geometry in EPSG:4326."""
        raise NotImplementedError

    def build_terrain(self, aoi) -> TerrainModels:
        """Build (or reuse) the DSM, DEM and their derivatives over the AOI."""
        raise NotImplementedError

    def load_parcels(self, aoi):
        """Parcels for PV fields and agrivoltaics, or None."""
        return None

    def detect_roofs(self, aoi, terrain: TerrainModels) -> Optional[RoofPlanes]:
        """Roof facets for rooftop PV, or None."""
        return None

    def acquisition_date(self, aoi) -> Optional[dt.date]:
        """When the terrain was surveyed; None leaves the canopy as surveyed."""
        return None

    def svf_options(self, class_path: Optional[str]) -> Dict[str, Any]:
        """Extra arguments for ``calculate_sky_view_factor``."""
        return {}

    def load_weather(self, aoi):
        """
        Hourly weather of the analysed years at the AOI, as
        ``generate_azimuth_altitude_pairs_by_*`` read it (GHI, DHI, DNI ...).

        ERA5-Land at the centroid, plus ERA5's own all-sky direct radiation
        (fdir) and cloud cover (tcc) for the same hours, so the beam/sky split
        can come from the reanalysis' clouds instead of a model. ERA5 is
        global, so a region overrides this only to use another source.
        """
        from ..weather import load_ERA5Land_with_direct_radiation

        cfg = self.config
        years = set(day.year for day in cfg.days)
        if cfg.year is not None:
            years.add(cfg.year)
        lat, lon = aoi.centroid.y, aoi.centroid.x
        with indent("Loading ERA5-Land data, with ERA5 direct radiation"):
            weather = load_ERA5Land_with_direct_radiation(
                data_dir=cfg.weather_data_dir,
                lat_range=[lat, lat],
                lon_range=[lon, lon],
                ym_range=[dt.datetime(min(years), 1, 1), dt.datetime(max(years), 12, 31)],
                update=False,
                use_point_location=True,
                tz_local=cfg.tz_local,
                mode="instantaneous",
            )
        success(f"Loaded weather data: {len(weather)} records")
        return weather

    # ------------------------------------------------------------------ run

    def run(self) -> PipelineResult:
        cfg = self.config
        with section_context(f"Processing {self.name}"):
            aoi = self.area()
            os.makedirs(self.out_dir, exist_ok=True)
            result = PipelineResult(name=self.name, out_dir=self.out_dir, aoi=aoi)
            centroid = aoi.centroid
            info(f"Centroid: {centroid.y:.3f}, {centroid.x:.3f}")

            result.weather = self._load_weather(aoi)

            if cfg.parcels:
                with section_context("Get rural parcels", level=2):
                    result.parcels = self.load_parcels(aoi)

            with section_context("Generating Digital Models", level=2):
                terrain = self.build_terrain(aoi)
                self._sky_view_factors(terrain)
                result.terrain = terrain

                if result.parcels is not None and cfg.crop_rows and terrain.has_vegetation:
                    result.crop_row_lines, result.crop_row_parcels = self._crop_rows(
                        result.parcels, terrain)

                if cfg.roofs:
                    result.roofs = self.detect_roofs(aoi, terrain)

            result.canopy = self._canopy(aoi, terrain)

            if cfg.days:
                with section_context("Solar Shading Analysis", level=2):
                    for day in cfg.days:
                        result.daily[day] = self._daily(day, aoi, terrain, result)

            if cfg.year is not None:
                with section_context(f"Annual Solar Analysis ({cfg.year})", level=2):
                    self._annual(aoi, terrain, result)

            if cfg.cleanup_xml:
                self._cleanup()

            success(f"Completed processing for {self.name}!")
        return result

    # ------------------------------------------------------------------ stages

    def _load_weather(self, aoi):
        if not self.config.include_weather:
            return None
        with section_context("Loading Weather Data", level=2):
            try:
                return self.load_weather(aoi)
            except Exception as e:
                error(f"Could not load weather data: {e}")
                warning("Continuing without weather analysis")
                return None

    def _sky_view_factors(self, terrain: TerrainModels):
        cfg = self.config
        jobs = [("svf", terrain.dsm, terrain.dsm_class, "Sky View Factor")]
        # The sky a PV field on a cleared parcel would see: the trees come out
        # of the horizon too.
        if terrain.has_vegetation:
            jobs.append(("svf_no_vegetation", terrain.dsm_no_vegetation,
                         terrain.dsm_no_vegetation_class,
                         "Sky View Factor without vegetation"))
        for attr, dsm, class_path, label in jobs:
            out_path = getattr(terrain, attr) or self.path(f"{attr}.tif")
            setattr(terrain, attr, out_path)
            if os.path.exists(out_path) and not cfg.recalculate_digital_models:
                info(f"Using existing {label}: {out_path}")
                continue
            with indent(f"Calculating {label}"):
                calculate_sky_view_factor(
                    dsm_path=dsm,
                    out_path=out_path,
                    search_radius=cfg.svf_search_radius_m,
                    angular_resolution=cfg.svf_angular_resolution_deg,
                    cuda=None,  # GPU only if CUDA (driver + libNVVM) is usable
                    class_path=class_path,
                    **self.svf_options(class_path),
                )
                success(f"{label} calculated: {out_path}")

    def _crop_rows(self, parcels, terrain: TerrainModels):
        # How each rural parcel is planted. On a woody crop the canopy height
        # model is a comb, and its direction and pitch are the plantation
        # geometry any agrivoltaic row has to fit between. On wheat or lettuce
        # there is no canopy to read, so the parcel reports its working axis
        # instead - the way the field is already driven. The rows are a
        # function of the digital models, so they are recomputed exactly when
        # those are; otherwise the GeoPackage on disk is the answer.
        cfg = self.config
        crop_rows_path = self.path("crop_rows.gpkg")
        with indent("Detecting the crop rows"):
            if os.path.exists(crop_rows_path) and not cfg.recalculate_digital_models:
                info(f"Using existing crop rows: {crop_rows_path}")
            lines, row_parcels = detect_crop_rows(
                parcels,
                dsm_path=terrain.dsm,
                dsm_no_vegetation_path=terrain.dsm_no_vegetation,
                # Without the class raster a farm shed is a tree, and two
                # sheds are a row.
                dsm_class_path=terrain.dsm_class,
                dem_slope_path=terrain.dem_slope,
                dem_aspect_path=terrain.dem_aspect,
                output_path=crop_rows_path,
                update=cfg.recalculate_digital_models,
                progress=True,
            )
            planted = row_parcels[row_parcels.row_pattern == "rows"]
            success(f"Crop rows: {len(lines)} rows over {len(planted)} "
                    f"planted parcels of {len(row_parcels)}")
            for pattern, group in row_parcels.groupby("row_pattern"):
                info(f"  {pattern}: {len(group)} parcels")
            if len(planted):
                info(f"  median spacing {planted.row_spacing_m.median():.2f} m, "
                     f"canopy {planted.canopy_height_p50_m.median():.2f} m tall and "
                     f"{planted.canopy_width_m.median():.2f} m wide, "
                     f"alley {planted.alley_width_m.median():.2f} m")
                # A parcel can hold more than one plantation, each with its
                # own direction; the parcel's azimuth is the largest block's.
                mixed = planted[planted.n_blocks > 1]
                info(f"  {len(mixed)} planted parcels hold more than one direction"
                     + (f", up to {mixed.row_azimuth_spread_deg.max():.0f} degrees apart"
                        if len(mixed) else ""))
        return lines, row_parcels

    def _canopy(self, aoi, terrain: TerrainModels):
        cfg = self.config
        if not (cfg.semitransparent_vegetation and terrain.has_vegetation
                and terrain.dsm_class is not None):
            return None
        with section_context("Semi-transparent canopy", level=2):
            growth = NO_GROWTH
            surveyed = (self.acquisition_date(aoi)
                        if cfg.grow_vegetation_to_simulated_year else None)
            if surveyed is not None:
                growth = vegetation_growth(
                    surveyed,
                    cfg.simulated_year,
                    height_rate_m_per_year=cfg.vegetation_height_growth_m_per_year,
                    crown_rate_m_per_year=cfg.vegetation_crown_growth_m_per_year,
                )
                info(f"Surveyed {surveyed:%Y-%m-%d}, simulating {cfg.simulated_year}: "
                     f"{growth.years:+.1f} y -> {growth.height_gain_m:+.2f} m height, "
                     f"{growth.crown_gain_m:+.2f} m crown diameter")
            canopy = build_canopy_model_from_rasters(
                dsm_path=terrain.dsm,
                dem_path=terrain.dem,
                class_path=terrain.dsm_class,
                under_canopy_path=terrain.dsm_no_vegetation,
                growth=growth,
            )
            success(f"{canopy.trees['row'].size} tree crowns detected")
            return canopy

    def _weather_pandas(self, weather):
        if weather is None:
            return None
        return weather.to_pandas() if hasattr(weather, "to_pandas") else weather

    def _daily(self, day: dt.date, aoi, terrain: TerrainModels, result: PipelineResult):
        cfg = self.config
        canopy = result.canopy
        leaf_type = cfg.vegetation_leaf_type if canopy is not None else None
        day_dir = self.path(day.strftime("%Y%m%d"))
        products = {}

        with section_context(f"Processing {day}", level=1):
            with indent("Generating solar positions"):
                solar = generate_azimuth_altitude_pairs_by_date(
                    date=day,
                    lat=aoi.centroid.y,
                    lon=aoi.centroid.x,
                    tz_name=cfg.tz_local,
                    granularity=cfg.daily_granularity,
                    solar_degrees_discretization=cfg.daily_solar_step_deg,
                    only_daylight=True,
                    weather_df=self._weather_pandas(result.weather),
                    irradiance_decomposition=cfg.irradiance_decomposition,
                )
                success(f"Generated {len(solar['azimuth_altitude_pairs'])} solar positions")

            sets = [("hillshades", terrain.dsm, canopy, leaf_type,
                     "Generating hillshades with cast shadows")]
            if terrain.has_vegetation:
                sets.append(("hillshades_no_vegetation", terrain.dsm_no_vegetation, None, None,
                             "Generating hillshades with cast shadows in a scenario of no vegetation"))
            for key, dsm, set_canopy, set_leaf_type, label in sets:
                hillshade_dir = os.path.join(day_dir, key)
                with indent(label):
                    paths = generate_multiple_hillshades_gdal(
                        raster_dir=hillshade_dir,
                        dsm_path=dsm,
                        aoi=aoi,
                        azimuth_altitude=solar["azimuth_altitude_pairs"],
                        related_datetimes=solar["df"]["datetime_local"].tolist(),
                        include_datetime_info=False,
                        qgis_temporal=False,
                        canopy=set_canopy,
                    )
                    success(f"Generated {len(paths)} hillshades")
                    if cfg.daily_videos:
                        info("Generating the hillshades video...")
                        generate_daily_shadow_mp4(
                            solar_df=solar["df"],
                            hillshade_dir=hillshade_dir,
                            out_mp4=os.path.join(day_dir, f"{key}.mp4"),
                            fps=10,
                            duration_seconds=10,
                            zoom=4,
                            frame_size=(1920, 1088),
                            overwrite=False,
                            canopy_leaf_type=set_leaf_type,
                        )
                products[key] = hillshade_dir

            # Shadows at a height above the ground, e.g. the top of a crop or
            # a person; meaningless when the DSM is the bare ground itself.
            if cfg.elevated_height_m is not None and terrain.dem != terrain.dsm:
                h = cfg.elevated_height_m
                elevated_dir = os.path.join(day_dir, f"hillshades_dem{h}m")
                with indent(f"Applying height-based shadows at {h}m elevation"):
                    elevated = apply_height_shadow_to_hillshades(
                        hillshade_dir=products["hillshades"],
                        dem_path=terrain.dem,
                        dsm_path=terrain.dsm,
                        aoi=aoi,
                        out_dir=elevated_dir,
                        height_m=h,
                        pattern="*.tif",
                        recursive=True,
                    )
                    success(f"Applied elevation shadows to {len(elevated)} files")
                    if cfg.daily_videos:
                        info("Generating the hillshades video...")
                        generate_daily_shadow_mp4(
                            solar_df=solar["df"],
                            hillshade_dir=elevated_dir,
                            out_mp4=os.path.join(day_dir, f"hillshades_dem{h}m.mp4"),
                            fps=10,
                            duration_seconds=10,
                            zoom=3,
                            frame_size=(1920, 1088),
                        )
                products["hillshades_elevated"] = elevated_dir

            if result.weather is None:
                warning("No weather data: daily solar energy skipped")
                return products
            with indent("Computing daily solar energy exposure"):
                products["daily_solar_energy"] = compute_surface_solar_energy(
                    hillshade_dir=products["hillshades"],
                    solar_df=solar["df"],
                    svf_path=terrain.svf,
                    aoi=aoi,
                    out_path=os.path.join(day_dir, "daily_solar_energy.tif"),
                    canopy_leaf_type=leaf_type,
                    overwrite=cfg.recalculate_solar_exposure,
                )
        return products

    def _annual(self, aoi, terrain: TerrainModels, result: PipelineResult):
        cfg = self.config
        year_dir = self.path(str(cfg.year))
        canopy = result.canopy
        plan_parcels = result.parcels is not None

        with indent("Generating annual solar positions"):
            solar = generate_azimuth_altitude_pairs_by_year(
                year=cfg.year,
                lat=aoi.centroid.y,
                lon=aoi.centroid.x,
                tz_name=cfg.tz_local,
                granularity=cfg.annual_granularity,
                solar_degrees_discretization=cfg.annual_solar_step_deg,
                only_daylight=True,
                weather_df=self._weather_pandas(result.weather),
                irradiance_decomposition=cfg.irradiance_decomposition,
            )
            success(f"Generated {len(solar['azimuth_altitude_pairs'])} unique "
                    f"solar positions for the year")

        # Modes, read in one pass over each hillshade stack:
        #  - "total": Wh on the surface itself, the plane-of-array irradiation
        #    of a module mounted flush with an inclined roof;
        #  - "fourier_poa": orientation-free basis, for racks whose tilt does
        #    not match the surface underneath (flat roofs, PV fields);
        #  - "vertical_basis" / "tracking_basis": vertical bifacial fences and
        #    single- and dual-axis trackers, whose orientation the fixed basis
        #    cannot cost. They also write a *_sun.csv the parcel planner uses
        #    for the shading between rows.
        modes = ["total", "fourier_poa", "fourier_decomposition"]
        bare_modes = ["fourier_poa", "vertical_basis", "tracking_basis"] if plan_parcels else []
        if not terrain.has_vegetation:
            modes += [m for m in bare_modes if m not in modes]

        with indent("Generating annual hillshades"):
            generate_multiple_hillshades_gdal(
                raster_dir=os.path.join(year_dir, "hillshades"),
                dsm_path=terrain.dsm,
                azimuth_altitude=solar["azimuth_altitude_pairs"],
                qgis_temporal=False,
                include_datetime_info=False,
                canopy=canopy,
            )
            success("Annual hillshades generation complete")

        # The irradiation is the weather's: without it there is nothing to
        # weight the hillshades with.
        if result.weather is None:
            warning("No weather data: annual solar energy and PV layouts skipped")
            return

        with indent("Creating annual average solar exposure"):
            result.annual = compute_surface_solar_energies(
                hillshade_dir=os.path.join(year_dir, "hillshades"),
                solar_df=solar["df"],
                aoi=aoi,
                svf_path=terrain.svf,
                modes=tuple(modes),
                out_path=os.path.join(year_dir, "annual_solar_exposure.tif"),
                canopy_leaf_type=cfg.vegetation_leaf_type if canopy is not None else None,
                overwrite=cfg.recalculate_solar_exposure,
            )
            success("Annual solar exposure rasters written")

        if result.roofs is not None:
            self._roof_pv(aoi, result)

        if not plan_parcels:
            return

        # PV fields and agrivoltaics over the parcels. Building either one
        # clears what grows there today, so the irradiance comes from
        # hillshades of the DSM without vegetation; the parcel's current crop
        # enters as the margin the scenario gives up, not as shade.
        if terrain.has_vegetation:
            with indent("Annual hillshades without vegetation"):
                generate_multiple_hillshades_gdal(
                    raster_dir=os.path.join(year_dir, "hillshades_no_vegetation"),
                    dsm_path=terrain.dsm_no_vegetation,
                    azimuth_altitude=solar["azimuth_altitude_pairs"],
                    qgis_temporal=False,
                    include_datetime_info=False,
                )
                result.annual_no_vegetation = compute_surface_solar_energies(
                    hillshade_dir=os.path.join(year_dir, "hillshades_no_vegetation"),
                    solar_df=solar["df"],
                    aoi=aoi,
                    svf_path=terrain.svf_no_vegetation,
                    modes=tuple(bare_modes),
                    out_path=os.path.join(year_dir, "annual_solar_exposure_no_vegetation.tif"),
                    overwrite=cfg.recalculate_solar_exposure,
                )
                success(f"No-vegetation POA basis: {result.annual_no_vegetation['fourier_poa']}")
                success("Vertical and tracking bases written")
        else:
            result.annual_no_vegetation = {m: result.annual[m] for m in bare_modes}

        self._parcel_pv(aoi, result)

    def _roof_pv(self, aoi, result: PipelineResult):
        cfg = self.config
        with indent("Planning the PV strings"):
            options = dict(
                # Roof capacity, not a wiring design: no electrical grouping,
                # no shading filter, no orientation refused, and the array
                # packed to 5 cm of the plane contour.
                string_size=None,
                min_poa_ratio=None,
                allowed_azimuth_range=None,
                setback_m=0.05,
                phase_steps=5,
                output_path=self.path("pv_modules.gpkg"),
                output_summary_path=self.path("pv_building_summary.csv"),
                progress=True,
            )
            options.update(cfg.roof_pv_options)
            modules, pv_summary = plan_pv_strings_in_buildings(
                result.roofs.planes,
                poa_basis_path=result.annual["fourier_poa"],
                annual_energy_path=result.annual["total"],
                dsm_path=result.roofs.dsm_path,
                latitude_deg=aoi.centroid.y,
                **options,
            )
            result.roof_pv_modules, result.roof_pv_summary = modules, pv_summary
            # One row per building and mounting scenario; "best" is the
            # per-plane optimum and the only one covering a whole roof.
            if len(pv_summary):
                best = pv_summary[pv_summary.scenario == "best"]
                success(f"Placed {(modules.scenario == 'best').sum()} modules "
                        f"({best.peak_kw.sum() / 1000:.2f} MWp, "
                        f"{best.energy_kwh.sum() / 1e6:.2f} GWh/year) "
                        f"over {len(best)} buildings")
                for scenario, rows in pv_summary.groupby("scenario"):
                    info(f"  {scenario}: {rows.n_modules.sum()} modules, "
                         f"{rows.energy_kwh.sum() / 1e6:.2f} GWh/year "
                         f"over {len(rows)} buildings")
            else:
                warning("No PV modules could be placed on the detected roofs")

    def _parcel_pv(self, aoi, result: PipelineResult):
        cfg = self.config
        bare = result.annual_no_vegetation
        with indent("Planning PV fields and agrivoltaics over the parcels"):
            # Every scenario in the catalogue - fixed, single- and dual-axis PV
            # fields, solar grazing, overhead, interspace, single-axis and
            # vertical bifacial agrivoltaics, each crop and pitch, rows N-S and
            # E-W - is laid out on every agricultural parcel and valued against
            # what the parcel earns today. Parcels already planted in rows also
            # get agrivoltaics built into that plantation; their alley tables
            # stand among the trees, so they are costed on the basis with
            # vegetation. Override through config.parcel_pv_options.
            options = dict(
                crop_rows=result.crop_row_lines,
                vegetated_basis_path=result.annual["fourier_poa"],
                vertical_basis_path=bare["vertical_basis"],
                tracking_basis_path=bare["tracking_basis"],
                economics=cfg.economics,
                criterion=cfg.criterion,
                latitude_deg=aoi.centroid.y,
                output_path=self.path("parcel_pv.gpkg"),
                output_summary_path=self.path("parcel_pv_scenarios.csv"),
                progress=True,
            )
            options.update(cfg.parcel_pv_options)
            scenarios, best, tables = plan_parcel_pv_scenarios(
                result.parcels, poa_basis_path=bare["fourier_poa"], **options)
            result.parcel_pv_scenarios = scenarios
            result.parcel_pv_best = best
            result.parcel_pv_tables = tables
            if len(best):
                viable = best[best.best_viable]
                success(f"{len(scenarios)} scenarios over {len(best)} parcels; "
                        f"{len(viable)} parcels have a scenario with positive NPV")
                for kind, group in viable.groupby("best_kind"):
                    info(f"  best is {kind}: {len(group)} parcels, "
                         f"{group.best_peak_kwp.sum() / 1000:.1f} MWp, "
                         f"{group.best_energy_mwh.sum() / 1000:.1f} GWh/year")
                if tables is not None and len(tables):
                    info(f"  {tables.scenario.nunique()} scenario layers, "
                         f"{(tables.scenario == 'best').sum()} tables in ppv_best")
            else:
                warning("No parcel could be laid out")

    def _cleanup(self):
        with indent("Removing temporary XML files"):
            deleted, errors = delete_matching_files(
                self.out_dir, "*.xml", mode="glob", dry_run=False)
            if deleted:
                success(f"Cleaned up {len(deleted)} temporary files")
            if errors:
                error(f"Errors during cleanup: {len(errors)}")


def run_pipelines(pipelines: Sequence[SolarPipeline]) -> Dict[str, PipelineResult]:
    """Run several pipelines one after the other and print a closing summary."""
    results = {}
    for pipeline in pipelines:
        results[pipeline.name] = pipeline.run()
    if pipelines:
        cfg = pipelines[0].config
        section("Analysis Complete")
        with indent():
            summary({
                "Areas processed": f"{len(results)} ({', '.join(results)})",
                "Output directory": cfg.wd,
                "Daily analysis": (f"✓ ({', '.join(d.strftime('%Y-%m-%d') for d in cfg.days)})"
                                   if cfg.days else "✗"),
                "Yearly analysis": f"✓ ({cfg.year})" if cfg.year is not None else "✗",
            })
    return results
