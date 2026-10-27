# -*- coding: utf-8 -*-
"""
greenshadow.parcel_pv

Synthetic PV and agrivoltaic layouts over SIGPAC parcels, costed and ranked.

Every agricultural parcel is laid out again from scratch, once per scenario: a
mounting system, a row direction - North-South or East-West - and a distance
between rows. Agrivoltaic systems take that distance from the crop that is to
be grown between or under the panels, so the same system appears once per
crop; a PV field takes the closest pitch at which no row shades the next at
winter-solstice noon. Each layout is costed against the annual plane-of-array
basis written by ``compute_surface_solar_energy(mode="fourier_poa")`` and
valued over the project's life, and the parcel's best scenario is chosen by a
selectable criterion - net present value by default, not energy: the most
energy is always the densest field, which is rarely the best investment once
the crop it displaces is counted.

The basis for a PV field should be the one built on the DSM **without
vegetation**: a field clears the parcel, and the trees standing on it today
are not going to shade the panels that replace them.

A parcel already planted in rows - the rows ``detect_crop_rows`` found - also
gets agrivoltaics built into that plantation, always as agrivoltaics: an
overhead structure over every k-th crop row, or low tables down the middle of
every k-th alley wide enough to take them. Those follow each block's own row
direction instead of North-South or East-West, keep the crop that is there,
and - in the alleys, among the trees - are costed on a basis built **with**
the vegetation, so the plantation's shade is counted.

Rows are laid on the basis raster's own grid. North-South rows are runs of
pixel columns and East-West rows runs of pixel rows, so a row is cut wherever
it leaves the parcel (set in by ``setback_m``) and kept only where it is at
least ``min_row_length_m`` long. The rows are aligned with the projection's
grid North, which in UTM differs from true North by the meridian convergence -
under two degrees across Catalonia, and nothing a row azimuth cares about.

Mountings and the annual basis
------------------------------
Fixed tables are costed on the ``fourier_poa`` basis, which stores the annual
direct beam projected onto three fixed axes: exact for any panel that never
sees the sun from behind - South-facing panels at any tilt, East- or
West-facing ones at a low one. Vertical bifacial fences and trackers need
sums that basis cannot give back, and read two more, written from the same
hillshades (``docs/pv_tracking_irradiance_plan.md``):

* ``vertical_basis`` - the lit beam split by which side of a vertical panel it
  reaches, so a fence's East face gets the mornings and its West face the
  afternoons;
* ``tracking_basis`` - direct, sky and ground terms for a dual-axis tracker
  and for a horizontal North-South single-axis one, accumulated at each sun
  position's own panel orientation.

Shading of one row by the next depends on the layout, not the pixel, so it
is worked out per layout over the sun table written beside those bases and
applied as a factor: backtracking for single-axis trackers, the share of a
dual-axis tracker its neighbours cover, the shadow a fence casts up the next.
Single-axis trackers are laid on North-South rows only. Dual-axis trackers
are not laid in rows at all: each stands on its own point of a lattice spaced
so that its neighbours barely shade it (``tracker_max_shading_loss``).

What comes out
--------------
Like ``plan_pv_strings_in_buildings``, every scenario costed survives into
the output with its panels: one polygon per table - each continuous stretch
of modules down a row, at its plan width - or per dual-axis tracker, written
one GeoPackage layer per scenario so they can be compared side by side, plus
``best`` and the best of each kind, which take each parcel's own winner.

Agronomy and economics
----------------------
Every number in ``DEFAULT_SYSTEMS``, ``DEFAULT_CROPS``, ``Economics`` and
``SIGPAC_USE_MARGIN_EUR_HA`` is a literature value or a placeholder for one,
and every one of them is meant to be replaced: pass your own catalogue, crop
table or economics. The crop response to shade is the piecewise-linear shape
of Laub et al. (2022): nothing lost up to a crop-specific reduction in
radiation, then a steady loss beyond it. How much radiation a layout takes off
the crop is a first-order estimate from the share of the ground under panels,
not a ray trace.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import LineString
try:
    from .logging_utils import tqdm
except ImportError:
    from greenshadow.logging_utils import tqdm

try:
    from .dimensionate import (
        DEFAULT_MODULES,
        _pixel_centres,
        _polygon_pixel_mask,
        _read_window,
        solar_min_altitude,
        write_scenario_layers,
    )
    from .rural_parcels import filter_agricultural_parcels, sigpac_parcel_id
    from .shader import reconstruct_poa_from_basis, read_sun_table
except ImportError:  # running from a checkout without the package installed
    from greenshadow.dimensionate import (
        DEFAULT_MODULES,
        _pixel_centres,
        _polygon_pixel_mask,
        _read_window,
        solar_min_altitude,
        write_scenario_layers,
    )
    from greenshadow.rural_parcels import filter_agricultural_parcels, sigpac_parcel_id
    from greenshadow.shader import reconstruct_poa_from_basis, read_sun_table


# ============================================================
# CATALOGUE
# ============================================================

#: Row directions. North-South rows carry modules split between East and West
#: faces; East-West rows carry them facing South.
DIRECTIONS = ("NS", "EW")


@dataclass(frozen=True)
class MountingSystem:
    """
    One way of putting modules on a parcel.

    ``kind`` is ``"pv_field"`` when the parcel stops being farmed and
    ``"agrivoltaic"`` when a crop or a flock stays on it.

    The table is ``modules_across`` portrait modules deep across the row. On
    East-West rows it faces South at ``tilt_south_deg``; on North-South rows it
    is an East-West "tent", ``modules_across`` modules down each side at
    ``tilt_east_west_deg``.

    Where the crop grows: under the panels when ``under_panel_cropped`` -
    overhead structures, and grazing under a field - losing only
    ``land_loss_frac`` of the ground to posts; otherwise only between the
    rows, minus the table's own footprint and ``crop_buffer_m`` either side of
    it that the machinery cannot work.

    ``mounting`` is ``"fixed"`` (the tilts above), ``"vertical"`` (a bifacial
    fence ``modules_across`` landscape modules high: East and West faces on
    North-South rows, South and North on East-West ones, the rear worth
    ``bifaciality`` of the front), ``"single_axis"`` (horizontal North-South
    axis, ``modules_across`` portrait modules across, backtracking at
    ``ground_cover_ratio`` when no crop sets the pitch) or ``"dual_axis"``
    (trackers ``modules_across`` landscape modules high and ``modules_along``
    wide, each on its own pole). The last three need the vertical or tracking
    basis.

    Dual-axis trackers are not laid in rows: each stands on a point of a
    rectangular lattice, at least ``tracker_spacing_m`` (East-West,
    North-South) apart and further if that is what it takes for its
    neighbours to cost it no more than ``tracker_max_shading_loss`` of its
    annual direct beam.

    ``layout`` says where the rows come from. ``"synthetic"`` systems are laid
    out from scratch, North-South and East-West. ``"rows_overhead"`` and
    ``"rows_alley"`` systems are built into a plantation that is already there,
    along the rows ``detect_crop_rows`` found: over every ``row_steps``-th crop
    row, or down the middle of every ``row_steps``-th alley - every row by
    default. They are always agrivoltaic, and the crop they keep is the one
    already growing. The plantation's spacing is fixed, so the table is fitted
    to it rather than the other way round: ``modules_across`` and the tilts are
    the most the table may take, and it is narrowed, turned to landscape and
    laid flatter until it fits (``_fit_table``).
    """

    key: str
    kind: str
    label: str
    tilt_south_deg: float
    tilt_east_west_deg: float
    modules_across: int
    clearance_m: float
    capex_eur_wp: float
    opex_eur_kwp_year: float
    bifacial_gain: float = 0.0
    under_panel_cropped: bool = False
    land_loss_frac: float = 0.0
    crop_buffer_m: float = 0.0
    layout: str = "synthetic"
    row_steps: tuple = ()
    mounting: str = "fixed"
    bifaciality: float = 0.0
    ground_cover_ratio: float = None
    modules_along: int = 0
    tracker_spacing_m: tuple = None
    tracker_max_rotation_deg: float = 60.0
    tracker_max_shading_loss: float = 0.02


@dataclass(frozen=True)
class CropProfile:
    """
    A crop - or a flock - that can share a parcel with panels.

    ``pitches_m`` maps a system key to the row pitches that crop allows under
    it, driven by the width of the machinery that has to pass between rows;
    ``None`` in place of a pitch means the system's own no-shading pitch.
    ``gross_margin_eur_ha`` is the margin per cropped hectare in the open, and
    ``shade_threshold`` / ``shade_slope`` the yield response to shade: no loss
    up to that reduction in radiation, ``shade_slope`` of relative yield per
    unit of reduction beyond it.
    """

    key: str
    label: str
    gross_margin_eur_ha: float
    shade_threshold: float
    shade_slope: float
    pitches_m: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Economics:
    """
    Project-life assumptions, all in real terms.

    The defaults are Spanish utility-scale values for 2025: solar captured just
    under 40 EUR/MWh on the spot market, and PPAs closed between 32 and 37.
    ``fixed_project_cost_eur`` - grid connection, permitting, fencing - does
    not scale with the plant and is what makes a small parcel a poor project;
    it is zero by default because it depends on where the nearest substation
    is, and that is not in this library's data.
    """

    electricity_price_eur_mwh: float = 40.0
    price_escalation: float = 0.0
    discount_rate: float = 0.06
    lifetime_years: int = 30
    degradation_per_year: float = 0.005
    fixed_project_cost_eur: float = 0.0


#: Phase-one catalogue. Capital costs are EUR per Wp installed: ground-mounted
#: fixed tilt about 0.55 (IRENA 2024 world average 691 USD/kW, Spain below
#: it); an overhead agrivoltaic structure roughly doubles the mounting and
#: site work (Fraunhofer ISE: 130-220 EUR/kW mounting, 250-350 EUR/kW site);
#: wide interspace rows spread the cabling and the civil works over fewer
#: modules. Category I overhead clearance is over 2.1 m (DIN SPEC 91434).
DEFAULT_SYSTEMS = (
    MountingSystem(
        key="pv_fixed",
        kind="pv_field",
        label="PV field, fixed tilt",
        tilt_south_deg=30.0,
        tilt_east_west_deg=10.0,
        modules_across=2,
        clearance_m=0.8,
        capex_eur_wp=0.55,
        opex_eur_kwp_year=12.0,
    ),
    MountingSystem(
        key="apv_grazing",
        kind="agrivoltaic",
        label="Solar grazing, fixed tilt",
        tilt_south_deg=30.0,
        tilt_east_west_deg=10.0,
        modules_across=2,
        clearance_m=1.2,
        capex_eur_wp=0.58,
        opex_eur_kwp_year=10.0,
        under_panel_cropped=True,
        land_loss_frac=0.02,
    ),
    MountingSystem(
        key="apv_overhead",
        kind="agrivoltaic",
        label="Elevated (overhead) agrivoltaics",
        tilt_south_deg=20.0,
        tilt_east_west_deg=10.0,
        modules_across=1,
        clearance_m=4.5,
        capex_eur_wp=0.95,
        opex_eur_kwp_year=15.0,
        bifacial_gain=0.07,
        under_panel_cropped=True,
        land_loss_frac=0.02,
    ),
    MountingSystem(
        key="apv_interspace",
        kind="agrivoltaic",
        label="Interspace agrivoltaics, crops between rows",
        tilt_south_deg=30.0,
        tilt_east_west_deg=15.0,
        modules_across=1,
        clearance_m=0.8,
        capex_eur_wp=0.62,
        opex_eur_kwp_year=13.0,
        crop_buffer_m=0.5,
    ),
    # Phase two: the mountings the fixed-orientation basis cannot cost. A
    # single-axis tracker is about 12 % dearer than a fixed rack and a
    # dual-axis one about 40 % (lifecycle surveys of utility trackers), with
    # more moving parts to maintain. Single-axis PV fields run at a ground
    # cover ratio of 0.35-0.45; agrivoltaic ones at 10-12 m pitch over a
    # 2.4 m axis (Rutgers agrivoltaic farms). Vertical bifacial fences stand
    # 8-12 m apart (Next2Sun), two landscape modules high. Dual-axis
    # trackers stand 21 x 23 m apart at least, and further where that still
    # leaves their neighbours more than 2 % of their beam: at 21 x 23 m they
    # take about 5 % under Lleida's sky, where a tenth of the beam comes
    # from below 10 degrees, and holding them to 1 % takes 60 m.
    MountingSystem(
        key="pv_single_axis",
        kind="pv_field",
        label="PV field, single-axis tracker",
        tilt_south_deg=0.0,
        tilt_east_west_deg=0.0,
        modules_across=1,
        clearance_m=1.5,
        capex_eur_wp=0.62,
        opex_eur_kwp_year=14.0,
        mounting="single_axis",
        ground_cover_ratio=0.35,
    ),
    MountingSystem(
        key="pv_dual_axis",
        kind="pv_field",
        label="PV field, dual-axis tracker",
        tilt_south_deg=0.0,
        tilt_east_west_deg=0.0,
        modules_across=5,
        clearance_m=1.0,
        capex_eur_wp=0.78,
        opex_eur_kwp_year=18.0,
        mounting="dual_axis",
        modules_along=6,
        tracker_spacing_m=(21.0, 23.0),
    ),
    MountingSystem(
        key="apv_single_axis",
        kind="agrivoltaic",
        label="Agrivoltaics, single-axis tracker over wide rows",
        tilt_south_deg=0.0,
        tilt_east_west_deg=0.0,
        modules_across=1,
        clearance_m=2.4,
        capex_eur_wp=0.70,
        opex_eur_kwp_year=15.0,
        crop_buffer_m=0.75,
        mounting="single_axis",
    ),
    MountingSystem(
        key="apv_vertical_bifacial",
        kind="agrivoltaic",
        label="Agrivoltaics, vertical bifacial fences",
        tilt_south_deg=90.0,
        tilt_east_west_deg=90.0,
        modules_across=2,
        clearance_m=0.8,
        capex_eur_wp=0.75,
        opex_eur_kwp_year=13.0,
        crop_buffer_m=0.5,
        mounting="vertical",
        bifaciality=0.75,
    ),
    # Over an existing plantation, on every row: an orchard or a vineyard is
    # planted 3.5-4 m apart because that is what its tractors need, so a PV
    # row can go over each crop row - the posts stand in the tree line - or
    # down an alley only where a tractor lane is left beside it. The overhead
    # structure is the one of a new field and has to clear the canopy; the
    # alley tables are the interspace ones.
    MountingSystem(
        key="apv_rows_overhead",
        kind="agrivoltaic",
        label="Overhead agrivoltaics over the existing crop rows",
        tilt_south_deg=20.0,
        tilt_east_west_deg=10.0,
        modules_across=1,
        clearance_m=4.5,
        capex_eur_wp=0.95,
        opex_eur_kwp_year=15.0,
        bifacial_gain=0.07,
        under_panel_cropped=True,
        land_loss_frac=0.02,
        layout="rows_overhead",
        row_steps=(1,),
    ),
    MountingSystem(
        key="apv_rows_alley",
        kind="agrivoltaic",
        label="Agrivoltaics in the alleys of the existing crop rows",
        tilt_south_deg=25.0,
        tilt_east_west_deg=15.0,
        modules_across=1,
        clearance_m=0.8,
        capex_eur_wp=0.62,
        opex_eur_kwp_year=13.0,
        crop_buffer_m=0.5,
        layout="rows_alley",
        row_steps=(1,),
    ),
)

#: How the crop already on a parcel answers to shade, by SIGPAC use, as
#: ``(threshold, slope)`` in the same sense as ``CropProfile``. Fruit trees
#: and vines hold their yield under a fair amount of shade (Laub et al. 2022,
#: fruits and berries), olive a little less, nuts less again; anything not
#: listed takes the cautious default.
SIGPAC_USE_SHADE_RESPONSE = {
    "FY": (0.30, 0.4), "CI": (0.30, 0.4), "FF": (0.30, 0.4), "CF": (0.30, 0.4),
    "CS": (0.30, 0.4), "OV": (0.25, 0.5), "OF": (0.25, 0.5), "OC": (0.25, 0.5),
    "VO": (0.25, 0.5), "VI": (0.25, 0.4), "VF": (0.25, 0.4), "CV": (0.25, 0.4),
    "FS": (0.20, 0.6), "FL": (0.20, 0.6),
}
DEFAULT_SHADE_RESPONSE = (0.15, 0.6)

#: What to grow with the panels. Margins are indicative EUR per cropped
#: hectare and year and should be replaced by regional farm accounts. The shade
#: response follows the crop groups of Laub et al. (2022): cereals and forage
#: lose yield steadily once radiation drops by 10-15 %, leafy vegetables and
#: fruit hold up to 25-30 % and lose slowly beyond it. Pitches are the distance
#: between PV rows the crop's machinery needs: a cereal combine wants 10-12 m,
#: a vegetable bed 6-8 m; overhead structures are spaced by the crop's light
#: needs instead, since the machinery passes underneath.
DEFAULT_CROPS = (
    CropProfile(
        key="cereal",
        label="Rainfed cereal (wheat, barley)",
        gross_margin_eur_ha=300.0,
        shade_threshold=0.10,
        shade_slope=0.9,
        pitches_m={"apv_interspace": (10.0, 12.0), "apv_overhead": (7.0,),
                   "apv_single_axis": (10.0, 12.0),
                   "apv_vertical_bifacial": (10.0, 12.0)},
    ),
    CropProfile(
        key="forage",
        label="Forage (alfalfa, grass)",
        gross_margin_eur_ha=500.0,
        shade_threshold=0.15,
        shade_slope=0.6,
        pitches_m={"apv_interspace": (8.0, 10.0), "apv_overhead": (6.0, 8.0),
                   "apv_single_axis": (8.0, 10.0),
                   "apv_vertical_bifacial": (8.0, 10.0)},
    ),
    CropProfile(
        key="vegetables",
        label="Leafy vegetables",
        gross_margin_eur_ha=3000.0,
        shade_threshold=0.25,
        shade_slope=0.5,
        pitches_m={"apv_interspace": (6.0, 8.0), "apv_overhead": (5.0, 7.0),
                   "apv_single_axis": (8.0, 10.0)},
    ),
    CropProfile(
        key="berries",
        label="Berries and soft fruit",
        gross_margin_eur_ha=6000.0,
        shade_threshold=0.30,
        shade_slope=0.4,
        pitches_m={"apv_overhead": (5.0, 7.0)},
    ),
    CropProfile(
        key="vineyard",
        label="Vineyard, a PV row every few vine rows",
        gross_margin_eur_ha=1500.0,
        shade_threshold=0.25,
        shade_slope=0.4,
        pitches_m={"apv_interspace": (7.5, 9.0)},
    ),
    CropProfile(
        key="grazing",
        label="Sheep grazing",
        gross_margin_eur_ha=80.0,
        shade_threshold=0.30,
        shade_slope=0.3,
        pitches_m={"apv_grazing": (None,)},
    ),
)

#: Indicative margin of what each SIGPAC use earns today, EUR per hectare and
#: year: the baseline every scenario is measured against, so that a field
#: built over an orchard pays for the orchard it removes. Placeholders to be
#: replaced by regional accounts; a use missing here counts as earning nothing.
SIGPAC_USE_MARGIN_EUR_HA = {
    "TA": 300.0,    # arable, mostly rainfed cereal
    "TH": 3000.0,   # market gardens
    "IV": 3000.0,   # greenhouses
    "FY": 2000.0,   # fruit trees
    "FS": 800.0,    # nuts
    "FL": 900.0,    # nuts and olive
    "FF": 1500.0,   # fruit and nuts
    "OV": 800.0,    # olive
    "VI": 1500.0,   # vineyard
    "VF": 1500.0,
    "VO": 1200.0,
    "OF": 1200.0,
    "OC": 1200.0,
    "CI": 2500.0,   # citrus
    "CF": 2000.0,
    "CS": 1500.0,
    "CV": 1500.0,
    "PS": 50.0,     # pasture
    "PR": 80.0,     # pasture with scrub
    "PA": 50.0,     # wooded pasture
}

#: What the best scenario is chosen by, and whether more is better.
CRITERIA = {
    "npv": True,
    "irr": True,
    "roi": True,
    "lcoe": False,
    "payback": False,
    "energy": True,
    "ler": True,
}

#: Gap kept between two modules down a row, and between the two sides of a
#: tent, in metres.
_MODULE_GAP_M = 0.02

#: Aisle between two North-South tents. Their low tilt throws a shadow only
#: when the sun is low in the East or West, and an aisle this wide is what
#: maintenance needs anyway.
_TENT_AISLE_M = 1.0

#: Gap left between two overhead tables over neighbouring crop rows, so that
#: rain and some light still reach the crop between them.
_OVERHEAD_GAP_M = 0.5

#: Row origin offsets tried per layout, as fractions of the pitch. The parcel
#: edge decides how many rows fit, and a third of a pitch either way can be
#: worth one.
_PHASES = (0.0, 1.0 / 3.0, 2.0 / 3.0)


# ============================================================
# ROW GEOMETRY
# ============================================================

def _table(system, direction, module):
    """
    Cross-section of one row: plan width, modules across, and their faces.

    Returns ``(plan_width_m, slope_length_m, tilt_deg, faces)``, ``faces``
    being ``(azimuth_deg, share_of_modules)`` pairs: South for East-West rows,
    East and West halves for North-South tents.
    """
    slope_length = system.modules_across * module.length_m

    if direction == "EW":
        tilt = system.tilt_south_deg
        plan = slope_length * np.cos(np.radians(tilt))
        return plan, slope_length, tilt, ((180.0, 1.0),)

    tilt = system.tilt_east_west_deg
    plan = 2.0 * slope_length * np.cos(np.radians(tilt)) + _MODULE_GAP_M
    return plan, slope_length, tilt, ((90.0, 0.5), (270.0, 0.5))


def _no_shading_pitch(system, direction, module, latitude_deg):
    """
    Closest pitch at which no row shades the next at winter-solstice noon.

    South-facing rows cast their shadow straight down the field at noon, so
    the gap is the table's height over the tangent of the solstice altitude -
    the same rule the roof racks use. A North-South tent's noon shadow falls
    along its own row, and its low tilt keeps the morning and evening ones
    short, so it only needs the aisle.
    """
    plan, slope_length, tilt, _ = _table(system, direction, module)

    if direction == "NS":
        return plan + _TENT_AISLE_M

    altitude = max(solar_min_altitude(latitude_deg), 5.0)
    rise = slope_length * np.sin(np.radians(tilt))

    return plan + rise / np.tan(np.radians(altitude))


#: Where each basis sits in the band stack ``_evaluate_parcel`` samples:
#: fourier_poa 0-4, vertical_basis 5-8, tracking_basis 9-14.
_VERTICAL_BAND = {90.0: 5, 270.0: 6, 0.0: 7, 180.0: 8}
_TRACKING_BAND = 9
_N_BANDS = 15

#: Which basis each mounting reads, beyond fourier_poa.
_MOUNTING_BASIS = {"fixed": None, "vertical": "vertical",
                   "single_axis": "tracking", "dual_axis": "tracking"}


def _directions(system):
    """
    How a system is laid: single-axis trackers turn on North-South rows,
    dual-axis ones stand on the points of a lattice, the rest go either way.
    """
    if system.mounting == "dual_axis":
        return ("points",)
    if system.mounting == "single_axis":
        return ("NS",)
    return DIRECTIONS


def _cross_section(system, direction, module):
    """
    What one row of a system is, across and along.

    A dict with ``plan`` (footprint across the row, m), ``tilt`` (reported),
    ``faces`` (fixed: ``(azimuth, share)`` pairs; vertical: ``(front,
    rear)``), ``unit_length`` / ``step`` / ``per_unit`` (a row holds
    ``floor((length - unit_length) / step) + 1`` units of ``per_unit``
    modules), ``height`` (top of the structure above the ground) and
    ``config`` (modules across and how they are turned, ``"2P"`` for two
    portrait). A dual-axis tracker is a ``panel`` (East-West, North-South
    size when stowed flat) of ``per_unit`` modules sweeping a circle of
    ``radius`` round its pole.
    """
    gap_step = module.width_m + _MODULE_GAP_M

    if system.mounting == "fixed":
        plan, slope_length, tilt, faces = _table(system, direction, module)
        return {
            "plan": plan, "tilt": tilt, "faces": faces,
            "unit_length": module.width_m, "step": gap_step,
            "per_unit": system.modules_across * (2 if direction == "NS" else 1),
            "height": system.clearance_m + slope_length * np.sin(np.radians(tilt)),
            "config": f"{system.modules_across}P",
        }

    if system.mounting == "vertical":
        faces = (90.0, 270.0) if direction == "NS" else (180.0, 0.0)
        return {
            "plan": 0.3, "tilt": 90.0, "faces": faces,
            "unit_length": module.length_m, "step": module.length_m + _MODULE_GAP_M,
            "per_unit": system.modules_across,
            "height": system.clearance_m + system.modules_across * module.width_m,
            "config": f"{system.modules_across}L",
        }

    if system.mounting == "single_axis":
        width = system.modules_across * module.length_m
        return {
            "plan": width, "tilt": np.nan, "faces": None,
            "unit_length": module.width_m, "step": gap_step,
            "per_unit": system.modules_across,
            "height": system.clearance_m + 0.5 * width,
            "config": f"{system.modules_across}P",
        }

    if system.mounting == "dual_axis":
        width = system.modules_along * module.length_m
        tall = system.modules_across * module.width_m
        return {
            "plan": width, "tilt": np.nan, "faces": None,
            "unit_length": tall, "step": float(system.tracker_spacing_m[1]),
            "per_unit": system.modules_along * system.modules_across,
            "height": system.clearance_m + tall, "panel": (width, tall),
            "radius": 0.5 * float(np.hypot(width, tall)),
            "config": f"{system.modules_across}Lx{system.modules_along}",
        }

    raise ValueError(f"unknown mounting {system.mounting!r}")


def _default_pitch(system, direction, module, cross, latitude_deg):
    """
    The pitch a system takes when no crop sets it.

    Fixed rows clear the solstice-noon shadow; single-axis trackers stand at
    their ground cover ratio and backtrack; dual-axis trackers at the East-West
    spacing ``_tracker_spacing`` found; a vertical fence, which only an
    agrivoltaic crop places, at four times its height.
    """
    if system.mounting == "fixed":
        return _no_shading_pitch(system, direction, module, latitude_deg)
    if system.mounting == "single_axis":
        return cross["plan"] / (system.ground_cover_ratio or 0.35)
    if system.mounting == "dual_axis":
        return float(cross["spacing"][0])
    return 4.0 * cross["height"]


def _sun_vectors(sun):
    """Unit vectors to every sun bin, and its beam and diffuse energy."""
    azimuth = np.radians(sun.azimuth_deg.to_numpy(dtype=float))
    altitude = np.radians(sun.altitude_deg.to_numpy(dtype=float))
    east = np.cos(altitude) * np.sin(azimuth)
    north = np.cos(altitude) * np.cos(azimuth)
    up = np.sin(altitude)
    return (east, north, up,
            sun.dni_wh_m2.to_numpy(dtype=float), sun.dhi_wh_m2.to_numpy(dtype=float))


def _dual_axis_lit(cross, dx, dy, vectors):
    """
    Share of each sun bin's beam that reaches a dual-axis tracker standing
    ``dx`` East-West and ``dy`` North-South from its eight neighbours.

    Every tracker faces the sun, so in the plane facing it each neighbour
    sunward of this one is the same rectangle, shifted; the share it covers
    is the overlap of the two.
    """
    east, north, up = vectors[:3]
    width, tall = cross["panel"]

    # In the plane facing the sun: across, horizontal; upwards, the rest.
    horizontal = np.hypot(east, north)
    across = np.stack([north / np.maximum(horizontal, 1e-9),
                       -east / np.maximum(horizontal, 1e-9),
                       np.zeros_like(east)])
    sunward = np.stack([east, north, up])
    upward = np.cross(sunward.T, across.T).T

    lit = np.ones_like(east)
    for ox in (-dx, 0.0, dx):
        for oy in (-dy, 0.0, dy):
            if ox == 0.0 and oy == 0.0:
                continue
            offset = np.array([ox, oy, 0.0])[:, None]
            ahead = (offset * sunward).sum(axis=0) > 0.0
            du = np.abs((offset * across).sum(axis=0))
            dv = np.abs((offset * upward).sum(axis=0))
            cover = (np.clip(1.0 - du / width, 0.0, 1.0)
                     * np.clip(1.0 - dv / tall, 0.0, 1.0))
            lit *= np.where(ahead, 1.0 - cover, 1.0)

    return lit


def _tracker_spacing(system, cross, sun, step_m=1.0):
    """
    East-West and North-South distance between dual-axis trackers.

    The closest lattice - least ground per tracker - at which its neighbours
    take no more than ``tracker_max_shading_loss`` of a tracker's annual
    direct beam, searched from ``tracker_spacing_m`` up to three times it in
    ``step_m`` steps. Never closer than the circles the panels sweep allow.
    Without a sun table, ``tracker_spacing_m`` itself.
    """
    base_x, base_y = (float(value) for value in system.tracker_spacing_m)
    clear = 2.0 * cross["radius"] + 1.0
    base_x, base_y = max(base_x, clear), max(base_y, clear)

    if sun is None or system.tracker_max_shading_loss is None:
        return base_x, base_y

    vectors = _sun_vectors(sun)
    dni = vectors[3]
    total = float(np.sum(dni))
    if total <= 0:
        return base_x, base_y

    target = 1.0 - system.tracker_max_shading_loss
    best, best_area = None, np.inf
    for dx in np.arange(base_x, 3.0 * base_x + 1e-9, step_m):
        for dy in np.arange(base_y, 3.0 * base_y + 1e-9, step_m):
            if dx * dy >= best_area:
                break
            lit = float(np.sum(dni * _dual_axis_lit(cross, dx, dy, vectors))) / total
            if lit >= target:
                best, best_area = (float(dx), float(dy)), dx * dy
                break

    return best if best is not None else (3.0 * base_x, 3.0 * base_y)


def _layout_factors(system, cross, direction, pitch, sun):
    """
    How much of a basis' direct (and sky) survives the neighbouring rows.

    The bases hold what a lone panel would get at each pixel, terrain and
    trees included. Shading by the next row depends on the layout only, so it
    is worked out once per layout over the sun table, unobstructed, and
    applied as a factor - assuming row shading and terrain shading do not
    favour the same hours much, which holds away from steep horizons.

    * Single axis: backtracking at the layout's ground cover ratio, so no row
      shades the next but each turns less than it would alone (pvlib's
      ``singleaxis``); ``direct`` and ``sky`` are the ratios with backtracking
      to without.
    * Dual axis: the share of each tracker the trackers sunward of it cover,
      its eight neighbours on the lattice (``cross["spacing"]``) projected
      on the plane facing the sun.
    * Vertical: the shadow each fence casts up the face of the next one,
      per face.

    Returns a dict of factors; empty for fixed rows, which are already spaced
    to clear the solstice shadow.
    """
    if sun is None or system.mounting == "fixed":
        return {}

    east, north, up, dni, dhi = _sun_vectors(sun)

    def ratio(numerator, denominator):
        total = float(np.sum(denominator))
        return float(np.sum(numerator)) / total if total > 0 else 1.0

    if system.mounting == "single_axis":
        import pvlib

        zenith = 90.0 - sun.altitude_deg.to_numpy(dtype=float)
        azimuth = sun.azimuth_deg.to_numpy(dtype=float)
        common = dict(axis_tilt=0.0, axis_azimuth=180.0,
                      max_angle=system.tracker_max_rotation_deg)
        free = pvlib.tracking.singleaxis(zenith, azimuth, backtrack=False, **common)
        back = pvlib.tracking.singleaxis(zenith, azimuth, backtrack=True,
                                         gcr=min(cross["plan"] / pitch, 0.99), **common)

        def cosine(frame, column):
            values = np.nan_to_num(np.asarray(frame[column], dtype=float), nan=90.0)
            return np.clip(np.cos(np.radians(values)), 0.0, None)

        return {
            "direct": ratio(dni * cosine(back, "aoi"), dni * cosine(free, "aoi")),
            "sky": ratio(dhi * (1.0 + cosine(back, "surface_tilt")),
                         dhi * (1.0 + cosine(free, "surface_tilt"))),
        }

    if system.mounting == "dual_axis":
        dx, dy = cross["spacing"]
        lit = _dual_axis_lit(cross, dx, dy, (east, north, up))
        return {"direct": ratio(dni * lit, dni)}

    # Vertical fences.
    top = cross["height"]
    bottom = system.clearance_m
    factors = {}
    for name, facing in zip(("front", "rear"), cross["faces"]):
        normal_east = np.sin(np.radians(facing))
        normal_north = np.cos(np.radians(facing))
        towards = east * normal_east + north * normal_north
        lit_face = towards > 1e-6
        shadow_top = top - pitch * up / np.where(lit_face, towards, 1.0)
        shaded = np.clip((shadow_top - bottom) / (top - bottom), 0.0, 1.0)
        beam = np.where(lit_face, dni * towards, 0.0)
        factors[name] = ratio(beam * (1.0 - shaded), beam)
    return factors


def _module_poa(system, cross, means, factors, albedo):
    """Annual irradiation on one module of a layout, kWh/m2."""
    if system.mounting == "fixed":
        return sum(
            share * reconstruct_poa_from_basis(
                *means[:5], tilt_deg=cross["tilt"], azimuth_deg=azimuth, albedo=albedo)
            for azimuth, share in cross["faces"]
        ) / 1000.0

    if system.mounting == "vertical":
        front, rear = cross["faces"]
        diffuse = 0.5 * means[3] + 0.5 * albedo * means[4]
        front_poa = means[_VERTICAL_BAND[front]] * factors.get("front", 1.0) + diffuse
        rear_poa = means[_VERTICAL_BAND[rear]] * factors.get("rear", 1.0) + diffuse
        return (front_poa + system.bifaciality * rear_poa) / 1000.0

    t = _TRACKING_BAND
    if system.mounting == "single_axis":
        return (means[t + 3] * factors.get("direct", 1.0)
                + means[t + 4] * factors.get("sky", 1.0)
                + albedo * means[t + 5]) / 1000.0

    return (means[t] * factors.get("direct", 1.0)
            + means[t + 1] + albedo * means[t + 2]) / 1000.0


def _units(length, cross):
    """Modules a row of this length holds."""
    if length < cross["unit_length"]:
        return 0
    units = int(np.floor((length - cross["unit_length"]) / cross["step"])) + 1
    return units * cross["per_unit"]


def _scenarios(systems, crops):
    """
    Every ``(system, crop, pitch)`` to lay out, before the direction.

    A PV field has no crop and one pitch - its no-shading one. An agrivoltaic
    system appears once per crop that lists it, at each pitch that crop
    allows; a system no crop lists is never laid out.
    """
    out = []

    for system in systems:
        if system.layout != "synthetic":
            continue

        if system.kind == "pv_field":
            out.append((system, None, None))
            continue

        for crop in crops:
            for pitch in crop.pitches_m.get(system.key, ()):
                out.append((system, crop, pitch))

    return out


def _row_runs(available, min_cells):
    """
    ``(start, stop)`` of every run of at least ``min_cells`` set cells.

    ``stop`` is exclusive.
    """
    padded = np.concatenate([[False], available, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, stops = edges[::2], edges[1::2]
    keep = (stops - starts) >= min_cells
    return list(zip(starts[keep].tolist(), stops[keep].tolist()))


def _lay_rows(buildable, values, *, resolution, pitch, plan_width, min_cells):
    """
    Rows of one pitch and footprint across a parcel, on the raster's own grid.

    ``buildable`` and ``values`` are already oriented so the rows run down the
    array's first axis: columns are row positions across, array rows the
    distance along. A row occupies the columns whose centre falls within its
    footprint, at least one, and is laid wherever every one of those is
    buildable; it is cut where the parcel or the valid raster ends and kept
    only in runs of at least ``min_cells``. The row origin is tried at a few
    offsets and the one that lays the most row length wins.

    Returns ``[(first_column, last_column, start, stop, means), ...]``, one per
    run, ``means`` holding the basis bands averaged under the run.
    """
    n_along, n_across = buildable.shape
    footprint = max(1, int(round(plan_width / resolution)))
    step = pitch / resolution

    best, best_length = [], -1

    for phase in _PHASES:
        runs = []
        length = 0

        position = phase * step
        while position + footprint <= n_across + 1e-9:
            first = int(np.floor(position))
            last = min(first + footprint, n_across)

            available = buildable[:, first:last].all(axis=1)

            for start, stop in _row_runs(available, min_cells):
                runs.append((first, last, start, stop))
                length += stop - start

            position += step

        if length > best_length:
            best, best_length = runs, length

    laid = []
    for first, last, start, stop in best:
        block = values[:, start:stop, first:last].reshape(values.shape[0], -1)
        laid.append((first, last, start, stop, block.mean(axis=1)))

    return laid


def _lay_points(buildable_shape, valid, values, transform, *, spacing, cross):
    """
    Dual-axis trackers on the points of a lattice, each clear of the parcel.

    A tracker stands wherever the whole circle its panel sweeps lies inside
    the buildable shape and every pixel under the stowed panel has data. The
    lattice is ``spacing`` (East-West, North-South) apart and its origin is
    tried at a few offsets either way; the one that stands the most trackers
    wins.

    Returns ``[(x, y, means), ...]``, ``means`` holding the basis bands
    averaged under the stowed panel.
    """
    inner = buildable_shape.buffer(-cross["radius"])
    if inner.is_empty:
        return []

    dx, dy = spacing
    width, tall = cross["panel"]
    resolution = abs(transform.a)
    half_cols = max(0, int(round(0.5 * width / resolution)))
    half_rows = max(0, int(round(0.5 * tall / resolution)))
    height_px, width_px = valid.shape
    inverse = ~transform
    minx, miny, maxx, maxy = inner.bounds

    def window(x, y):
        column, row = (int(np.floor(value)) for value in inverse * (x, y))
        top, bottom = row - half_rows, row + half_rows + 1
        left, right = column - half_cols, column + half_cols + 1
        if top < 0 or left < 0 or bottom > height_px or right > width_px:
            return None
        if not valid[top:bottom, left:right].all():
            return None
        return slice(top, bottom), slice(left, right)

    best = []
    for phase_x in _PHASES:
        for phase_y in _PHASES:
            xs = np.arange(minx + phase_x * dx, maxx + 1e-9, dx)
            ys = np.arange(miny + phase_y * dy, maxy + 1e-9, dy)
            if not len(xs) or not len(ys):
                continue
            grid_x, grid_y = np.meshgrid(xs, ys)
            inside = _polygon_pixel_mask(inner, grid_x, grid_y)
            standing = []
            for x, y in zip(grid_x[inside], grid_y[inside]):
                rows, columns = window(x, y) or (None, None)
                if rows is not None:
                    standing.append((float(x), float(y), rows, columns))
            if len(standing) > len(best):
                best = standing

    return [
        (x, y, values[:, rows, columns].reshape(values.shape[0], -1).mean(axis=1))
        for x, y, rows, columns in best
    ]


# ============================================================
# AGRONOMY AND ECONOMICS
# ============================================================

def _crop_effect(system, crop, cross, pitch):
    """
    ``(crop_area_frac, radiation_reduction, relative_yield)`` of one layout.

    Under an overhead structure the crop loses about as much light as the
    share of the ground the panels cover. Between ground-level rows it only
    grows on the strip the table and its buffers leave free, and that strip
    is shaded only for part of the day, so it loses about half as much. A
    vertical fence takes almost no ground but shades by its height: at eight
    metres crops keep about 75 % of their light and at ten about 80 %
    (vertical bifacial field trials), which ``0.6 * height / pitch`` follows.
    """
    plan_width = cross["plan"]
    cover = min(plan_width / pitch, 1.0)

    if system.mounting == "vertical":
        area = max(0.0, 1.0 - (plan_width + 2.0 * system.crop_buffer_m) / pitch)
        reduction = min(0.6 * cross["height"] / pitch, 1.0)
    elif system.under_panel_cropped:
        area = 1.0 - system.land_loss_frac
        reduction = cover
    else:
        area = max(0.0, 1.0 - (plan_width + 2.0 * system.crop_buffer_m) / pitch)
        reduction = 0.5 * cover

    if crop is None:
        return 0.0, reduction, 0.0

    loss = crop.shade_slope * max(0.0, reduction - crop.shade_threshold)

    return area, reduction, float(np.clip(1.0 - loss, 0.0, 1.0))


def _irr(cashflows):
    """
    Internal rate of return by bisection, NaN when there is none.
    """
    years = np.arange(len(cashflows))

    def npv(rate):
        return float(np.sum(cashflows / (1.0 + rate) ** years))

    low, high = -0.9, 1.0
    f_low, f_high = npv(low), npv(high)

    if not (np.isfinite(f_low) and np.isfinite(f_high)) or f_low * f_high > 0:
        return np.nan

    for _ in range(100):
        middle = 0.5 * (low + high)
        f_middle = npv(middle)
        if f_low * f_middle <= 0:
            high = middle
        else:
            low, f_low = middle, f_middle

    return 0.5 * (low + high)


def project_cashflows(*, peak_kwp, energy_mwh, capex_eur, opex_eur_year,
                      agri_delta_eur_year, economics):
    """
    Year-by-year cash flows of one scenario and what they add up to.

    Year 0 is the investment; years 1..``lifetime_years`` carry the
    electricity sold, less the operating cost, plus whatever the farm earns
    above (or, usually, below) what the parcel earns today. Energy decays by
    ``degradation_per_year`` and the price moves by ``price_escalation``.

    Returns a dict: ``npv_eur``, ``irr``, ``roi``, ``payback_years``,
    ``lcoe_eur_mwh`` and ``cashflows``.
    """
    years = np.arange(1, economics.lifetime_years + 1)

    energy = energy_mwh * (1.0 - economics.degradation_per_year) ** (years - 1)
    price = economics.electricity_price_eur_mwh * (
        1.0 + economics.price_escalation) ** (years - 1)

    flows = energy * price - opex_eur_year + agri_delta_eur_year
    cashflows = np.concatenate([[-capex_eur], flows])

    discount = (1.0 + economics.discount_rate) ** -years

    cumulative = np.cumsum(cashflows)
    paid = np.flatnonzero(cumulative >= 0.0)
    if paid.size:
        year = int(paid[0])
        payback = float(year - cumulative[year] / cashflows[year]) if year else 0.0
    else:
        payback = np.nan

    discounted_energy = float(np.sum(energy * discount))

    return {
        "npv_eur": float(-capex_eur + np.sum(flows * discount)),
        "irr": _irr(cashflows),
        "roi": float(np.sum(flows) / capex_eur - 1.0) if capex_eur > 0 else np.nan,
        "payback_years": payback,
        "lcoe_eur_mwh": (
            float((capex_eur + np.sum(opex_eur_year * discount)) / discounted_energy)
            if discounted_energy > 0 else np.nan
        ),
        "cashflows": cashflows,
    }


# ============================================================
# PER-PARCEL EVALUATION
# ============================================================

def _orient(array, direction):
    """
    View an (..., rows, columns) array so the rows of ``direction`` run down
    its second-to-last axis.

    North-South rows already do: a raster column is a line of constant x.
    East-West rows are lines of constant y, which are the raster's rows, so
    the array is transposed.
    """
    if direction == "NS":
        return array
    return np.swapaxes(array, -1, -2)


def _run_line(transform, direction, first, last, start, stop):
    """
    Centreline of one run, in map coordinates.
    """
    middle = 0.5 * (first + last)

    if direction == "NS":
        column, rows = middle, (start, stop)
        points = [transform * (column, row) for row in rows]
    else:
        row, columns = middle, (start, stop)
        points = [transform * (column, row) for column in columns]

    return LineString(points)


def _trimmed(line, length):
    """
    The middle ``length`` metres of a line: where the modules of a run sit.
    """
    if length >= line.length:
        return line
    from shapely.ops import substring
    margin = 0.5 * (line.length - length)
    return substring(line, margin, line.length - margin)


def _table_shape(line, plan_width):
    """
    Footprint of a table laid along ``line``: a rectangle ``plan_width`` wide.
    """
    return line.buffer(0.5 * max(plan_width, 0.01), cap_style=2, join_style=2)


def _table_length(n_modules, cross):
    """
    Length a table of ``n_modules`` takes down its row.
    """
    units = n_modules // cross["per_unit"]
    return cross["unit_length"] + (units - 1) * cross["step"] if units else 0.0


def _evaluate_parcel(geometry, *, basis_src, scenarios, module, config):
    """
    Every scenario on one parcel: its tables, energy and crop.

    The fourier_poa basis is stacked with the vertical and tracking bases
    when they were given (``config["sources"]``), so every row is sampled
    once for all of them; a system whose basis is missing is not laid out.

    Returns ``(records, layouts)``: one summary dict per scenario and
    direction, and the tables behind each, keyed by scenario id, for the
    caller to turn into geometry once it knows which ones it wants.
    """
    block = _read_window(basis_src, geometry, [1, 2, 3, 4, 5])

    if block is None:
        return [], {}

    values, transform, valid = block
    resolution = abs(transform.a)

    stack = np.full((_N_BANDS,) + values.shape[1:], np.nan, dtype=np.float32)
    stack[:5] = values
    for name, first, count in (("vertical", 5, 4), ("tracking", _TRACKING_BAND, 6)):
        src = config["sources"].get(name)
        if src is None:
            continue
        extra = _read_window(src, geometry, list(range(1, count + 1)))
        if extra is None or extra[0].shape[1:] != values.shape[1:]:
            raise ValueError(f"the {name} basis is not on the fourier_poa basis' grid")
        stack[first:first + count] = extra[0]
        valid &= extra[2]
    values = stack

    grid_x, grid_y = _pixel_centres(transform, values.shape[1:])

    buildable_shape = geometry.buffer(-config["setback_m"])
    if buildable_shape.is_empty:
        return [], {}

    buildable = _polygon_pixel_mask(buildable_shape, grid_x, grid_y) & valid

    if not buildable.any():
        return [], {}

    buildable_area = float(buildable.sum()) * resolution * resolution
    min_cells = max(1, int(np.ceil(config["min_row_length_m"] / resolution)))
    module_area = module.area_m2
    module_kw = module.peak_kw(config["module_efficiency"])

    records, out_layouts = [], {}
    layouts = {}

    for system, crop, pitch in scenarios:
        basis = _MOUNTING_BASIS[system.mounting]
        if basis is not None and config["sources"].get(basis) is None:
            continue

        for direction in _directions(system):

            cross = _cross_section(system, direction, module)
            if system.mounting == "dual_axis":
                spacing_key = ("spacing", system.key)
                if spacing_key not in config["factor_cache"]:
                    config["factor_cache"][spacing_key] = _tracker_spacing(
                        system, cross, config["sun"])
                cross["spacing"] = config["factor_cache"][spacing_key]
            plan, tilt = cross["plan"], cross["tilt"]

            if pitch is None:
                spacing = _default_pitch(
                    system, direction, module, cross, config["latitude_deg"])
            else:
                spacing = float(pitch)

            if spacing < plan:
                continue

            # A layout depends on the geometry alone, so crops that share a
            # system, direction and pitch share its tables and its energy.
            key = (system.key, direction, round(spacing, 3))

            if key not in layouts:
                factors = config["factor_cache"].get(key)
                if factors is None:
                    factors = _layout_factors(
                        system, cross, direction, spacing, config["sun"])
                    config["factor_cache"][key] = factors

                if direction == "points":
                    placed = [
                        ((x, y), cross["per_unit"], means)
                        for x, y, means in _lay_points(
                            buildable_shape, buildable, values, transform,
                            spacing=cross["spacing"], cross=cross)
                    ]
                else:
                    placed = [
                        ((first, last, start, stop),
                         _units((stop - start) * resolution, cross), means)
                        for first, last, start, stop, means in _lay_rows(
                            _orient(buildable, direction),
                            _orient(values, direction),
                            resolution=resolution,
                            pitch=spacing,
                            plan_width=plan,
                            min_cells=min_cells,
                        )
                    ]

                tables = []
                for where, count, means in placed:
                    if count == 0 or not np.all(np.isfinite(means[:5])):
                        continue
                    poa = _module_poa(system, cross, means, factors, config["albedo"])
                    if not np.isfinite(poa):
                        continue
                    energy = (count * module_area * poa
                              * config["module_efficiency"]
                              * config["performance_ratio"]
                              * (1.0 + system.bifacial_gain))
                    tables.append((where, count, poa, energy))

                layouts[key] = {
                    "tables": tables,
                    "factors": factors,
                    "cross": cross,
                    "n_modules": sum(count for _, count, _, _ in tables),
                    "energy_kwh": sum(energy for _, _, _, energy in tables),
                    "row_length_m": 0.0 if direction == "points" else float(sum(
                        stop - start for (_, _, start, stop), _, _, _ in tables
                    )) * resolution,
                }

            layout = layouts[key]

            if layout["n_modules"] == 0:
                continue

            area, reduction, relative = _crop_effect(system, crop, cross, spacing)

            if direction == "points":
                dx, dy = cross["spacing"]
                scenario = f"{system.key}_points_{dx:.0f}x{dy:.0f}m"
            else:
                scenario = f"{system.key}_{direction}_{spacing:.1f}m"
            if crop is not None:
                scenario += f"_{crop.key}"

            if direction == "points":
                cover = cross["panel"][0] * cross["panel"][1] / (dx * dy)
            else:
                cover = min(plan / spacing, 1.0)

            records.append({
                "scenario": scenario,
                "system": system.key,
                "kind": system.kind,
                "crop": crop.key if crop is not None else None,
                "direction": direction,
                "table_config": cross["config"],
                "tilt_deg": float(tilt),
                "pitch_m": float(spacing),
                "spacing_along_m": (float(cross["spacing"][1])
                                    if direction == "points" else np.nan),
                "n_rows": 0 if direction == "points" else len(layout["tables"]),
                "n_trackers": len(layout["tables"]) if direction == "points" else 0,
                "row_length_m": layout["row_length_m"],
                "n_modules": layout["n_modules"],
                "peak_kwp": layout["n_modules"] * module_kw,
                "energy_mwh": layout["energy_kwh"] / 1000.0,
                "ground_cover_ratio": cover,
                "mounting": system.mounting,
                "row_shading_factor": layout["factors"].get(
                    "direct", layout["factors"].get("front", 1.0)),
                "buildable_area_m2": buildable_area,
                "crop_area_frac": area,
                "radiation_reduction": reduction,
                "relative_crop_yield": relative,
                "land_loss_frac": 1.0 - area if crop is not None else 1.0,
                "_system": system,
                "_crop": crop,
            })

            out_layouts[scenario] = {
                "shape": "points" if direction == "points" else "grid",
                "direction": direction,
                "tables": layout["tables"],
                "cross": cross,
                "pitch_m": float(spacing),
                "transform": transform,
            }

    return records, out_layouts


def _layout_tables(layout, module_kw):
    """
    The tables of one parcel's layout, as records with their footprints.

    Grid runs are trimmed to the length their modules take and centred in
    the run; a dual-axis tracker is drawn as its panel stowed flat, centred
    on its pole. Existing-row layouts already carry their own lines.
    """
    out = []
    shape = layout["shape"]

    for number, table in enumerate(layout["tables"]):
        if shape == "lines":
            line, count, poa, energy, plan, tilt, faces, config = table
            footprint = _table_shape(line, plan)
            length = float(line.length)
        elif shape == "points":
            (x, y), count, poa, energy = table
            width, tall = layout["cross"]["panel"]
            from shapely.geometry import box
            footprint = box(x - 0.5 * width, y - 0.5 * tall,
                            x + 0.5 * width, y + 0.5 * tall)
            length, plan = width, tall
            tilt, faces, config = np.nan, None, layout["cross"]["config"]
        else:
            (first, last, start, stop), count, poa, energy = table
            cross = layout["cross"]
            line = _trimmed(
                _run_line(layout["transform"], layout["direction"],
                          first, last, start, stop),
                _table_length(count, cross))
            plan = cross["plan"]
            footprint = _table_shape(line, plan)
            length = float(line.length)
            tilt, faces, config = cross["tilt"], cross["faces"], cross["config"]

        azimuth = np.nan
        if faces and isinstance(faces[0], tuple) and len(faces) == 1:
            azimuth = float(faces[0][0])
        elif faces and not isinstance(faces[0], tuple):
            azimuth = float(faces[0])

        out.append({
            "table_index": number,
            "table_config": config,
            "n_modules": int(count),
            "peak_kw": count * module_kw,
            "length_m": length,
            "width_m": float(plan),
            "tilt_deg": float(tilt),
            "azimuth_deg": azimuth,
            "poa_kwh_m2": float(poa),
            "energy_kwh": float(energy),
            "geometry": footprint,
        })

    return out


def _fit_table(row_azimuth_deg, system, module, *, pitch, max_plan, latitude_deg):
    """
    The largest table of ``system`` that fits rows of this bearing and pitch.

    The table faces whichever side of the row is nearer South. Rows running
    within 45 degrees of North-South would have it facing East or West at a
    tilt the annual basis cannot cost honestly - the sun spends half the day
    behind it - so those get the low East-West tent instead.

    In a plantation the pitch is the crop's, not the table's, so the table is
    fitted to it: from ``modules_across`` portrait modules down to one
    landscape module, and on South-facing rows from the system's tilt down to
    flat in 5 degree steps, the first that is at most ``max_plan`` wide and
    clears the next table - the solstice-noon shadow for a South-facing
    table, the maintenance aisle for a tent - is taken. Wider beats steeper:
    the energy goes with the module area much more than with the tilt.

    Returns a dict with ``plan``, ``tilt``, ``faces``, ``per_side`` (modules
    across the row), ``step`` (one module's length down the row, gap
    included) and ``config`` (``"1P"``, ``"2L"`` ...), or None when not even
    one landscape module fits.
    """
    sides = ((row_azimuth_deg + 90.0) % 360.0, (row_azimuth_deg + 270.0) % 360.0)
    off_south = [min(abs(side - 180.0), 360.0 - abs(side - 180.0)) for side in sides]
    south_facing = min(off_south) <= 45.0
    altitude = np.radians(max(solar_min_altitude(latitude_deg), 5.0))

    sizes = sorted(
        ((count * across, count, name, along)
         for count in range(system.modules_across, 0, -1)
         for name, across, along in (("P", module.length_m, module.width_m),
                                     ("L", module.width_m, module.length_m))),
        reverse=True,
    )

    for slope_length, count, name, along in sizes:
        config = f"{count}{name}"
        step = along + _MODULE_GAP_M

        if south_facing:
            facing = sides[int(np.argmin(off_south))]
            for tilt in np.arange(system.tilt_south_deg, -1e-9, -5.0):
                plan = slope_length * np.cos(np.radians(tilt))
                shadow = slope_length * np.sin(np.radians(tilt)) / np.tan(altitude)
                if plan <= max_plan and plan + shadow <= pitch:
                    return {"plan": plan, "tilt": float(tilt),
                            "faces": ((facing, 1.0),), "per_side": count,
                            "step": step, "config": config}
            continue

        tilt = system.tilt_east_west_deg
        plan = 2.0 * slope_length * np.cos(np.radians(tilt)) + _MODULE_GAP_M
        if plan <= max_plan and plan + _TENT_AISLE_M <= pitch:
            return {"plan": plan, "tilt": float(tilt),
                    "faces": ((sides[0], 0.5), (sides[1], 0.5)),
                    "per_side": 2 * count, "step": step,
                    "config": f"tent {config}"}

    return None


def _clear_of(lines, keep_out, min_length):
    """
    The pieces of ``lines`` outside ``keep_out``, at least ``min_length`` long.
    """
    if keep_out is None or keep_out.is_empty:
        return [line for line in lines if line.length >= min_length]

    out = []
    for line in lines:
        rest = line.difference(keep_out)
        for part in getattr(rest, "geoms", [rest]):
            if part.geom_type == "LineString" and part.length >= min_length:
                out.append(part)
    return out


def _lattice_rows(block_rows, azimuth_deg, spacing):
    """
    A block's detected rows, gathered into lattice rows ordered across it.

    A detected row can come in several pieces where a hole cut it; pieces
    whose offsets across the block are within half a spacing are the same
    row. Returns ``[(offset_m, [LineString, ...]), ...]`` sorted by offset,
    the offset measured along the row normal.
    """
    theta = np.radians(azimuth_deg)
    normal = np.array([np.cos(theta), -np.sin(theta)])

    pieces = sorted(
        ((float(np.dot(np.asarray(line.centroid.coords[0]), normal)), line)
         for line in block_rows
         if line is not None and not line.is_empty),
        key=lambda item: item[0],
    )

    lattice = []
    for offset, line in pieces:
        if lattice and offset - lattice[-1][0] <= 0.5 * spacing:
            lattice[-1][1].append(line)
        else:
            lattice.append((offset, [line]))

    return lattice


def _sample_lines(lines, plan_width, src, geometry):
    """
    Basis bands averaged under each line's footprint, shaped (n, 5).

    One window is read for the parcel and every footprint rasterised into it
    with its own label; a footprint narrower than a pixel that catches no
    pixel centre falls back to the pixel under its midpoint.
    """
    from rasterio.features import rasterize

    block = _read_window(src, geometry.buffer(plan_width), [1, 2, 3, 4, 5])
    if block is None:
        return np.full((len(lines), 5), np.nan)

    values, transform, valid = block
    footprints = [line.buffer(0.5 * plan_width, cap_style=2) for line in lines]

    labels = rasterize(
        ((footprint, index + 1) for index, footprint in enumerate(footprints)),
        out_shape=valid.shape, transform=transform, fill=0, dtype="int32",
    )
    labels = np.where(valid, labels, 0)

    counts = np.bincount(labels.ravel(), minlength=len(lines) + 1)
    out = np.full((len(lines), 5), np.nan)

    for band in range(5):
        sums = np.bincount(labels.ravel(), weights=values[band].ravel(),
                           minlength=len(lines) + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, band] = (sums / counts)[1:]

    inverse = ~transform
    for index in np.flatnonzero(counts[1:] == 0):
        point = lines[index].interpolate(0.5, normalized=True)
        column, row = (int(np.floor(c)) for c in inverse * (point.x, point.y))
        if 0 <= row < valid.shape[0] and 0 <= column < valid.shape[1] and valid[row, column]:
            out[index] = values[:, row, column]

    return out


def _evaluate_existing_rows(geometry, parcel_rows, *, use, basis_src,
                            vegetated_src, systems, module, margins, config):
    """
    Agrivoltaic scenarios built into the plantation already on a parcel.

    ``parcel_rows`` are this parcel's rows from ``detect_crop_rows``, one block
    or several. An overhead system goes over every ``row_steps``-th crop row -
    every one by default - with its posts in the tree line; an alley system
    down the middle of every such alley that still leaves a tractor lane of
    ``config["tractor_lane_m"]`` beside the table, its buffers and the canopy.
    Each block keeps its own direction and spacing, and the table on it is
    the largest that fits them (``_fit_table``): an overhead table at most
    ``_OVERHEAD_GAP_M`` short of the pitch, an alley one at most what the
    alley leaves. The origin row is chosen per block to lay the most length.

    No PV row crosses another: a row is cut where it would run into a table
    already laid, by its own block or another, and an alley table where it
    would come within its buffer of a crop row.

    Overhead panels stand above the canopy, so they are costed on the basis
    without vegetation. Alley panels stand among the trees and are costed on
    ``vegetated_src`` - the trees shade them - and are skipped without it.

    The crop is the one already there, valued at the margin its use earns
    today and its shade response (``SIGPAC_USE_SHADE_RESPONSE``).

    Returns ``(records, layouts)`` like ``_evaluate_parcel``.
    """
    from shapely.ops import unary_union

    buildable = geometry.buffer(-config["setback_m"])
    if buildable.is_empty or parcel_rows.empty:
        return [], {}

    buildable_area = float(buildable.area)
    module_kw = module.peak_kw(config["module_efficiency"])
    min_length = config["min_row_length_m"]
    threshold, slope = SIGPAC_USE_SHADE_RESPONSE.get(use, DEFAULT_SHADE_RESPONSE)
    existing = CropProfile(
        key=f"existing_{use}",
        label=f"Existing {use} plantation",
        gross_margin_eur_ha=margins.get(use, 0.0),
        shade_threshold=threshold,
        shade_slope=slope,
    )

    blocks = []
    for _, block_rows in parcel_rows.groupby("block_index"):
        first = block_rows.iloc[0]
        spacing = float(first["row_spacing_m"])
        if not np.isfinite(spacing) or spacing <= 0:
            continue
        width = float(first.get("canopy_width_m", np.nan))
        blocks.append({
            "azimuth": float(first["row_azimuth_deg"]),
            "spacing": spacing,
            "canopy_width": width if np.isfinite(width) else 0.0,
            "canopy_height": float(first.get("canopy_height_p90_m", np.nan)),
            "lattice": _lattice_rows(
                list(block_rows.geometry), float(first["row_azimuth_deg"]), spacing),
        })

    blocks = [block for block in blocks if block["lattice"]]
    if not blocks:
        return [], {}

    crop_lines = unary_union([
        line for block in blocks for _, pieces in block["lattice"] for line in pieces])

    records, layouts = [], {}

    for system in systems:
        if system.layout not in ("rows_overhead", "rows_alley"):
            continue

        alley = system.layout == "rows_alley"
        src = vegetated_src if alley else basis_src
        if src is None:
            continue

        for step in system.row_steps:
            tables = []
            covers, weights, fitted = [], [], []
            laid = []

            for block in blocks:
                pitch = step * block["spacing"]
                width = block["canopy_width"]
                lane = config["tractor_lane_m"]

                if alley:
                    max_plan = (block["spacing"] - width - 2.0 * system.crop_buffer_m
                                - lane)
                else:
                    max_plan = pitch - _OVERHEAD_GAP_M

                fit = _fit_table(block["azimuth"], system, module, pitch=pitch,
                                 max_plan=max_plan, latitude_deg=config["latitude_deg"])
                if fit is None:
                    continue
                plan = fit["plan"]

                theta = np.radians(block["azimuth"])
                normal = np.array([np.cos(theta), -np.sin(theta)])
                lattice = block["lattice"]

                if alley:
                    # An alley is the strip between two neighbouring rows, and
                    # it only takes the table where what the canopy leaves of
                    # it is wide enough for the table, its buffers and a lane
                    # for the tractor beside them.
                    candidates = []
                    for (left, pieces), (right, _) in zip(lattice[:-1], lattice[1:]):
                        gap = right - left
                        if not (0.6 * block["spacing"] <= gap <= 1.5 * block["spacing"]):
                            candidates.append(None)
                            continue
                        if gap - width - 2.0 * system.crop_buffer_m - lane < plan:
                            candidates.append(None)
                            continue
                        shift = normal * (0.5 * gap)
                        candidates.append([
                            LineString(np.asarray(piece.coords) + shift)
                            for piece in pieces
                        ])
                else:
                    candidates = [pieces for _, pieces in lattice]

                best, best_length = [], 0.0
                for phase in range(step):
                    chosen = []
                    for index in range(phase, len(candidates), step):
                        if candidates[index] is None:
                            continue
                        for piece in candidates[index]:
                            clipped = piece.intersection(buildable)
                            for part in getattr(clipped, "geoms", [clipped]):
                                if (part.geom_type == "LineString"
                                        and part.length >= min_length):
                                    chosen.append(part)
                    length = sum(part.length for part in chosen)
                    if length > best_length:
                        best, best_length = chosen, length

                # Nothing may cross: down an alley, not the crop rows either
                # side; and nowhere a table already laid, by this block - a
                # detected row can come in overlapping pieces, or lean into
                # its neighbour - or by another one. Longest first, so it is
                # the stubs that give way.
                if alley:
                    best = _clear_of(best, crop_lines.buffer(
                        0.5 * width + 0.5 * plan + system.crop_buffer_m), min_length)
                clear = []
                for line in sorted(best, key=lambda line: -line.length):
                    near = [footprint for footprint in laid
                            if footprint.distance(line) < 0.5 * plan + _MODULE_GAP_M]
                    pieces = _clear_of(
                        [line],
                        unary_union(near).buffer(0.5 * plan + _MODULE_GAP_M) if near else None,
                        min_length)
                    clear += pieces
                    laid += [_table_shape(piece, plan) for piece in pieces]
                best = clear

                if not best:
                    continue

                means = _sample_lines(best, plan, src, geometry)
                placed = 0
                for line, band_means in zip(best, means):
                    units = int(np.floor((line.length + _MODULE_GAP_M) / fit["step"]))
                    count = units * fit["per_side"]
                    if count == 0 or not np.all(np.isfinite(band_means)):
                        continue
                    poa = sum(
                        share * reconstruct_poa_from_basis(
                            *band_means, tilt_deg=fit["tilt"], azimuth_deg=azimuth,
                            albedo=config["albedo"])
                        for azimuth, share in fit["faces"]
                    ) / 1000.0
                    energy = (count * module.area_m2 * poa
                              * config["module_efficiency"]
                              * config["performance_ratio"]
                              * (1.0 + system.bifacial_gain))
                    line = _trimmed(line, units * fit["step"] - _MODULE_GAP_M)
                    tables.append((line, count, poa, energy, plan, fit["tilt"],
                                   fit["faces"], fit["config"]))
                    placed += 1

                if not placed:
                    continue

                covers.append(min(plan / pitch, 1.0))
                weights.append(best_length)
                fitted.append((len(lattice), fit, pitch))

            n_modules = sum(table[1] for table in tables)
            if n_modules == 0:
                continue

            energy_kwh = sum(table[3] for table in tables)
            cover = float(np.average(covers, weights=weights))

            if alley:
                area, reduction = 1.0, 0.5 * cover
            else:
                area, reduction = 1.0 - system.land_loss_frac, cover

            relative = float(np.clip(
                1.0 - slope * max(0.0, reduction - threshold), 0.0, 1.0))

            # What the record reports is the table of the largest block.
            _, fit, pitch = max(fitted, key=lambda item: item[0])
            dominant = max(blocks, key=lambda block: len(block["lattice"]))

            required = (dominant["canopy_height"] + 1.0
                        if not alley and np.isfinite(dominant["canopy_height"]) else np.nan)

            scenario = f"{system.key}_k{step}"

            records.append({
                "scenario": scenario,
                "system": system.key,
                "kind": "agrivoltaic",
                "crop": existing.key,
                "direction": "rows",
                "table_config": fit["config"],
                "tilt_deg": float(fit["tilt"]),
                "pitch_m": float(pitch),
                "spacing_along_m": np.nan,
                "n_rows": len(tables),
                "n_trackers": 0,
                "row_length_m": float(sum(table[0].length for table in tables)),
                "n_modules": n_modules,
                "peak_kwp": n_modules * module_kw,
                "energy_mwh": energy_kwh / 1000.0,
                "ground_cover_ratio": cover,
                "mounting": system.mounting,
                "row_shading_factor": 1.0,
                "buildable_area_m2": buildable_area,
                "crop_area_frac": area,
                "radiation_reduction": reduction,
                "relative_crop_yield": relative,
                "land_loss_frac": 1.0 - area,
                "required_clearance_m": required,
                "clearance_ok": (bool(system.clearance_m >= required)
                                 if np.isfinite(required) else None),
                "_system": system,
                "_crop": existing,
            })

            layouts[scenario] = {"shape": "lines", "direction": "rows",
                                 "tables": tables}

    return records, layouts


def _value(record, *, parcel_area_m2, baseline_margin, economics):
    """
    Price one scenario: what it costs, earns and leaves of the farm.
    """
    system, crop = record.pop("_system"), record.pop("_crop")

    peak_kwp = record["peak_kwp"]
    capex = peak_kwp * 1000.0 * system.capex_eur_wp + economics.fixed_project_cost_eur
    opex = peak_kwp * system.opex_eur_kwp_year

    # The crop keeps the ground the layout does not take: the parcel outside
    # the buildable area, and the cropped share of the buildable area.
    if crop is not None:
        cropped_m2 = (parcel_area_m2 - record["buildable_area_m2"]) + (
            record["buildable_area_m2"] * record["crop_area_frac"])
        agri_margin = (crop.gross_margin_eur_ha * record["relative_crop_yield"]
                       * cropped_m2 / 10_000.0)
        agri_share = record["relative_crop_yield"] * cropped_m2 / parcel_area_m2
    else:
        agri_margin, agri_share = 0.0, 0.0

    flows = project_cashflows(
        peak_kwp=peak_kwp,
        energy_mwh=record["energy_mwh"],
        capex_eur=capex,
        opex_eur_year=opex,
        agri_delta_eur_year=agri_margin - baseline_margin,
        economics=economics,
    )

    record.update({
        "capex_eur": capex,
        "opex_eur_year": opex,
        "electricity_revenue_eur_year": (
            record["energy_mwh"] * economics.electricity_price_eur_mwh),
        "agri_margin_eur_year": agri_margin,
        "baseline_margin_eur_year": baseline_margin,
        "agri_share": agri_share,
        "specific_yield_kwh_kwp": (
            record["energy_mwh"] * 1000.0 / peak_kwp if peak_kwp else np.nan),
        "npv_eur": flows["npv_eur"],
        "irr": flows["irr"],
        "roi": flows["roi"],
        "payback_years": flows["payback_years"],
        "lcoe_eur_mwh": flows["lcoe_eur_mwh"],
    })

    return record


def _rank(table, criterion):
    """
    Index of the best row of ``table`` by ``criterion``, or None.
    """
    column = "energy_mwh" if criterion == "energy" else (
        "payback_years" if criterion == "payback" else
        "lcoe_eur_mwh" if criterion == "lcoe" else
        "npv_eur" if criterion == "npv" else criterion)

    scores = table[column].astype(float)
    if not CRITERIA[criterion]:
        scores = -scores

    scores = scores[np.isfinite(scores)]

    return None if scores.empty else scores.idxmax()


# ============================================================
# MAIN ENGINE
# ============================================================

def plan_parcel_pv_scenarios(
    parcels,
    *,
    poa_basis_path,
    crop_rows=None,
    vegetated_basis_path=None,
    vertical_basis_path=None,
    tracking_basis_path=None,
    systems=DEFAULT_SYSTEMS,
    crops=DEFAULT_CROPS,
    economics=None,
    baseline_margin_eur_ha=None,
    criterion="npv",
    parcel_id_column=None,
    module=DEFAULT_MODULES[0],
    module_efficiency=0.21,
    performance_ratio=0.80,
    albedo=0.2,
    latitude_deg=None,
    setback_m=5.0,
    min_row_length_m=20.0,
    min_parcel_area_m2=2000.0,
    tractor_lane_m=2.0,
    tables="all",
    output_path=None,
    output_summary_path=None,
    layer_prefix="ppv_",
    progress=False,
):
    """
    Lay out, cost and rank PV and agrivoltaic scenarios on every parcel.

    Each agricultural parcel gets every scenario the catalogue allows - each
    system, each crop it suits, each pitch that crop allows, in both row
    directions - laid out on synthetic rows from one side of the parcel to the
    other, or, for dual-axis trackers, on a lattice of separate points. Each
    is costed over the project's life against what the parcel earns today,
    and the best is chosen by ``criterion``.

    As in ``plan_pv_strings_in_buildings``, every scenario costed survives
    into the output with its panels, one GeoPackage layer per scenario -
    ``ppv_pv_fixed_EW_8.9m``, ``ppv_apv_overhead_NS_6.0m_forage``,
    ``ppv_apv_rows_overhead_k1`` ... - plus ``ppv_best`` and the best of each
    kind, which take each parcel's own winner. The panels come as tables:
    one polygon per continuous stretch of modules down a row, at its plan
    width and as long as its modules, or per dual-axis tracker. A scenario's
    name is the same on every parcel, so its layer compares them directly.

    Parameters
    ----------
    parcels : GeoDataFrame or str
        SIGPAC recintos. Only the agricultural uses are laid out
        (``filter_agricultural_parcels``); they are reprojected to the basis
        raster's CRS.
    poa_basis_path : str
        Raster written by ``compute_surface_solar_energy(mode="fourier_poa")``,
        built on hillshades of the DSM **without vegetation**: the scenarios
        replace whatever grows on the parcel today. Overhead structures over
        an existing plantation use it too - they stand above its canopy.
    crop_rows : GeoDataFrame or str, optional
        Rows from ``detect_crop_rows`` - the frame, or the GeoPackage it wrote.
        Where a parcel has them, the ``"rows_overhead"`` and ``"rows_alley"``
        systems are also built into that plantation, as agrivoltaics that keep
        the crop already growing.
    vegetated_basis_path : str, optional
        ``fourier_poa`` basis built **with** vegetation. Panels in the alleys
        of a plantation stand among its trees and are costed on it, so their
        shade is counted; without it the alley scenarios are skipped.
    vertical_basis_path, tracking_basis_path : str, optional
        ``compute_surface_solar_energy(mode="vertical_basis")`` and
        ``mode="tracking_basis"`` rasters, built on the same hillshades as
        ``poa_basis_path``. They cost the vertical bifacial fences and the
        single- and dual-axis trackers; a system whose basis is not given is
        not laid out. The ``*_sun.csv`` written next to them supplies the sun
        for the shading between rows.
    systems : sequence of MountingSystem, default=DEFAULT_SYSTEMS
    crops : sequence of CropProfile, default=DEFAULT_CROPS
        The catalogue. An agrivoltaic system is laid out only with the crops
        that list a pitch for it.
    economics : Economics, optional
        Prices, discount rate and life. ``Economics()`` when omitted.
    baseline_margin_eur_ha : dict, optional
        Margin per hectare each ``uso_sigpac`` earns today, defaulting to
        ``SIGPAC_USE_MARGIN_EUR_HA``. Every scenario is valued as the change
        it makes: the electricity it sells, less what it costs, plus whatever
        the farm earns above that baseline - so building over an orchard pays
        for the orchard.
    criterion : {"npv", "irr", "roi", "lcoe", "payback", "energy", "ler"}
        What picks a parcel's best scenario. ``"lcoe"`` and ``"payback"`` are
        minimised, the rest maximised. ``"ler"`` is the land-equivalent ratio:
        the scenario's energy over the parcel's best PV field's, plus the
        share of the open-field crop it keeps.
    setback_m : float, default=5.0
        Rows stop this far inside the parcel: fence, access track, and the
        headland a tractor turns on.
    min_row_length_m : float, default=20.0
        Shorter pieces of row are dropped.
    min_parcel_area_m2 : float, default=2000.0
        Smaller parcels are not laid out.
    tractor_lane_m : float, default=2.0
        Free width a tractor needs down an alley of an existing plantation -
        what an orchard or a vineyard planted 3.5-4 m apart leaves between
        its canopies. A table goes down an alley only where this is left
        beside it, its buffers and the canopy.
    tables : {"all", "best_per_kind", "best", None}, default="all"
        Which scenarios' tables come back as geometry. ``"all"``: every
        scenario, plus ``best``, ``best_pv_field``, ``best_agrivoltaic`` and
        ``best_existing_rows``; ``"best_per_kind"``: only those four;
        ``"best"``: only ``best``; None: none. Every scenario is in the
        summary table regardless.
    output_path : str, optional
        GeoPackage with the parcels and their best scenarios
        (``parcel_pv_best``) and one layer of tables per scenario
        (``<layer_prefix><scenario>``).
    output_summary_path : str, optional
        CSV of every scenario on every parcel.
    layer_prefix : str, default="ppv_"
        Prefix of the scenario layers, so that they sort together.

    Returns
    -------
    tuple
        ``(scenarios_df, parcels_gdf, tables_gdf)``. ``scenarios_df`` is long:
        one row per parcel and scenario. ``parcels_gdf`` holds one row per
        parcel laid out, with the best scenario overall and the best of each
        kind; ``best_viable`` says whether the best one actually beats leaving
        the parcel as it is (positive NPV), since a ranking names a winner
        even among losers. ``tables_gdf`` is long too: one polygon per table
        and scenario, ``scenario`` naming the layer it belongs to and
        ``source_scenario`` the scenario it was laid out for - the same, but
        on the ``best*`` layers. Summing it across scenarios counts a parcel
        several times; filter to one first.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {sorted(CRITERIA)}")
    if tables not in {"best", "best_per_kind", "all", None}:
        raise ValueError("tables must be 'all', 'best_per_kind', 'best' or None")

    economics = economics or Economics()
    margins = (SIGPAC_USE_MARGIN_EUR_HA if baseline_margin_eur_ha is None
               else baseline_margin_eur_ha)

    if isinstance(parcels, str):
        parcels = gpd.read_file(parcels)

    parcels = filter_agricultural_parcels(parcels).copy()

    if parcel_id_column is None:
        parcels["parcel_id"] = sigpac_parcel_id(parcels)
        parcel_id_column = "parcel_id"

    scenarios = _scenarios(systems, crops)

    if isinstance(crop_rows, str):
        crop_rows = gpd.read_file(crop_rows, layer="crop_rows")

    vegetated_src = (rasterio.open(vegetated_basis_path)
                     if vegetated_basis_path is not None else None)

    sources = {
        "vertical": (rasterio.open(vertical_basis_path)
                     if vertical_basis_path is not None else None),
        "tracking": (rasterio.open(tracking_basis_path)
                     if tracking_basis_path is not None else None),
    }

    sun = None
    for path in (tracking_basis_path, vertical_basis_path):
        if path is not None:
            try:
                sun = read_sun_table(path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{path} has no sun table next to it; rebuild it with "
                    f"compute_surface_solar_energy, which writes one") from None
            break

    skipped = sorted({
        system.key for system in systems
        if _MOUNTING_BASIS.get(system.mounting) is not None
        and sources[_MOUNTING_BASIS[system.mounting]] is None
    })
    if skipped:
        import warnings
        warnings.warn(f"no basis for {', '.join(skipped)}: pass vertical_basis_path "
                      f"and tracking_basis_path to lay them out")

    with rasterio.open(poa_basis_path) as basis_src:

        parcels = parcels.to_crs(basis_src.crs)

        rows_by_parcel = {}
        if crop_rows is not None and len(crop_rows):
            rows_by_parcel = dict(tuple(
                crop_rows.to_crs(basis_src.crs).groupby(parcel_id_column)))
        parcels = parcels[parcels.geometry.area >= min_parcel_area_m2]

        if latitude_deg is None:
            import pyproj
            bounds = basis_src.bounds
            _, latitude_deg = pyproj.Transformer.from_crs(
                basis_src.crs, "EPSG:4326", always_xy=True,
            ).transform((bounds.left + bounds.right) / 2.0,
                        (bounds.bottom + bounds.top) / 2.0)

        config = {
            "setback_m": setback_m,
            "min_row_length_m": min_row_length_m,
            "module_efficiency": module_efficiency,
            "performance_ratio": performance_ratio,
            "albedo": albedo,
            "latitude_deg": latitude_deg,
            "tractor_lane_m": tractor_lane_m,
            "sources": sources,
            "sun": sun,
            "factor_cache": {},
        }

        all_records, best_rows, table_records = [], [], []
        module_kw = module.peak_kw(module_efficiency)

        iterator = parcels.iterrows()
        if progress:
            iterator = tqdm(iterator, total=len(parcels),
                            desc="Planning parcel PV", unit="parcel")

        for _, parcel in iterator:
            geometry = parcel.geometry
            if geometry is None or geometry.is_empty:
                continue

            records, layouts = _evaluate_parcel(
                geometry,
                basis_src=basis_src,
                scenarios=scenarios,
                module=module,
                config=config,
            )

            planted = rows_by_parcel.get(parcel[parcel_id_column])
            if planted is not None:
                more, more_layouts = _evaluate_existing_rows(
                    geometry, planted,
                    use=parcel.get("uso_sigpac"),
                    basis_src=basis_src,
                    vegetated_src=vegetated_src,
                    systems=systems,
                    module=module,
                    margins=margins,
                    config=config,
                )
                records += more
                layouts.update(more_layouts)

            if not records:
                continue

            area = float(geometry.area)
            baseline = margins.get(parcel.get("uso_sigpac"), 0.0) * area / 10_000.0

            table = pd.DataFrame([
                _value(record, parcel_area_m2=area,
                       baseline_margin=baseline, economics=economics)
                for record in records
            ])

            fields = table[table.kind == "pv_field"]
            reference = fields.energy_mwh.max() if len(fields) else table.energy_mwh.max()
            table["ler"] = (table.energy_mwh / reference if reference > 0 else 0.0) \
                + table.agri_share

            table.insert(0, parcel_id_column, parcel[parcel_id_column])
            table.insert(1, "uso_sigpac", parcel.get("uso_sigpac"))
            table["parcel_area_m2"] = area

            best = _rank(table, criterion)
            summary = {parcel_id_column: parcel[parcel_id_column],
                       "uso_sigpac": parcel.get("uso_sigpac"),
                       "parcel_area_m2": area,
                       "criterion": criterion,
                       "geometry": geometry}

            chosen = {}
            if best is not None:
                chosen["best"] = best
                summary.update({
                    f"best_{column}": table.at[best, column]
                    for column in ("scenario", "kind", "system", "crop",
                                   "direction", "pitch_m", "peak_kwp",
                                   "energy_mwh", "npv_eur", "irr", "roi",
                                   "lcoe_eur_mwh", "payback_years", "ler",
                                   "relative_crop_yield")
                })
                # The ranking always names a winner, even where every
                # scenario loses money against what the parcel earns today;
                # this says whether building it beats leaving the parcel be.
                summary["best_viable"] = bool(table.at[best, "npv_eur"] > 0.0)

            for kind in ("pv_field", "agrivoltaic", "existing_rows"):
                subset = (table[table.direction == "rows"] if kind == "existing_rows"
                          else table[table.kind == kind])
                index = _rank(subset, criterion) if len(subset) else None
                if index is None:
                    continue
                chosen[kind] = index
                for column in ("scenario", "energy_mwh", "npv_eur", "irr",
                               "lcoe_eur_mwh", "ler"):
                    summary[f"{kind}_{column}"] = table.at[index, column]

            best_rows.append(summary)
            all_records.append(table)

            if tables is None:
                continue

            # layer name -> scenario it shows on this parcel.
            wanted = {}
            if tables == "all":
                wanted.update({scenario: scenario for scenario in table.scenario})
            for role in (("best",) if tables == "best" else
                         ("best", "pv_field", "agrivoltaic", "existing_rows")):
                if role in chosen:
                    layer = "best" if role == "best" else f"best_{role}"
                    wanted[layer] = table.at[chosen[role], "scenario"]

            details = table.set_index("scenario")
            for layer, scenario in wanted.items():
                row = details.loc[scenario]
                for record in _layout_tables(layouts[scenario], module_kw):
                    record.update({
                        "scenario": layer,
                        "source_scenario": scenario,
                        parcel_id_column: parcel[parcel_id_column],
                        "system": row["system"],
                        "kind": row["kind"],
                        "crop": row["crop"],
                        "mounting": row["mounting"],
                        "direction": row["direction"],
                        "pitch_m": row["pitch_m"],
                    })
                    table_records.append(record)

        crs = parcels.crs

    if vegetated_src is not None:
        vegetated_src.close()
    for src in sources.values():
        if src is not None:
            src.close()

    scenarios_df = (pd.concat(all_records, ignore_index=True)
                    if all_records else pd.DataFrame())

    parcels_gdf = gpd.GeoDataFrame(
        best_rows, geometry="geometry", crs=crs) if best_rows else \
        gpd.GeoDataFrame({parcel_id_column: []}, geometry=[], crs=crs)

    table_columns = ["scenario", "source_scenario", parcel_id_column, "system",
                     "kind", "crop", "mounting", "direction", "table_index",
                     "table_config", "n_modules", "peak_kw", "length_m",
                     "width_m", "tilt_deg", "azimuth_deg", "pitch_m",
                     "poa_kwh_m2", "energy_kwh"]
    if table_records:
        tables_gdf = gpd.GeoDataFrame(
            table_records, geometry="geometry", crs=crs)[table_columns + ["geometry"]]
    else:
        tables_gdf = gpd.GeoDataFrame(
            {column: [] for column in table_columns}, geometry=[], crs=crs)

    if output_path is not None:
        write_scenario_layers(tables_gdf, output_path, layer_prefix=layer_prefix)
        parcels_gdf.to_file(output_path, driver="GPKG", layer="parcel_pv_best",
                            mode="a")

    if output_summary_path is not None:
        scenarios_df.to_csv(output_summary_path, index=False)

    return scenarios_df, parcels_gdf, tables_gdf
