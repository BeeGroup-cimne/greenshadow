"""
greenshadow.pv

Advanced PV layout and energy optimisation engine.

Features:
- Vector-based panel packing
- True solar minimum altitude spacing
- Fourier POA reconstruction
- Shading factor per panel
- Tilt optimisation per roof
- Per-building aggregation
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate
from rasterio.features import shapes
from scipy.ndimage import label
import pandas as pd

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PVPanelSpec:
    name: str
    width_m: float
    height_m: float
    peak_power_kw: float


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def solar_min_altitude(latitude: float) -> float:
    """
    Winter solstice minimum solar altitude at noon.
    """
    return 90 - latitude - 23.45


# ============================================================
# ROOF EXTRACTION
# ============================================================

def extract_roof_polygons(
    classification_array: np.ndarray,
    transform,
    crs,
) -> gpd.GeoDataFrame:

    mask = classification_array == 6
    labeled, _ = label(mask)

    polygons = []
    ids = []

    for geom, val in shapes(
        labeled.astype(np.int32),
        mask=labeled > 0,
        transform=transform
    ):
        polygons.append(Polygon(geom["coordinates"][0]))
        ids.append(val)

    return gpd.GeoDataFrame(
        {"roof_id": ids},
        geometry=polygons,
        crs=crs,
    )


# ============================================================
# FOURIER POA EVALUATION
# ============================================================

def evaluate_poa(
    panel_geom: Polygon,
    fourier_raster_path: str,
    tilt_deg: float,
    azimuth_deg: float,
) -> float:
    """
    Evaluate annual POA energy from Fourier coefficients.
    Raster bands:
        0 A0
        1 A1c
        2 A1s
        3 A2
        4 D0
        5 D1
    """

    theta = np.radians(azimuth_deg)
    beta = np.radians(tilt_deg)

    with rasterio.open(fourier_raster_path) as src:
        out, _ = rasterio.mask.mask(src, [panel_geom], crop=True)

        A0  = out[0]
        A1c = out[1]
        A1s = out[2]
        A2  = out[3]
        D0  = out[4]
        D1  = out[5]

    direct = (
        A0
        + A1c * np.cos(theta)
        + A1s * np.sin(theta)
        + A2  * np.sin(beta)
    )

    diffuse = D0 + D1 * np.sin(beta)

    total = np.nansum(direct + diffuse)

    return float(total)


# ============================================================
# PACKING FUNCTIONS
# ============================================================

def pack_flat_roof_physical(
    roof_polygon: Polygon,
    panel: PVPanelSpec,
    tilt_deg: float,
    azimuth_deg: float,
    latitude: float,
) -> List[Polygon]:

    panels = []

    alpha_min = solar_min_altitude(latitude)

    spacing = (
        panel.height_m * np.sin(np.radians(tilt_deg))
        / np.tan(np.radians(alpha_min))
    )

    minx, miny, maxx, maxy = roof_polygon.bounds

    x = minx
    while x + panel.width_m <= maxx:

        y = miny
        while y + panel.height_m <= maxy:

            p = box(0, 0, panel.width_m, panel.height_m)
            p = rotate(p, azimuth_deg, origin=(0, 0), use_radians=False)
            p = translate(p, xoff=x, yoff=y)

            if roof_polygon.contains(p):
                panels.append(p)

            y += panel.height_m + spacing

        x += panel.width_m

    return panels


def pack_pitched_roof(
    roof_polygon: Polygon,
    panel: PVPanelSpec,
    ridge_angle_deg: float,
) -> List[Polygon]:

    panels = []

    minx, miny, maxx, maxy = roof_polygon.bounds

    x = minx
    while x + panel.width_m <= maxx:

        y = miny
        while y + panel.height_m <= maxy:

            p = box(0, 0, panel.width_m, panel.height_m)
            p = rotate(p, ridge_angle_deg, origin=(0, 0), use_radians=False)
            p = translate(p, xoff=x, yoff=y)

            if roof_polygon.contains(p):
                panels.append(p)

            y += panel.height_m

        x += panel.width_m

    return panels


# ============================================================
# TILT OPTIMISATION
# ============================================================

def optimise_flat_roof(
    roof_polygon: Polygon,
    panel_specs: List[PVPanelSpec],
    fourier_raster: str,
    latitude: float,
    tilt_candidates: Tuple[int, ...] = (20, 25, 30, 35, 40),
    azimuth: float = 180,
):

    best_energy = 0
    best_panels = []
    best_tilt = None

    for tilt in tilt_candidates:

        roof_energy = 0
        roof_panels = []

        for panel in panel_specs:

            panels = pack_flat_roof_physical(
                roof_polygon,
                panel,
                tilt,
                azimuth,
                latitude,
            )

            for p in panels:
                E = evaluate_poa(p, fourier_raster, tilt, azimuth)
                roof_energy += E
                roof_panels.append((p, panel, E))

        if roof_energy > best_energy:
            best_energy = roof_energy
            best_panels = roof_panels
            best_tilt = tilt

    return best_panels, best_tilt, best_energy


# ============================================================
# MAIN ENGINE
# ============================================================

def run_pv_layout_engine_advanced(
    roof_gdf,
    fourier_raster,
    latitude,
    output_panel_path,
    output_building_summary_path,
    slope_raster_path,
    aspect_raster_path,
    slope_threshold_deg: float = 5.0,
    flat_tilts=(20, 25, 30, 35, 40),
    allowed_orientations=(135, 150, 165, 180, 195, 210, 225, 90, 270),
):
    """
    Advanced PV layout engine with:
    - Flat vs pitched detection
    - Tilt optimisation
    - Fourier POA reconstruction
    - True solar minimum spacing
    - Non-overlapping panel placement
    """

    import rasterio
    import rasterio.mask
    import numpy as np
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from shapely.affinity import rotate, translate

    # ------------------------------------------------------------
    # Load Fourier raster once
    # ------------------------------------------------------------

    with rasterio.open(fourier_raster) as src:
        fourier_data = src.read()
        fourier_transform = src.transform

    # ------------------------------------------------------------
    # Load slope and aspect rasters once
    # ------------------------------------------------------------

    with rasterio.open(slope_raster_path) as src:
        slope_data = src.read(1)
        slope_transform = src.transform

    with rasterio.open(aspect_raster_path) as src:
        aspect_data = src.read(1)

    # ------------------------------------------------------------
    # Panel specs
    # ------------------------------------------------------------

    panel_specs = [
        PVPanelSpec("2.3x1.1", 2.3, 1.1, 0.55),
        PVPanelSpec("2.1x1.05", 2.1, 1.05, 0.50),
    ]

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------

    def evaluate_poa_fast(panel_geom, tilt_deg, azimuth_deg):

        theta = np.radians(azimuth_deg)
        beta = np.radians(tilt_deg)

        minx, miny, maxx, maxy = panel_geom.bounds

        row_min, col_min = rasterio.transform.rowcol(
            fourier_transform, minx, maxy
        )
        row_max, col_max = rasterio.transform.rowcol(
            fourier_transform, maxx, miny
        )

        row_min = max(row_min, 0)
        col_min = max(col_min, 0)
        row_max = min(row_max, fourier_data.shape[1])
        col_max = min(col_max, fourier_data.shape[2])

        if row_min >= row_max or col_min >= col_max:
            return 0.0

        A0  = fourier_data[0, row_min:row_max, col_min:col_max]
        A1c = fourier_data[1, row_min:row_max, col_min:col_max]
        A1s = fourier_data[2, row_min:row_max, col_min:col_max]
        A2  = fourier_data[3, row_min:row_max, col_min:col_max]
        D0  = fourier_data[4, row_min:row_max, col_min:col_max]
        D1  = fourier_data[5, row_min:row_max, col_min:col_max]

        direct = (
            A0
            + A1c * np.cos(theta)
            + A1s * np.sin(theta)
            + A2  * np.sin(beta)
        )

        diffuse = D0 + D1 * np.sin(beta)

        return float(np.nansum(direct + diffuse))

    def solar_min_altitude(latitude):
        return 90 - latitude - 23.45

    # ------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------

    all_records = []
    all_geoms = []

    for _, roof in roof_gdf.iterrows():

        roof_polygon = roof.geometry

        if roof_polygon.area < 15:
            continue

        # --------------------------------------------------------
        # Mean slope & aspect using raster mask
        # --------------------------------------------------------

        slope_masked, _ = rasterio.mask.mask(
            rasterio.open(slope_raster_path),
            [roof_polygon],
            crop=True
        )

        aspect_masked, _ = rasterio.mask.mask(
            rasterio.open(aspect_raster_path),
            [roof_polygon],
            crop=True
        )

        mean_slope = np.degrees(np.nanmean(slope_masked))
        mean_aspect = np.degrees(np.nanmean(aspect_masked))

        # ========================================================
        # PITCHED ROOF
        # ========================================================

        if mean_slope >= slope_threshold_deg:

            if mean_aspect < 90 or mean_aspect > 270:
                continue

            ridge_angle = mean_aspect + 90
            available_area = roof_polygon.buffer(0)

            for panel in panel_specs:

                minx, miny, maxx, maxy = roof_polygon.bounds

                x = minx
                while x + panel.width_m <= maxx:

                    y = miny
                    while y + panel.height_m <= maxy:

                        p = box(0, 0, panel.width_m, panel.height_m)
                        p = rotate(p, ridge_angle, origin=(0, 0))
                        p = translate(p, xoff=x, yoff=y)

                        if available_area.contains(p):

                            E = evaluate_poa_fast(
                                p,
                                mean_slope,
                                mean_aspect,
                            )

                            all_geoms.append(p)
                            all_records.append({
                                "roof_id": roof.roof_id,
                                "panel_type": panel.name,
                                "tilt_deg": mean_slope,
                                "azimuth_deg": mean_aspect,
                                "energy_kWh": E / 1000,
                                "peak_kW": panel.peak_power_kw,
                                "shading_factor": E / (panel.peak_power_kw * 1500),
                            })

                            available_area = available_area.difference(p)

                        y += panel.height_m

                    x += panel.width_m

        # ========================================================
        # FLAT ROOF (Tilt optimisation)
        # ========================================================

        else:

            best_energy = 0
            best_panels = []
            best_tilt = None
            best_azimuth = None

            for tilt in flat_tilts:

                alpha_min = solar_min_altitude(latitude)

                spacing = (
                    panel_specs[0].height_m
                    * np.sin(np.radians(tilt))
                    / np.tan(np.radians(alpha_min))
                )

                for az in allowed_orientations:

                    candidate_available = roof_polygon.buffer(0)
                    roof_energy = 0
                    temp_panels = []

                    for panel in panel_specs:

                        minx, miny, maxx, maxy = roof_polygon.bounds

                        x = minx
                        while x + panel.width_m <= maxx:

                            y = miny
                            while y + panel.height_m <= maxy:

                                p = box(0, 0, panel.width_m, panel.height_m)
                                p = rotate(p, az, origin=(0, 0))
                                p = translate(p, xoff=x, yoff=y)

                                if candidate_available.contains(p):

                                    E = evaluate_poa_fast(p, tilt, az)
                                    roof_energy += E
                                    temp_panels.append((p, panel, E))

                                    candidate_available = candidate_available.difference(p)

                                y += panel.height_m + spacing

                            x += panel.width_m

                    if roof_energy > best_energy:
                        best_energy = roof_energy
                        best_panels = temp_panels
                        best_tilt = tilt
                        best_azimuth = az

            for geom, panel, E in best_panels:

                all_geoms.append(geom)
                all_records.append({
                    "roof_id": roof.roof_id,
                    "panel_type": panel.name,
                    "tilt_deg": best_tilt,
                    "azimuth_deg": best_azimuth,
                    "energy_kWh": E / 1000,
                    "peak_kW": panel.peak_power_kw,
                    "shading_factor": E / (panel.peak_power_kw * 1500),
                })

    # ------------------------------------------------------------
    # Output creation
    # ------------------------------------------------------------

    if len(all_records) == 0:

        panel_gdf = gpd.GeoDataFrame(
            {
                "roof_id": [],
                "panel_type": [],
                "tilt_deg": [],
                "azimuth_deg": [],
                "energy_kWh": [],
                "peak_kW": [],
                "shading_factor": [],
                "geometry": [],
            },
            geometry="geometry",
            crs=roof_gdf.crs,
        )

        building_summary = pd.DataFrame(
            columns=["roof_id", "energy_kWh", "peak_kW", "num_panels"]
        )

    else:

        panel_gdf = gpd.GeoDataFrame(
            all_records,
            geometry=all_geoms,
            crs=roof_gdf.crs,
        )

        building_summary = (
            panel_gdf.groupby("roof_id", as_index=False)
            .agg(
                energy_kWh=("energy_kWh", "sum"),
                peak_kW=("peak_kW", "sum"),
                num_panels=("geometry", "count"),
            )
        )

    panel_gdf.to_file(output_panel_path)
    building_summary.to_csv(
        output_building_summary_path.replace(".gpkg", ".csv"),
        index=False,
    )

    return panel_gdf, building_summary

