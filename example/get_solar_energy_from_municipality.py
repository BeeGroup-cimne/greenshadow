# -*- coding: utf-8 -*-
"""
greenshadow Example: solar energy and PV potential of Catalan municipalities

One call runs the whole chain for each municipality (greenshadow.pipelines):

1. Municipality boundary from the census tracts, and ERA5-Land weather at it
2. ICGC LiDAR tiles, cadastral buildings and SIGPAC rural parcels
3. DSM, DSM without vegetation, DEM and sky view factors
4. Crop rows of the planted parcels, roof planes of the buildings
5. Semi-transparent tree canopy, aged to the simulated year
6. Daily hillshades, shadow videos and irradiation for the chosen days
7. Annual irradiation, PV on the roofs, and PV fields and agrivoltaics on
   the parcels, ranked against what each parcel earns today

Every product is cached under {wd}/{municipality}, so a second run only
computes what is missing. The same is available from the shell:

    python -m greenshadow.pipelines catalonia 25023 --name Alpicat \\
        --wd /mnt/data/greenshadow --year 2023 --day 2023-12-22

The tuned parameters of each step (DSM classes, roof detection, SVF ...) live
in greenshadow/pipelines/catalonia.py; subclass CataloniaMunicipalityPipeline
to change them.
"""

import datetime as dt

import greenshadow as gs

municipal_codes = {
    "Alpicat": "25023",
    # "Barcelona": "08900",
}

config = gs.CataloniaConfig(
    wd="/mnt/data/greenshadow",
    # Annual analysis: aggregated shades and irradiation of the year, and the
    # PV layouts on roofs and parcels. Daily analysis: shades, videos and
    # irradiation of specific days, e.g. the solstices. Either or both.
    year=2023,
    days=[dt.date(2023, 12, 22), dt.date(2023, 6, 21)],
    rural_parcels_year=2025,
    # "disc" | "dirint" | "era5" | "erbs"; see docs/irradiance_era5.md
    irradiance_decomposition="disc",
    # Trees as a semi-transparent medium, aged from the LiDAR flight to the
    # simulated year. False keeps them opaque.
    semitransparent_vegetation=True,
    vegetation_leaf_type=0,  # 0 = deciduous, 1 = evergreen
    # Rebuild from scratch instead of reusing what is on disk.
    recalculate_digital_models=False,
    recalculate_solar_exposure=False,
    # Interactive 3D page of the segmented roofs, to check the detection.
    render_roof_3d_view=False,
    # Assumptions of the parcel scenarios: prices, discount rate, life ...
    economics=gs.Economics(),
    criterion="npv",  # or "irr", "roi", "lcoe", "payback", "energy", "ler"
)

results = gs.run_catalonia_municipalities(municipal_codes, config)

# Everything the run produced is in results[name]: rasters on disk, frames in
# memory, e.g. results["Alpicat"].parcel_pv_best or .annual["fourier_poa"].
