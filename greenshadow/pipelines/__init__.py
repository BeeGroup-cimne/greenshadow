# -*- coding: utf-8 -*-
"""
End-to-end pipelines: from an area name to irradiation rasters and PV layouts.

``SolarPipeline`` holds the region-agnostic stages (weather, sky view factor,
hillshades, irradiation, roof and parcel PV); a regional pipeline only says
where the area, the terrain, the parcels and the roofs come from.

- ``CataloniaMunicipalityPipeline``: ICGC LiDAR, Spanish cadastre and SIGPAC.
"""

from .base import (PipelineConfig, PipelineResult, RoofPlanes, SolarPipeline,
                   TerrainModels, run_pipelines)
from .catalonia import (CataloniaConfig, CataloniaMunicipalityPipeline,
                        run_catalonia_municipalities)

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "RoofPlanes",
    "SolarPipeline",
    "TerrainModels",
    "run_pipelines",
    "CataloniaConfig",
    "CataloniaMunicipalityPipeline",
    "run_catalonia_municipalities",
]
