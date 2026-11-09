# Example run: Alpicat, execution times

Runs of the Catalan pipeline on the municipality of Alpicat (Lleida,
cadastral code 25023): digital models, sky view factors, crop rows, roof
planes and roof obstacles, the semi-transparent canopy, two daily analyses
(the solstices), the annual irradiation of 2023, rooftop PV, and PV fields
and agrivoltaics on the parcels. It shows how long each stage takes and what
it produces, for a mid-size rural municipality on a single workstation, from
scratch and when rerun on what is already on disk.

## 1. Setup

**Machine.** AMD Ryzen 9 3950X (16 cores / 32 threads), 64 GB RAM, NVIDIA
RTX PRO 5000 Blackwell (48 GB). Working directory on a SATA SSD (Samsung
850 EVO).

**Configuration.** [`example/get_solar_energy_from_municipality.py`](../example/get_solar_energy_from_municipality.py)
as shipped:

```python
config = gs.CataloniaConfig(
    wd="/mnt/data/greenshadow",
    year=2023,
    days=[dt.date(2023, 12, 22), dt.date(2023, 6, 21)],
    rural_parcels_year=2025,
    irradiance_decomposition="disc",
    semitransparent_vegetation=True,
    vegetation_leaf_type=0,
    recalculate_digital_models=False,
    recalculate_solar_exposure=False,
    render_roof_3d_view=False,
    roof_obstacles=gs.ObstacleConfig(
        imagery=True,
        sam3=True,
        sam3_python="/home/gmor/PyVenvs/sam3/bin/python",
        lidar=False,
    ),
    economics=gs.Economics(),
    criterion="npv",
)
results = gs.run_catalonia_municipalities({"Alpicat": "25023"}, config)
```

**Area.** 41 ICGC LiDAR tiles. Every raster is 12264 × 13167 px at 0.5 m
(161 Mpx, 6.1 × 6.6 km). The area holds 1891 cadastral buildings and 2077
agricultural parcels.

**Runs.** The times come from three runs:

* **Full run** (2026-10-26, 15:59–17:39). The 41 LAZ tiles, the cadastre,
  the census tracts and the Copernicus DEM used by `hypercadaster_ES` were
  already on disk; every greenshadow product was computed from scratch.
  The roof obstacle stage did not exist yet. Downloading the LiDAR the
  first time adds the transfer of the 41 tiles from the ICGC.
* **Rerun** (2026-11-05, 16:17–16:42). The digital models, the sky view
  factors, the crop rows, the hillshades and the irradiation rasters of the
  full run were reused. The roof planes, the roof obstacles, the rooftop PV
  and the parcel PV were computed again.
* **SAM 3 inference** (2026-11-04). The rerun found every SAM 3 window in
  the cache left by a run the day before, whose cache files date the
  inference itself.

The wall-clock times come from the progress bars of the logs and from the
write times of the outputs.

## 2. Times from scratch

Each stage is timed in the run that last computed it: the roof planes, the
roof obstacles, the rooftop PV and the parcel PV in the rerun, the rest in
the full run.

| Stage | Wall clock | Share | Notes |
|---|---:|---:|---|
| Weather, boundary, parcels | not timed | | ERA5-Land + ERA5 direct radiation (8760 h), SIGPAC parcels of province 25 |
| LiDAR tiles (cached) | 0:07 | 0.1 % | 41 tiles found, already downloaded |
| DSM | 2:05 | 1.8 % | Halo strips 0:34, tiles 1:14, 11 workers; slope, aspect, classes |
| DSM without vegetation | 2:09 | 1.9 % | Halo strips 0:30, tiles 1:23 |
| DEM | 1:46 | 1.6 % | Halo strips 0:30, tiles 1:00 |
| Sky view factor | 3:33 | 3.1 % | CUDA hybrid: coarse 0:07, full resolution on 62 % of pixels 3:06 |
| Sky view factor without vegetation | 3:01 | 2.7 % | Full resolution on 53 % of pixels 2:35 |
| Crop rows | 4:22 | 3.9 % | 2077 parcels at 7.9 parcels/s |
| Roof planes | 10:49 | 9.5 % | 914 building clusters in 24 tiles of 1 km: 10:28 with 7 workers, 4.6 GB peak per tile |
| Roof obstacles | ~12:30 | 11.0 % | 8:45 with the SAM 3 cache warm; see below |
| Canopy and first sun table | 0:59 | 0.9 % | 15645 tree crowns, grown 1.1 years since the flight |
| **Day 2023-12-22** (36 sun positions) | 10:54 | 9.6 % | see below |
| **Day 2023-06-21** (61 sun positions) | 16:45 | 14.8 % | see below |
| Annual hillshades (462 sun positions) | 14:51 | 13.1 % | 1.86 s per position on the GPU |
| Annual irradiation | 15:35 | 13.7 % | `total`, `fourier_poa`, `fourier_decomposition` in one pass |
| Rooftop PV | 0:30 | 0.4 % | 1452 buildings at 59 buildings/s, 8 mounting scenarios each |
| Annual hillshades without vegetation | 6:12 | 5.5 % | 0.8 s per position |
| Vertical and tracking bases | 4:00 | 3.5 % | `fourier_poa`, `vertical_basis`, `tracking_basis` in one pass |
| Parcel PV and agrivoltaics | 3:16 | 2.9 % | 876 parcels at 4.9 parcels/s |
| **Total** | **~1:53:24** | 100 % | |

By group:

| Group | Wall clock | Share |
|---|---:|---:|
| Digital models (DSM, DSM without vegetation, DEM) | 6:00 | 5 % |
| Sky view factors | 6:34 | 6 % |
| Crop rows, roof planes, roof obstacles, canopy | ~28:40 | 25 % |
| Daily analyses (2 days) | 27:39 | 24 % |
| Annual analysis and PV | 44:24 | 39 % |

### Inside the roof obstacles

| Step | Wall clock | Result |
|---|---:|---|
| Sky view factor | 0:06 | 1879 obstacles |
| Orthophoto co-registration | 0:05 | 1528 of 1528 roofs registered on `ortofoto-rgb-25cm-2022`, median shift 0.61 m |
| Orthophoto sun | 0:33 | azimuth 168°, altitude 56° |
| SAM 3, cache warm | 6:05 | 1600 windows at 4.4 windows/s; 1725 obstacles on 508 roofs |
| SAM 3, inference | 9:50 | the same 1600 windows at 2.7 windows/s (2026-11-04) |
| Shadows | 0:07 | 6659 obstacles on 1042 roofs |
| Terraces | 0:02 | 1605 planes, 31,737 m² |
| Orthophoto and SAM 3 start-up, fusion, usable area, writing | 1:47 | 7461 obstacles; usable area 382,807 of 481,391 m² |
| **Total**, cache warm | **8:45** | |
| **Total**, first run | **~12:30** | with the 9:50 of inference in place of the 6:05 |

The SAM 3 cache saves less than it seems: its key is the pixels of each
window, so a warm rerun still reads and resamples all 1600 windows from the
orthophoto, on one reader thread. Only the inference, about 4 minutes, is
saved.

### Inside a daily analysis

| Step | 2023-12-22 (36 positions) | 2023-06-21 (61 positions) |
|---|---:|---:|
| Hillshades with canopy (CUDA) | 1:06 | 1:42 |
| Video | 2:07 | 3:09 |
| Hillshades without vegetation (CUDA) | 0:28 | 0:31 |
| Video | 0:38 | 0:52 |
| Elevated shadows at 1.5 m | 3:57 | 6:29 |
| Video | 0:59 | 1:27 |
| Daily irradiation (CUDA) | 1:11 | 2:01 |

The optional products dominate the daily analyses. The three videos took
9:12 over both days and the elevated shadows took 10:26, against 3:47 for
the hillshades themselves.

## 3. Times of a rerun

With the digital models, the hillshades and the irradiation on disk, the
rerun of 2026-11-05 took **25:20**, and 90 % of it went to three of the
stages it computed again: the roof planes, the roof obstacles and the
parcel PV.

| Stage | Wall clock | Notes |
|---|---:|---|
| Weather, boundary, parcels | not timed | before 16:17:17 |
| LiDAR, DSM, DEM, sky view factors, crop rows (reused) | 0:14 | |
| Roof planes | 10:49 | computed again |
| Roof obstacles | 8:45 | computed again, SAM 3 cache warm |
| Canopy, both days, annual hillshades and irradiation (reused) | 1:41 | the canopy is rebuilt on every run (about 1 min) |
| Rooftop PV | 0:30 | computed again |
| Annual bases without vegetation (reused) | 0:05 | |
| Parcel PV and agrivoltaics | 3:16 | computed again |
| **Total** | **25:20** | |

Every reused product says so in the log, as `Using existing ...` with its
path, or with the count of files kept for a set such as the hillshades of a
day.

## 4. What it produced

| Product | Result |
|---|---|
| Crop rows | 8866 rows over 345 planted parcels; median spacing 4.41 m, canopy 2.16 m tall and 2.27 m wide |
| Roof planes | 10976 facets on 1528 roofs |
| Roof obstacles | 7461 obstacles; usable area 382,807 of 481,391 m² (80 %) |
| Rooftop PV (best per building) | 114443 modules, 49.53 MWp, 61.38 GWh/year over 1333 buildings |
| Parcel scenarios | 39941 scenarios over 837 parcels; 803 parcels with a positive-NPV scenario |
| Best is a PV field | 609 parcels, 701.0 MWp, 1104.7 GWh/year |
| Best is agrivoltaic | 194 parcels, 15.1 MWp, 23.0 GWh/year |

Disk use under `{wd}/Alpicat/`: **112 GB**. Most of it is hillshades:

| Directory | Size |
|---|---:|
| `2023/hillshades` (462 files) | 69 GB |
| `2023/hillshades_no_vegetation` (462 files) | 18 GB |
| `2023/` annual rasters | 4.3 GB |
| `20230621/` | 9.7 GB |
| `20231222/` | 6.7 GB |
| Digital models, SVF, slope and aspect | ~5 GB |
| `sam3_cache/` (1600 windows) | 30 MB |

Hillshades with the semi-transparent canopy hold fractional values, so they
compress about 4× worse than the opaque ones without vegetation.

## 5. Reading the numbers

* **The annual analysis is the largest part.** Its cost scales with the
  number of distinct sun positions (462 for a year at the default
  sampling), not with the 8760 weather hours.
* **Irradiation is bound by disk reads, not by the GPU.** Summing the 462
  annual hillshades takes about as long as generating them (1.94 s against
  1.86 s per position), and most of that time goes to reading and decoding
  the ZSTD tiles. A faster disk shortens it; a faster GPU does not. Each
  pass sums every mode it is given, so adding a mode costs almost nothing.
* **The canopy costs 2–2.5×.** Hillshades without vegetation run at 0.8 s
  per position against 1.86 s with the semi-transparent canopy, and take a
  quarter of the disk. Use `semitransparent_vegetation=False` when trees
  do not matter.
* **The digital models are cheap.** All three models take 6 minutes. The
  roof planes (11 min) and the roof obstacles (12.5 min, of which SAM 3
  takes 10) are the most expensive geometric steps.
* **The daily options are costly.** The videos take a third of each day
  and the elevated shadows almost 40 %. Turn them off when you don't need
  them: `daily_videos=False` (`--no-videos`) and `elevated_height_m=None`.
  With both off, a solstice day takes about 3–4 minutes.
* **Rerunning costs what is computed again.** Every product is cached:
  the reused ones took 2 minutes of the 25-minute rerun, the rest went to
  the roof planes, roof obstacles and PV planners it recomputed. A second
  call with the same configuration only loads the results, and a new day
  computes only that day. A change upstream does not invalidate what was
  derived from it; force it with the `recalculate_*` flags. See
  [pipelines.md](pipelines.md).
