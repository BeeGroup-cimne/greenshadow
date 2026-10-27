# Solar irradiance from ERA5-Land: alignment, beam/sky split and validation

How greenshadow turns the ERA5-Land weather into the direct and diffuse
irradiance its solar tables, hillshades and PV planners use. The split is
checked against PVGIS-SARAH3 at Lleida (Alpicat) for 2023.

## 1. What the weather provides

ERA5-Land provides one solar variable, `ssrd` (surface solar radiation
downwards). It is the **all-sky global horizontal irradiance**, with clouds
already included, so using it as GHI is correct. It does not say how much of
that light came straight from the sun (the beam, DNI) and how much came from
the sky (the diffuse, DHI). That split decides two things:

* how much a shadow costs, since a shadow removes the beam and keeps most of
  the diffuse;
* how much a tilted, vertical or tracking panel gains over a flat one.

`hypercadaster_ES` reads `ssrd` as `GHI`. greenshadow redoes the split itself
(`shader._attach_weather`, used by `generate_azimuth_altitude_pairs_by_date`
and `..._by_year`) in three steps.

### 1.1 Time alignment

An ERA5 or ERA5-Land value stamped *T* is the mean over the hour **ending**
at *T*. For 2023 at Lleida, the hourly GHI matches pvlib's clear-sky GHI best
when each value is placed at *T* − 30 min:

| Value stamped T placed at | T − 1 h | **T − 30 min** | T | T + 30 min |
|---|---|---|---|---|
| Correlation with clear sky | 0.960 | **0.974** | 0.964 | 0.931 |

In the same data, the GHI-weighted timestamps fall 0.47 h after solar noon.
Each record is therefore moved to the middle of its hour
(`weather_time_offset_minutes=-30`, the default) before it is interpolated
onto the solar table's times. Pairing each hour with the sun at its end, as
before, put every hour's sunshine half an hour late. Combined with the
diffuse-model split done at that late sun position, the annual beam came out
leaning towards the West.

### 1.2 The beam/sky split (`irradiance_decomposition`)

The split is computed at the solar table's exact, undiscretised sun position:

| Option | Source of the split |
|---|---|
| `"disc"` (**default**) | DISC model (Maxwell 1987): DNI from the clearness index |
| `"dirint"` | DIRINT (Perez et al. 1992): the clearness index plus its hour-to-hour variability, which separates broken cloud from haze |
| `"erbs"` | Erbs et al. (1982) diffuse-fraction fit |
| `"era5"` | ERA5's own all-sky direct radiation (§ 1.3) |
| `"weather"` | whatever DNI/DHI the weather frame already carries |

DNI is never allowed to put more energy on the horizontal than GHI does. DHI
is what remains, `GHI − DNI·cos z`, and is never negative.

### 1.3 ERA5's own direct radiation (`"era5"`)

ERA5 single levels, the 0.25° reanalysis that ERA5-Land is downscaled from,
also publishes `fdir`, the all-sky direct radiation at the surface that its
own clouds let through, and `tcc`, the total cloud cover.
`greenshadow.weather` downloads them as a point time series, in the same way
hypercadaster downloads the ERA5-Land one:

```python
weather_df = gs.load_ERA5Land_with_direct_radiation(
    data_dir=weather_data_dir, lat_range=[lat, lat], lon_range=[lon, lon],
    ym_range=ym_range, tz_local=tz_local, update=False,
    use_point_location=True, mode="instantaneous")
```

* Dataset: `reanalysis-era5-single-levels-timeseries`. Variables:
  `surface_solar_radiation_downwards`,
  `total_sky_direct_solar_radiation_at_surface` and `total_cloud_cover`.
  Format: CSV. It needs the same `~/.cdsapirc` as ERA5-Land.
* The point is snapped to the 0.25° grid. Files are cached as
  `era5_radiation_<start>_<end>_<lat>_<lon>.csv` and reused while they cover
  the period.
* The ERA5-Land frame is returned unchanged, plus `GHI_ERA5`,
  `directHorizontal` (W/m², hourly mean) and `totalCloudCover` (0–1). If the
  download fails, the plain frame is returned with a warning.

With `"era5"`, the hour's **beam fraction** `directHorizontal / GHI_ERA5` is
applied to the ERA5-Land GHI. The finer ERA5-Land global keeps the total, and
ERA5's clouds decide how it splits. The two GHI series are nearly the same
(hourly r = 0.9998; 1,764 vs 1,760 kWh/m² in 2023). DNI is capped at the
extraterrestrial value, because a sunrise-hour mean divided by one instant's
cosine can exceed it. Without the ERA5 columns, `"era5"` falls back to
`"dirint"` with a warning.

## 2. Validation against PVGIS-SARAH3

### Method

`eval_irradiance_pvgis.py` makes this comparison for any point and year:

```bash
python eval_irradiance_pvgis.py --lat 41.66 --lon 0.55 --year 2023 \
    --weather-dir /mnt/data/greenshadow/weather --out lleida2023
```

* **Reference.** PVGIS 5.3, database PVGIS-SARAH3, hourly, terrain horizon
  off (`usehorizon=0`), so both sides describe an open sky. The horizontal
  series gives GHI, beam and diffuse; the two-axis-tracker series (a plane
  that always faces the sun) gives DNI.
* **Estimates.** ERA5-Land at the municipality centroid under every split,
  evaluated at the middle of each hour.
* SARAH3 is satellite-derived, so it is a reference, **not ground truth**.
  Differences of a few percent are within what separates independent
  irradiance products. Lleida has no public ground measurement of DNI or DHI
  to settle that.

### Annual totals, 2023 (kWh/m²)

| Source | GHI | Beam on horizontal | DHI | DNI | Diffuse fraction | DNI vs SARAH3 | DHI vs SARAH3 |
|---|---|---|---|---|---|---|---|
| PVGIS-SARAH3 | 1,781 | 1,237 | 544 | 2,173 | 0.305 | — | — |
| ERA5-Land + `era5` | 1,760 | 1,304 | 456 | 2,434 | 0.259 | **+12.0 %** | −16.1 % |
| ERA5-Land + `dirint` | 1,760 | 1,150 | 611 | 2,066 | 0.347 | −4.9 % | +12.3 % |
| ERA5-Land + `disc` | 1,760 | 1,164 | 596 | 2,128 | 0.339 | **−2.0 %** | +9.6 % |
| ERA5-Land + `erbs` | 1,760 | 1,144 | 617 | 1,945 | 0.350 | −10.5 % | +13.4 % |

The global agrees to within 1.1 %, so what differs is the split.

### Monthly DNI, 2023 (kWh/m²)

| | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SARAH3 | 136 | 158 | 210 | 235 | 220 | 181 | 231 | 240 | 177 | 142 | 134 | 109 |
| `era5` | 159 | 175 | 228 | 250 | 234 | 201 | 257 | 263 | 200 | 167 | 154 | 146 |
| `dirint` | 134 | 160 | 211 | 215 | 194 | 149 | 205 | 226 | 174 | 140 | 132 | 126 |
| `disc` | 146 | 169 | 217 | 224 | 201 | 151 | 201 | 224 | 175 | 146 | 140 | 135 |
| `erbs` | 118 | 144 | 194 | 210 | 196 | 156 | 203 | 214 | 161 | 127 | 114 | 106 |

ERA5's excess is systematic: it is higher every month, by +34 % in
December. DISC and DIRINT are short in early summer (June −17 % and −18 %),
when convective cloud is broken, and long in December (+24 % and +16 %).
Erbs is short all year.

### Hourly agreement, daylight hours

| Split | DNI MBE (W/m²) | DNI RMSE (W/m²) | DNI r | DHI MBE (W/m²) | DHI RMSE (W/m²) | DHI r |
|---|---|---|---|---|---|---|
| `era5` | +49 | 213 | **0.750** | −21 | **57** | **0.735** |
| `dirint` | −25 | 227 | 0.722 | +15 | 73 | 0.707 |
| `disc` | −10 | 226 | 0.714 | +12 | 70 | 0.719 |
| `erbs` | −53 | 217 | 0.757 | +17 | 58 | 0.724 |

GHI itself agrees hour by hour only to r = 0.93 (RMSE 103 W/m², rRMSE 25 %).
This is the floor set by the two products placing clouds differently, plus
SARAH3 being an instant a few minutes past the hour while ERA5 is an hourly
mean. The splits differ from each other by much less than that floor. ERA5's
`fdir` follows the hours slightly better, but carries the bias above.

### What a panel receives, 2023 (kWh/m², isotropic sky, albedo 0.2)

Every source here goes through the same transposition, so only the split
differs:

| Source | South 30° | Vertical South | Vertical East |
|---|---|---|---|
| PVGIS-SARAH3 | 2,062 | 1,395 | 1,036 |
| `era5` | +0.9 % | +3.2 % | +4.7 % |
| `dirint` | −1.1 % | **+0.9 %** | **+1.1 %** |
| `disc` | **−0.5 %** | +2.5 % | +2.2 % |
| `erbs` | −2.0 % | −2.3 % | −4.8 % |

For a fixed field near the optimum tilt, every split is within ±2 %. The
choice matters more for vertical and East/West faces, for trackers (which
work almost only on the beam), and for shading losses (a shadow removes the
beam).

## 3. Conclusions

* **Default: `"disc"`.** It is closest to SARAH3 on the annual DNI (−2 %) and
  on the South-30° plane (−0.5 %), and within 2.5 % on vertical faces. Over
  DNI, DHI and the three planes, its mean absolute deviation is 3.4 %, against
  4.1 % for `dirint`, 6.6 % for `erbs` and 7.4 % for `era5`.
* **`"dirint"`** is the better choice for vertical bifacial fences and
  East/West faces (within 1 % on both).
* **`"era5"`** follows the hours best but puts 12 % too much energy in the
  beam and 16 % too little in the diffuse at this site. That overstates both
  tracker gains and shading losses. It is kept as an option, and is the one to
  revisit if ground DNI becomes available.
* **The global itself is sound.** ERA5-Land's annual GHI is within 1.1 % of
  SARAH3.

## 4. Limitations

* One site and one year. Re-run `eval_irradiance_pvgis.py` before relying on
  the ranking elsewhere, especially in mountain or coastal municipalities,
  where both products are less certain.
* SARAH3 is not ground truth (§ 2).
* The solar tables floor sun positions to their discretisation step (3° in the
  annual run). The hillshades and the energy rasters both use the floored
  positions, which lowers annual beam on the horizontal by about 2.4 % at
  Lleida. That is independent of the split.
* The transposition behind the panel figures is isotropic, as is
  `reconstruct_poa_from_basis`. PVGIS's own POA uses the Muneer model, so its
  published PV yields differ from these figures by more than the split does.

## 5. History

Before 2026-10-19, `compute_surface_solar_energy` weighted each sun position
by the time to the next row of the solar table. A daylight-only table jumps
the night after each day's last hour, so those evening positions counted for
12–16 hours each. The annual basis at Lleida then read 1,305 kWh/m² of beam on
the horizontal and 998 of diffuse, against the weather's 1,083 and 625, and
leaned strongly West. Every row now counts for one step of the table. Energy
rasters written before that date overstate the annual energy by about a third
and must be recomputed (`recalculate_solar_exposure=True` in the pipeline
configuration, `overwrite=True` when calling the energy functions);
the hillshades remain valid.

## References

* Maxwell, E. L. (1987). *A quasi-physical model for converting hourly global
  horizontal to direct normal insolation.* SERI/TR-215-3087.
* Perez, R., Ineichen, P., Maxwell, E., Seals, R., Zelenka, A. (1992).
  *Dynamic global-to-direct irradiance conversion models.* ASHRAE
  Transactions 98(1), 354–369.
* Erbs, D. G., Klein, S. A., Duffie, J. A. (1982). *Estimation of the diffuse
  radiation fraction for hourly, daily and monthly-average global radiation.*
  Solar Energy 28(4), 293–302.
* Hersbach, H. et al. (2020). *The ERA5 global reanalysis.* QJRMS 146,
  1999–2049.
* Muñoz-Sabater, J. et al. (2021). *ERA5-Land: a state-of-the-art global
  reanalysis dataset for land applications.* ESSD 13, 4349–4383.
* PVGIS 5.3 and the SARAH3 database: European Commission, Joint Research
  Centre, https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en
