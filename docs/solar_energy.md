# Irradiation rasters

How `greenshadow.shader` sums the hillshades, the sky view factor and the
weather into irradiation: on the surface itself, or as a basis from which
the irradiation of any panel orientation, a vertical fence or a tracker is
reconstructed exactly.

```python
paths = gs.compute_surface_solar_energies(
    solar_df=year["df"], hillshade_dir=".../2023/hillshades",
    svf_path="svf.tif", aoi=aoi,
    modes=("total", "fourier_poa", "fourier_decomposition"),
    out_path=".../2023/annual_solar_exposure.tif",
    canopy_leaf_type=0)              # only for canopy hillshades
paths["fourier_poa"]                 # .../annual_solar_exposure_fourier_poa.tif
```

`compute_surface_solar_energy(mode=...)` does one mode. The mode is
appended to the file name (`foo.tif` → `foo_total.tif`); always use the
returned path. An existing raster is returned as it is unless
`overwrite=True`.

## 1. What is summed

For each discretised sun position `t` (azimuth θ, altitude φ) of the solar
table, the weather gives the energy that position stands for:
`DNI_t · Δt` and `DHI_t · Δt`, where `Δt` is the table's step. A position
that recurs on many days carries the sum of all of them. At each pixel:

* **beam**: `DNI_t · hill_t`, where `hill_t` is the hillshade - the
  cosine of the incidence angle on the surface, 0 in a cast shadow, and
  attenuated by the canopy when there is one;
* **diffuse**: `DHI_t · SVF`.

Canopy hillshades are resolved per month, since the same sun position is
attenuated differently in February and in July; plain ones once per bin.

**History.** Before 2026-10-19 each row was weighted by the time to the
next one. A daylight-only table jumps the night after each day's last hour,
so those evening positions counted for 12-16 hours: annual energy came out
about a third too high and leaning West. Rasters written before that date
must be recomputed (`recalculate_solar_exposure=True`); the hillshades
remain valid.

## 2. Modes

| Mode | Bands | Units | Use |
|---|---|---|---|
| `total` | direct, diffuse | **Wh per pixel** | Irradiation on the surface as it is, e.g. a module flush with an inclined roof |
| `monthly` | 12 direct, 12 diffuse, annual direct, annual diffuse | Wh per pixel | Seasonal maps |
| `fourier_decomposition` | A0, A1c, A1s, A2 (direct), D0, D1 (diffuse) | Wh per pixel | Coarse PV screening (`pv_potential_from_fourier_raster`) |
| `fourier_poa` | `B_z`, `B_east`, `B_north`, `D_sky`, `GHI` | **Wh/m²** | Any fixed panel orientation (§ 3) |
| `vertical_basis` | `H_east`, `H_west`, `H_north`, `H_south` | Wh/m² | Vertical bifacial fences |
| `tracking_basis` | `T2_dir`, `T2_sky`, `T2_gnd`, `T1_dir`, `T1_sky`, `T1_gnd` | Wh/m² | Dual-axis and horizontal N-S single-axis trackers |

The last two also write `<raster>_sun.csv`, the sun bins and their
unobstructed beam and diffuse energy, from which the parcel planner works
out the shading between rows. Both are described, and validated against
pvlib, in [pv_tracking_irradiance_plan.md](pv_tracking_irradiance_plan.md).

### Units

`total`, `monthly` and `fourier_decomposition` are multiplied by the pixel
area: a band value is the Wh falling on that pixel over the period. Divide
by the pixel area for Wh/m² - at 0.5 m, Alpicat's mean annual direct of
235,538 Wh per pixel is 942 kWh/m². The value of the same irradiation
therefore scales with the resolution: 3,600 times larger on a 30 m grid
than on a 0.5 m one. The three bases are per m², so a panel's energy is the
reconstructed value times its area.

## 3. Any fixed panel: `fourier_poa`

Instead of the surface's own incidence, this basis recovers the shadow mask
(`hill_t / sin φ_t`, exact on near-horizontal surfaces) and projects the lit
beam on three fixed axes, with the sun vector
`s = (cos φ sin θ, cos φ cos θ, sin φ)`:

    B_z = Σ E_t sin φ     B_east = Σ E_t cos φ sin θ     B_north = Σ E_t cos φ cos θ

plus `D_sky = Σ DHI_t · SVF` and `GHI`. For a panel of tilt β and azimuth γ
(0 = North, clockwise):

    POA = B_z cos β + sin β (B_north cos γ + B_east sin γ)
        + D_sky (1 + cos β)/2 + GHI · albedo (1 − cos β)/2

`reconstruct_poa_from_basis(b_z, b_east, b_north, d_sky, ghi, tilt_deg=,
azimuth_deg=, albedo=0.2)` evaluates it for scalars or whole arrays. It is
exact for the beam as long as the panel never sees the sun from behind
(South-facing panels at any tilt, East/West ones at low tilt); at β = 0 it
collapses to `GHI`, the cheapest self-check. A module coplanar with an
inclined roof already has the roof's orientation, so for it the `total`
raster *is* the POA. Vertical faces and trackers break the condition, hence
the two other bases.

`pv_potential_from_fourier_raster` (on `fourier_decomposition`) cannot
separate the panel's orientation from the ground's, and is kept only for
quick whole-raster screening where the panels follow the terrain.

## 4. Reading the hillshades once

Reading the hillshades is most of the cost, and every mode is a sum over
the same ones, so `compute_surface_solar_energies(modes=...)` accumulates
any set of modes in one read; three modes take about as long as one.

* **Backends.** `cuda=None` sums on the GPU when it is usable and the
  accumulators fit in 90 % of its free memory, and on the CPU (Numba)
  otherwise; `test_solar_energy_backends.py` checks they agree.
* **I/O.** While GDAL decodes one hillshade (multi-threaded), the next file
  is read sequentially in the background to warm the page cache. On the full
  Alpicat grid a canopy hillshade costs about 0.6-1.3 s to decode and
  0.08 s of GPU kernels for three modes: the step is bound by reading, not
  by computing. Faster runs come from lighter hillshades (ZSTD, see
  [hillshades_and_sky_view.md](hillshades_and_sky_view.md#2-hillshades-with-cast-shadows)),
  not from more kernels.
* **Output.** ZSTD level 9 with the floating-point predictor, tiled 256 ×
  256, written with all cores (a 5-band raster: 40 s single-threaded,
  9 s threaded).
* **Memory.** Accumulators are float32 grids: 2, 24, 6, 5, 4 and 6 bands for
  the modes above. On Alpicat (10,265 × 11,169 px) one band is 0.46 GB.

## 5. Which hillshades for which product

| Product | Hillshades | SVF |
|---|---|---|
| Irradiation of the area as it is, rooftop PV | DSM with vegetation (canopy) | `svf.tif` |
| PV fields and agrivoltaics that clear the parcel | DSM without vegetation | `svf_no_vegetation.tif` |
| Agrivoltaic tables in the alleys of a plantation | DSM with vegetation | `svf.tif` |

## 6. Limitations

* Isotropic sky in both the diffuse sum and the transposition (no
  circumsolar or horizon brightening); PVGIS uses the Muneer model, so its
  POA figures differ by more than the beam/sky split does.
* The lit fraction `hill / sin φ` assumes a near-horizontal surface, which
  is where the bases are used.
* Sun positions are floored to the table's step (3° in the annual run),
  about −2.4 % of annual horizontal beam at Lleida.
* The weather is a single point; see
  [irradiance_era5.md](irradiance_era5.md) for its accuracy.
