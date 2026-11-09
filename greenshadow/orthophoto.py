# -*- coding: utf-8 -*-
"""
greenshadow.orthophoto

Orthophotos for the roof obstacle detector: where they come from, a local
tile cache, and crops read onto any grid of the project CRS, optionally
displaced to undo the per-building misregistration found by
``roof_obstacles.coregister_orthophoto``.

    ortho = icgc_orthophoto(planes.geometry, "/mnt/data/greenshadow/icgc_orthophoto",
                            year=2022)
    image, transform = read_orthophoto(ortho, bounds, crs=planes.crs)

**Sources.**

* ICGC territorial orthophoto (Catalonia): RGB, 25 cm, EPSG:25831, one
  GeoPackage per year for the whole territory (2009 onwards, ~210 GB each)
  on datacloud.icgc.cat, CC BY 4.0. Only the windows that hold buildings are
  read, through HTTP range requests (GDAL ``/vsicurl/``), and kept as local
  1 km GeoTIFF tiles - native pixels, never WMS. The flight window of every
  sheet is published for the current edition only; an older year is dated
  1 July of that year.
* Anything else GDAL reads - PNOA sheets from the CNIG download centre, a
  VRT, a COG URL - through ``local_orthophoto``. The CNIG centre has no
  scriptable download, so PNOA sheets are fetched by hand.

Crops are reprojected to the requested CRS at the native resolution
(bilinear), so an orthophoto in another CRS than the LiDAR needs no
preprocessing.
"""

import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds, calculate_default_transform
from rasterio.windows import Window, from_bounds

try:
    from .logging_utils import info, warning, tqdm
    from .utils import _session, UA
except ImportError:
    from greenshadow.logging_utils import info, warning, tqdm
    from greenshadow.utils import _session, UA


ICGC_ORTHO_DIRECTORY = "https://datacloud.icgc.cat/datacloud/orto-territorial/gpkg_unzip/"
ICGC_ORTHO_URL = ICGC_ORTHO_DIRECTORY + "ortofoto-rgb-25cm-catalunya-{year}.gpkg"
ICGC_ORTHO_CURRENT_URL = ("https://datacloud.icgc.cat/datacloud/orto-territorial/vigent/"
                          "gpkg_unzip/ortofoto-rgb-25cm-catalunya.gpkg")
# Flight window of every sheet of the current edition.
ICGC_ORTHO_METADATA_URL = "https://datacloud.ide.cat/metadades/ortofoto-25cm-v4r0-metadades.zip"

RASTER_EXTENSIONS = (".tif", ".tiff", ".vrt", ".jp2", ".ecw", ".gpkg", ".img")

# GDAL options for reading over HTTP: no directory listings (the datacloud
# answers them with the whole index), retries, timeouts (a stalled range
# request otherwise hangs its tile for good), and a user agent the server
# accepts.
_VSICURL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=",".join(RASTER_EXTENSIONS),
    GDAL_HTTP_USERAGENT=UA,
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    GDAL_HTTP_CONNECTTIMEOUT="30",
    GDAL_HTTP_TIMEOUT="120",
)


@dataclass
class Orthophoto:
    """
    An orthophoto: the rasters it is made of and when it was flown.

    ``paths`` are GDAL-readable rasters (local files, ``/vsicurl/`` URLs),
    in any CRS; ``bands`` the red, green and blue bands. ``sheets``, when
    known, holds the flight window of each sheet (``date_start``,
    ``date_end``, polygons); otherwise ``date`` dates the whole image.
    """
    paths: List[str]
    date: Optional[pd.Timestamp] = None
    sheets: Optional[gpd.GeoDataFrame] = None
    name: str = "orthophoto"
    bands: Tuple[int, ...] = (1, 2, 3)

    def dates_at(self, geometries: gpd.GeoSeries) -> pd.Series:
        """Flight date under each geometry: its sheet's mid-window, else ``date``."""
        dates = pd.Series(pd.NaT, index=geometries.index, dtype="datetime64[ns]")
        if self.sheets is not None and len(self.sheets) and len(geometries):
            sheets = self.sheets.to_crs(geometries.crs)
            points = gpd.GeoDataFrame(geometry=geometries.representative_point(),
                                      crs=geometries.crs)
            hit = gpd.sjoin(points, sheets[["date_start", "date_end", "geometry"]],
                            how="left", predicate="within")
            hit = hit[~hit.index.duplicated()]
            mid = hit.date_start + (hit.date_end - hit.date_start) / 2
            dates.loc[mid.index] = mid.fillna(hit.date_start).fillna(hit.date_end).values
        if self.date is not None:
            dates = dates.fillna(pd.Timestamp(self.date))
        return dates

    def resolution(self, crs) -> float:
        """Native pixel size of the first raster, in the units of ``crs``."""
        with rasterio.Env(**_VSICURL_ENV), _open(self.paths[0]) as src:
            if src.crs is None or rasterio.crs.CRS.from_user_input(crs) == src.crs:
                return float(abs(src.res[0]))
            _, width, _ = calculate_default_transform(src.crs, crs, src.width, src.height,
                                                      *src.bounds)
            left, _, right, _ = transform_bounds(src.crs, crs, *src.bounds)
            return float((right - left) / width)


# ============================================================
# SOURCES
# ============================================================

def local_orthophoto(source: Union[str, Sequence[str]], *, date=None,
                     name: Optional[str] = None,
                     bands: Tuple[int, ...] = (1, 2, 3)) -> Orthophoto:
    """
    An orthophoto from files already at hand: one raster, a list of them, or
    a directory of them (GeoTIFF, VRT, JPEG 2000, ECW when GDAL has it ...).
    A ``http(s)://`` path is read with ``/vsicurl/``.

    Example:
        >>> ortho = local_orthophoto("/data/pnoa/2023/", date="2023-07-15")
    """
    if isinstance(source, str) and os.path.isdir(source):
        paths = sorted(os.path.join(source, f) for f in os.listdir(source)
                       if f.lower().endswith(RASTER_EXTENSIONS))
    elif isinstance(source, str):
        paths = [source]
    else:
        paths = list(source)
    paths = [_vsicurl(p) for p in paths]
    if not paths:
        raise FileNotFoundError(f"No orthophoto raster in {source}")
    if name is None:
        name = os.path.basename(os.path.normpath(str(source if isinstance(source, str)
                                                     else paths[0])))
    return Orthophoto(paths=paths, date=pd.Timestamp(date) if date is not None else None,
                      name=name, bands=tuple(bands))


def icgc_orthophoto_years() -> List[int]:
    """Years of the 25 cm ICGC RGB orthophoto on the datacloud."""
    try:
        listing = _session().get(ICGC_ORTHO_DIRECTORY, timeout=60).text
        years = sorted({int(y) for y in re.findall(r"ortofoto-rgb-25cm-catalunya-(\d{4})\.gpkg",
                                                   listing)})
        if years:
            return years
    except Exception as exc:                                    # pragma: no cover - offline
        warning(f"Could not list the ICGC orthophotos ({exc})")
    return list(range(2009, pd.Timestamp.today().year))


def icgc_orthophoto_sheets(cache_dir: str, *, refresh: bool = False) -> gpd.GeoDataFrame:
    """
    Flight window of each sheet of the current ICGC orthophoto edition:
    ``sheet``, ``date_start``, ``date_end`` and the sheet polygon (EPSG:25831).
    Downloaded once into ``cache_dir/orthophoto_metadata``.
    """
    cache = os.path.join(cache_dir, "orthophoto_metadata")
    os.makedirs(cache, exist_ok=True)
    archive = os.path.join(cache, os.path.basename(ICGC_ORTHO_METADATA_URL))
    if refresh or not os.path.exists(archive) or os.path.getsize(archive) == 0:
        info(f"Downloading the ICGC orthophoto metadata from {ICGC_ORTHO_METADATA_URL}")
        with _session().get(ICGC_ORTHO_METADATA_URL, stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(archive + ".part", "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        os.replace(archive + ".part", archive)
    with zipfile.ZipFile(archive) as zf:
        shp = sorted(n for n in zf.namelist() if n.lower().endswith(".shp"))
        if not shp:
            raise FileNotFoundError(f"No shapefile inside {archive}")
        zf.extractall(cache)
    sheets = gpd.read_file(os.path.join(cache, shp[-1]))
    return gpd.GeoDataFrame({
        "sheet": sheets["CODI_UNIT"],
        "date_start": pd.to_datetime(sheets["DATA_INICI"], errors="coerce"),
        "date_end": pd.to_datetime(sheets["DATA_FI"], errors="coerce"),
    }, geometry=sheets.geometry, crs=sheets.crs)


def nearest_icgc_orthophoto_year(date, years: Optional[Iterable[int]] = None) -> int:
    """The orthophoto year closest to ``date`` (the LiDAR flight); the later on a tie."""
    years = sorted(years if years is not None else icgc_orthophoto_years())
    when = pd.Timestamp(date)
    target = when.year + (when.dayofyear - 0.5) / 365.25
    return min(years, key=lambda y: (abs(y + 0.5 - target), -y))


def icgc_orthophoto(footprints, out_dir: str, *, year: Union[int, str] = "current",
                    tile_size_m: float = 1000.0, buffer_m: float = 25.0,
                    workers: int = 8, progress: bool = False) -> Orthophoto:
    """
    The ICGC 25 cm RGB orthophoto over ``footprints``, as local tiles.

    Every ``tile_size_m`` tile of a fixed grid that comes within
    ``buffer_m`` of a footprint is read from the territory-wide GeoPackage of
    ``year`` and written to ``out_dir/ortofoto-rgb-25cm-{year}/`` (JPEG,
    quality 90, with the coverage as an internal mask), unless it is there
    already: the cache is shared by every municipality.

    Args:
        footprints: GeoSeries / GeoDataFrame (any CRS) of what will be read,
            normally the roof planes.
        out_dir: Cache directory.
        year: An edition (2009 onwards) or "current". The current edition
            carries the flight window of each of its sheets; an older one is
            dated 1 July of its year.
        tile_size_m: Tile side, in metres of EPSG:25831.
        buffer_m: Margin around the footprints (co-registration search and
            crop padding).
        workers: Parallel downloads.

    Returns:
        ``Orthophoto`` over the tiles.
    """
    footprints = gpd.GeoSeries(getattr(footprints, "geometry", footprints))
    if footprints.crs is None:
        raise ValueError("footprints need a CRS")
    footprints = footprints.to_crs(25831)

    sheets = icgc_orthophoto_sheets(out_dir)
    current_year = int(sheets.date_start.dropna().dt.year.mode().iloc[0])
    if year == "current":
        year = current_year
    year = int(year)
    if year == current_year:
        url, date = ICGC_ORTHO_CURRENT_URL, None
    else:
        url, date, sheets = ICGC_ORTHO_URL.format(year=year), pd.Timestamp(year, 7, 1), None

    name = f"ortofoto-rgb-25cm-{year}"
    tile_dir = os.path.join(out_dir, name)
    os.makedirs(tile_dir, exist_ok=True)
    tiles = _covering_tiles(footprints, tile_size_m, buffer_m)
    paths = [os.path.join(tile_dir, f"{name}_{int(x0)}_{int(y0)}.tif") for x0, y0 in tiles]
    missing = [(t, p) for t, p in zip(tiles, paths) if not os.path.exists(p)]
    if missing:
        info(f"Reading {len(missing)} of {len(tiles)} orthophoto tiles of {year} "
             f"from {url}")
        remote = _vsicurl(url)

        def fetch(item):
            (x0, y0), path = item
            _download_tile(remote, (x0, y0, x0 + tile_size_m, y0 + tile_size_m), path)

        with ThreadPoolExecutor(max(1, workers)) as pool:
            done = pool.map(fetch, missing)
            if progress:
                done = tqdm(done, total=len(missing), desc="Orthophoto tiles")
            list(done)
    return Orthophoto(paths=paths, date=date, sheets=sheets, name=name)


def _covering_tiles(footprints: gpd.GeoSeries, tile_size_m, buffer_m):
    tiles = set()
    for minx, miny, maxx, maxy in footprints.bounds.itertuples(index=False):
        if not np.isfinite(minx):
            continue
        for ix in range(int(np.floor((minx - buffer_m) / tile_size_m)),
                        int(np.floor((maxx + buffer_m) / tile_size_m)) + 1):
            for iy in range(int(np.floor((miny - buffer_m) / tile_size_m)),
                            int(np.floor((maxy + buffer_m) / tile_size_m)) + 1):
                tiles.add((ix * tile_size_m, iy * tile_size_m))
    return sorted(tiles)


def _download_tile(remote, bounds, path):
    with rasterio.Env(**_VSICURL_ENV), rasterio.open(remote) as src:
        window = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        data = src.read(window=window, boundless=True, fill_value=0)
        mask = src.read_masks(1, window=window, boundless=True)
        profile = dict(driver="GTiff", width=data.shape[2], height=data.shape[1], count=3,
                       dtype="uint8", crs=src.crs, transform=src.window_transform(window),
                       tiled=True, blockxsize=256, blockysize=256, compress="JPEG",
                       photometric="YCBCR", jpeg_quality=90)
    # Written beside and renamed: another process (or municipality) may be
    # reading the cache, and an interrupted write must not look finished.
    part = path + ".part.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(part, "w", **profile) as dst:
        dst.write(data[:3])
        dst.write_mask(mask)
    os.replace(part, path)


# ============================================================
# READING
# ============================================================

def read_orthophoto(ortho: Orthophoto, bounds, *, crs, resolution: Optional[float] = None,
                    shift: Tuple[float, float] = (0.0, 0.0),
                    resampling: Resampling = Resampling.bilinear):
    """
    The orthophoto on a north-up grid of ``crs``.

    ``bounds`` (minx, miny, maxx, maxy in ``crs``) are widened to whole
    pixels of ``resolution`` (default: native) on a grid anchored at the
    origin. ``shift`` (dx, dy) is where the imagery shows things relative to
    the LiDAR - the displacement ``coregister_orthophoto`` measures - and is
    undone: the crop is read ``shift`` away and returned at ``bounds``.

    Returns:
        (image, transform): float32 (bands, rows, cols), NaN where there is
        no imagery; the affine transform of the grid.
    """
    if resolution is None:
        resolution = ortho.resolution(crs)
    minx, miny, maxx, maxy = bounds
    minx, miny = np.floor(minx / resolution) * resolution, np.floor(miny / resolution) * resolution
    maxx, maxy = np.ceil(maxx / resolution) * resolution, np.ceil(maxy / resolution) * resolution
    width = max(1, int(round((maxx - minx) / resolution)))
    height = max(1, int(round((maxy - miny) / resolution)))
    transform = from_origin(minx, maxy, resolution, resolution)
    read_transform = from_origin(minx + shift[0], maxy + shift[1], resolution, resolution)
    read_bounds = (minx + shift[0], miny + shift[1], maxx + shift[0], maxy + shift[1])

    image = np.full((len(ortho.bands), height, width), np.nan, dtype=np.float32)
    with rasterio.Env(**_VSICURL_ENV):
        for path in ortho.paths:
            _read_into(image, path, ortho.bands, crs, read_bounds, read_transform, resampling)
    return image, transform


def _read_into(image, path, bands, crs, read_bounds, read_transform, resampling):
    # Fills the NaN pixels of image that this raster covers.
    src = _cached_open(path)
    src_bounds = (transform_bounds(crs, src.crs, *read_bounds, densify_pts=5)
                  if src.crs != rasterio.crs.CRS.from_user_input(crs) else read_bounds)
    left, bottom, right, top = src.bounds
    if (src_bounds[0] >= right or src_bounds[2] <= left or
            src_bounds[1] >= top or src_bounds[3] <= bottom):
        return
    # A margin of a few source pixels for the interpolation kernel.
    pad = 3 * abs(src.res[0])
    window = from_bounds(src_bounds[0] - pad, src_bounds[1] - pad, src_bounds[2] + pad,
                         src_bounds[3] + pad, src.transform)
    window = window.round_offsets().round_lengths().intersection(
        Window(0, 0, src.width, src.height))
    data = src.read(list(bands), window=window, masked=True)
    if data.mask.all():
        return
    data = data.astype(np.float32).filled(np.nan)
    piece = np.full_like(image, np.nan)
    for b in range(len(bands)):
        reproject(data[b], piece[b], src_transform=src.window_transform(window),
                  src_crs=src.crs, src_nodata=np.nan, dst_transform=read_transform,
                  dst_crs=crs, dst_nodata=np.nan, resampling=resampling)
    fill = np.isnan(image) & ~np.isnan(piece)
    image[fill] = piece[fill]


_HANDLES = {}


def _cached_open(path):
    # One handle per process and path: GDAL handles do not survive fork, and
    # over HTTP every open costs a round trip. A local file rewritten since
    # (another run, a re-downloaded tile) is opened again.
    stamp = os.path.getmtime(path) if os.path.exists(path) else None
    key = (os.getpid(), path)
    cached = _HANDLES.get(key)
    if cached is None or cached[0] != stamp:
        if cached is not None:
            cached[1].close()
        _HANDLES[key] = (stamp, _open(path))
    return _HANDLES[key][1]


def _open(path):
    # Callers hold rasterio.Env(**_VSICURL_ENV): the HTTP options are read
    # on every request, not only when the file is opened.
    return rasterio.open(path)


def _vsicurl(path: str) -> str:
    return "/vsicurl/" + path if path.startswith(("http://", "https://")) else path
