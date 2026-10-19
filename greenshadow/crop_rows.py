# -*- coding: utf-8 -*-
"""
greenshadow.crop_rows

Plantation rows inside rural parcels, read off the canopy height model.

The canopy is whatever the vegetated DSM has that the bare one does not::

    chm = dsm - dsm_no_vegetation

In a woody parcel - olive, vineyard, fruit, nuts - that surface is a comb, and
the comb's direction and pitch are the plantation geometry. They are recovered
by sweeping a direction, projecting the canopy pixels onto its normal, and
scoring how periodic the resulting profile is; the sharpest direction wins and
its dominant period is the row spacing.

Two details decide whether this works at all:

- The profile is a **fraction**, canopy pixels over all parcel pixels at the
  same offset. A raw count carries the parcel's own chord-length envelope,
  whose shape turns with the sweep and quietly favours the parcel's long axis.
  Dividing cancels the envelope; detrending only blurs it.
- The spacing is proposed by the profile's spectrum, band-limited by what the
  crop could plausibly be, but the direction is chosen by **how much of the
  profile's variance folding at that period explains**. Scoring on the power in
  the fundamental instead rewards a *sinusoidal* profile over a sharp one, and
  a well-kept orchard is sharp: its rows throw most of their power into
  harmonics, putting a local minimum of that score exactly on the true
  direction. Folding is indifferent to the waveform and peaks where the rows
  really are. It is indifferent enough to fold a comb at twice its period,
  though, so a direction is only eligible once some power actually sits where
  it claims a period - otherwise a sweep *along* the rows wins on the trees'
  own spacing down the row, whose fundamental falls below the search band and
  leaves only a sub-harmonic inside it.

A parcel is not one plantation, though. Blocks get replanted, a terrace or an
access corridor cuts the field, and the two halves are then worked in different
directions. So the fitting is per **row block**: one connected string of rows
with its own azimuth and its own pitch. Blocks are peeled off one at a time -
sweep, take the strings the winning lattice explains, remove their pixels,
sweep what is left - which works because a mixed parcel has no compromise
direction to fall into: folding at an angle between two plantations explains
nothing, so the sweep's maximum always sits on one of the real ones.

What counts as a block is decided in the lattice's own coordinates, not on the
ground: each canopy pixel is filed under ``(nearest row, cell along the row)``
and the connected cells are labelled. A corridor is then a run of empty cells
right across the lattice, which is what separates two plantations; a few dead
trees is a short gap inside one row, which is bridged, so the row stays one
line.

The lattice says where the rows are; each row's own trees say what it does
along its length. Rows planted across a slope wander, and neighbours need not
wander together - they converge round a terrace, or fan across a field that is
not a rectangle - so a row is tracked on its own pixels, as the median
cross-row position of the trees about each of a handful of nodes. Straightness
is then imposed rather than assumed: the nodes are at least ten metres apart,
which makes a row a chain of long straight runs and leaves nowhere for the
half-metre sampling grid to write a staircase.

Where a row ends is not where its last tree is. A plantation is planted up to
its headland and then loses its outermost trees first - to the track, the
machinery, the shade of whatever is over the boundary - so every lattice row is
laid from one side of the parcel to the other, set in a couple of metres from
the boundary, and only broken by a hole: bare ground that is long and lined up
across several neighbouring rows, which is a track or a cleared corner rather
than a few dead trees. The rows are laid by following the canopy node by node,
not by snapping to the lattice, so a spacing that drifts over a long field or
rows that bend round the terrain stay one row each. Blocks pointing the same
way are followed as one planting; blocks pointing different ways share the
parcel out between them, so no row is drawn across another planting.

A different direction has to be earned, though. Trees standing apart are in
rows along both axes of their grid, and which one a sweep prefers can flip from
one corner of an orchard to the next. So a block pointing another way is first
folded along the larger planting's direction, and if its trees are in rows that
way too, it follows that direction. Only canopy that is flat along the other
direction keeps its own: a hedgerow, or a planting that really crosses the
other. Finally no two rows are left closer than two metres. Where crops of
different spacing share a parcel, a row can end up squeezed between another
planting's rows, and the one standing on fewer trees is deleted.

The sweep runs only on the woody uses in ``SIGPAC_SPACING_PRIORS``. Herbaceous
crops - cereals, vegetables, tubers - have no canopy for the CHM to see, and
anything else with canopy in it is scrub or margin rather than a plantation;
either way a periodicity found there would describe the scatter, not a planting
geometry. Those parcels still get a row direction, taken from the parcel
outline's principal axis: the way the field is already worked, and so the way
machinery has to keep moving between whatever is built on it.
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window
import shapely
from shapely import STRtree
from shapely.affinity import translate
from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union
from tqdm import tqdm

try:
    from .dimensionate import _read_window, _pixel_centres, _polygon_pixel_mask
    from .buildings_infer import _dominant_orientation
    from .rural_parcels import sigpac_parcel_id, filter_agricultural_parcels
except ImportError:  # running from a checkout without the package installed
    from greenshadow.dimensionate import (
        _read_window,
        _pixel_centres,
        _polygon_pixel_mask,
    )
    from greenshadow.buildings_infer import _dominant_orientation
    from greenshadow.rural_parcels import (
        sigpac_parcel_id,
        filter_agricultural_parcels,
    )


# ============================================================
# LAND-USE KNOWLEDGE
# ============================================================

#: LiDAR classes carrying vegetation returns, as used throughout the library.
VEGETATION_CLASSES = (3, 4, 5)

#: Uses whose canopy a 0.5 m CHM can resolve, and the row spacing to expect.
#: Narrowing the search band is what stops a harmonic winning: an olive grove
#: at 7 m has a strong 3.5 m second harmonic, and only the prior rules it out.
SIGPAC_SPACING_PRIORS = {
    "VI": (1.8, 3.5),    # vineyard
    "OV": (5.0, 12.0),   # olive
    "FY": (3.0, 6.5),    # fruit trees
    "FS": (5.5, 8.5),    # nuts
    "FL": (5.0, 12.0),   # nuts and olive
    "FF": (3.0, 8.0),    # fruit and nuts
    "CI": (3.0, 6.5),    # citrus
    "CF": (3.0, 6.5),
    "CS": (3.0, 8.0),
    "CV": (1.8, 6.5),
    "VF": (1.8, 6.5),
    "VO": (1.8, 12.0),
    "OF": (3.0, 12.0),
    "OC": (3.0, 12.0),
    "OP": (1.8, 12.0),   # other permanent crops
}

#: Closest spacing a superintensive planting is put at. Olive and almond are
#: also planted as continuous hedges three to five metres apart, which the
#: traditional bands above refuse outright. A parcel of those uses is swept a
#: second time down to this, and a spacing below its traditional band is taken
#: only where the canopy really is a hedge (``_is_hedgerow``): the harmonic of
#: a grid of separate trees is exactly what the band was there to rule out.
HEDGEROW_MIN_SPACING_M = 3.0

#: Share of the below-band canopy that has to be hedge for the
#: superintensive reading to replace the traditional one.
#: Not all of it: a corner of a few hundred pixels reads however it likes.
_HEDGE_CANOPY_SHARE = 0.75

#: A block also counts as a hedge when it folds along its rows to less than
#: this share of what it folds across them. A small parcel's few short rows
#: make the fold along them mostly row ends, which no fixed bar can tell from
#: trees; but a hedge still folds three to five times better across its rows
#: than along them, and a misread grid of separate trees does not.
_HEDGE_ALONG_RATIO = 0.4

WOODY_USES = tuple(SIGPAC_SPACING_PRIORS)


# ============================================================
# RASTER SAMPLING
# ============================================================

def _sample_raster_at(src, grid_x, grid_y, band=1):
    """
    Nearest-pixel sample of one band at arbitrary world coordinates.

    Returns ``(values, valid)`` on the shape of ``grid_x``. Going through
    coordinates rather than a shared window means the companion rasters need
    only cover the same ground as the reference, not sit on the identical grid,
    so a DEM at a different resolution still lines up.
    """
    rows, cols = rowcol(src.transform, grid_x.ravel(), grid_y.ravel())

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    values = np.full(rows.size, np.nan, dtype=np.float32)
    valid = np.zeros(rows.size, dtype=bool)

    inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

    if inside.any():
        row_lo, row_hi = rows[inside].min(), rows[inside].max() + 1
        col_lo, col_hi = cols[inside].min(), cols[inside].max() + 1

        block = src.read(
            band,
            window=Window(col_lo, row_lo, col_hi - col_lo, row_hi - row_lo),
        ).astype(np.float32)

        sampled = block[rows[inside] - row_lo, cols[inside] - col_lo]

        good = np.isfinite(sampled)
        if src.nodata is not None:
            good &= ~np.isclose(sampled, src.nodata)

        values[inside] = sampled
        valid[inside] = good

    return values.reshape(grid_x.shape), valid.reshape(grid_x.shape)


# ============================================================
# PARCEL OUTLINE
# ============================================================

def _working_axis_deg(geometry):
    """
    Compass bearing of the parcel's principal axis, folded to [0, 90).

    ``_dominant_orientation`` quadruples the edge angles before averaging, so
    that the four sides of a rectangle land on the same phase. That makes the
    result inherently **modulo 90 degrees**: it names a pair of perpendicular
    directions, ``a`` and ``a + 90``, and cannot say which of the two the field
    is worked along. For machinery that is not a defect - both runs are equally
    drivable - but a consumer has to try both, and an 80 degree disagreement
    with a detected row azimuth is no disagreement at all.
    """
    if geometry is None or geometry.is_empty:
        return np.nan

    polygon = max(geometry.geoms, key=lambda part: part.area) if (
        geometry.geom_type == "MultiPolygon"
    ) else geometry

    coords = np.asarray(polygon.exterior.coords)

    angle = _dominant_orientation(coords)

    if angle is None:
        return np.nan

    return float((90.0 - np.degrees(angle)) % 90.0)


# ============================================================
# DIRECTIONAL SWEEP
# ============================================================

def _directional_profile(s_canopy, s_all, bin_size):
    """
    Canopy fraction as a function of offset along one direction.

    Returns ``(profile, support, origin)``, or None when too little of the
    parcel carries enough pixels to measure. Bins whose support is thin - the
    parcel's corners, where the chord is short - are filled with the mean of
    the rest rather than a noisy ratio, so they neither invent a peak nor punch
    a hole in the spectrum, and ``support`` marks them so the fold score can
    ignore them entirely.

    The profile is also trimmed to the run of bins the canopy actually reaches.
    Ground past the last tree is not a wide alley, it is somewhere else, and
    left in it is a step - the largest feature in the profile, explained by no
    period at all. It inflates the variance the fold score is a share of, and
    the sweep then prefers a flat profile in some other direction, where a
    little noise folds into a large share of almost nothing. Interior gaps stay
    where they are: a corridor through a plantation is structure, not an edge.
    """
    lo = float(s_all.min())
    hi = float(s_all.max())

    n_bins = int(np.ceil((hi - lo) / bin_size))

    if n_bins < 12:
        return None

    edges = lo + np.arange(n_bins + 1) * bin_size

    count_all, _ = np.histogram(s_all, bins=edges)
    count_canopy, _ = np.histogram(s_canopy, bins=edges)

    planted = count_canopy > 0

    if not planted.any():
        return None

    # A quarter of a typical planted bin, so that a stray pixel beyond the edge
    # of the plantation does not hold the trim open.
    reached = np.flatnonzero(
        count_canopy >= max(1.0, 0.25 * float(np.median(count_canopy[planted])))
    )

    if reached.size == 0:
        return None

    first, last = int(reached[0]), int(reached[-1]) + 1

    if last - first < 12:
        return None

    count_all = count_all[first:last]
    count_canopy = count_canopy[first:last]

    lo += first * bin_size
    n_bins = last - first

    occupied = count_all > 0

    if not occupied.any():
        return None

    support = count_all >= max(3.0, 0.25 * np.median(count_all[occupied]))

    if support.sum() < 12:
        return None

    profile = np.zeros(n_bins, dtype=np.float64)
    profile[support] = count_canopy[support] / count_all[support]
    profile[~support] = profile[support].mean()

    return profile, support, lo


def _fold_variance_ratio(profile, support, period_bins):
    """
    Share of the profile's variance explained by folding it at one period.

    A one-way analysis of variance with the fold phase as the grouping factor:
    1 when every bin at the same phase agrees, 0 when the period explains
    nothing. Unlike the power in a single Fourier peak it does not care what
    shape the repeating unit has, so a sharp comb of trees scores at least as
    well as a smooth one - which is what makes it safe to choose a direction
    with.
    """
    values = profile[support]

    if values.size < 8 or period_bins <= 1:
        return 0.0

    total = values.var()

    if total <= 0:
        return 0.0

    n_phase = max(4, int(round(period_bins)))

    phase = np.mod(np.flatnonzero(support), period_bins)
    index = np.clip((phase / period_bins * n_phase).astype(int), 0, n_phase - 1)

    counts = np.bincount(index, minlength=n_phase)
    sums = np.bincount(index, weights=values, minlength=n_phase)

    occupied = counts > 0
    means = sums[occupied] / counts[occupied]

    between = np.sum(counts[occupied] * (means - values.mean()) ** 2) / values.size

    return float(np.clip(between / total, 0.0, 1.0))


def _spectral_score(profile, bin_size, spacing_range, pad=4, min_cycles=4.0):
    """
    Periodicity of a profile: how much of its power sits in one narrow peak.

    Returns ``(purity, spacing_m, phase, n_fft, k_peak)`` or None. The power is
    summed over the whole main lobe of the peak, whose width is set by the
    window and the padding at ``2 * n_fft / n`` bins either side. Taking a
    fixed few bins instead makes the score scallop by 15% with nothing but the
    sub-bin position of the peak, which on a 1 degree sweep is far larger than
    the angular signal being measured and lands the answer a couple of degrees
    off the true row direction.
    """
    n = profile.size

    windowed = (profile - profile.mean()) * np.hanning(n)

    if not np.any(windowed):
        return None

    n_fft = int(2 ** np.ceil(np.log2(max(n * pad, 16))))

    spectrum = np.fft.rfft(windowed, n_fft)
    power = np.abs(spectrum) ** 2
    frequency = np.fft.rfftfreq(n_fft, d=bin_size)

    total = power[1:].sum()

    if total <= 0:
        return None

    # A spacing as long as the parcel is not a plantation: one or two periods
    # fit, folding at them explains everything by construction, and the score
    # saturates at 1 on a parcel with no rows at all. Requiring the period to
    # repeat min_cycles times is what keeps that degenerate answer out.
    frequency_floor = max(1.0 / spacing_range[1], min_cycles / (n * bin_size))

    band = (frequency >= frequency_floor) & (frequency <= 1.0 / spacing_range[0])
    band[0] = False

    if not band.any():
        return None

    k = int(np.argmax(np.where(band, power, -1.0)))

    half_lobe = max(2, int(round(2.0 * n_fft / n)))
    lo = max(1, k - half_lobe)
    hi = min(power.size, k + half_lobe + 1)
    purity = float(power[lo:hi].sum() / total)

    # Parabolic interpolation on the log spectrum, so the reported spacing is
    # not quantised to the padded bin width. The shift is clamped to half a
    # bin, which is all it can legitimately be: where the argmax sits on the
    # edge of the search band the three samples need not form a peak at all,
    # and the unclamped vertex then flies far outside the band and reports a
    # spacing the search was explicitly told to exclude.
    peak = float(k)
    if 0 < k < power.size - 1:
        left, centre, right = (np.log(power[k - 1] + 1e-300),
                               np.log(power[k] + 1e-300),
                               np.log(power[k + 1] + 1e-300))
        denominator = left - 2.0 * centre + right
        if denominator < 0:
            peak = k + float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    frequency_peak = peak / (n_fft * bin_size)

    if frequency_peak <= 0:
        return None

    return purity, float(1.0 / frequency_peak), float(np.angle(spectrum[k])), n_fft, k


def _sweep_azimuth(
    x_canopy, y_canopy, x_all, y_all, *,
    bin_size,
    spacing_range,
    step_deg,
    refine_step_deg,
    min_purity=0.0,
    theta_centre=None,
    theta_span=None,
):
    """
    Direction whose canopy profile is most periodic.

    ``theta`` is the angle of the projection axis - the row *normal* - measured
    from East counterclockwise, so the rows themselves run along
    ``theta + 90``. Returns a dict of the winning direction's measurements, or
    None when no direction produced a usable profile.

    ``theta_centre`` and ``theta_span`` narrow the search to one window, which
    is how a single block re-fits the direction it was found on: the answer is
    already known to a degree or two, and searching the full half-circle again
    would only let a small block wander onto some other structure.

    ``min_purity`` is the one thing the fold score cannot decide for itself.
    Folding is indifferent to waveform, which is what makes it the right judge
    of direction - but it is equally happy to fold a comb at twice its period,
    and a sweep *along* the rows meets exactly that: the trees are spaced along
    the row too, usually tighter than the rows are apart, so their comb falls
    below the search band and only its sub-harmonic is in it. That direction
    folds beautifully and has no spectral peak at all where it claims one.
    Requiring some power to actually sit at the folded period rules it out
    without letting purity choose the direction.
    """
    def evaluate(theta_deg):
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        built = _directional_profile(
            x_canopy * cos_t + y_canopy * sin_t,
            x_all * cos_t + y_all * sin_t,
            bin_size,
        )

        if built is None:
            return None

        profile, support, origin = built

        scored = _spectral_score(profile, bin_size, spacing_range)

        if scored is None:
            return None

        purity, spacing, phase, n_fft, k = scored

        if purity < min_purity:
            return None

        return {
            "theta_deg": float(theta_deg),
            # The spectrum proposes the period; the fold decides the direction.
            "periodicity": _fold_variance_ratio(profile, support, spacing / bin_size),
            "purity": purity,
            "spacing_m": spacing,
            "phase": phase,
            "n_fft": n_fft,
            "k_peak": k,
            "origin": origin,
            "bin_size": bin_size,
            "profile": profile,
        }

    if theta_centre is None:
        grid = np.arange(0.0, 180.0, step_deg)
    else:
        grid = np.arange(
            theta_centre - theta_span, theta_centre + theta_span + 1e-9, step_deg
        )

    best = None

    for theta_deg in grid:
        candidate = evaluate(theta_deg % 180.0)
        if candidate is not None and (
            best is None or candidate["periodicity"] > best["periodicity"]
        ):
            best = candidate

    if best is None:
        return None

    coarse = best["theta_deg"]

    for theta_deg in np.arange(coarse - step_deg, coarse + step_deg + 1e-9, refine_step_deg):
        candidate = evaluate(theta_deg % 180.0)
        if candidate is not None and candidate["periodicity"] > best["periodicity"]:
            best = candidate

    return best


# ============================================================
# ROW GEOMETRY
# ============================================================

def _lattice_strength(s, periods):
    """
    How strongly a set of offsets repeats at each trial period.

    The magnitude of the circular mean of their phase: 1 when every pixel sits
    at the same point of the cycle, 0 when they are spread over it.
    """
    angle = 2.0 * np.pi * s[None, :] / np.asarray(periods, float)[:, None]

    return np.abs(np.exp(1j * angle).mean(axis=1))


def _refine_lattice(s, spacing, *, span=0.12, coarse=160, fine=40,
                    max_points=20000):
    """
    Spacing and phase to far better than the spectrum can give them.

    The sweep reads the period off a padded FFT of a binned profile, which is
    accurate to a fraction of a bin - and a fraction of a bin is not nearly
    enough. A spacing 1% short is invisible on any single row, but it slips a
    whole row over a hundred metres of parcel, so the lattice starts on the
    trees at one end and sits in the alleys at the other. Half a plantation
    comes back misplaced from an error too small to see.

    So the period is refined directly on the pixels, with no binning at all:
    the circular mean of their phase at a trial period, scanned around the
    spectrum's estimate. Its magnitude peaks where the rows really repeat and
    its argument is the phase, which places the rows in one step. The scan is
    deliberately narrow - a wider one would simply find the half-period, which
    fits a comb just as well.

    Returns ``(spacing, phase)``.
    """
    if s.size == 0 or not np.isfinite(spacing) or spacing <= 0:
        return spacing, 0.0

    if s.size > max_points:
        s = s[::int(np.ceil(s.size / max_points))]

    centred = s - s.mean()

    periods = np.linspace(spacing * (1 - span), spacing * (1 + span), coarse)
    best = periods[int(np.argmax(_lattice_strength(centred, periods)))]

    step = periods[1] - periods[0]
    periods = np.linspace(best - step, best + step, fine)
    best = float(periods[int(np.argmax(_lattice_strength(centred, periods)))])

    vector = np.exp(2j * np.pi * centred / best).mean()

    # Back to the original origin: the phase was measured about the mean.
    phase = float(np.angle(vector) + 2.0 * np.pi * s.mean() / best)

    return best, phase


def _lattice_offsets(s_min, s_max, spacing, phase):
    """
    Row centres of the refined lattice, across the span the block covers.

    The rows sit where the canopy's phase is stationary, ``s = spacing *
    (k + phase / 2.pi)``, one period apart by construction.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        return np.array([])

    base = spacing * phase / (2.0 * np.pi)

    first = int(np.floor((s_min - base) / spacing))
    last = int(np.ceil((s_max - base) / spacing))

    return base + np.arange(first, last + 1) * spacing


def _half_modulation_width(counts, bin_width):
    """
    Full width at half **modulation** of one folded cross-row histogram.

    Halfway between the curve's trough and its crest, not half its maximum.
    Measuring from zero would be right only for an orchard whose alleys are
    bare; a mature one where the crowns have closed over the alley keeps the
    whole curve above half its maximum and would report the entire spacing as
    canopy. Working from the trough keeps the row and the alley distinct.

    Returns NaN for an empty histogram. A perfectly flat one - no modulation
    at all - returns the full span, which the caller reads as a failure to
    separate row from alley rather than as a very wide row.
    """
    if counts.size == 0 or counts.max() <= 0:
        return np.nan

    half = (counts.max() + counts.min()) * 0.5

    return float((counts >= half).sum()) * bin_width


def _folded_width(s_canopy, offsets, spacing, bin_size):
    """
    Canopy width across the row, from the phase-folded profile.

    Every canopy pixel in the parcel is folded into a single period, so the
    width is measured on one well-populated curve instead of row by row, and
    the alley is what is left of the spacing. Reported, never enforced: it is
    what a planner wants to know about a block, not a shape every row of it has
    to have.
    """
    if offsets.size == 0 or spacing <= 0:
        return np.nan, np.nan

    phase = np.mod(s_canopy - offsets[0] + spacing * 0.5, spacing)

    n_bins = max(8, int(np.ceil(spacing / bin_size)))
    counts, edges = np.histogram(phase, bins=n_bins, range=(0.0, spacing))

    width = _half_modulation_width(counts, spacing / n_bins)

    if not np.isfinite(width):
        return np.nan, np.nan

    centre = float(edges[np.argmax(counts)] + 0.5 * spacing / n_bins)

    return width, centre


# ============================================================
# ROW BLOCKS
# ============================================================
#
# A parcel is not always one plantation. Blocks get replanted, a terrace or an
# access corridor cuts the field in two, and the halves are then worked in
# different directions. Fitting one azimuth to the whole parcel splits the
# difference and lands the rows on nothing, which is exactly what a single
# global sweep does on a parcel like that.
#
# So the parcel is fitted as a set of **row blocks**: one connected string of
# parallel, evenly spaced rows each, with its own azimuth, its own spacing and
# its own pitch. The blocks are found by peeling - sweep, take the
# strings the winning lattice explains, remove their pixels, sweep what is
# left - because a mixed parcel has no compromise direction to be trapped by:
# folding at an angle between two plantations explains nothing, so the sweep's
# maximum always sits on one of the real directions.


def _near_mask(x, y, x_reference, y_reference, cell, min_share=0.2):
    """
    Which of ``(x, y)`` lie within about one ``cell`` of the reference points.

    The profile's denominator has to cover the same ground as its numerator.
    Once part of the canopy has been peeled off, the rest of the parcel is no
    longer the right denominator: the bins over ground the residual does not
    reach would read as empty alley and drown the periodicity that is left.

    A coarse occupancy grid grown by one cell, rather than a distance
    transform - the bound only has to be about right, and the cell is a row
    spacing wide, so a bare alley still counts as covered ground.

    A cell has to be properly occupied, not merely touched: ``min_share`` of
    what a typical occupied cell holds. Peeling always leaves a thin scatter
    behind - the edge of a block, a few trees the lattice never claimed - and a
    grid that marked a cell on one stray point would hand the next pass the
    whole parcel back as its denominator, which is the same failure the
    restriction exists to prevent.
    """
    if x_reference.size == 0 or cell <= 0:
        return np.zeros(x.size, dtype=bool)

    lo_x = float(x_reference.min()) - cell
    lo_y = float(y_reference.min()) - cell

    n_x = int((float(x_reference.max()) - lo_x) / cell) + 2
    n_y = int((float(y_reference.max()) - lo_y) / cell) + 2

    counts = np.zeros((n_y, n_x), dtype=float)
    np.add.at(
        counts,
        (
            ((y_reference - lo_y) / cell).astype(int),
            ((x_reference - lo_x) / cell).astype(int),
        ),
        1.0,
    )

    filled = counts[counts > 0]
    occupied = counts >= max(1.0, min_share * float(np.median(filled)))

    if not occupied.any():
        occupied = counts > 0

    padded = np.zeros((n_y + 2, n_x + 2), dtype=bool)
    padded[1:-1, 1:-1] = occupied

    grown = np.zeros_like(occupied)
    for down in range(3):
        for right in range(3):
            grown |= padded[down:down + n_y, right:right + n_x]

    column = np.floor((x - lo_x) / cell).astype(int)
    row = np.floor((y - lo_y) / cell).astype(int)

    within = (column >= 0) & (column < n_x) & (row >= 0) & (row < n_y)

    near = np.zeros(x.size, dtype=bool)
    near[within] = grown[row[within], column[within]]

    return near


def _label_grid(occupied):
    """
    Eight-connected components of a small boolean grid, and how many.

    Written out rather than pulled from scipy: the grid here is rows by
    along-row cells, a few tens on a side, and the flood fill costs nothing at
    that size.
    """
    labels = np.zeros(occupied.shape, dtype=int)
    n_rows, n_cols = occupied.shape
    current = 0

    for seed_row, seed_col in zip(*np.nonzero(occupied)):
        if labels[seed_row, seed_col]:
            continue

        current += 1
        labels[seed_row, seed_col] = current
        stack = [(int(seed_row), int(seed_col))]

        while stack:
            row, col = stack.pop()
            for down in (-1, 0, 1):
                for right in (-1, 0, 1):
                    neighbour_row, neighbour_col = row + down, col + right
                    if not (0 <= neighbour_row < n_rows and 0 <= neighbour_col < n_cols):
                        continue
                    if occupied[neighbour_row, neighbour_col] and not labels[
                        neighbour_row, neighbour_col
                    ]:
                        labels[neighbour_row, neighbour_col] = current
                        stack.append((neighbour_row, neighbour_col))

    return labels, current


def _lattice_strings(s, t, offsets, spacing, *, cell_m, bridge_cells,
                     min_cell_points, min_coherence, min_rows):
    """
    Connected strings of rows, found in the lattice's own coordinates.

    Each canopy pixel is put in the cell ``(nearest row, along-row cell)``, and
    labelling the connected cells is what a "string of rows" means:
    neighbouring rows that overlap along their length belong together, and a
    corridor - a run of empty cells right across the lattice - separates two
    strings the way it separates two plantations on the ground.

    Doing it here rather than on the raster is what makes gaps tractable. A
    missing stretch inside a row is bridged when it is shorter than
    ``bridge_cells``, so a row stays one continuous line through a few dead
    trees; anything longer is a real break and the string ends there.

    What makes a cell planted is not how much canopy it holds but **where in
    the period that canopy sits**: the mean of ``cos`` of the pixels' lattice
    phase, which is high only when they pile up on the row and near zero when
    they are spread across the band. Counting alone is what lets a second
    plantation at another angle be swallowed whole - its trees are dense, so
    every cell of the wrong lattice looks planted, and the two blocks come back
    as one compromise. Its trees are not *in phase*, and that is the difference
    this reads.

    Returns ``(member, footprint)`` pairs of boolean masks over ``s``, largest
    string first. ``member`` is the string's own canopy, which is what its
    geometry is measured from; ``footprint`` adds the halo of one cell around
    it, which is the ground the string occupies and which the caller takes out
    of play so the next pass is not handed this plantation's fringe to sweep.
    """
    n_rows = offsets.size

    if n_rows == 0 or s.size == 0:
        return []

    index = np.rint((s - offsets[0]) / spacing).astype(int)
    valid = (index >= 0) & (index < n_rows)

    if not valid.any():
        return []

    t_lo = float(t[valid].min())
    cell = np.floor((t - t_lo) / cell_m).astype(int)

    valid &= cell >= 0

    if not valid.any():
        return []

    n_cells = int(cell[valid].max()) + 1

    counts = np.zeros((n_rows, n_cells), dtype=float)
    np.add.at(counts, (index[valid], cell[valid]), 1.0)

    phase = 2.0 * np.pi * (s - offsets[0]) / spacing

    coherence = np.zeros((n_rows, n_cells), dtype=float)
    np.add.at(coherence, (index[valid], cell[valid]), np.cos(phase[valid]))

    populated = counts >= min_cell_points

    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = np.where(populated, coherence / np.maximum(counts, 1.0), 0.0)

    # The bar is set by the lattice's own plantation rather than in the
    # abstract: half of what a well-planted cell on this lattice achieves. An
    # absolute threshold has to be low enough for a mature grove whose crowns
    # have closed - whose cells are genuinely less peaked - and that same
    # threshold then lets a second plantation at another angle in, because
    # "not very coherent" and "not coherent at all" are a factor of two apart,
    # not an order of magnitude.
    if populated.any():
        strong = float(np.percentile(coherence[populated], 75))
    else:
        strong = 0.0

    # Scattered pixels give a coherence of order 1/sqrt(n) by luck alone, so a
    # thinly populated cell has to clear a proportionately higher bar. Only
    # barely, though: pixels inside one crown are not independent draws, so
    # counting them as such sets a bar that rises with the raster's resolution
    # and quietly shortens every block on a coarser grid.
    floor = np.maximum(
        max(min_coherence, 0.6 * strong),
        1.0 / np.sqrt(np.maximum(counts, 1.0)),
    )

    occupied = populated & (coherence >= floor)

    if bridge_cells > 0:
        for row in range(n_rows):
            planted = np.flatnonzero(occupied[row])
            if planted.size < 2:
                continue
            for start, stop in zip(planted[:-1], planted[1:]):
                if stop - start - 1 <= bridge_cells:
                    occupied[row, start + 1:stop] = True

    labels, n_labels = _label_grid(occupied)

    row_index = np.clip(index, 0, n_rows - 1)
    cell_index = np.clip(cell, 0, n_cells - 1)

    strings = []

    for label in range(1, n_labels + 1):
        in_label = labels == label

        if int(in_label.any(axis=1).sum()) < min_rows:
            continue

        member = valid & in_label[row_index, cell_index]

        if not member.any():
            continue

        padded = np.zeros((n_rows + 2, n_cells + 2), dtype=bool)
        padded[1:-1, 1:-1] = in_label

        halo = np.zeros_like(in_label)
        for down in range(3):
            for right in range(3):
                halo |= padded[down:down + n_rows, right:right + n_cells]

        strings.append((member, valid & halo[row_index, cell_index]))

    strings.sort(key=lambda pair: int(pair[0].sum()), reverse=True)

    return strings


def _row_polylines(s, t, offsets, spacing, *, segment_m, min_points):
    """
    One polyline per lattice row, following its own trees.

    The lattice says where the rows are; the trees say what each one does along
    its length. A row planted across a slope wanders, and neighbouring rows in
    the same block need not wander together - they converge round a terrace,
    or fan out across a field that is not a rectangle - so each is tracked on
    its own pixels.

    Straightness is imposed here rather than assumed: the nodes are at least
    ``segment_m`` apart, so a row is a chain of long straight runs and there is
    nowhere for the half-metre sampling grid to write a staircase. Each node is
    the median cross-row position of the trees around it, which a few stray
    pixels cannot move, and it may sit at most a third of a spacing off the
    lattice - further than that is not a bending row, it is the next one.

    Returns ``(offset, node_t, node_s, n_points)`` per row.
    """
    if offsets.size == 0 or s.size == 0:
        return []

    index = np.rint((s - offsets[0]) / spacing).astype(int)
    valid = (index >= 0) & (index < offsets.size)

    limit = spacing / 3.0
    rows = []

    for position in range(offsets.size):
        selected = valid & (index == position)
        n_points = int(selected.sum())

        if n_points < min_points:
            continue

        t_row, s_row = t[selected], s[selected]
        t_lo, t_hi = float(t_row.min()), float(t_row.max())

        if t_hi - t_lo < segment_m:
            continue

        n_nodes = max(2, int((t_hi - t_lo) // segment_m) + 1)
        node_t = np.linspace(t_lo, t_hi, n_nodes)

        node_s = np.full(n_nodes, np.nan)
        for node in range(n_nodes):
            near = np.abs(t_row - node_t[node]) <= segment_m * 0.75
            if int(near.sum()) >= 3:
                node_s[node] = np.median(s_row[near])

        known = np.isfinite(node_s)

        if int(known.sum()) < 2:
            continue

        node_s = np.interp(node_t, node_t[known], node_s[known])
        node_s = offsets[position] + np.clip(
            node_s - offsets[position], -limit, limit)

        rows.append((float(offsets[position]), node_t, node_s, n_points))

    return rows


def _row_line(node_t, node_s, theta_deg):
    """
    A row's nodes, back in the parcel's own axes.
    """
    theta = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    return LineString(np.column_stack([
        node_s * cos_t - node_t * sin_t,
        node_s * sin_t + node_t * cos_t,
    ]))


def _line_pieces(geometry):
    """
    The LineStrings inside whatever an intersection handed back.

    Clipping a row to its parcel gives one segment on a convex parcel and
    several on a parcel with a notch cut out of it, and an empty geometry where
    the row missed entirely.
    """
    if geometry is None or geometry.is_empty:
        return []

    if geometry.geom_type == "LineString":
        return [geometry]

    if geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        return [
            part for part in geometry.geoms
            if part.geom_type == "LineString" and not part.is_empty
        ]

    return []


def _clip_row(line, bounds, relaxed):
    """
    Cut a row to the parcel, without letting the setback line chop it up.

    The ends are cut at ``bounds``, the parcel set in by the full margin. But
    a row running along that line - the outermost row, planted right at the
    setback - wanders a few centimetres either side of it, and clipping it
    there leaves a row of dashes. In between its ends a row only has to stay
    within ``relaxed``, set in by half the margin.
    """
    pieces = []

    for piece in _line_pieces(line.intersection(relaxed)):
        inner = _line_pieces(piece.intersection(bounds))
        if not inner:
            continue
        start = min(piece.project(Point(part.coords[0])) for part in inner)
        stop = max(piece.project(Point(part.coords[-1])) for part in inner)
        if stop > start:
            pieces.append(substring(piece, start, stop))

    return pieces


def _regularise(line, minimum):
    """
    Drop the vertices that leave a run shorter than ``minimum``.

    The nodes a row is built on are already that far apart, but cutting the row
    to its parcel lands wherever the boundary happens to be and can leave a
    stub of a metre or two hanging off the end. Absorbing it into the run
    before it keeps the promise the nodes made - every straight piece of a row
    is a long one - and moves the line by less than the stub was long.
    """
    coords = np.asarray(line.coords, dtype=float)

    if coords.shape[0] < 3:
        return line

    keep = [coords[0]]

    for point in coords[1:-1]:
        if float(np.hypot(*(point - keep[-1]))) >= minimum:
            keep.append(point)

    while len(keep) > 1 and float(np.hypot(*(coords[-1] - keep[-1]))) < minimum:
        keep.pop()

    keep.append(coords[-1])

    return LineString(keep)


def _measure_block(members, parent, x_canopy, y_canopy, x_all, y_all, *,
                   bin_size, spacing_range, config):
    """
    Everything one string of rows is worth, measured on its own pixels.

    The string was found on the parent sweep's lattice, which was fitted to
    every block still in play. Here it gets its own: a narrow re-sweep around
    the parent direction, then the period and phase refined on the pixels
    themselves, so a block a few degrees or a few percent off the parent's
    compromise ends up on its own rows rather than near them.
    """
    x_block, y_block = x_canopy[members], y_canopy[members]

    sweep = parent

    near = _near_mask(x_all, y_all, x_block, y_block, parent["spacing_m"])

    if int(near.sum()) >= 24:
        refined = _sweep_azimuth(
            x_block, y_block, x_all[near], y_all[near],
            bin_size=bin_size,
            spacing_range=spacing_range,
            step_deg=config["azimuth_refine_step_deg"],
            refine_step_deg=config["azimuth_refine_step_deg"],
            min_purity=config["min_spectral_purity"],
            theta_centre=parent["theta_deg"],
            theta_span=config["azimuth_step_deg"],
        )
        if refined is not None and abs(
            refined["spacing_m"] - parent["spacing_m"]
        ) <= 0.25 * parent["spacing_m"]:
            sweep = refined

    theta = np.radians(sweep["theta_deg"])

    s = x_block * np.cos(theta) + y_block * np.sin(theta)
    t = -x_block * np.sin(theta) + y_block * np.cos(theta)

    spacing, phase = _refine_lattice(s, sweep["spacing_m"])

    block = {
        "accepted": False,
        "members": members,
        "theta_deg": sweep["theta_deg"],
        "azimuth_deg": float((-sweep["theta_deg"]) % 180.0),
        "spacing_m": float(spacing),
        "periodicity": float(sweep["periodicity"]),
        "purity": float(sweep["purity"]),
        "width_m": np.nan,
        "alley_m": np.nan,
        "captured": 0.0,
        "n_points": int(members.sum()),
        "n_rows": 0,
        "rows": [],
        "lattice_base": np.nan,
    }

    offsets = _lattice_offsets(float(s.min()), float(s.max()), spacing, phase)

    if offsets.size == 0:
        return block

    block["lattice_base"] = float(offsets[0])

    # Reported, not enforced: how wide the planted strip is and how much alley
    # is left between two of them is what a planner wants to know, but holding
    # every row of a block to one width only ever threw away rows that a weed,
    # a dead stretch or a wider crown had made the wrong shape.
    width, _ = _folded_width(s, offsets, spacing, bin_size)

    block["width_m"] = width
    if np.isfinite(width):
        block["alley_m"] = float(max(spacing - width, 0.0))
        distance = np.abs(
            np.mod(s - offsets[0] + spacing * 0.5, spacing) - spacing * 0.5)
        block["captured"] = float(np.mean(distance <= width * 0.5))

    rows = _row_polylines(
        s, t, offsets, spacing,
        segment_m=config["row_segment_m"],
        min_points=config["min_points_per_row"],
    )

    block["rows"] = rows
    block["accepted"] = (
        len(rows) >= config["min_rows"]
        and sweep["periodicity"] >= config["min_row_periodicity"]
        and block["captured"] >= config["min_captured_fraction"]
    )

    return block


def _fit_row_blocks(x_canopy, y_canopy, x_all, y_all, *,
                    bin_size, spacing_range, pixel_area, config):
    """
    Peel the parcel's canopy into row blocks, strongest plantation first.

    Each pass sweeps whatever canopy is still unexplained, splits the winning
    lattice into connected strings of rows, and takes those pixels out of play
    - accepted or not, so a string that failed its own checks cannot be found
    again on the next pass and spin the loop.

    Returns ``(accepted, rejected)``, both largest first. The rejected strings
    are kept because they are the diagnosis: a parcel that came back without
    rows can still say how periodic the best thing in it was.
    """
    remaining = np.ones(x_canopy.size, dtype=bool)
    minimum_points = config["min_rows"] * config["min_points_per_row"]

    blocks = []
    rejected = []

    for _ in range(config["max_row_blocks"]):
        if int(remaining.sum()) < minimum_points:
            break

        x_left, y_left = x_canopy[remaining], y_canopy[remaining]

        # Numerator and denominator have to describe the same ground. The
        # footprint is built from the residual canopy and then applied to both,
        # so the scatter peeling left outside it is not counted as canopy over
        # ground the profile has no support on - a fraction with nothing under
        # it, which is how a residual sweep ends up on a period that is not
        # there.
        footprint = _near_mask(x_left, y_left, x_left, y_left, spacing_range[1])
        near = _near_mask(x_all, y_all, x_left, y_left, spacing_range[1])

        if int(near.sum()) < 24 or int(footprint.sum()) < minimum_points:
            break

        x_left, y_left = x_left[footprint], y_left[footprint]

        sweep = _sweep_azimuth(
            x_left, y_left, x_all[near], y_all[near],
            bin_size=bin_size,
            spacing_range=spacing_range,
            step_deg=config["azimuth_step_deg"],
            refine_step_deg=config["azimuth_refine_step_deg"],
            min_purity=config["min_spectral_purity"],
        )

        if sweep is None:
            break

        theta = np.radians(sweep["theta_deg"])
        s = x_left * np.cos(theta) + y_left * np.sin(theta)
        t = -x_left * np.sin(theta) + y_left * np.cos(theta)

        spacing, phase = _refine_lattice(s, sweep["spacing_m"])
        offsets = _lattice_offsets(float(s.min()), float(s.max()), spacing, phase)

        if offsets.size == 0:
            break

        width, _ = _folded_width(s, offsets, spacing, bin_size)
        reference = width if (np.isfinite(width) and width > 0) else 0.4 * spacing

        cell_m = max(spacing, 2.0)

        # A cell is planted once it holds this fraction of the canopy a fully
        # planted cell would. Tying it to the pixel footprint keeps the same
        # threshold meaningful at any resolution, and after the parcel has been
        # subsampled.
        min_cell_points = max(8, int(round(
            config["min_cell_cover_frac"] * reference * cell_m / pixel_area
        )))

        strings = _lattice_strings(
            s, t, offsets, spacing,
            cell_m=cell_m,
            bridge_cells=max(1, int(round(config["row_gap_bridge_m"] / cell_m))),
            min_cell_points=min_cell_points,
            min_coherence=config["min_cell_coherence"],
            min_rows=config["min_rows"],
        )

        if not strings:
            break

        index = np.flatnonzero(remaining)[footprint]
        consumed = np.zeros(x_canopy.size, dtype=bool)

        for string, footprint in strings:
            members = np.zeros(x_canopy.size, dtype=bool)
            members[index[string]] = True

            consumed[index[footprint]] = True

            block = _measure_block(
                members, sweep, x_canopy, y_canopy, x_all, y_all,
                bin_size=bin_size,
                spacing_range=spacing_range,
                config=config,
            )

            (blocks if block["accepted"] else rejected).append(block)

        if not consumed.any():
            break

        remaining &= ~consumed

    blocks.sort(key=lambda block: len(block["rows"]), reverse=True)
    rejected.sort(key=lambda block: block["n_points"], reverse=True)

    return blocks, rejected


# ============================================================
# ROW LAYOUT
# ============================================================
#
# The blocks say where the plantation's lattice is; they do not say where its
# rows end. A row is laid from one side of the parcel to the other, set in by
# the margin, and only broken where the ground says it has to be: a bare
# stretch that is long **and** repeated on several neighbouring rows - a
# track, a cleared corner, a pond. A few dead trees, a gap in a single row, a
# headland stripped of its outermost trees are all still the same row, and
# reporting them as breaks turns a solid block into a fringe of stubs.

#: Along-row resolution of the hole test. Fine enough that a cut lands within
#: a couple of metres of where the bare ground starts; coarse enough that the
#: gap between two trees down a row is a handful of cells, not a hole.
_LAYOUT_CELL_M = 2.0

#: Share of a well-planted cell's canopy a cell must hold to count as planted.
_PLANTED_CELL_SHARE = 0.15

#: Share of its length a row must carry canopy on to count as planted. A block
#: is laid from its first planted row to its last, every row in between
#: included. A row of olives seven metres apart along the row covers about
#: half its cells; a bare strip at the side of the field covers none.
_EDGE_ROW_MIN_SHARE = 0.25


def _follow_direction(block, leader, x_canopy, y_canopy, x_all, y_all, *,
                      bin_size, spacing_range, config):
    """
    The leader's direction, swept on this block's own canopy - or None when
    the block's trees are not in rows that way.

    Folding a block's canopy along another direction explains nothing when
    its rows are continuous - a hedgerow is flat along its own length - or
    planted at an angle that is no direction of its lattice. Trees standing
    apart are different: they are in rows along every short vector of their
    grid, so a block found square to its neighbour is often the same orchard,
    read the other way. The two cases sit far apart: across this library's
    survey ground a trellised hedgerow folds to under 0.15 along its own
    length, separate olive or nut trees to 0.2 and more. A block is turned
    once the leader's way explains ``hedgerow_max_periodicity`` of its
    profile - once it is not flat that way.
    """
    members = block["members"]
    x_block, y_block = x_canopy[members], y_canopy[members]

    near = _near_mask(x_all, y_all, x_block, y_block, block["spacing_m"])

    if int(near.sum()) < 24:
        return None

    sweep = _sweep_azimuth(
        x_block, y_block, x_all[near], y_all[near],
        bin_size=bin_size,
        spacing_range=spacing_range,
        step_deg=config["azimuth_refine_step_deg"],
        refine_step_deg=config["azimuth_refine_step_deg"],
        min_purity=config["min_spectral_purity"],
        theta_centre=leader["theta_deg"],
        theta_span=config["azimuth_step_deg"],
    )

    if sweep is None:
        return None

    if sweep["periodicity"] < config["hedgerow_max_periodicity"]:
        return None

    return sweep


def _is_hedgerow(block, x_canopy, y_canopy, x_all, y_all, *, bin_size, config):
    """
    Whether a block's canopy is flat along its own rows.

    A hedge is one strip of canopy from end to end, so folding it along the
    row explains little (``hedgerow_max_periodicity``), and far less than
    folding it across (``_HEDGE_ALONG_RATIO``); a grid of separate trees folds
    along the row about as well as across it. That difference is what tells a
    superintensive planting at four metres from the second harmonic of a
    traditional one at eight, which no spacing band can. The search along the
    row starts at ``HEDGEROW_MIN_SPACING_M``: the plants inside a hedge stand a
    metre or two apart, and that is a hedge's grain, not a row of trees.
    """
    members = block["members"]
    x_block, y_block = x_canopy[members], y_canopy[members]

    near = _near_mask(x_all, y_all, x_block, y_block, block["spacing_m"])

    if int(near.sum()) < 24:
        return False

    along = _sweep_azimuth(
        x_block, y_block, x_all[near], y_all[near],
        bin_size=bin_size,
        spacing_range=(
            max(HEDGEROW_MIN_SPACING_M, config["row_spacing_range_m"][0]),
            config["row_spacing_range_m"][1],
        ),
        step_deg=config["azimuth_refine_step_deg"],
        refine_step_deg=config["azimuth_refine_step_deg"],
        min_purity=config["min_spectral_purity"],
        theta_centre=(block["theta_deg"] + 90.0) % 180.0,
        theta_span=config["azimuth_step_deg"],
    )

    return along is None or along["periodicity"] < max(
        config["hedgerow_max_periodicity"],
        _HEDGE_ALONG_RATIO * block["periodicity"],
    )


def _merge_blocks(blocks, x_canopy, y_canopy, x_all, y_all, *,
                  bin_size, spacing_range, config):
    """
    One block per planting direction.

    Peeling splits a plantation wherever one straight lattice stopped fitting
    it - a run of dead trees across several rows, a spacing that drifts by a
    few centimetres a row over a long field, rows bending round the terrain -
    and the pieces come back as separate blocks pointing the same way, often
    half a spacing out of step with each other. Laid out separately, each
    would either run its rows across the others or carve the field into
    islands.

    So blocks within ``block_azimuth_tolerance_deg`` of each other are re-fitted
    as one: the lattice here only has to be close, because the rows are laid
    by following the canopy (``_track_rows``), not by snapping to it.

    A block pointing another way is first asked whether it has to
    (``_follow_direction``). Trees standing apart on a grid are in rows both
    ways, and which way a sweep happens to prefer flips from one corner of the
    orchard to the next; reporting each corner its own way paints one planting
    as a patchwork of crossed rows. Such a block follows the larger planting's
    direction - joining it outright when its rows that way are as far apart,
    and as its own block on that direction when they are not, a second crop
    planted the same way. Only a block whose canopy is *not* in rows that way
    - a hedgerow, a vineyard, a planting that really does cross the other -
    keeps a direction of its own.
    """
    tolerance = config["block_azimuth_tolerance_deg"]
    groups = []

    for block in sorted(blocks, key=lambda block: block["n_points"], reverse=True):
        for group in groups:
            if _angular_difference_180(
                group[0]["azimuth_deg"], block["azimuth_deg"]
            ) <= tolerance:
                group.append(block)
                break
        else:
            for group in groups:
                sweep = _follow_direction(
                    block, group[0], x_canopy, y_canopy, x_all, y_all,
                    bin_size=bin_size,
                    spacing_range=spacing_range,
                    config=config,
                )

                if sweep is None:
                    continue

                if abs(sweep["spacing_m"] - group[0]["spacing_m"]) <= (
                    0.25 * group[0]["spacing_m"]
                ):
                    group.append(block)
                    break

                turned = _measure_block(
                    block["members"], sweep, x_canopy, y_canopy, x_all, y_all,
                    bin_size=bin_size,
                    spacing_range=spacing_range,
                    config=config,
                )
                if np.isfinite(turned["lattice_base"]):
                    groups.append([turned])
                    break
            else:
                groups.append([block])

    merged = []

    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue

        members = np.zeros(x_canopy.size, dtype=bool)
        for block in group:
            members |= block["members"]

        joint = _measure_block(
            members, group[0], x_canopy, y_canopy, x_all, y_all,
            bin_size=bin_size,
            spacing_range=spacing_range,
            config=config,
        )

        # The joint fit is the better summary of the planting, but only if it
        # found a lattice at all; the largest block's is the fallback.
        if not np.isfinite(joint["lattice_base"]):
            joint = dict(group[0], members=members, n_points=int(members.sum()))

        merged.append(joint)

    merged.sort(key=lambda block: block["n_points"], reverse=True)

    return merged


def _territories(blocks, x_canopy, y_canopy, x_all, y_all, *, cell, gap):
    """
    Which block each part of the parcel belongs to, on a coarse grid.

    Every cell goes to the block whose own canopy is nearest - grown outwards
    from each block's trees one ring at a time, a discrete Voronoi diagram -
    so a row runs up to the plantation planted the other way and no further.
    A strip ``gap`` cells wide either side of each border belongs to nobody:
    two plantations at an angle meet at a turning headland, not at a seam.

    Returns ``(lo_x, lo_y, labels)``; ``labels`` holds the block index plus
    one, and zero for the strip between blocks.
    """
    lo_x = float(x_all.min()) - cell
    lo_y = float(y_all.min()) - cell

    n_x = int((float(x_all.max()) - lo_x) / cell) + 2
    n_y = int((float(y_all.max()) - lo_y) / cell) + 2

    counts = np.zeros((len(blocks), n_y, n_x), dtype=float)

    for index, block in enumerate(blocks):
        members = block["members"]
        np.add.at(
            counts[index],
            (
                ((y_canopy[members] - lo_y) / cell).astype(int),
                ((x_canopy[members] - lo_x) / cell).astype(int),
            ),
            1.0,
        )

    labels = np.where(counts.max(axis=0) > 0, counts.argmax(axis=0) + 1, 0)

    shifts = [(down, right) for down in (-1, 0, 1) for right in (-1, 0, 1)
              if down or right]

    def neighbours(grid):
        padded = np.pad(grid, 1)
        for down, right in shifts:
            yield padded[1 + down:1 + down + n_y, 1 + right:1 + right + n_x]

    while not labels.all():
        grown = labels.copy()
        for shifted in neighbours(labels):
            take = (grown == 0) & (shifted > 0)
            grown[take] = shifted[take]
        if np.array_equal(grown, labels):
            break
        labels = grown

    disputed = np.zeros(labels.shape, dtype=bool)

    for label in range(1, len(blocks) + 1):
        other = (labels > 0) & (labels != label)
        for _ in range(gap):
            spread = other.copy()
            for shifted in neighbours(other):
                spread |= shifted
            other = spread
        disputed |= (labels == label) & other

    labels[disputed] = 0

    return lo_x, lo_y, labels


def _owner_at(territory, x, y):
    """
    The territory label under each point, zero off the grid.
    """
    lo_x, lo_y, labels = territory
    n_y, n_x = labels.shape

    column = np.floor((x - lo_x) / _LAYOUT_CELL_M).astype(int)
    row = np.floor((y - lo_y) / _LAYOUT_CELL_M).astype(int)

    within = (column >= 0) & (column < n_x) & (row >= 0) & (row < n_y)

    owner = np.zeros(x.shape, dtype=int)
    owner[within] = labels[row[within], column[within]]

    return owner


def _runs(mask, min_cells):
    """
    Runs of at least ``min_cells`` set cells along each row of ``mask``.

    One list per row, each run ``(first, last)`` in cell indices, inclusive.
    """
    runs = []

    for line in mask:
        padded = np.concatenate([[False], line, [False]])
        edges = np.flatnonzero(np.diff(padded.astype(int)))
        runs.append([
            (int(first), int(stop - 1))
            for first, stop in zip(edges[::2], edges[1::2])
            if stop - first >= min_cells
        ])

    return runs


def _shared_holes(runs, min_rows):
    """
    The long bare runs that line up across at least ``min_rows`` rows.

    Runs on neighbouring rows that overlap along the row are the same hole.
    One long gap in a single row is a stretch of dead trees and is bridged;
    the same gap on row after row is ground nobody planted, and that is what
    is allowed to break a row.
    """
    nodes = [(row, run) for row, row_runs in enumerate(runs) for run in row_runs]

    if not nodes:
        return []

    parent = list(range(len(nodes)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    by_row = {}
    for node, (row, _) in enumerate(nodes):
        by_row.setdefault(row, []).append(node)

    for node, (row, (first, last)) in enumerate(nodes):
        for other in by_row.get(row + 1, []):
            other_first, other_last = nodes[other][1]
            if first <= other_last and other_first <= last:
                parent[find(node)] = find(other)

    components = {}
    for node in range(len(nodes)):
        components.setdefault(find(node), []).append(node)

    holes = []
    for members in components.values():
        if len({nodes[node][0] for node in members}) >= min_rows:
            holes.extend(nodes[node] for node in members)

    return holes


def _advance_rows(previous, slab, *, spacing, min_count):
    """
    Move every row on to the next node, following the canopy there.

    Each row looks for canopy within a third of a spacing of where it was,
    and moves to the median of what it finds. A row that finds too little -
    dead trees, a gap, ground past the end of the planting - moves with its
    neighbours instead, so a row never stops just because its own trees did.
    Two rows that would end up closer than half a spacing are both treated as
    having found nothing: one of them has strayed onto the other's trees.

    Returns ``(positions, measured)``.
    """
    half = spacing / 3.0

    lo = np.searchsorted(slab, previous - half)
    hi = np.searchsorted(slab, previous + half)
    measured = (hi - lo) >= min_count

    shift = np.zeros(previous.size)
    for row in np.flatnonzero(measured):
        shift[row] = float(np.median(slab[lo[row]:hi[row]])) - previous[row]

    def borrow(known):
        moved = shift.copy()
        if known.any():
            moved[~known] = np.interp(
                np.flatnonzero(~known), np.flatnonzero(known), shift[known])
        else:
            moved[:] = 0.0
        return moved

    shift = borrow(measured)
    positions = previous + shift

    crowded = np.diff(positions) < 0.5 * spacing
    if crowded.any():
        clash = np.zeros(previous.size, dtype=bool)
        clash[:-1] |= crowded
        clash[1:] |= crowded
        measured &= ~clash
        shift = borrow(measured)
        positions = previous + shift

    return positions, measured


def _track_rows(s, t, nodes, *, spacing, base, s_lo, s_hi, reach):
    """
    Every row of a block, followed across the parcel node by node.

    A rigid lattice is right about a plantation locally and wrong about it
    over a long field: the spacing drifts by a few centimetres a row, rows
    bend round the terrain, a replanted strip sits half a spacing out of step
    with the rest. Snapping to the lattice then loses rows, or splits one
    planting into blocks that tile the field in islands. Following the rows
    does not.

    The rows are seeded where the canopy is densest, by walking across the
    plantation from its best-planted row, so the spacing is free to change
    from one row to the next. Each row is then carried along to both ends of
    the parcel by ``_advance_rows``, at most a third of a spacing sideways
    per node, so a row can bend but never jumps onto its neighbour.

    ``s_lo`` and ``s_hi`` are the cross-row extent the rows must cover at the
    seed. Returns ``positions`` of shape ``(rows, nodes)`` - each row's
    cross-row coordinate at each node - or None when there is no canopy.
    """
    slabs = [np.sort(s[np.abs(t - node) <= reach]) for node in nodes]
    sizes = np.array([slab.size for slab in slabs])

    if not sizes.any():
        return None

    seed = int(np.argmax(sizes))
    slab = slabs[seed]
    half = spacing / 3.0

    lattice = base + spacing * np.arange(
        np.floor((s_lo - base) / spacing), np.ceil((s_hi - base) / spacing) + 1)

    counts = (np.searchsorted(slab, lattice + half)
              - np.searchsorted(slab, lattice - half))

    if not (counts > 0).any():
        return None

    # A row is following its own trees when its window holds a fifth of what
    # a well-planted row holds there. Relative, so it means the same at any
    # resolution and after the parcel has been subsampled.
    min_count = max(3.0, 0.2 * float(np.percentile(counts[counts > 0], 75)))

    def refine(centre):
        lo, hi = np.searchsorted(slab, [centre - half, centre + half])
        if hi - lo >= min_count:
            return float(np.median(slab[lo:hi])), True
        return centre, False

    origin = float(lattice[np.argmax(counts)])
    for _ in range(3):
        origin, found = refine(origin)

    seeded = {0: origin}

    for direction in (1, -1):
        current, step, index = origin, spacing, 0
        while s_lo - spacing <= current + direction * step <= s_hi + spacing:
            index += direction
            position, found = refine(current + direction * step)
            if found:
                gap = float(np.clip(abs(position - current),
                                    0.8 * spacing, 1.25 * spacing))
                position = current + direction * gap
                step = 0.5 * step + 0.5 * gap
            seeded[index] = position
            current = position

    order = sorted(seeded)
    positions = np.empty((len(order), nodes.size))
    positions[:, seed] = [seeded[index] for index in order]

    for sequence in (range(seed + 1, nodes.size), range(seed - 1, -1, -1)):
        previous = positions[:, seed]
        for node in sequence:
            positions[:, node], _ = _advance_rows(
                previous, slabs[node], spacing=spacing, min_count=min_count)
            previous = positions[:, node]

    return positions


def _layout_rows(block, label, x_canopy, y_canopy, bounds, territory, config):
    """
    The block's rows, laid across the whole of its ground.

    The rows are followed across the parcel on the canopy (``_track_rows``)
    and every one of them between the block's outermost planted rows gets a
    line from one side of ``bounds`` to the other - including a row dead
    along its whole length, which follows its neighbours. Rows beyond those
    are laid only while they still carry canopy: a bare strip at the side of
    the field is not a row.

    A row is then cut in only two places: where the ground belongs to another
    block (``territory``, when the parcel is planted in more than one
    direction), and where it crosses a hole - a bare stretch at least
    ``row_hole_min_length_m`` long, shared by at least ``row_hole_min_rows``
    neighbouring rows. The ends are left long; the caller clips them to the
    parcel.

    Returns ``LineString`` pieces in the parcel's local axes.
    """
    theta = np.radians(block["theta_deg"])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    spacing = block["spacing_m"]
    base = block["lattice_base"]

    if not (np.isfinite(base) and np.isfinite(spacing) and spacing > 0):
        return []

    polygons = getattr(bounds, "geoms", [bounds])
    corners = np.vstack([
        np.asarray(ring.coords)
        for polygon in polygons
        for ring in [polygon.exterior, *polygon.interiors]
    ])

    s_bounds = corners[:, 0] * cos_t + corners[:, 1] * sin_t
    t_bounds = -corners[:, 0] * sin_t + corners[:, 1] * cos_t

    t_lo, t_hi = float(t_bounds.min()), float(t_bounds.max())
    segment = config["row_segment_m"]
    nodes = np.linspace(t_lo, t_hi, max(2, int(np.ceil((t_hi - t_lo) / segment)) + 1))

    # Any canopy on the block's own ground counts, not only the pixels that
    # won it its lattice: a row runs through a patch of weeds or a stretch of
    # replanted saplings, and the edge rows the lattice strings did not claim
    # are still rows.
    if territory is not None:
        own = _owner_at(territory, x_canopy, y_canopy) == label + 1
    else:
        own = np.ones(x_canopy.size, dtype=bool)

    s_canopy = x_canopy[own] * cos_t + y_canopy[own] * sin_t
    t_canopy = -x_canopy[own] * sin_t + y_canopy[own] * cos_t

    # Rows drift as they go, so the seed has to cover more than the parcel's
    # own width there or the far end of a slanting planting runs out of rows.
    drift = spacing + 0.1 * (t_hi - t_lo)

    positions = _track_rows(
        s_canopy, t_canopy, nodes,
        spacing=spacing,
        base=base,
        s_lo=float(s_bounds.min()) - drift,
        s_hi=float(s_bounds.max()) + drift,
        reach=0.75 * segment,
    )

    if positions is None:
        return []

    cell = _LAYOUT_CELL_M
    n_cells = max(1, int(np.ceil((t_hi - t_lo) / cell)))
    t_cells = t_lo + (np.arange(n_cells) + 0.5) * cell

    s_cells = np.vstack([np.interp(t_cells, nodes, row) for row in positions])

    x_cells = s_cells * cos_t - t_cells[None, :] * sin_t
    y_cells = s_cells * sin_t + t_cells[None, :] * cos_t

    inside = _polygon_pixel_mask(bounds, x_cells, y_cells)

    if territory is not None:
        mine = _owner_at(territory, x_cells, y_cells) == label + 1
    else:
        mine = np.ones(inside.shape, dtype=bool)

    # Which row each canopy pixel is on, column by column: the rows are not a
    # lattice any more, so the nearest one is looked up rather than computed.
    cover = np.zeros(inside.shape, dtype=float)

    along = np.floor((t_canopy - t_lo) / cell).astype(int)
    on_grid = (along >= 0) & (along < n_cells)
    order = np.argsort(along[on_grid], kind="stable")
    columns = along[on_grid][order]
    s_sorted = s_canopy[on_grid][order]
    starts = np.searchsorted(columns, np.arange(n_cells + 1))

    for column in range(n_cells):
        s_column = s_sorted[starts[column]:starts[column + 1]]
        if s_column.size == 0:
            continue
        centres = s_cells[:, column]
        row = np.clip(
            np.searchsorted(0.5 * (centres[:-1] + centres[1:]), s_column),
            0, centres.size - 1)
        close = np.abs(s_column - centres[row]) <= 0.5 * spacing
        np.add.at(cover[:, column], row[close], 1.0)

    # A cell is planted when it holds a real share of what a planted cell
    # holds here, not merely a pixel: the odd weed above the canopy threshold
    # is scattered over bare ground too, and counting it would bridge every
    # hole in the field.
    held = cover[cover > 0]
    planted = cover >= max(
        2.0,
        _PLANTED_CELL_SHARE * float(np.percentile(held, 75)) if held.size else 0.0,
    )

    ground = (inside & mine).sum(axis=1)
    share = np.where(
        ground > 0,
        (planted & inside & mine).sum(axis=1) / np.maximum(ground, 1),
        0.0,
    )

    # The block spans from its first planted row to its last, and every row
    # in between is laid whatever is on it.
    planted_rows = np.flatnonzero(share >= _EDGE_ROW_MIN_SHARE)

    if planted_rows.size == 0:
        return []

    lo, hi = int(planted_rows[0]), int(planted_rows[-1])

    runs = _runs(
        inside & mine & ~planted,
        max(1, int(np.ceil(config["row_hole_min_length_m"] / cell))),
    )

    keep = ~inside | mine
    for row, (start, stop) in _shared_holes(runs, config["row_hole_min_rows"]):
        keep[row, start:stop + 1] = False

    keep[:lo] = False
    keep[hi + 1:] = False

    pieces = []

    for row in range(lo, hi + 1):
        for start, stop in _runs(keep[row:row + 1], 1)[0]:
            a = max(t_lo, t_lo + start * cell)
            b = min(t_hi, t_lo + (stop + 1) * cell)
            if b - a <= 0:
                continue
            node_t = np.concatenate([[a], nodes[(nodes > a) & (nodes < b)], [b]])
            node_s = np.interp(node_t, nodes, positions[row])
            pieces.append(_row_line(node_t, node_s, block["theta_deg"]))

    return pieces


def _drop_crowded_rows(laid, x_canopy, y_canopy, config):
    """
    No two of a parcel's rows closer than ``min_row_separation_m``.

    Two rows that close are one row drawn twice. It happens where crops of
    different spacing share a parcel: a row followed off one planting into
    the other ends up squeezed between that planting's own rows, and the
    blocks' borders let a row of each run alongside the other. The bar is
    also never more than three quarters of either row's spacing, so a
    vineyard planted closer than the bar keeps every row.

    Of two crowded rows the one standing on more canopy per metre is the
    real one. The other is deleted when it runs within the bar of a kept row
    for at least ``row_segment_m``; when it only touches one - an end
    reaching the next block's headland, a brief swerve - just that stretch is
    cut out of it.

    ``laid`` holds ``(block_index, block, line)`` in the parcel's local axes,
    the same axes as the canopy; the survivors come back in the same order.
    """
    if len(laid) < 2:
        return laid

    segment = config["row_segment_m"]

    def bar(a, b):
        return min(
            config["min_row_separation_m"],
            0.75 * min(laid[a][1]["spacing_m"], laid[b][1]["spacing_m"]),
        )

    lines = [line for _, _, line in laid]
    left, right = STRtree(lines).query(
        lines, predicate="dwithin", distance=config["min_row_separation_m"])

    crowded = {
        int(k)
        for i, j in zip(left, right) if i < j
        and lines[i].distance(lines[j]) < bar(i, j)
        for k in (i, j)
    }

    if not crowded:
        return laid

    canopy = STRtree(shapely.points(x_canopy, y_canopy))

    def support(k):
        radius = max(0.25 * laid[k][1]["spacing_m"], 0.5)
        hits = canopy.query(lines[k], predicate="dwithin", distance=radius)
        return hits.size / max(lines[k].length, 1e-9)

    kept = {k: [lines[k]] for k in range(len(laid)) if k not in crowded}

    for k in sorted(crowded, key=support, reverse=True):
        zones = [
            line.buffer(bar(k, other))
            for other, pieces in kept.items()
            for line in pieces
            if line.distance(lines[k]) < bar(k, other)
        ]

        if not zones:
            kept[k] = [lines[k]]
            continue

        zone = unary_union(zones)

        if lines[k].intersection(zone).length >= segment:
            continue

        kept[k] = [
            _regularise(piece, segment)
            for piece in _line_pieces(shapely.line_merge(lines[k].difference(zone)))
            if piece.length >= segment
        ]

    return [
        (index, block, piece)
        for k, (index, block, _) in enumerate(laid)
        for piece in kept.get(k, [])
    ]


# ============================================================
# PER-PARCEL ANALYSIS
# ============================================================

def _empty_attributes():
    """
    The measured columns, before anything has been measured.

    One definition for both the parcels that go through ``_analyse_parcel``
    and the ones that never get there - too small, no geometry - so the two
    can never drift apart and leave a column missing from half the output.
    """
    return {
        "row_pattern": "empty",
        "row_confidence": 0.0,
        "row_azimuth_deg": np.nan,
        "row_spacing_m": np.nan,
        "n_rows": 0,
        "n_blocks": 0,
        "row_azimuth_spread_deg": np.nan,
        "row_periodicity": np.nan,
        "spectral_purity": np.nan,
        "captured_fraction": np.nan,
        "canopy_height_p50_m": np.nan,
        "canopy_height_p90_m": np.nan,
        "canopy_width_m": np.nan,
        "alley_width_m": np.nan,
        "canopy_cover_frac": 0.0,
        "chm_max_m": np.nan,
        "n_canopy_px": 0,
        "slope_deg": np.nan,
        "aspect_deg": np.nan,
    }


def _analyse_parcel(
    geometry,
    uso_sigpac,
    *,
    dsm_src,
    nv_src,
    class_src,
    slope_src,
    aspect_src,
    config,
):
    """
    Everything measurable about one parcel's planting geometry.

    Returns ``(attributes, lines)``. ``attributes`` is always populated - a
    parcel with no canopy is a result, not a failure - and ``lines`` is empty
    unless the parcel was classified as planted in rows. Each entry of
    ``lines`` is ``(block_index, block, geometry)``: the rows carry their own
    block's azimuth, spacing and width, which are the parcel's only where the
    parcel is a single block.
    """
    attributes = _empty_attributes()

    block = _read_window(dsm_src, geometry, [1])

    if block is None:
        return attributes, []

    dsm, transform, valid = block
    dsm = dsm[0]

    grid_x, grid_y = _pixel_centres(transform, dsm.shape)

    inside = _polygon_pixel_mask(geometry, grid_x, grid_y) & valid

    if inside.sum() < 24:
        return attributes, []

    bare, bare_valid = _sample_raster_at(nv_src, grid_x, grid_y)
    inside &= bare_valid

    if inside.sum() < 24:
        return attributes, []

    chm = np.clip(dsm - bare, 0.0, None)

    if slope_src is not None:
        slope, slope_valid = _sample_raster_at(slope_src, grid_x, grid_y)
        if (inside & slope_valid).any():
            # The slope/aspect GeoTIFFs are written in radians, unlike the
            # degree-valued columns the roof planes carry.
            attributes["slope_deg"] = float(
                np.degrees(np.mean(slope[inside & slope_valid]))
            )

    if aspect_src is not None:
        aspect, aspect_valid = _sample_raster_at(aspect_src, grid_x, grid_y)
        use = inside & aspect_valid
        if use.any():
            angles = aspect[use]
            attributes["aspect_deg"] = float(
                np.degrees(np.arctan2(
                    np.mean(np.sin(angles)), np.mean(np.cos(angles))
                )) % 360.0
            )

    canopy = inside & (chm > config["min_canopy_height_m"]) & (
        chm < config["max_canopy_height_m"]
    )

    if class_src is not None and config["use_class_filter"]:
        classes, class_valid = _sample_raster_at(class_src, grid_x, grid_y)
        # Sheds, silos and parked machinery all stand proud of the bare DSM.
        # Two of them in one parcel will fabricate a convincing row, so the
        # class filter is what keeps the detector honest on a working farm.
        canopy &= class_valid & np.isin(
            np.rint(classes), np.asarray(VEGETATION_CLASSES, dtype=float)
        )

    attributes["chm_max_m"] = float(np.nanmax(chm[inside])) if inside.any() else np.nan
    attributes["n_canopy_px"] = int(canopy.sum())
    attributes["canopy_cover_frac"] = float(canopy.sum() / inside.sum())

    if canopy.any():
        heights = chm[canopy]
        attributes["canopy_height_p50_m"] = float(np.percentile(heights, 50))
        attributes["canopy_height_p90_m"] = float(np.percentile(heights, 90))

    if attributes["canopy_cover_frac"] < config["min_canopy_cover_frac"]:
        attributes["row_pattern"] = (
            "herbaceous" if attributes["canopy_cover_frac"] > 0.0 else "empty"
        )
        return attributes, []

    # Only a woody crop is planted in rows a canopy model can see. The use
    # code says so where it is right, and the canopy says so where it is not:
    # SIGPAC carries orchards as arable land for years after they are planted,
    # and refusing to look at those parcels loses real plantations.
    #
    # What the canopy has to show is a *crop*. Cover alone would not do it -
    # woodland and tree-lined streets have more of it than any orchard - so
    # the height decides: across this library's own survey ground, plantations
    # sit around two metres and change, while forest, wooded pasture and urban
    # trees sit at seven to ten. A parcel taller than an orchard has trees on
    # it, not rows.
    #
    # Nothing is swept without that evidence. Whatever periodicity a sweep
    # found on scrub would be a property of the scatter, not of a plantation,
    # and the parcel still carries its ``working_axis_deg``, which is what a
    # planner needs on those.
    if config["woody_only"] and uso_sigpac not in SIGPAC_SPACING_PRIORS:
        height = attributes["canopy_height_p50_m"]
        planted = (
            attributes["canopy_cover_frac"] >= config["woody_min_cover_frac"]
            and np.isfinite(height)
            and config["woody_min_height_m"] <= height <= config["woody_max_height_m"]
        )
        if not planted:
            attributes["row_pattern"] = "scattered"
            return attributes, []

    if canopy.sum() < 40:
        attributes["row_pattern"] = "scattered"
        return attributes, []

    resolution = max(abs(transform.a), abs(transform.e))
    bin_size = max(resolution, 0.25)

    spacing_range = config["row_spacing_range_m"]
    hedge_range = None

    if config["use_sigpac_prior"] and uso_sigpac in SIGPAC_SPACING_PRIORS:
        prior = SIGPAC_SPACING_PRIORS[uso_sigpac]
        if prior[0] > HEDGEROW_MIN_SPACING_M:
            hedge_range = (
                max(spacing_range[0], HEDGEROW_MIN_SPACING_M),
                min(spacing_range[1], prior[1]),
            )
        spacing_range = (
            max(spacing_range[0], prior[0]),
            min(spacing_range[1], prior[1]),
        )

    # A period needs several pixels to exist at all. Reporting a spacing below
    # this would be reading structure out of the sampling grid.
    spacing_floor = 4.0 * resolution
    spacing_range = (max(spacing_range[0], spacing_floor), spacing_range[1])
    if hedge_range is not None:
        hedge_range = (max(hedge_range[0], spacing_floor), hedge_range[1])

    if spacing_range[0] >= spacing_range[1]:
        attributes["row_pattern"] = "scattered"
        return attributes, []

    centroid = geometry.centroid
    x_canopy = grid_x[canopy] - centroid.x
    y_canopy = grid_y[canopy] - centroid.y
    x_all = grid_x[inside] - centroid.x
    y_all = grid_y[inside] - centroid.y

    pixel_area = resolution * resolution

    if x_canopy.size > config["max_pixels_per_parcel"]:
        stride = int(np.ceil(x_canopy.size / config["max_pixels_per_parcel"]))
        x_canopy, y_canopy = x_canopy[::stride], y_canopy[::stride]
        # Each surviving pixel now stands for `stride` of them, and the cell
        # occupancy test downstream is a density, so it has to be told.
        pixel_area *= stride

    if x_all.size > config["max_pixels_per_parcel"]:
        stride = int(np.ceil(x_all.size / config["max_pixels_per_parcel"]))
        x_all, y_all = x_all[::stride], y_all[::stride]

    blocks, rejected = _fit_row_blocks(
        x_canopy, y_canopy, x_all, y_all,
        bin_size=bin_size,
        spacing_range=spacing_range,
        pixel_area=pixel_area,
        config=config,
    )

    # Olive and nut are planted both ways: traditional groves of separate
    # trees, which the SIGPAC band is for, and superintensive hedges three to
    # five metres apart, which it refuses. The parcel is swept again down to
    # the hedge spacing, and that reading wins only when it finds rows the
    # band could not hold *and* most of their canopy is a hedge, flat along
    # its length. A grid of separate trees found at half its spacing, or read
    # across its rows down the line of its trees, folds along the row as well
    # as across it, and keeps the traditional reading.
    if hedge_range is not None:
        hedges, hedges_rejected = _fit_row_blocks(
            x_canopy, y_canopy, x_all, y_all,
            bin_size=bin_size,
            spacing_range=hedge_range,
            pixel_area=pixel_area,
            config=config,
        )

        close = [block for block in hedges
                 if block["spacing_m"] < spacing_range[0]]

        credible = sum(
            block["n_points"] for block in close
            if _is_hedgerow(block, x_canopy, y_canopy, x_all, y_all,
                            bin_size=bin_size, config=config)
        )

        if close and credible >= _HEDGE_CANOPY_SHARE * sum(
            block["n_points"] for block in close
        ):
            blocks, rejected = hedges, hedges_rejected
            spacing_range = hedge_range

    if not blocks:
        attributes["row_pattern"] = "scattered"

        # Say why, when there is something to say. The best string that was
        # tried and refused carries its own numbers - a low periodicity means
        # nothing was planted in rows, a low width consistency means something
        # was but not to one width - and a parcel that reports neither is a
        # parcel nobody can check.
        if rejected:
            best = rejected[0]
            attributes["row_periodicity"] = best["periodicity"]
            attributes["spectral_purity"] = best["purity"]
            attributes["canopy_width_m"] = best["width_m"]
            attributes["alley_width_m"] = best["alley_m"]
            attributes["captured_fraction"] = best["captured"]

        return attributes, []

    blocks = _merge_blocks(
        blocks, x_canopy, y_canopy, x_all, y_all,
        bin_size=bin_size,
        spacing_range=spacing_range,
        config=config,
    )

    # Rows run to the parcel's own outline, set in from it: the row nearest
    # the edge is a row, not a fence line, and nothing is planted into the
    # track. Everything here is in the parcel's local axes, the same ones the
    # blocks were fitted in.
    shift = np.array([centroid.x, centroid.y])
    bounds = translate(
        geometry.buffer(-config["parcel_margin_m"]), -centroid.x, -centroid.y)

    if bounds.is_empty:
        bounds = translate(geometry, -centroid.x, -centroid.y)

    relaxed = translate(
        geometry.buffer(-0.5 * config["parcel_margin_m"]), -centroid.x, -centroid.y)

    if relaxed.is_empty:
        relaxed = bounds

    # A parcel planted two ways is shared out between its blocks, so neither
    # one's rows are drawn across the other's. One direction - nearly every
    # parcel - has the whole of it.
    territory = None
    if len(blocks) > 1:
        territory = _territories(
            blocks, x_canopy, y_canopy, x_all, y_all,
            cell=_LAYOUT_CELL_M,
            gap=max(1, int(round(
                0.5 * config["parcel_margin_m"] / _LAYOUT_CELL_M))),
        )

    azimuths = [block["azimuth_deg"] for block in blocks]
    spread = max(
        (_angular_difference_180(a, b) for a in azimuths for b in azimuths),
        default=0.0,
    )

    laid = []

    for index, block in enumerate(blocks):
        block["n_rows"] = 0

        for line in _layout_rows(
            block, index, x_canopy, y_canopy, bounds, territory, config
        ):
            for piece in _clip_row(line, bounds, relaxed):
                if piece.length >= config["row_segment_m"]:
                    laid.append((
                        index, block,
                        _regularise(piece, config["row_segment_m"]),
                    ))

    laid = _drop_crowded_rows(laid, x_canopy, y_canopy, config)

    lines = []

    for index, block, piece in laid:
        block["n_rows"] += 1
        lines.append((
            index, block, LineString(np.asarray(piece.coords) + shift)))

    blocks = [block for block in blocks if block["n_rows"] >= config["min_rows"]]

    if not blocks:
        attributes["row_pattern"] = "scattered"
        return attributes, []

    # Dropping a block renumbers the rest, so the tags have to be rebuilt
    # rather than carried over from the enumeration above.
    keep = {id(block) for block in blocks}
    order = {id(block): index for index, block in enumerate(blocks)}
    lines = [
        (order[id(block)], block, piece)
        for _, block, piece in lines if id(block) in keep
    ]

    # The parcel is summarised by its largest block - the plantation a planner
    # would actually be laying out against - with the spread saying how much
    # the rest of the parcel disagrees. A parcel of two blocks 40 degrees apart
    # has no single row azimuth, and a spread near zero is the promise that it
    # does.
    dominant = blocks[0]

    n_rows = len(lines)
    in_blocks = np.zeros(x_canopy.size, dtype=bool)
    for block in blocks:
        in_blocks |= block["members"]

    attributes["row_pattern"] = "rows"
    attributes["n_blocks"] = len(blocks)
    attributes["n_rows"] = n_rows
    attributes["row_azimuth_deg"] = dominant["azimuth_deg"]
    attributes["row_azimuth_spread_deg"] = float(spread)
    attributes["row_spacing_m"] = dominant["spacing_m"]
    attributes["row_periodicity"] = dominant["periodicity"]
    attributes["spectral_purity"] = dominant["purity"]
    attributes["canopy_width_m"] = dominant["width_m"]
    attributes["alley_width_m"] = dominant["alley_m"]
    attributes["captured_fraction"] = float(
        in_blocks.sum() / x_canopy.size
    ) if x_canopy.size else 0.0

    confidence = (
        0.5 * min(dominant["periodicity"] / 0.6, 1.0)
        + 0.3 * attributes["captured_fraction"]
        + 0.2 * min(n_rows / 6.0, 1.0)
    )
    attributes["row_confidence"] = float(np.clip(confidence, 0.0, 1.0))

    return attributes, lines

# ============================================================
# MAIN ENGINE
# ============================================================

def detect_crop_rows(
    parcels,
    *,
    dsm_path,
    dsm_no_vegetation_path,
    dsm_class_path=None,
    dem_slope_path=None,
    dem_aspect_path=None,
    parcel_id_column=None,
    min_canopy_height_m=0.8,
    max_canopy_height_m=25.0,
    min_canopy_cover_frac=0.02,
    row_spacing_range_m=(1.5, 14.0),
    azimuth_step_deg=1.0,
    azimuth_refine_step_deg=0.25,
    min_rows=4,
    min_row_periodicity=0.25,
    min_captured_fraction=0.45,
    min_points_per_row=12,
    max_row_blocks=3,
    row_gap_bridge_m=12.0,
    row_segment_m=10.0,
    row_hole_min_length_m=30.0,
    row_hole_min_rows=5,
    block_azimuth_tolerance_deg=10.0,
    hedgerow_max_periodicity=0.16,
    min_row_separation_m=2.0,
    parcel_margin_m=2.0,
    min_cell_cover_frac=0.10,
    min_cell_coherence=0.15,
    min_spectral_purity=0.05,
    woody_only=True,
    woody_min_cover_frac=0.15,
    woody_min_height_m=1.0,
    woody_max_height_m=6.0,
    use_sigpac_prior=True,
    use_class_filter=True,
    max_pixels_per_parcel=400_000,
    min_parcel_area_m2=500.0,
    min_pv_area_m2=10_000.0,
    exclusions=None,
    output_path=None,
    update=False,
    progress=False,
):
    """
    Find the plantation rows in every parcel, and how the rest are worked.

    Parameters
    ----------
    parcels : GeoDataFrame or str
        SIGPAC recintos from ``load_spanish_sigpac_parcels``, or a path to the
        GeoPackage. Reprojected to a metric CRS by the caller: every length
        here is in the parcels' own units.
    dsm_path : str
        Surface model **with** vegetation.
    dsm_no_vegetation_path : str
        The same surface with the vegetation classes left out. The difference
        of the two is the canopy this works on, so both must describe the same
        ground - they are two outputs of one ``build_digital_model_from_lidar``
        run.
    dsm_class_path : str, optional
        Per-pixel LiDAR class. Strongly recommended: without it a farm shed is
        indistinguishable from a tree, and two sheds make a row.
    dem_slope_path, dem_aspect_path : str, optional
        Terrain slope and aspect, **in radians** as the digital-model builder
        writes them. Reported per parcel for the planners downstream.
    parcel_id_column : str, optional
        Existing column to use as the identifier. When omitted, the SIGPAC
        7-tuple is joined into ``parcel_id`` by ``sigpac_parcel_id``.
    min_canopy_height_m : float, default=0.8
        Below this a CHM pixel is ground noise rather than a plant.
    min_canopy_cover_frac : float, default=0.02
        A parcel with less canopy than this is called herbaceous without a
        sweep - there is nothing periodic to find.
    row_spacing_range_m : tuple, default=(1.5, 14.0)
        Outer bounds of the search. Narrowed per parcel by the land-use prior
        and by the resolution floor of four pixels to a period.
    min_rows : int, default=4
        Fewer rows than this is a hedge or a field boundary, not a plantation.
        Applied per block, not per parcel.
    max_row_blocks : int, default=3
        How many times the parcel is re-swept after peeling off what the last
        sweep explained. Each pass can yield several blocks, so this bounds the
        number of *directions* looked for, not the number of blocks.
    row_gap_bridge_m : float, default=12.0
        While the blocks are being found, a missing stretch inside a row
        shorter than this keeps the block in one piece; longer ends the string
        of rows there. It only decides how the canopy is split into blocks -
        where a drawn row breaks is ``row_hole_min_length_m``'s business.
    row_segment_m : float, default=10.0
        Shortest straight run a row is allowed, and so the shortest row. Rows
        follow their own trees, so a row planted across a slope bends; its
        nodes are spaced at least this far apart, which keeps the bend gentle
        and leaves nowhere for the sampling grid to write a staircase. Anything
        shorter than one run is a stub at the edge of a block, not a row.
    row_hole_min_length_m : float, default=30.0
    row_hole_min_rows : int, default=5
        Every row is laid from one side of the parcel to the other, and broken
        only by a hole: bare ground at least ``row_hole_min_length_m`` long
        along the row, lined up across at least ``row_hole_min_rows``
        neighbouring rows. Anything less - dead trees, a gap in one row, a
        headland that has lost its outermost trees - is bridged, so a solid
        block never comes back as a fringe of stubs. A row with no trees of
        its own at all is still laid, following its neighbours, unless it is
        part of such a hole.
    block_azimuth_tolerance_deg : float, default=10.0
        Blocks this close in direction are re-fitted as one lattice, as long
        as that lattice still sits on every one of their rows. Blocks further
        apart share the parcel out between them, each row stopping short of
        the other planting.
    hedgerow_max_periodicity : float, default=0.16
        Below this share of explained variance, a canopy folded along a
        direction is flat that way - a hedge, not separate trees. Across the
        Alpicat survey, hedgerow orchards fold to 0.13 at most along their
        rows and grids of separate olive or nut trees to 0.19 and more. It
        decides two things. A block pointing a different way from a larger
        one is turned to the larger one's direction unless it is flat that
        way: trees standing apart are in rows both ways, and without this an
        orchard comes back as a patchwork of crossed blocks. And an olive or
        nut parcel is read as a superintensive planting, at a spacing below
        its traditional SIGPAC band, only when its blocks are flat along their
        own rows. Set above 1 to never turn a block.
    min_row_separation_m : float, default=2.0
        No two rows of a parcel are left closer than this - nor than three
        quarters of their spacing, which is what keeps a vineyard planted
        under two metres whole. Where two run that close the one standing on
        fewer trees is deleted; where they only touch, at a row end or a
        border between blocks, it is cut back instead.
    parcel_margin_m : float, default=2.0
        Rows are cut to the parcel set in by this much. The row nearest the
        boundary is a row, not a fence line, and nothing is planted into the
        track. Half of it is also kept clear either side of the border between
        two blocks planted in different directions.
    min_cell_cover_frac : float, default=0.10
        Share of a fully planted cell's canopy that a cell must hold before it
        is looked at at all when the blocks are labelled. Lower lets a sparse
        block hang together; higher keeps scattered trees from bridging two
        blocks.
    min_cell_coherence : float, default=0.15
        Floor on how strongly a cell's canopy must pile up on the row - the
        mean cosine of its pixels' lattice phase - before the cell joins a
        block. The bar actually applied is the larger of this and half of what
        a well-planted cell on the same lattice reaches, so it adapts to a
        grove whose crowns have closed without letting a second plantation at
        another angle in: its trees are dense but not in phase, and the two
        cases differ by a factor of two, not an order of magnitude.
    min_spectral_purity : float, default=0.05
        How much of the cross-row profile's power must sit at the period the
        fold is scored on, before that direction is eligible at all. The fold
        still chooses the direction; this only refuses one where nothing is
        there to fold - a sweep along the rows, where the trees' own spacing
        down the row lies below the search band and lends its sub-harmonic a
        perfect fold over an empty spectrum.
    min_row_periodicity : float, default=0.25
        Share of the cross-row profile's variance that folding at the detected
        period must explain before a block counts as planted in rows.
    min_captured_fraction : float, default=0.45
        Share of a block's canopy pixels that must fall within its fitted rows.
    woody_only : bool, default=True
        Estimate rows only where the crop is woody. A use in
        ``SIGPAC_SPACING_PRIORS`` says so outright; any other parcel has to
        show it in the canopy instead, because SIGPAC carries an orchard as
        arable land for years after it is planted. Set False to sweep every
        parcel that carries enough canopy at all.
    woody_min_cover_frac : float, default=0.15
        Canopy cover a parcel outside the woody uses must show before it is
        swept. Arable land runs an order of magnitude below this; a plantation
        runs two to three times above it.
    woody_min_height_m : float, default=1.0
    woody_max_height_m : float, default=6.0
        The band a plantation's median canopy height has to fall in. The upper
        bound is what separates a crop from woodland, wooded pasture and
        street trees, which carry more cover than any orchard but stand two to
        four times taller.
    use_sigpac_prior : bool, default=True
        Narrow the spacing band by ``uso_sigpac``. This is what stops an olive
        grove at 7 m being reported at its 3.5 m second harmonic.
    min_pv_area_m2 : float, default=10000.0
        Threshold behind the ``min_pv_area_ok`` flag. Nothing is filtered by it.
    exclusions : GeoDataFrame, optional
        Preserved areas, as returned by ``load_spanish_preserved_regions``.
        Used only to flag: a protected parcel is kept, with ``is_preserved``
        set, because protection constrains a project rather than ruling it
        out.
    output_path : str, optional
        GeoPackage. Written as two layers, ``crop_rows`` and
        ``crop_row_parcels``, and read straight back on a later call - see
        ``update``.
    update : bool, default=False
        Recompute even though ``output_path`` already exists. Left False, an
        existing GeoPackage is returned as it stands and not one raster is
        opened: the rows are a function of the digital models, so the time to
        recompute them is when those are rebuilt, not every run. Pass the same
        flag that rebuilds the digital models. Ignored when ``output_path`` is
        None, which has nothing to reuse.

    Returns
    -------
    tuple
        ``(rows_gdf, parcels_gdf)``. ``rows_gdf`` holds one straight two-point
        line per detected row, tagged with the ``block_index`` it belongs to
        and carrying **that block's** azimuth, spacing and width rather than
        the parcel's summary of them. ``parcels_gdf`` holds every **agricultural**
        input parcel - a FEGA agricultural ``uso_sigpac``, see
        ``filter_agricultural_parcels`` - including the ones with no rows,
        carrying the classification, the confidence and the eligibility flags.
        Roads, buildings, water, forest, scrubland, urban zones and the rest
        of the non-agricultural uses are dropped before anything is read;
        a protected or undersized parcel is flagged, not removed.

        A parcel's single-valued row columns describe its **largest block**:
        ``row_azimuth_deg``, ``row_spacing_m``, ``canopy_width_m``,
        ``alley_width_m`` and ``row_periodicity`` are all read off it,
        while ``n_rows`` counts
        every row in the parcel and ``n_blocks`` says how many plantations they
        came from. ``row_azimuth_spread_deg`` is the widest disagreement
        between any two of those blocks: near zero the parcel really does have
        one row direction, and large it does not and the rows must be read per
        block. ``captured_fraction`` is the share of the parcel's canopy that
        ended up inside some block.

        ``row_azimuth_deg`` is a compass bearing in [0, 180) and means the
        direction the rows run, so the panel azimuth a planner wants is
        ``row_azimuth_deg + 90``, taking whichever of the two candidates falls
        in [90, 270]. ``working_axis_deg`` is in [0, 90) and names a *pair* of
        perpendicular directions - see ``_working_axis_deg`` - so a planner
        must try both. It is populated on every parcel; ``row_azimuth_deg``
        only where ``row_pattern == "rows"``.
    """
    # Reading the whole canopy of a municipality back off disk costs minutes,
    # so a GeoPackage that is already there is the answer unless the caller
    # says the ground under it has changed.
    if output_path is not None and not update and os.path.exists(output_path):
        return _read_crop_rows(output_path, parcel_id_column)

    if isinstance(parcels, str):
        parcels = gpd.read_file(parcels)

    # Only farmland is a candidate. A road, a river or a building has no crop
    # rows to find and no business carrying agrivoltaics, and leaving them in
    # would only put them in front of every planner downstream.
    parcels = filter_agricultural_parcels(parcels).copy()

    if parcel_id_column is None:
        parcels["parcel_id"] = sigpac_parcel_id(parcels)
        parcel_id_column = "parcel_id"

    config = {
        "min_canopy_height_m": min_canopy_height_m,
        "max_canopy_height_m": max_canopy_height_m,
        "min_canopy_cover_frac": min_canopy_cover_frac,
        "row_spacing_range_m": tuple(row_spacing_range_m),
        "azimuth_step_deg": azimuth_step_deg,
        "azimuth_refine_step_deg": azimuth_refine_step_deg,
        "min_rows": min_rows,
        "min_row_periodicity": min_row_periodicity,
        "min_captured_fraction": min_captured_fraction,
        "min_points_per_row": min_points_per_row,
        "max_row_blocks": max_row_blocks,
        "row_gap_bridge_m": row_gap_bridge_m,
        "row_segment_m": row_segment_m,
        "row_hole_min_length_m": row_hole_min_length_m,
        "row_hole_min_rows": row_hole_min_rows,
        "block_azimuth_tolerance_deg": block_azimuth_tolerance_deg,
        "hedgerow_max_periodicity": hedgerow_max_periodicity,
        "min_row_separation_m": min_row_separation_m,
        "parcel_margin_m": parcel_margin_m,
        "min_cell_cover_frac": min_cell_cover_frac,
        "min_cell_coherence": min_cell_coherence,
        "min_spectral_purity": min_spectral_purity,
        "woody_only": woody_only,
        "woody_min_cover_frac": woody_min_cover_frac,
        "woody_min_height_m": woody_min_height_m,
        "woody_max_height_m": woody_max_height_m,
        "use_class_filter": use_class_filter,
        "use_sigpac_prior": use_sigpac_prior,
        "max_pixels_per_parcel": max_pixels_per_parcel,
    }

    dsm_src = rasterio.open(dsm_path)
    nv_src = rasterio.open(dsm_no_vegetation_path)
    class_src = rasterio.open(dsm_class_path) if dsm_class_path else None
    slope_src = rasterio.open(dem_slope_path) if dem_slope_path else None
    aspect_src = rasterio.open(dem_aspect_path) if dem_aspect_path else None

    parcel_rows = []
    row_records = []
    row_geometries = []

    iterator = parcels.iterrows()
    if progress:
        iterator = tqdm(iterator, total=len(parcels), desc="Detecting crop rows",
                        unit="parcel")

    try:
        for _, parcel in iterator:

            geometry = parcel.geometry
            uso = parcel.get("uso_sigpac")

            record = {
                parcel_id_column: parcel[parcel_id_column],
                "uso_sigpac": uso,
                "working_axis_deg": _working_axis_deg(geometry),
            }

            area = float(geometry.area) if geometry is not None else 0.0

            if geometry is None or geometry.is_empty or area < min_parcel_area_m2:
                record.update(_empty_attributes())
                lines = []
            else:
                attributes, lines = _analyse_parcel(
                    geometry, uso,
                    dsm_src=dsm_src,
                    nv_src=nv_src,
                    class_src=class_src,
                    slope_src=slope_src,
                    aspect_src=aspect_src,
                    config=config,
                )
                record.update(attributes)

            record["azimuth_vs_parcel_axis_deg"] = _angular_difference(
                record["row_azimuth_deg"], record["working_axis_deg"]
            )

            record["min_pv_area_ok"] = area >= min_pv_area_m2
            record["parcel_area_m2"] = area

            parcel_rows.append(record)

            for index, (block_index, block, line) in enumerate(lines):
                row_records.append({
                    parcel_id_column: parcel[parcel_id_column],
                    "row_index": index,
                    "block_index": block_index,
                    # The block's own geometry, not the parcel's summary: on a
                    # parcel of two plantations the summary belongs to neither.
                    "row_azimuth_deg": block["azimuth_deg"],
                    "row_spacing_m": block["spacing_m"],
                    "canopy_width_m": block["width_m"],
                    "row_length_m": float(line.length),
                    "canopy_height_p50_m": record["canopy_height_p50_m"],
                    "canopy_height_p90_m": record["canopy_height_p90_m"],
                    "canopy_cover_frac": record["canopy_cover_frac"],
                    "n_canopy_px": record["n_canopy_px"],
                })
                row_geometries.append(line)

    finally:
        for source in (dsm_src, nv_src, class_src, slope_src, aspect_src):
            if source is not None:
                source.close()

    measured = pd.DataFrame(parcel_rows).drop(columns=["uso_sigpac"])

    keep = [
        column for column in parcels.columns
        if column not in measured.columns or column == parcel_id_column
    ]

    parcels_out = parcels[keep].merge(measured, on=parcel_id_column, how="left")
    parcels_out = gpd.GeoDataFrame(
        parcels_out, geometry=parcels.geometry.values, crs=parcels.crs
    )

    parcels_out = _flag_preserved(parcels_out, exclusions)

    if row_records:
        rows_gdf = gpd.GeoDataFrame(
            row_records, geometry=row_geometries, crs=parcels.crs
        )
    else:
        rows_gdf = _empty_rows(parcel_id_column, parcels.crs)

    if output_path is not None:
        if os.path.exists(output_path):
            os.remove(output_path)

        parcels_out.to_file(output_path, driver="GPKG", layer=PARCEL_LAYER)
        if len(rows_gdf):
            rows_gdf.to_file(output_path, driver="GPKG", layer=ROW_LAYER, mode="a")

    return rows_gdf, parcels_out


#: Layer names inside the GeoPackage ``detect_crop_rows`` writes and reads back.
ROW_LAYER = "crop_rows"
PARCEL_LAYER = "crop_row_parcels"


def _empty_rows(parcel_id_column, crs):
    """
    The row layer's columns, with nothing in them.

    A municipality of nothing but cereal really does produce no rows, and the
    caller should still get a frame it can concatenate and query rather than
    something it has to special-case.
    """
    return gpd.GeoDataFrame(
        {
            column: []
            for column in (
                parcel_id_column, "row_index", "block_index",
                "row_azimuth_deg", "row_spacing_m", "canopy_width_m",
                "row_length_m", "canopy_height_p50_m",
                "canopy_height_p90_m",
                "canopy_cover_frac", "n_canopy_px",
            )
        },
        geometry=[],
        crs=crs,
    )


def _read_crop_rows(output_path, parcel_id_column):
    """
    Read back a GeoPackage this function wrote earlier.

    The row layer is absent when the last run found no rows anywhere - nothing
    is written for an empty frame - so its absence is a result, not a damaged
    file, and it comes back as the empty frame it was.
    """
    parcels_out = gpd.read_file(output_path, layer=PARCEL_LAYER)

    try:
        rows_gdf = gpd.read_file(output_path, layer=ROW_LAYER)
    except Exception:
        rows_gdf = _empty_rows(parcel_id_column or "parcel_id", parcels_out.crs)

    # A file written before non-agricultural parcels were dropped still holds
    # them; filtering on the way out keeps the answer the same either way.
    id_column = parcel_id_column or "parcel_id"
    parcels_out = filter_agricultural_parcels(parcels_out)
    rows_gdf = rows_gdf[rows_gdf[id_column].isin(parcels_out[id_column])]

    return rows_gdf, parcels_out


def _angular_difference_180(a_deg, b_deg):
    """
    Angle between two row bearings, in degrees, 0 to 90.

    Bearings live in [0, 180): a row running at 175 degrees and one at 5 are 10
    degrees apart, not 170. Unlike ``_angular_difference`` this does not fold
    to 45 - it compares two row directions, not a row against a parcel axis
    that is only known modulo 90.
    """
    if not np.isfinite(a_deg) or not np.isfinite(b_deg):
        return np.nan

    difference = abs(a_deg - b_deg) % 180.0

    return float(min(difference, 180.0 - difference))


def _angular_difference(a_deg, b_deg):
    """
    Angle between a row bearing and a parcel axis, in degrees, 0 to 45.

    The parcel axis is only known modulo 90, so the comparison has to be too:
    rows running square across the field are as aligned with its geometry as
    rows running along it, and folding to 0-45 is what says so. A value near 45
    is the genuinely odd case - rows planted diagonally to every boundary - and
    is the one to look at when a detection is in doubt.
    """
    if not np.isfinite(a_deg) or not np.isfinite(b_deg):
        return np.nan

    difference = abs(a_deg - b_deg) % 90.0

    return float(min(difference, 90.0 - difference))


def _flag_preserved(parcels, exclusions):
    """
    Attach the protection columns, defaulting them when nothing was supplied.

    Flags only. A protected parcel stays in the output with everything else
    measured, because whether protection rules out a PV plant is a planning
    judgement rather than a geometric one.
    """
    if exclusions is None or len(exclusions) == 0:
        parcels["is_preserved"] = False
        parcels["preserved_resource"] = None
        parcels["preserved_area_m2"] = 0.0
        parcels["preserved_area_ratio"] = 0.0
        return parcels

    try:
        from .rural_parcels import tag_parcels_with_preserved_areas
    except ImportError:
        from greenshadow.rural_parcels import tag_parcels_with_preserved_areas

    return tag_parcels_with_preserved_areas(parcels, exclusions)
