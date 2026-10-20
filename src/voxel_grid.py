"""
Voxel-grid likelihood: bins detected binaries into a fixed 4D grid of
(H, delta-magnitude, separation, position-angle) and compares the observed
and simulated per-voxel counts with an independent Poisson likelihood in
each voxel, marginalized over the simulation's own counting noise (see
voxel_grid_log_likelihood). Replaces emcee_walker.get_log_likelihood()'s
old single aggregate P(N_obs_total | N_sim_total) count-only term: summing every
voxel's simulated count already recovers that same total-count
information, while the per-voxel comparison now also scores the *shape*
of the simulated population (across primary brightness, delta-magnitude,
separation, and position angle) against the real one.

Bin edges (H_EDGES/DM_EDGES/SEP_EDGES/PA_EDGES below) are fixed constants
by science-team specification, not runprops-driven like the moon/wide/
orb_param draw expressions -- ask before changing them.

NOTE: "separation" was originally specified as 0-2, 2-4, 4-8, 8-16, 16-32
"pixels", on the assumption of a detector pixel scale that was never
pinned down -- Population.compute_separation() only ever produces arcsec.
SEP_EDGES below converts that original pixel-based spec to arcsec via the
HST WFC3/UVIS plate scale (0.04 arcsec/pixel, a common choice for TNO
binary imaging surveys): the *shape* (doubling bins) is unchanged, only
the units/scale. Confirmed against the Pluto reference population: its
observed separations (0.09-0.80 arcsec) now spread across the bins
(counts [1, 3, 18, 111, 314], not all 447 collapsed into one) -- before
this conversion, every observed *and* simulated separation fell inside
the single first bin (arcsec 0-2 covers far more range than the ~0-1
arcsec these binaries actually span), leaving this axis unable to
constrain the fit at all.
"""
import numpy as np
from scipy.stats import nbinom

# Position angle: 0-360 deg, 8 equal-width bins.
PA_EDGES = np.linspace(0.0, 360.0, 9)

# Primary absolute magnitude H: -2 to 10, as one wide [-2, 2) bin then four
# width-2 bins.
H_EDGES = np.array([-2.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# Delta magnitude: unit-width bins from 0-4, then one wide [5, 10) bin.
# NOTE: [4, 5) is intentionally NOT a gap -- it's covered by its own bin
# here (unlike a literal reading of "0-1, 1-2, 2-3, 3-4, 5-10", which
# would skip it).
DM_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])

# Separation in arcsec (see module docstring): the original 0-2, 2-4, 4-8,
# 8-16, 16-32 "pixel" bins, converted via the HST WFC3/UVIS plate scale
# (0.04 arcsec/pixel).
SEP_PLATE_SCALE = 0.04  # arcsec/pixel, HST WFC3/UVIS
SEP_EDGES = np.array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0]) * SEP_PLATE_SCALE

GRID_AXES = ("H", "dm", "sep", "pa")
GRID_EDGES = (H_EDGES, DM_EDGES, SEP_EDGES, PA_EDGES)
GRID_SHAPE = tuple(len(edges) - 1 for edges in GRID_EDGES)


def voxel_counts(popu):
    """
    4D histogram of `popu`'s H/dm/sep/pa columns into the fixed
    H_EDGES/DM_EDGES/SEP_EDGES/PA_EDGES grid.

    Rows with any of H/dm/sep/pa NaN (e.g. undetected binaries, or
    singles, which have no sep/pa/dm) are dropped first. Values outside
    every edges' [min, max) range (e.g. sep >= 32 arcsec, or exactly
    pa == 360) are silently excluded by numpy.histogramdd, same as a
    voxel simply having zero members there.

    Parameters
    ----------
    popu : pd.DataFrame
        Must have columns H, dm, sep, pa -- e.g. a simulated population
        filtered to detected binaries, or an observed/reference table
        with the same columns.

    Returns
    -------
    ndarray, shape GRID_SHAPE
        Float counts per voxel (numpy.histogramdd's native dtype).
    """
    rows = popu.dropna(subset=list(GRID_AXES))
    counts, _ = np.histogramdd(rows[list(GRID_AXES)].to_numpy(dtype=float), bins=GRID_EDGES)
    return counts


def voxel_grid_log_likelihood(simulated_popu, observed_popu=None, n_obs=None,
                              n_realizations=1, alpha=0.5):
    """
    Sum, over every voxel of the fixed 4D (H, dm, sep, pa) grid, the
    log-probability of the observed count given `n_realizations` (K)
    stacked simulated realizations of the observed sample.

    The per-voxel rate lambda (expected detections per one realization)
    is not known exactly, only estimated from the K-realization count
    n_sim ~ Poisson(K * lambda). Marginalizing Poisson(n_obs | lambda)
    over the posterior lambda | n_sim ~ Gamma(n_sim + alpha, rate=K) gives
    a negative binomial,

        n_obs ~ NB(r = n_sim + alpha, p = K / (K + 1)),

    with mean (n_sim + alpha) / K. This replaces the old plug-in
    Poisson(n_obs | mu=n_sim) with its 1e-300 floor for empty voxels:
    that floor cost ln(1e-300) ~ -691 per observed binary landing in a
    voxel the (single, noisy) realization happened to leave empty, which
    made the log-likelihood a step function of the realization's luck
    (std ~1400 at fixed parameters) and froze the emcee walkers on lucky
    draws. The negative binomial penalizes an empty voxel only as much as
    K realizations' worth of evidence warrants, and its noise shrinks as
    K grows.

    Parameters
    ----------
    simulated_popu : pd.DataFrame
        Simulated, already-detected binaries pooled over all K
        realizations. Must have columns H, dm, sep, pa.
    observed_popu : pd.DataFrame, optional
        Real observed binaries, same required columns. Ignored if `n_obs`
        is given.
    n_obs : ndarray, shape GRID_SHAPE, optional
        Precomputed voxel_counts(observed_popu) -- pass this instead of
        `observed_popu` when the observed population is the same across
        many calls (e.g. every emcee likelihood evaluation in a run).
        Exactly one of `observed_popu`/`n_obs` must be given.
    n_realizations : int
        K, the number of realizations of the observed sample pooled into
        `simulated_popu`.
    alpha : float
        Gamma prior shape on each voxel's rate (0.5 ~ Jeffreys); sets how
        heavily an empty simulated voxel is penalized.

    Returns
    -------
    float
        Total log-likelihood.
    """
    if n_obs is None:
        n_obs = voxel_counts(observed_popu)
    n_sim = voxel_counts(simulated_popu)
    k = float(n_realizations)
    return float(np.sum(nbinom.logpmf(n_obs, n_sim + alpha, k / (k + 1.0))))
