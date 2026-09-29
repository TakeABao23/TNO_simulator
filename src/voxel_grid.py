"""
Voxel-grid likelihood: bins detected binaries into a fixed 4D grid of
(H, delta-magnitude, separation, position-angle) and compares the observed
and simulated per-voxel counts with an independent Poisson likelihood in
each voxel. Replaces emcee_walker.get_log_likelihood()'s old single
aggregate P(N_obs_total | N_sim_total) count-only term: summing every
voxel's simulated count already recovers that same total-count
information, while the per-voxel comparison now also scores the *shape*
of the simulated population (across primary brightness, delta-magnitude,
separation, and position angle) against the real one.

Bin edges (H_EDGES/DM_EDGES/SEP_EDGES/PA_EDGES below) are fixed constants
by science-team specification, not runprops-driven like the moon/wide/
orb_param draw expressions -- ask before changing them.

NOTE: "separation" here is binned directly in arcsec, the same units
Population.compute_separation() already returns -- despite "pixel
separation" in the original request, no plate-scale (arcsec/pixel)
conversion is applied, by explicit choice. If a real detector's pixel
scale should be used instead, SEP_EDGES needs converting (divide by that
scale) and this note updated.
"""
import numpy as np
from scipy.stats import poisson

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

# Separation in arcsec (see module docstring): doubling bins from 0-32.
SEP_EDGES = np.array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0])

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


def voxel_grid_log_likelihood(simulated_popu, observed_popu=None, n_obs=None):
    """
    Sum, over every voxel of the fixed 4D (H, dm, sep, pa) grid, the
    Poisson log-probability of the observed count given the simulated
    count as its rate: sum_voxel log Poisson(n_obs | mu=n_sim).

    Parameters
    ----------
    simulated_popu : pd.DataFrame
        Simulated, already-detected binaries (e.g.
        simulated_pop.popu[simulated_pop.popu['detected']]). Must have
        columns H, dm, sep, pa.
    observed_popu : pd.DataFrame, optional
        Real observed binaries, same required columns. Ignored if `n_obs`
        is given.
    n_obs : ndarray, shape GRID_SHAPE, optional
        Precomputed voxel_counts(observed_popu) -- pass this instead of
        `observed_popu` when the observed population is the same across
        many calls (e.g. every emcee likelihood evaluation in a run: the
        real/reference data never changes step to step), to avoid
        re-histogramming it from scratch on every single call. Exactly one
        of `observed_popu`/`n_obs` must be given.

    Returns
    -------
    float
        Total log-likelihood. A voxel with n_sim == 0 uses the same
        1e-300 floor emcee_walker.get_log_likelihood() used for its old
        total-count term, rather than 0, so an observed count there
        contributes a large-but-finite penalty instead of an outright
        -inf (which would reject the walker step outright rather than
        merely disfavoring it).
    """
    if n_obs is None:
        n_obs = voxel_counts(observed_popu)
    n_sim = voxel_counts(simulated_popu)
    mu = np.where(n_sim > 0, n_sim, 1e-300)
    return float(np.sum(poisson.logpmf(n_obs, mu=mu)))
