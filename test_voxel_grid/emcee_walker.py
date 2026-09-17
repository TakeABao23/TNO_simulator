import os
import population_class
from population_class import *
from population_plotter import *
import TNO_sim_lib
from TNO_sim_lib import log_prior, moon_params_to_array, moon_params_from_array, draw_moon_params
import logging
import emcee
from schwimmbad import MPIPool
import runprops

# Set to a runs/<objectname>/<run_file> directory (relative to this
# notebook's location, e.g. "runs/Pluto_test/000") to load run config from
# its runprops.txt. Leave as None to use the hardcoded defaults in
# emcee_walker() below instead. Overridable via TNO_RUN_DIR (e.g. by
# run_emcee_walker.py) instead of editing this file directly -- MPIPool has
# to pickle log_posterior by reference to ship it to worker ranks, which
# only works if this module is import()ed normally (one real module object
# in sys.modules), so callers can no longer patch RUN_DIR by exec()ing a
# hand-edited copy of this file into a separate namespace.
RUN_DIR = os.environ.get("TNO_RUN_DIR", "runs/Pluto_test/000")

run_config = runprops.load_runprops(RUN_DIR) if RUN_DIR else None

# Used only when emcee_walker()/run_once() are called with no real runprops
# (the "hardcoded defaults" path) -- there's no more params.py to fall back
# on, so the moon/wide/orbital parameter definitions have to live somewhere
# even without a runprops.txt. Mirrors runs/Pluto_test/000/runprops.txt.
FALLBACK_RUNPROPS = {
    "moon_param_names": ["fb", "ka", "ae", "ke", "ai", "ki", "mdm", "sdm"],
    "moon_fb": "rng.uniform(0.0, 1.0)",
    "moon_ka": "rng.uniform(0.1, 5.0)",
    "moon_ae": "rng.uniform(0.5, 5.0)",
    "moon_ke": "rng.uniform(0.5, 5.0)",
    "moon_ai": "rng.uniform(1.9, 2.1)",
    "moon_ki": "rng.uniform(4.9, 5.1)",
    "moon_mdm": "rng.uniform(0.0, 4.0)",
    "moon_sdm": "rng.uniform(0.1, 2.0)",
    "wide_fb": "0.0",
    "wide_ka": "np.nan",
    "wide_ae": "np.nan",
    "wide_ke": "np.nan",
    "wide_ai": "np.nan",
    "wide_ki": "np.nan",
    "wide_mdm": "np.nan",
    "wide_sdm": "np.nan",
    "orb_param_names": ["a", "e", "i", "w", "Om", "mu"],
    "a": "lambda: rng.power(params['ka']) * 10000",
    "e": "lambda: rng.beta(params['ae'], params['ke'])",
    "i": "lambda: 180.0 * rng.beta(params['ai'], params['ki'])",
    "w": "lambda: rng.uniform(0.0, 360.0)",
    "Om": "lambda: rng.uniform(0.0, 360.0)",
    "mu": "lambda: rng.uniform(0.0, 360.0)",
    "param_bounds": {
        "fb": [0.0, 1.0], "ka": [0.1, None], "ae": [0.5, None], "ke": [0.5, None],
        "ai": [0.5, None], "ki": [0.5, None], "mdm": [None, None], "sdm": [0.1, None],
    },
}

def det_prob(sep, dm):
    if sep < 0.1:
        return False
    elif sep < 0.5:
        if dm > 12.5 * sep - 1.25:
            return False
    elif dm > 5:
        return False
    return True


class ObservedPopulation:
    """
    Thin wrapper so a real observed-binary csv (loaded via runprops'
    "reference_pop" field) exposes the same `.popu`/`__str__` interface
    that Pluto() and Population() provide to get_log_likelihood().
    """
    def __init__(self, popu):
        self.popu = popu

    def __str__(self):
        return f'Observed population\n Total: {len(self.popu)}\n'

def emcee_walker(runprops=None):
    """
    Run the emcee ensemble sampler for the moonlike-binary population fit.

    If `runprops` (e.g. the `runprops` module's `.runprops` dict) is given,
    run parameters come from it; otherwise falls back to the hardcoded
    defaults below.

    Parallelized across MPI ranks via schwimmbad.MPIPool, the same pattern
    multimoon's mm_run_multi.py uses: every rank enters the pool, worker
    ranks immediately park in pool.wait() (returning None to their caller
    once the master rank's `with` block below exits and closes the pool),
    and only the master rank builds the reference population and runs the
    sampler, with each walker's log_posterior call farmed out to a worker.
    Works the same under a plain `python emcee_walker.py` (schwimmbad falls
    back to a single-rank/serial pool) as it does under
    `mpiexec -n <numprocs> python emcee_walker.py`.
    """
    if runprops is not None:
        nwalkers       = runprops.get("nwalkers", 100)
        ndim           = runprops.get("ndim", 8)
        step_count     = runprops.get("nsteps", 1000)
        burn_count     = runprops.get("nburnin", 500)
        numpy_seed     = runprops.get("numpy_seed")
        ref_pop_name   = runprops.get("reference_pop", "Pluto")
        param_config   = runprops
    else:
        nwalkers       = 10
        ndim           = 8 # number of params
        step_count     = 100 # how many times we run the whole simulation
        burn_count     = 50 # iterations to do and toss before starting the real run
        numpy_seed     = None
        ref_pop_name   = "Pluto"
        param_config   = FALLBACK_RUNPROPS

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            return None

        det_prob_fn = globals().get(runprops.get("det_prob_function"), det_prob) if runprops is not None else det_prob

        if numpy_seed is not None:
            np.random.seed(numpy_seed)

        if ref_pop_name == "Pluto":
            reference_pop = Pluto()
        else:
            reference_pop = ObservedPopulation(pd.read_csv(ref_pop_name))

        p0 = np.array([moon_params_to_array(draw_moon_params(param_config)) for _ in range(nwalkers)])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool,
                                         args=[reference_pop, det_prob_fn, param_config])
        print(f"Burn-in: {burn_count} steps x {nwalkers} walkers")
        state = sampler.run_mcmc(p0, burn_count, progress=True)
        sampler.reset()
        print(f"Sampling: {step_count} steps x {nwalkers} walkers")
        sampler.run_mcmc(state, step_count, progress=True)
        # Stashed so callers (e.g. run_emcee_walker.py) can plot posterior
        # results against the same reference population/detection function the
        # run actually used, without re-deriving them from runprops themselves.
        sampler.reference_pop = reference_pop
        sampler.det_prob_fn = det_prob_fn
        # walker 0's actual pre-burn-in starting position, i.e. what
        # param_config drew before emcee moved any walkers -- distinct from
        # the first post-burn-in posterior sample.
        sampler.initial_params = moon_params_from_array(p0[0], param_config)
        # TODO: multimoon has functions to graph emcee results
        # TODO: verbose mode and plotting mode
        return sampler

def get_log_likelihood(theta, reference_pop, det_prob, param_config, verbose = False):
    """
    UNTESTED
    Log-likelihood of the observed binary population given one simulated realization.

    Combines two terms:
      1. Poisson likelihood on binary count: P(N_obs | N_det_sim)
      2. Voxel-grid likelihood on the joint (sep, pa, dm, H) distribution of
         detected binaries: sum_i log p_sim(voxel containing observed binary i)

    The (sep, pa, dm, H) space is partitioned into a grid of voxels sized to
    span the combined range of the simulated detected sample and the real
    observations, so every observed binary always lands inside the grid.
    Each voxel's probability is the (Laplace-smoothed) fraction of simulated
    detected binaries that land in it; each observed binary contributes the
    log-probability of its own voxel. Smoothing keeps an empty voxel
    strongly disfavored rather than impossible (see voxel_smoothing below) --
    without it, bins_per_dim^ndim voxels vastly outnumber a typical n_det,
    so nearly every observed binary would land in an empty voxel regardless
    of theta and the whole likelihood surface would go flat at -inf.

    Parameters
    ----------
    theta : array-like, shape (ndim,)
        Flat moonlike parameter vector for one emcee walker, in
        param_config['moon_param_names'] order (fb, ka, ae, ke, ai, ki, mdm, sdm).
    reference_pop : Population
        Real observed binaries. reference_pop.popu must have columns:
        sep, pa, dm, H.
    det_prob : callable
        Detection probability function passed through to Population().
    param_config : dict
        runprops (or FALLBACK_RUNPROPS) -- supplies the moon/wide/orbital
        parameter draw expressions Population() needs. May set
        "voxel_bins_per_dim" (default 5) to control grid resolution per
        active dimension, and "voxel_smoothing" (default 1.0) as the
        per-voxel Laplace pseudo-count.

    Returns
    -------
    float
        Total log-likelihood. Returns -inf if the simulation doesn't detect
        enough binaries to define a shape grid over the active
        (non-degenerate) columns while observations exist -- under-detecting
        is penalized, not given a free pass on the shape term.
    """
    shape_columns = ['sep', 'pa', 'dm', 'H']
    init_params = moon_params_from_array(theta, param_config)
    simulated_pop = Population(reference_pop.popu, init_params, det_prob, runprops=param_config)
    det = simulated_pop.popu[simulated_pop.popu['detected']].dropna(subset=shape_columns)
    ref = reference_pop.popu.dropna(subset=shape_columns)
    n_det = len(det)
    n_ref = len(ref)

    # Poisson likelihood on the binary count
    ll_count = poisson.logpmf(n_ref, mu=max(n_det, 1e-300))

    # Voxel-grid likelihood on the joint (sep, pa, dm, H) shape of the
    # detected binaries, evaluated at each observed binary.
    if n_ref == 0:
        ll_shape = 0.0
    elif n_det < 2:
        ll_shape = -np.inf
    else:
        det_values = det[shape_columns].to_numpy()
        ref_values = ref[shape_columns].to_numpy()
        # Drop any dimension that's (numerically) constant in the simulated
        # sample -- e.g. H is fixed for every Pluto_i test object -- a
        # zero-width voxel edge in that dimension is undefined.
        variable = np.std(det_values, axis=0) > 1e-12
        active_idx = np.flatnonzero(variable)
        # A voxel grid needs strictly more detected binaries than active
        # dimensions for its bins to carry any real information (e.g. a
        # low-fb walker proposal can detect only 2-3 binaries out of 500
        # simulated TNOs). Treated as -inf rather than skipped (0.0): a free
        # pass here would make under-detecting walkers dodge the shape
        # penalty entirely, biasing fb toward the degenerate
        # too-few-detections corner instead of the real posterior.
        if active_idx.size == 0 or n_det <= active_idx.size:
            ll_shape = -np.inf
        else:
            det_active = det_values[:, active_idx]
            ref_active = ref_values[:, active_idx]
            bins_per_dim = param_config.get('voxel_bins_per_dim', 5)
            # Grid spans the combined range of simulated + observed values in
            # each active dimension, so every observed point lands inside the
            # grid instead of silently falling outside it.
            combined = np.vstack([det_active, ref_active])
            edges = [np.linspace(combined[:, j].min(), combined[:, j].max(), bins_per_dim + 1)
                     for j in range(active_idx.size)]
            counts, _ = np.histogramdd(det_active, bins=edges)

            # Voxel index per active dimension for each observed binary,
            # clipped so a point exactly on the outer edge lands in the last
            # bin (matching histogramdd's closed-on-both-ends last bin)
            # instead of falling outside the grid.
            ref_voxel = tuple(
                np.clip(np.searchsorted(edges[j], ref_active[:, j], side='right') - 1,
                        0, bins_per_dim - 1)
                for j in range(active_idx.size)
            )
            # Laplace/additive smoothing: without it, any voxel with zero
            # simulated hits is exactly impossible (log 0 = -inf), and with
            # bins_per_dim=5 over 3-4 dimensions there are far more voxels
            # than typical n_det, so nearly every observed binary lands in
            # an empty voxel regardless of theta -- the whole likelihood
            # surface goes flat at -inf and emcee has nothing to sample on.
            # Adding a small pseudo-count to every voxel keeps empty voxels
            # heavily disfavored (a tiny probability, not a impossible one)
            # while restoring a real gradient across parameter space.
            alpha = param_config.get('voxel_smoothing', 1.0)
            n_voxels = counts.size
            probs = (counts[ref_voxel] + alpha) / (n_det + alpha * n_voxels)
            ll_shape = np.sum(np.log(probs))

    ll_count = ll_count + ll_shape
    simulated_pop.ll_count = ll_count
    '''
    if verbose is True:
        print(reference_pop.__str__())
        print(simulated_pop.__str__())
        plot_sep_vs_dm(simulated_pop, reference_pop.popu)
        plot_pa_vs_sep(simulated_pop, reference_pop.popu)
        plot_pa_vs_sep(simulated_pop)
        plot_orbits(simulated_pop)
    '''
    return ll_count

def log_posterior(theta, reference_pop, det_prob, param_config, verbose = False):
    """
    log_prior(theta) + get_log_likelihood(theta, ...); this is what emcee
    should sample, so that walker proposals outside param_bounds (e.g. a
    negative ka/ae/ke/ai/ki, which crashes rng.power/rng.beta) are rejected
    via -inf instead of reaching the likelihood function at all.

    Returns (log_posterior, ll_count) -- the ll_count is returned as an
    emcee blob so run_emcee_walker.py can write it out alongside each
    posterior sample without recomputing it.
    """
    lp = log_prior(theta, param_config)
    if not np.isfinite(lp):
        return -np.inf, np.nan
    ll_count = get_log_likelihood(theta, reference_pop, det_prob, param_config, verbose)
    return lp + ll_count, ll_count

def run_once():
    reference_pop = Pluto()
    moon_params = draw_moon_params(FALLBACK_RUNPROPS)
    theta = moon_params_to_array(moon_params)
    ll_count = get_log_likelihood(theta, reference_pop, det_prob, FALLBACK_RUNPROPS, True)
    simulated_pop = Population(reference_pop.popu, moon_params, det_prob, runprops=FALLBACK_RUNPROPS)

if __name__ == '__main__':
    if run_config is not None:
        print("run config going!")
        emcee_walker(run_config)
    else:
        print("no run config, running once")
        run_once()