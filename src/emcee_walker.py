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
from runprops import FALLBACK_RUNPROPS, fallback_det_prob

# Set to a runs/<objectname>/<run_file> directory (relative to this
# notebook's location, e.g. "runs/Pluto_test/000") to load run config from
# its runprops.txt. Leave as None to use the hardcoded defaults in
# emcee_walker() below instead. Overridable via TNO_RUN_DIR (e.g. by
# run_emcee_walker.py) instead of editing this file directly -- MPIPool has
# to pickle log_posterior by reference to ship it to worker ranks, which
# only works if this module is import()ed normally (one real module object
# in sys.modules), so callers can no longer patch RUN_DIR by exec()ing a
# hand-edited copy of this file into a separate namespace.


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

def emcee_walker(run_config=None):
    """
    Run the emcee ensemble sampler for the moonlike-binary population fit.

    If `run_config` (e.g. the `runprops` module's `.runprops` dict) is given,
    run parameters come from it; otherwise falls back to the hardcoded
    defaults.

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
    # load config file, if any
    if run_config is not None:
        nwalkers       = run_config.get("nwalkers", 100)
        ndim           = run_config.get("ndim", 8)
        step_count     = run_config.get("nsteps", 1000)
        burn_count     = run_config.get("nburnin", 500)
        numpy_seed     = run_config.get("numpy_seed")
        ref_pop_name   = run_config.get("reference_pop", "Pluto")
        param_config   = run_config
    else:
        print("No run_config detected. Falling back on defaults.")
        nwalkers       = 10
        ndim           = 8 # number of params
        step_count     = 100 # how many times we run the whole simulation
        burn_count     = 50 # iterations to do and toss before starting the real run
        numpy_seed     = None
        ref_pop_name   = "Pluto"
        param_config   = FALLBACK_RUNPROPS

    # create MPIPool and its contents
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            return None

        if run_config is not None:
            det_prob_name = run_config.get("det_prob_function")
            det_prob_fn = globals().get(det_prob_name, fallback_det_prob)
            if det_prob_fn is fallback_det_prob:
                print(f"No det_prob function found in run_config "
                      f"(det_prob_function={det_prob_name!r}); using default fallback_det_prob.")
        else:
            det_prob_fn = fallback_det_prob

        if numpy_seed is not None:
            np.random.seed(numpy_seed)

        if ref_pop_name == "Pluto":
            print("Using synthetic Pluto population as reference")
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
      2. KDE likelihood on the (sep, dm) distribution of detected binaries:
         sum_i log f_sim(sep_i, dm_i) for each observed binary i

    The KDE is built from the simulated detected population and evaluated at
    each observed binary's (sep, dm). This is an unbinned likelihood, so no
    binning choices are needed.

    Parameters
    ----------
    theta : array-like, shape (ndim,)
        Flat moonlike parameter vector for one emcee walker, in
        param_config['moon_param_names'] order (fb, ka, ae, ke, ai, ki, mdm, sdm).
    reference_pop : Population
        Real observed binaries. reference_pop.popu must have columns: sep, dm.
    det_prob : callable
        Detection probability function passed through to Population().
    param_config : dict
        runprops (or FALLBACK_RUNPROPS) -- supplies the moon/wide/orbital
        parameter draw expressions Population() needs.

    Returns
    -------
    float
        Total log-likelihood. Returns -inf if the simulation produces fewer
        than 2 detected binaries (KDE undefined) while observations exist.
    """
    init_params = moon_params_from_array(theta, param_config)
    simulated_pop = Population(reference_pop.popu, init_params, det_prob, runprops=param_config)
    det = simulated_pop.popu[simulated_pop.popu['detected']].dropna(subset=['sep', 'dm'])
    ref = reference_pop.popu.dropna(subset=['sep', 'dm'])
    n_det = len(det)
    n_ref = len(ref)

    # Poisson likelihood on the binary count
    ll_count = poisson.logpmf(n_ref, mu=max(n_det, 1e-300))
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
    ll_count = get_log_likelihood(theta, reference_pop, fallback_det_prob, FALLBACK_RUNPROPS, True)
    simulated_pop = Population(reference_pop.popu, moon_params, fallback_det_prob, runprops=FALLBACK_RUNPROPS)

if __name__ == '__main__':
    RUN_DIR = os.environ.get("TNO_RUN_DIR", "runs/Pluto_test/000")
    run_config = runprops.load_runprops(RUN_DIR) if RUN_DIR else None
    
    if run_config is not None:
        print("run config going!")
        emcee_walker(run_config)
    else:
        print("no run config, running once")
        run_once()