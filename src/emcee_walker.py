import os
from TNO_simulator.src.params import *
import TNO_simulator.src.population_class as population_class
from TNO_simulator.src.population_class import *
from TNO_simulator.src.population_plotter import *
from TNO_simulator.src.TNO_sim_lib import *
import logging
import emcee
from schwimmbad import MPIPool
import TNO_simulator.src.runprops as runprops
import TNO_simulator.src.param_versions as param_versions

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
        params_version = runprops.get("params_version")
    else:
        nwalkers       = 10
        ndim           = 8 # number of params
        step_count     = 100 # how many times we run the whole simulation
        burn_count     = 50 # iterations to do and toss before starting the real run
        numpy_seed     = None
        ref_pop_name   = "Pluto"
        params_version = None

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            return None

        if params_version:
            # Locks this run to a frozen Parameters/<version>.py snapshot instead
            # of whatever params.py currently contains. population_class.py
            # resolves OrbitParamDist/InitWideParams through its own module
            # globals (from its own `from params import *`), so both namespaces
            # need patching for the swap to actually reach Population().
            param_versions.apply_params_version(params_version, [globals(), vars(population_class)])
            print(f"Using parameter model '{params_version}' from Parameters/")

        det_prob_fn = globals().get(runprops.get("det_prob_function"), det_prob) if runprops is not None else det_prob

        if numpy_seed is not None:
            np.random.seed(numpy_seed)

        if ref_pop_name == "Pluto":
            reference_pop = Pluto()
        else:
            reference_pop = ObservedPopulation(pd.read_csv(ref_pop_name))

        p0 = np.array([InitMoonParams().to_array() for _ in range(nwalkers)])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool, args=[reference_pop, det_prob_fn])
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
        # params.py (or the swapped-in params_version) drew before emcee moved
        # any walkers -- distinct from the first post-burn-in posterior sample.
        sampler.initial_params = moon_params_from_array(p0[0])
        # TODO: multimoon has functions to graph emcee results
        # TODO: verbose mode and plotting mode
        return sampler

def get_log_likelihood(theta, reference_pop, det_prob, verbose = False):
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
        Flat moonlike parameter vector for one emcee walker, in MOON_PARAM_NAMES
        order (fb, ka, ae, ke, ai, ki, mdm, sdm).
    reference_pop : Population
        Real observed binaries. reference_pop.popu must have columns: sep, dm.
    det_prob : callable
        Detection probability function passed through to Population().

    Returns
    -------
    float
        Total log-likelihood. Returns -inf if the simulation produces fewer
        than 2 detected binaries (KDE undefined) while observations exist.
    """
    init_params = moon_params_from_array(theta)
    simulated_pop = Population(reference_pop.popu, init_params, det_prob)
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

def log_posterior(theta, reference_pop, det_prob, verbose = False):
    """
    log_prior(theta) + get_log_likelihood(theta, ...); this is what emcee
    should sample, so that walker proposals outside PARAM_BOUNDS (e.g. a
    negative ka/ae/ke/ai/ki, which crashes rng.power/rng.beta) are rejected
    via -inf instead of reaching the likelihood function at all.

    Returns (log_posterior, ll_count) -- the ll_count is returned as an
    emcee blob so run_emcee_walker.py can write it out alongside each
    posterior sample without recomputing it.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf, np.nan
    ll_count = get_log_likelihood(theta, reference_pop, det_prob, verbose)
    return lp + ll_count, ll_count

def run_once():
    reference_pop = Pluto()
    moon_params = InitMoonParams()
    ll_count = get_log_likelihood(moon_params.to_array(), reference_pop, det_prob, True)
    simulated_pop = Population(reference_pop.popu, moon_params.moon_params, det_prob)

if __name__ == '__main__':
    if run_config is not None:
        print("run config going!")
        emcee_walker(run_config)
    else:
        print("no run config, running once")
        run_once()