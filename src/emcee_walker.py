import os
import population_class
from population_class import *
from population_plotter import *
import TNO_sim_lib
from TNO_sim_lib import (log_prior, moon_params_to_array, moon_params_from_array, draw_moon_params,
                          save_walker_positions, load_walker_positions, load_det_prob)
import voxel_grid
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

    The sampler is never reset() after burn-in -- burn-in and sampling run
    into one continuous chain (the same approach multimoon's mm_run_multi.py
    takes), with `sampler.burnin_steps` stashed so callers can tell the two
    phases apart via get_chain(discard=...) instead of losing the burn-in
    steps outright. This is what lets chain_plotter.plot_trace_full() show
    the walkers actually converging, not just the already-converged tail.

    If `run_config` has an `init_positions_file` (a path, absolute or
    relative to the run directory, to a CSV written by
    TNO_sim_lib.save_walker_positions()), walkers start from the positions
    in that file instead of drawing fresh from the moon_<name> expressions
    -- e.g. to resume from a previous run's end-of-burn-in snapshot
    (results_folder/burnin_end.csv, written right after burn-in below)
    instead of re-paying for burn-in every time, or from its end-of-run
    snapshot (results_folder/final_positions.csv, written after sampling).

    If `run_config` has a `det_prob` eval'able lambda expression (same
    pattern as the moon/wide/orb_param_names draw expressions), it's used
    as the detection-probability function; otherwise falls back to the
    older `det_prob_function`-named-lookup mechanism, and then to
    runprops.fallback_det_prob.
    """
    # load config file, if any
    if run_config is not None:
        nwalkers            = run_config.get("nwalkers", 100)
        ndim                = run_config.get("ndim", 8)
        step_count          = run_config.get("nsteps", 1000)
        burn_count          = run_config.get("nburnin", 500)
        numpy_seed          = run_config.get("numpy_seed")
        ref_pop_name        = run_config.get("reference_pop", "Pluto")
        init_positions_file = run_config.get("init_positions_file")
        results_folder      = run_config.get("results_folder")
        param_config        = run_config
    else:
        print("No run_config detected. Falling back on defaults.")
        nwalkers            = 10
        ndim                = 8 # number of params
        step_count          = 100 # how many times we run the whole simulation
        burn_count          = 50 # iterations to do and toss before starting the real run
        numpy_seed          = None
        ref_pop_name        = "Pluto"
        init_positions_file = None
        results_folder      = None
        param_config        = FALLBACK_RUNPROPS

    if init_positions_file and run_config is not None and not os.path.isabs(init_positions_file):
        init_positions_file = os.path.join(run_config["runs_file"], init_positions_file)

    # create MPIPool and its contents
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            return None

        if run_config is not None and "det_prob" in run_config:
            # Preferred: runprops-driven, same eval'd-lambda pattern as the
            # moon/wide/orb_param draw expressions -- a run's own
            # runprops.txt is then a complete record of its detection
            # function too, and doesn't depend on a same-named Python
            # function already existing in this module's globals (see the
            # det_prob_function fallback below, which does and is easy to
            # silently miss).
            det_prob_fn = load_det_prob(run_config)
        elif run_config is not None:
            det_prob_name = run_config.get("det_prob_function")
            det_prob_fn = globals().get(det_prob_name, fallback_det_prob)
            if det_prob_fn is fallback_det_prob:
                print(f"No det_prob function found in run_config "
                      f"(det_prob_function={det_prob_name!r}); using default fallback_det_prob.")
        else:
            det_prob_fn = fallback_det_prob

        if numpy_seed is not None:
            np.random.seed(numpy_seed)

        if ref_pop_name in SYNTHETIC_REFERENCES:
            print(f"Using synthetic {ref_pop_name} population as reference")
            reference_pop = SYNTHETIC_REFERENCES[ref_pop_name](
                runprops=param_config, detection_prob=det_prob_fn)
        else:
            # runprops resolves the csv against the run directory
            # (reference_pop_file); a bare ref_pop_name would be read
            # relative to the cwd instead.
            ref_pop_path = (run_config or {}).get("reference_pop_file",
                                                  ref_pop_name)
            reference_pop = ObservedPopulation(pd.read_csv(ref_pop_path))
        # Precomputed once, here, before reference_pop is ever pickled into
        # a sampler.args task: reference_pop.popu never changes across the
        # whole run, but get_log_likelihood() calls this on every single
        # walker/step otherwise -- baking it in now means every worker's
        # freshly-unpickled copy already has it, not just this process'.
        reference_pop.voxel_n_obs = voxel_grid.voxel_counts(reference_pop.popu)

        if init_positions_file:
            print(f"Seeding {nwalkers} walkers from {init_positions_file}")
            p0 = load_walker_positions(init_positions_file, param_config, nwalkers)
        else:
            p0 = np.array([moon_params_to_array(draw_moon_params(param_config)) for _ in range(nwalkers)])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool,
                                         args=[reference_pop, det_prob_fn, param_config])
        print(f"Burn-in: {burn_count} steps x {nwalkers} walkers")
        state = sampler.run_mcmc(p0, burn_count, progress=True)
        if results_folder:
            runprops.ensure_results_folder(results_folder)
            burnin_end_path = save_walker_positions(state.coords, param_config, results_folder)
            print(f"End-of-burn-in walker positions written to {burnin_end_path}")
        print(f"Sampling: {step_count} steps x {nwalkers} walkers")
        final_state = sampler.run_mcmc(state, step_count, progress=True)
        if results_folder:
            runprops.ensure_results_folder(results_folder)
            # Same format as burnin_end.csv, so a later run can point its
            # init_positions_file here to continue from this run's end.
            final_path = save_walker_positions(final_state.coords, param_config, results_folder,
                                               filename="final_positions.csv")
            print(f"Final walker positions written to {final_path}")
        # Stashed so callers (e.g. run_emcee_walker.py) can plot posterior
        # results against the same reference population/detection function the
        # run actually used, without re-deriving them from runprops themselves.
        sampler.reference_pop = reference_pop
        sampler.det_prob_fn = det_prob_fn
        # walker 0's actual pre-burn-in starting position -- either drawn
        # fresh from param_config, or loaded from init_positions_file --
        # distinct from the first post-burn-in posterior sample.
        sampler.initial_params = moon_params_from_array(p0[0], param_config)
        # walker 0's position right at the end of burn-in (same moment
        # burnin_end.csv above snapshots for every walker) -- what
        # plot_posterior_result.plot_first_last_comparison() now compares
        # against the final posterior sample, instead of initial_params
        # above (the pre-burn-in draw), so the comparison plots show what
        # burn-in itself achieved rather than being dominated by it.
        sampler.burnin_end_params = moon_params_from_array(state.coords[0], param_config)
        # How many leading steps of sampler.get_chain()/get_log_prob() are
        # burn-in -- callers pass this as discard= to get the post-burn-in
        # chain (what used to be the only chain left, back when this reset()
        # the sampler right after burn-in), or discard=0 for the full chain
        # including burn-in (see chain_plotter.sampler_to_full_chain()).
        sampler.burnin_steps = burn_count
        sampler.autocorr_time = check_autocorr(sampler, burn_count, step_count,
                                               param_config["moon_param_names"],
                                               results_folder=results_folder)
        # TODO: verbose mode and plotting mode
        return sampler


def check_autocorr(sampler, burn_count, step_count, param_names, n_tau=50,
                   results_folder=None, stuck_af=0.05):
    """
    Estimate each parameter's integrated autocorrelation time tau on the
    post-burn-in chain and warn if the run was shorter than n_tau * tau
    (emcee docs' rule of thumb: ~50 tau for a reliable tau estimate and
    posterior). tol=0 so emcee returns its estimate instead of raising
    AutocorrError on a short chain -- which is exactly the case this
    check exists to flag. The likelihood is stochastic (a fresh simulated
    population every call), so treat tau as a lower bound on how long
    walkers actually take to decorrelate.

    If `results_folder` is given, also writes results_folder/autocorr.csv:
    one row per parameter with its tau, the run's nsteps, the
    n_tau * max(tau) steps needed, and whether nsteps met it, plus the
    post-burn-in acceptance fraction summarized over walkers (mean, min,
    and how many walkers fell below stuck_af) -- per-walker, not
    per-parameter, so the same values repeat on every row. Walkers stuck
    near zero acceptance usually mean they're sitting on a lucky
    (stochastically high) likelihood draw, which also inflates tau.

    Returns
    -------
    ndarray, shape (ndim,)
        tau per parameter, in param_names order.
    """
    tau = sampler.get_autocorr_time(discard=burn_count, tol=0)
    print("Autocorrelation time (steps): "
          + ", ".join(f"{name}={t:.1f}" for name, t in zip(param_names, tau)))
    needed = n_tau * np.nanmax(tau)
    # Post-burn-in only (sampler.acceptance_fraction would include
    # burn-in): a step counts as accepted if the walker moved at all.
    chain = sampler.get_chain(discard=burn_count)
    moved = np.any(chain[1:] != chain[:-1], axis=2)
    af = moved.mean(axis=0)
    n_stuck = int(np.sum(af < stuck_af))
    print(f"Acceptance fraction: mean={af.mean():.3f}, min={af.min():.3f}, "
          f"{n_stuck}/{len(af)} walkers below {stuck_af}")
    if not np.isfinite(needed) or step_count < needed:
        print(f"WARNING: not converged -- nsteps={step_count} < "
              f"{n_tau} * max(tau) = {needed:.0f}. Increase nsteps "
              f"(or continue from final_positions.csv) before trusting "
              f"these posteriors.")
    else:
        print(f"Chain length OK: nsteps={step_count} >= "
              f"{n_tau} * max(tau) = {needed:.0f}.")
    if results_folder:
        runprops.ensure_results_folder(results_folder)
        autocorr_path = os.path.join(results_folder, "autocorr.csv")
        pd.DataFrame({
            "param": param_names,
            "tau": tau,
            "nsteps": step_count,
            "needed_steps": needed,
            "converged": bool(np.isfinite(needed) and step_count >= needed),
            "acceptance_mean": af.mean(),
            "acceptance_min": af.min(),
            "n_stuck_walkers": n_stuck,
            "nwalkers": len(af),
        }).to_csv(autocorr_path, index=False)
        print(f"Autocorrelation times written to {autocorr_path}")
    return tau

    
def get_log_likelihood(init_params, reference_pop, det_prob, param_config, verbose = False):
    """
    Log-likelihood of the observed binary population given
    runprops' n_realizations (K, default 1) simulated realizations, via voxel_grid.voxel_grid_log_likelihood(): bins detected
    binaries into a fixed 4D grid of (H, delta-magnitude, separation,
    position-angle) and sums each voxel's Poisson log-probability of the
    observed count given the simulated count as its rate. Summing every
    voxel's simulated count already recovers the same total-detected-count
    constraint the old single aggregate P(N_obs | N_det_sim) term gave, on
    top of now also scoring the simulated population's *shape* across all
    four dimensions against the real one -- see voxel_grid.py.

    Parameters
    ----------
    init_params : structured scalar
        Moon-like hyperparameter record (see moon_params_from_array()) for
        one emcee walker -- log_posterior() builds this once from theta
        and passes the same record here and to log_prior(), rather than
        each independently reconstructing it from theta.
    reference_pop : Population
        Real observed binaries. reference_pop.popu must have columns:
        H, dm, sep, pa.
    det_prob : callable
        Detection probability function passed through to Population().
    param_config : dict
        runprops (or FALLBACK_RUNPROPS) -- supplies the moon/wide/orbital
        parameter draw expressions Population() needs, plus optional
        n_realizations (K) and voxel_prior_alpha for the voxel likelihood.

    Returns
    -------
    float
        Total log-likelihood (see voxel_grid.voxel_grid_log_likelihood).
    """
    # K independent realizations of the whole observed sample, pooled into
    # one Population (each reference TNO repeated K times) -- the
    # negative-binomial voxel likelihood accounts for K, and its noise at
    # fixed parameters shrinks as K grows (see voxel_grid).
    n_real = param_config.get("n_realizations", 1)
    alpha = param_config.get("voxel_prior_alpha", 0.5)
    ref_popu = reference_pop.popu
    if n_real > 1:
        ref_popu = ref_popu.loc[ref_popu.index.repeat(n_real)]
    # Common random numbers: if runprops sets crn_seed, every evaluation
    # gives each object the same uniforms, so log-L is a smooth,
    # deterministic function of theta and walkers compare parameters
    # rather than lucky/unlucky simulation draws (see
    # TNO_sim_lib.crn_uniforms()). The posterior is then conditioned on
    # that one fixed set of K realizations -- larger n_realizations
    # shrinks that effect.
    crn_seed = param_config.get("crn_seed")
    uniforms = (None if crn_seed is None
                else TNO_sim_lib.crn_uniforms(crn_seed, len(ref_popu), param_config))
    simulated_pop = Population(ref_popu, init_params, det_prob, runprops=param_config,
                               uniforms=uniforms)
    det = simulated_pop.popu[simulated_pop.popu['detected']]

    # reference_pop.voxel_n_obs, if emcee_walker() precomputed it (it does,
    # right after building reference_pop), skips re-histogramming the
    # observed population from scratch on every single call -- it never
    # changes across a run. Falls back to computing it here (still
    # correct, just not optimized) for any caller that built its own
    # reference_pop without going through emcee_walker(), e.g. tests.
    n_obs = getattr(reference_pop, "voxel_n_obs", None)
    ll_count = voxel_grid.voxel_grid_log_likelihood(det, reference_pop.popu, n_obs=n_obs,
                                                    n_realizations=n_real, alpha=alpha)
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
    log_prior(params) + get_log_likelihood(params, ...); this is what emcee
    should sample, so that walker proposals outside param_bounds (e.g. a
    negative ka/ae/ke/ai/ki, which crashes rng.power/rng.beta) are rejected
    via -inf instead of reaching the likelihood function at all.

    Converts theta to a moon-like structured record exactly once here, and
    passes that same record to both log_prior() and get_log_likelihood()
    -- they used to each independently reconstruct it from theta via
    moon_params_from_array(), the same conversion done twice on every
    single evaluation.

    Returns (log_posterior, ll_count) -- the ll_count is returned as an
    emcee blob so run_emcee_walker.py can write it out alongside each
    posterior sample without recomputing it.
    """
    init_params = moon_params_from_array(theta, param_config)
    lp = log_prior(init_params, param_config)
    if not np.isfinite(lp):
        return -np.inf, np.nan
    ll_count = get_log_likelihood(init_params, reference_pop, det_prob, param_config, verbose)
    return lp + ll_count, ll_count

def run_once():
    reference_pop = Pluto(runprops=FALLBACK_RUNPROPS, detection_prob=fallback_det_prob)
    moon_params = draw_moon_params(FALLBACK_RUNPROPS)
    ll_count = get_log_likelihood(moon_params, reference_pop, fallback_det_prob, FALLBACK_RUNPROPS, True)
    simulated_pop = Population(reference_pop.popu, moon_params, fallback_det_prob, runprops=FALLBACK_RUNPROPS)

if __name__ == '__main__':
    RUN_DIR = os.environ.get("TNO_RUN_DIR", "runs/Pluto_test/000")
    run_config = (runprops.load_runprops(RUN_DIR, make_results_folder=runprops.is_mpi_driver())
                  if RUN_DIR else None)
    
    if run_config is not None:
        print("run config going!")
        emcee_walker(run_config)
    else:
        print("no run config, running once")
        run_once()