#!/usr/bin/env python3
"""
Run emcee_walker.py against a runprops.txt without hand-editing the
script's RUN_DIR line.

Usage:
    python run_emcee_walker.py runs/Pluto_test/000
    mpiexec -n <numprocs> python run_emcee_walker.py runs/Pluto_test/000

`run_dir` is a runs/<objectname>/<run_file> directory, relative to this
script's own directory (same convention as emcee_walker.py's RUN_DIR).
"""
import argparse
import os

import commentjson
import numpy as np

import plot_posterior_result
import runprops

TNO_SIM_DIR = os.path.dirname(os.path.abspath(__file__))


def write_frozen_runprops(last_sample, param_names, results_folder):
    """
    Write a frozen_runprops.txt into `results_folder` that mirrors the run's
    own runprops.txt (already copied there by runprops.load_runprops), but
    with each moon_<name> draw expression replaced by its fixed value from
    `last_sample` (e.g. the last row of posteriors.csv) instead of drawing a
    random walker start -- a reproducible snapshot of the run's final
    estimate. Replaces the old params.py-freezing approach now that the
    parameter draw expressions live in runprops.txt itself, not a separate
    params.py module.
    """
    source_path = os.path.join(results_folder, "runprops.txt")
    with open(source_path) as f:
        frozen = commentjson.load(f)

    for name, value in zip(param_names, last_sample):
        frozen[f"moon_{name}"] = repr(float(value))

    output_path = os.path.join(results_folder, "frozen_runprops.txt")
    with open(output_path, "w") as f:
        commentjson.dump(frozen, f, indent=4)
    return output_path


def plot_and_write(sampler, run_config, results_folder):
    # moon_param_names now lives in runprops itself (run_config), not a
    # module-level constant -- there's no more params.py/param_versions.py
    # to have swapped it out from under us.
    param_names = run_config.get("moon_param_names")
    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    nsteps, nwalkers, _ = chain.shape
    samples = chain.reshape(-1, chain.shape[-1])  # matches get_chain(flat=True)
    # One row per (step, walker); step varies slowest, matching the
    # chain's own flatten order, so this lines up with `samples`.
    step_numbers = np.repeat(np.arange(nsteps), nwalkers)
    ll_counts = np.asarray(sampler.get_blobs(flat=True), dtype=float)

    extra_columns = ["step", "ll_count"]
    header = ",".join(extra_columns + list(param_names)) if param_names else ",".join(extra_columns)
    posteriors_path = os.path.join(results_folder, "posteriors.csv")
    full_data = np.column_stack([step_numbers, ll_counts, samples])
    # column_stack upcasts step_numbers to float alongside the other
    # columns, so it needs its own integer format to write back out as
    # an int rather than e.g. "0.000000000000000000e+00".
    fmt = ["%d"] + ["%.18e"] * (full_data.shape[1] - 1)
    np.savetxt(posteriors_path, full_data, delimiter=",", header=header, comments="", fmt=fmt)
    print(f"Posterior samples written to {posteriors_path}")

    # Compare the run's actual starting point (param_config's draw
    # expressions) against its last posterior sample, rather than only
    # ever looking at the last one (as write_frozen_runprops below does).
    plot_posterior_result.plot_first_last_comparison(
        posteriors_path, sampler.reference_pop, run_config, sampler.det_prob_fn,
        first_params=sampler.initial_params, results_folder=results_folder
    )

    if param_names:
        try:
            frozen_runprops_path = write_frozen_runprops(
                samples[-1], param_names, results_folder
            )
            print(f"Frozen runprops written to {frozen_runprops_path}")
        except (OSError, RuntimeError) as exc:
            print(f"Could not write a frozen runprops: {exc}")

    return results_folder


def run(run_dir):
    """
    Run emcee_walker.py against the runprops.txt in `run_dir`.

    `run_dir` is a runs/<objectname>/<run_file> directory, relative to this
    script's own directory. Sets TNO_RUN_DIR so emcee_walker.py resolves its
    RUN_DIR, then imports and runs it: emcee_walker.emcee_walker(run_config)
    if a run_config was found, otherwise emcee_walker.run_once().

    On completion, writes posteriors.csv (one row per (step, walker) of the
    chain, plus a leading step/ll_count column) and a first-vs-last posterior
    comparison plot into the run's results_folder, and -- if the run reports
    param_names -- a frozen_runprops.txt snapshotting the last posterior
    sample as fixed parameter values.

    Returns the results_folder path, or None if run_config didn't resolve
    one.
    """
    run_dir_abs = os.path.join(TNO_SIM_DIR, run_dir)
    if not os.path.exists(os.path.join(run_dir_abs, "runprops.txt")):
        raise FileNotFoundError(
            f"No runprops.txt found in {run_dir_abs} -- expected a "
            "runs/<objectname>/<run_file> directory."
        )

    # Passed via env var (read by emcee_walker.py's own RUN_DIR line) rather
    # than exec()ing a hand-patched copy of the module into a separate
    # namespace: emcee_walker()'s MPIPool has to pickle log_posterior by
    # reference to ship it to worker ranks, which only resolves correctly
    # if this is the one real `emcee_walker` module object in sys.modules,
    # not a lookalike copy living in its own namespace dict.
    os.environ["TNO_RUN_DIR"] = run_dir

    original_cwd = os.getcwd()
    os.chdir(TNO_SIM_DIR)
    try:
        import emcee_walker
        run_config = runprops.load_runprops(run_dir) if run_dir else None
        if run_config is not None:
            print("run config going!")
            sampler = emcee_walker.emcee_walker(run_config)
            results_folder = run_config.get("results_folder")
        else:
            print("no run config, running once")
            emcee_walker.run_once()
            sampler = None
            results_folder = None
    finally:
        os.chdir(original_cwd)

    if not results_folder:
        print("Run complete, but no runprops results_folder was reported "
              "(RUN_DIR may not have resolved to a valid run).")
        return results_folder, None, None # ends run here
    else:
        print(f"Run complete. Results in {results_folder}")
        
    return results_folder, sampler, run_config


def main(run_dir):
    results_folder, sampler, run_config = run(run_dir)
    if sampler is not None:
        plot_and_write(sampler, run_config, results_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir",
                         help="runs/<objectname>/<run_file> directory, relative to TNO_simulator/")
    args = parser.parse_args()
    main(args.run_dir)
