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
import re

import numpy as np

import TNO_simulator.src.plot_posterior_result as plot_posterior_result

TNO_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DOCSTRING_PATTERN = re.compile(r'^"""[\s\S]*?"""\n')


def write_frozen_params(last_sample, param_names, results_folder, params_version=None):
    """
    Write a params.py into `results_folder` that mirrors the format of
    TNO_simulator/params.py (or the Parameters/<params_version>.py it was
    swapped in from), but with InitMoonParams frozen to `last_sample`
    (e.g. the last row of posteriors.csv) instead of drawing a random
    walker start -- a reproducible snapshot of the run's final estimate.
    """
    if params_version:
        source_path = os.path.join(TNO_SIM_DIR, "Parameters", params_version + ".py")
    else:
        source_path = os.path.join(TNO_SIM_DIR, "params.py")

    with open(source_path) as f:
        source = f.read()

    for name, value in zip(param_names, last_sample):
        draw_pattern = re.compile(rf'^(\s*self\.{re.escape(name)}\s*=\s*)rng\..*$', re.MULTILINE)
        source, count = draw_pattern.subn(rf'\g<1>{float(value)!r}', source, count=1)
        if count == 0:
            raise RuntimeError(
                f"Could not find an InitMoonParams draw line for '{name}' in {source_path}"
            )

    header = (
        '"""\n'
        'Frozen parameter snapshot -- InitMoonParams below is fixed to the\n'
        'last emcee posterior sample from this run, instead of drawing a\n'
        f'random walker start. Generated from {os.path.relpath(source_path, TNO_SIM_DIR)}.\n'
        '"""\n'
    )
    source = MODULE_DOCSTRING_PATTERN.sub(header, source, count=1)

    output_path = os.path.join(results_folder, "params.py")
    with open(output_path, "w") as f:
        f.write(source)
    return output_path


def run(run_dir):
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
        import TNO_simulator.src.emcee_walker as emcee_walker

        run_config = emcee_walker.run_config
        if run_config is not None:
            print("run config going!")
            sampler = emcee_walker.emcee_walker(run_config)
        else:
            print("no run config, running once")
            emcee_walker.run_once()
            sampler = None
    finally:
        os.chdir(original_cwd)

    results_folder = run_config.get("results_folder") if run_config else None

    if not results_folder:
        print("Run complete, but no runprops results_folder was reported "
              "(RUN_DIR may not have resolved to a valid run).")
        return results_folder

    print(f"Run complete. Results in {results_folder}")

    if sampler is not None:
        # MOON_PARAM_NAMES is read after emcee_walker() runs, since a
        # runprops "params_version" can swap it out via param_versions.py
        # partway through that call.
        param_names = getattr(emcee_walker, "MOON_PARAM_NAMES", None)
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

        # Compare the run's actual params.py starting point against its
        # last posterior sample, rather than only ever looking at the last
        # one (as write_frozen_params below does).
        plot_posterior_result.plot_first_last_comparison(
            posteriors_path, sampler.reference_pop, sampler.det_prob_fn,
            first_params=sampler.initial_params, results_folder=results_folder
        )

        if param_names:
            try:
                frozen_params_path = write_frozen_params(
                    samples[-1], param_names, results_folder,
                    run_config.get("params_version")
                )
                print(f"Frozen params.py written to {frozen_params_path}")
            except (OSError, RuntimeError) as exc:
                print(f"Could not write a frozen params.py: {exc}")

    return results_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir",
                         help="runs/<objectname>/<run_file> directory, relative to TNO_simulator/")
    args = parser.parse_args()
    run(args.run_dir)
