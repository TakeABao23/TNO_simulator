#!/usr/bin/env python3
"""
Run emcee_walker.py against a runprops.txt without hand-editing the
script's RUN_DIR line.

Usage:
    python run_emcee_walker.py runs/Pluto_test/000

`run_dir` is a runs/<objectname>/<run_file> directory, relative to this
script's own directory (same convention as emcee_walker.py's RUN_DIR).
"""
import argparse
import os
import re

import numpy as np

import plot_posterior_result

TNO_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(TNO_SIM_DIR, "emcee_walker.py")
RUN_DIR_PATTERN = re.compile(r"^RUN_DIR\s*=.*$", re.MULTILINE)
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

    with open(MODULE_PATH) as f:
        source = f.read()

    patched_source, count = RUN_DIR_PATTERN.subn(
        f'RUN_DIR = "{run_dir}"', source, count=1
    )
    if count == 0:
        raise RuntimeError("Could not find the RUN_DIR assignment in emcee_walker.py")

    # __name__ deliberately left as something other than "__main__" so
    # emcee_walker.py's own `if __name__ == '__main__':` block does not run
    # itself -- we call emcee_walker(run_config) ourselves below so we keep
    # a handle on the returned sampler (and thus its posterior samples).
    namespace = {"__name__": "emcee_walker", "__file__": MODULE_PATH}
    code = compile(patched_source, MODULE_PATH, "exec")

    original_cwd = os.getcwd()
    os.chdir(TNO_SIM_DIR)
    try:
        exec(code, namespace)

        run_config = namespace.get("run_config")
        if run_config is not None:
            print("run config going!")
            sampler = namespace["emcee_walker"](run_config)
        else:
            print("no run config, running once")
            namespace["run_once"]()
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
        param_names = namespace.get("MOON_PARAM_NAMES")
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
        np.savetxt(posteriors_path, full_data, delimiter=",", header=header, comments="")
        print(f"Posterior samples written to {posteriors_path}")

        # Compare the first and last posterior samples side by side, rather
        # than only ever looking at the last one (as write_frozen_params
        # below does).
        plot_posterior_result.plot_first_last_comparison(
            posteriors_path, sampler.reference_pop, sampler.det_prob_fn
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
