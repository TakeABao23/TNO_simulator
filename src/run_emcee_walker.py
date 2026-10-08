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
import json
import os
import re

import commentjson
import numpy as np

import chain_plotter
import plot_posterior_result
import runprops

TNO_SIM_DIR = os.path.dirname(os.path.abspath(__file__))


def _set_runprops_value(text, key, value):
    """
    Return `text` (a runprops.txt's contents) with `key`'s value replaced
    by `value`, leaving every other line -- comments included -- as is.
    Returns None if `key` isn't in `text`.
    """
    match = re.search(r'^\s*"%s"\s*:\s*' % re.escape(key), text, re.M)
    if match is None:
        return None
    _, value_end = json.JSONDecoder().raw_decode(text, match.end())
    return text[:match.end()] + json.dumps(value) + text[value_end:]


def _add_runprops_value(text, key, value, comment):
    """
    Insert `key`: `value` (with a preceding `comment` line) into `text`,
    right after the "numpy_seed" line, or after the opening brace if there
    is none.
    """
    entry = f'    # {comment}\n    "{key}": {json.dumps(value)},\n'
    match = re.search(r'^\s*"numpy_seed".*\n', text, re.M)
    if match is None:
        match = re.search(r"\{[^\n]*\n", text)
    return text[:match.end()] + entry + text[match.end():]


def _set_comment_above(text, key, comment_lines):
    """
    Replace the contiguous block of comment lines directly above `key`'s
    line in `text` with `comment_lines` (inserting one if there is none).
    Returns `text` unchanged if `key` isn't in it.
    """
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if re.match(r'\s*"%s"\s*:' % re.escape(key), line):
            break
    else:
        return text
    start = idx
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    indent = re.match(r"\s*", lines[idx]).group()
    block = [f"{indent}# {comment}" for comment in comment_lines]
    return "\n".join(lines[:start] + block + lines[idx:])


def write_next_runprops(best_sample, param_names, run_config, results_folder):
    """
    Write a next_runprops.txt into `results_folder` that a later run can
    use directly as its runs/<objectname>/<run_file>/runprops.txt to
    continue from this one.

    It is this run's own runprops.txt (already copied into results_folder
    by runprops.load_runprops), comments and all, with only these changed:
      init_positions_file  this run's final_positions.csv, relative to the
                           runs/<objectname>/<run_file> directory (so it
                           works from any sibling run directory)
      moon_<name>          `best_sample` (the maximum-posterior sample) --
                           unused while init_positions_file is set, but a
                           record of this run's best fit
      run_file             incremented, e.g. "002" -> "003"
      RunGoal              notes which run this continues from
    and the comment above "moon_param_names" rewritten to say where the
    moon_<name> values came from.
    """
    source_path = os.path.join(results_folder, "runprops.txt")
    with open(source_path) as f:
        text = f.read()

    replacements = {f"moon_{name}": repr(float(value))
                    for name, value in zip(param_names, best_sample)}

    run_file = str(run_config.get("run_file"))
    if run_file.isdigit():
        replacements["run_file"] = str(int(run_file) + 1).zfill(len(run_file))
    replacements["RunGoal"] = (f"Continue from run {run_file} "
                               f"({os.path.basename(results_folder)})")

    final_path = os.path.join(results_folder, "final_positions.csv")
    if os.path.exists(final_path):
        replacements["init_positions_file"] = os.path.relpath(
            final_path, run_config["runs_file"])
    else:
        print(f"No {final_path}; next_runprops.txt will draw fresh "
              "walker positions instead of continuing from this run.")

    for key, value in replacements.items():
        updated = _set_runprops_value(text, key, value)
        if updated is None:
            updated = _add_runprops_value(
                text, key, value,
                f"Added by run {run_file}'s write_next_runprops()")
        text = updated

    text = _set_comment_above(text, "moon_param_names", [
        "Moon-like binary parameters: names (fixes the structured-array field",
        "order and emcee's flat theta order) and one draw expression per name,",
        "each eval'd with `rng` (a numpy Generator) in scope. Unused while",
        "init_positions_file is set; these are run "
        f"{run_file}'s maximum-posterior values",
        f"({os.path.basename(results_folder)}).",
    ])

    # Fail here, not at the start of the next run, if the edit broke it.
    commentjson.loads(text)

    output_path = os.path.join(results_folder, "next_runprops.txt")
    with open(output_path, "w") as f:
        f.write(text)
    return output_path


def plot_and_write(sampler, run_config, results_folder):
    # moon_param_names now lives in runprops itself (run_config), not a
    # module-level constant -- there's no more params.py/param_versions.py
    # to have swapped it out from under us.
    param_names = run_config.get("moon_param_names")
    # sampler.burnin_steps (see emcee_walker.emcee_walker()) is how many
    # leading steps are burn-in -- the sampler is no longer reset() after
    # burn-in (so chain_plotter.plot_trace_full() can still see it), so
    # posteriors.csv has to discard those steps explicitly here instead.
    burnin_steps = getattr(sampler, "burnin_steps", 0)
    chain = sampler.get_chain(discard=burnin_steps)  # (nsteps, nwalkers, ndim), post-burn-in only
    nsteps, nwalkers, _ = chain.shape
    samples = chain.reshape(-1, chain.shape[-1])  # matches get_chain(flat=True)
    # One row per (step, walker); step varies slowest, matching the
    # chain's own flatten order, so this lines up with `samples`.
    step_numbers = np.repeat(np.arange(nsteps), nwalkers)
    ll_counts = np.asarray(sampler.get_blobs(discard=burnin_steps, flat=True), dtype=float)

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

    # Trace/corner/parameter-vs-likelihood diagnostic plots of the chain,
    # straight off the live sampler (so they use the full log-posterior,
    # not just the ll_count blob posteriors.csv records).
    if param_names:
        chain_plotter.plot_diagnostics_from_sampler(sampler, param_names, results_folder=results_folder)

    # Compare where burn-in left off against the chain's last posterior
    # sample, rather than only ever looking at the last one -- shows what
    # sampling itself achieved, without burn-in's own (usually much larger)
    # movement dominating the comparison.
    plot_posterior_result.plot_first_last_comparison(
        posteriors_path, sampler.reference_pop, run_config, sampler.det_prob_fn,
        first_params=sampler.burnin_end_params, results_folder=results_folder
    )

    if param_names:
        log_probs = sampler.get_log_prob(discard=burnin_steps, flat=True)
        best_sample = samples[np.argmax(log_probs)]
        try:
            next_runprops_path = write_next_runprops(
                best_sample, param_names, run_config, results_folder
            )
            print(f"Next-run runprops written to {next_runprops_path}")
        except (OSError, ValueError) as exc:
            print(f"Could not write a next-run runprops: {exc}")

    return results_folder


def run(run_dir):
    """
    Run emcee_walker.py against the runprops.txt in `run_dir`.

    `run_dir` is a runs/<objectname>/<run_file> directory, relative to this
    script's own directory. Sets TNO_RUN_DIR so emcee_walker.py resolves its
    RUN_DIR, then imports and runs it: emcee_walker.emcee_walker(run_config)
    if a run_config was found, otherwise emcee_walker.run_once().

    On completion, writes posteriors.csv (one row per (step, walker) of the
    chain, plus a leading step/ll_count column), trace/corner/parameter-vs-
    likelihood diagnostic plots plus a burn-in-included full trace plot and
    a per-parameter percentile/best-fit summary table (see chain_plotter.py),
    and a first-vs-last posterior comparison plot into the run's
    results_folder, and -- if the run reports param_names -- a
    next_runprops.txt that a later run can use as its runprops.txt to
    continue from this one (see write_next_runprops()).

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
