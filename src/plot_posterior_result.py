#!/usr/bin/env python3
"""
Take the first and last posterior samples from a run_emcee_walker.py
results folder's posteriors.csv, build a simulated Population from each,
and plot them side by side against the Pluto reference population for an
easy before/after comparison of the chain.

Usage:
    python plot_posterior_result.py results/Pluto_test/Pluto_test_.../posteriors.csv
"""
import argparse
import csv
import os

import commentjson

from TNO_simulator.src.population_class import Population, Pluto
from TNO_simulator.src.population_plotter import plot_sep_vs_dm_comparison, plot_pa_vs_sep_comparison


def det_prob(sep, dm):
    if sep < 0.1:
        return False
    elif sep < 0.5:
        if dm > 12.5 * sep - 1.25:
            return False
    elif dm > 5:
        return False
    return True


def load_first_and_last_sample(posteriors_path):
    """
    Read a posteriors.csv in a single pass and return (first_sample,
    last_sample), each a dict mapping parameter name to value.
    """
    with open(posteriors_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)
        reader = reversed(list(csv.reader(f)))
        last_row = next(reader)
    if first_row is None:
        raise ValueError(f"No posterior samples found in {posteriors_path}")

    def to_dict(row):
        return dict(zip(header, (float(v) for v in row)))

    return to_dict(first_row), to_dict(last_row)


def load_run_runprops(posteriors_path):
    """
    Load the runprops.txt that sits alongside posteriors.csv (as
    run_emcee_walker.py leaves in a run's results folder) -- needed so the
    Population objects built here draw orbital elements using the same
    moon_param_names/orb_param_names/draw-expression definitions the run
    actually used.
    """
    runprops_path = os.path.join(os.path.dirname(posteriors_path), "runprops.txt")
    with open(runprops_path) as f:
        return commentjson.load(f)


def save_plots(dm_fig, pa_fig, results_folder):
    """
    Save the sep-vs-dm and pa-vs-sep comparison figures to results_folder
    as PNGs, alongside the run's other output (posteriors.csv, params.py).

    Returns (dm_path, pa_path).
    """
    dm_path = os.path.join(results_folder, "sep_vs_dm_comparison.png")
    pa_path = os.path.join(results_folder, "pa_vs_sep_comparison.png")
    dm_fig.savefig(dm_path, dpi=150)
    pa_fig.savefig(pa_path, dpi=150)
    return dm_path, pa_path


def plot_first_last_comparison(posteriors_path, reference_pop, run_runprops, det_prob_fn=det_prob,
                                first_params=None, results_folder=None):
    """
    Build simulated Populations from a "start" and "end" parameter set and
    plot them next to each other (and against `reference_pop`) so the start
    and end of the chain are easy to compare.

    `run_runprops` supplies the moon_param_names/orb_param_names/draw
    expressions Population() needs to draw orbital elements -- the same
    runprops the run itself used (see load_run_runprops).

    `first_params`, if given (e.g. run_emcee_walker.py's
    sampler.initial_params, the actual pre-burn-in draw from runprops), is
    used as the "start" set instead of posteriors_path's first row -- the
    first row is only the first post-burn-in *posterior* sample, not the
    values the run actually started from.

    `results_folder`, if given, saves the two comparison plots there via
    save_plots() (e.g. the same results folder posteriors.csv came from).

    Returns (first_pop, last_pop).
    """
    csv_first_params, last_params = load_first_and_last_sample(posteriors_path)
    if first_params is not None:
        first_label = "Initial params.py"
    else:
        first_params = csv_first_params
        first_label = "First posterior"

    first_pop = Population(reference_pop.popu, first_params, det_prob_fn, runprops=run_runprops)
    last_pop = Population(reference_pop.popu, last_params, det_prob_fn, runprops=run_runprops)

    named_pops = [(first_label, first_pop), ("Last posterior", last_pop)]
    dm_fig = plot_sep_vs_dm_comparison(named_pops, reference_pop.popu)
    pa_fig = plot_pa_vs_sep_comparison(named_pops, reference_pop.popu)

    if results_folder is not None:
        dm_path, pa_path = save_plots(dm_fig, pa_fig, results_folder)
        print(f"Comparison plots saved to {dm_path} and {pa_path}")

    return first_pop, last_pop


def main(posteriors_path):
    run_runprops = load_run_runprops(posteriors_path)
    reference_pop = Pluto()
    print(reference_pop)

    results_folder = os.path.dirname(posteriors_path)
    first_pop, last_pop = plot_first_last_comparison(posteriors_path, reference_pop, run_runprops,
                                                       results_folder=results_folder)

    print("First posterior sample:")
    print(first_pop)
    print("Last posterior sample:")
    print(last_pop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("posteriors_csv",
                         help="Path to a posteriors.csv written by run_emcee_walker.py")
    args = parser.parse_args()
    main(args.posteriors_csv)
