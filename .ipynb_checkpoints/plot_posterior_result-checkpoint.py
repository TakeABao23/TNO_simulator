#!/usr/bin/env python3
"""
Take the last posterior sample from a run_emcee_walker.py results folder's
posteriors.csv, build a simulated Population from it, and plot it against
the Pluto reference population.

Usage:
    python plot_posterior_result.py results/Pluto_test/Pluto_test_.../posteriors.csv
"""
import argparse
import csv
import os

import commentjson

import param_versions
import population_class
from params import det_prob
from population_class import Population, Pluto
from population_plotter import plot_sep_vs_dm, plot_pa_vs_sep


def load_last_sample(posteriors_path):
    with open(posteriors_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        last_row = None
        for row in reader:
            if row:
                last_row = row
    if last_row is None:
        raise ValueError(f"No posterior samples found in {posteriors_path}")
    return dict(zip(header, (float(v) for v in last_row)))


def load_params_version(posteriors_path):
    """
    If a runprops.txt sits alongside posteriors.csv (as run_emcee_walker.py
    leaves in a run's results folder) and names a params_version, swap in
    that Parameters/<version>.py snapshot so the Population we build here
    matches what the run actually used.
    """
    runprops_path = os.path.join(os.path.dirname(posteriors_path), "runprops.txt")
    if not os.path.exists(runprops_path):
        return
    with open(runprops_path) as f:
        runprops = commentjson.load(f)
    params_version = runprops.get("params_version")
    if params_version:
        param_versions.apply_params_version(params_version, [globals(), vars(population_class)])
        print(f"Using parameter model '{params_version}' from Parameters/")


def main(posteriors_path):
    load_params_version(posteriors_path)
    moon_params = load_last_sample(posteriors_path)

    reference_pop = Pluto()
    simulated_pop = Population(reference_pop.popu, moon_params, det_prob)

    print(reference_pop)
    print(simulated_pop)

    plot_sep_vs_dm(simulated_pop, reference_pop.popu)
    plot_pa_vs_sep(simulated_pop, reference_pop.popu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("posteriors_csv",
                         help="Path to a posteriors.csv written by run_emcee_walker.py")
    args = parser.parse_args()
    main(args.posteriors_csv)
