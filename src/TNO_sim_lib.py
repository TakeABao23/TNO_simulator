"""
Runprops-driven replacement for params.py/param_versions.py/Parameters/.

Instead of a separate params.py module defining InitMoonParams/OrbitParamDist
classes, every moon-like/wide/orbital parameter distribution is defined
directly in runprops.txt as an eval'able Python expression string (with
`rng`, a numpy Generator, and -- for orbital params -- `params`, the current
moon-level hyperparameter record, in scope). This means a run's runprops.txt
alone is the complete, reproducible record of the model that produced it
(it's already copied into the run's results folder), so there's no more
need for a separate Parameters/<version>.py snapshot + param_versions.py
swapping mechanism to keep old model definitions reproducible.

Moon-like/wide parameter sets and per-object orbital-element draws are
represented as numpy structured scalars (e.g. `params['fb']`) rather than
class instances -- dict-like field access, but a real ndarray dtype, which
converts to/from the flat float array emcee's theta requires via
numpy.lib.recfunctions instead of hand-written zip/dict-comprehension code.
"""
import os

import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rfn

# Process-wide Generator, deliberately not reseeded by runprops' numpy_seed
# (each process/MPI rank gets its own instance from OS entropy at import
# time, then it advances across every draw that process makes for the rest
# of the run) -- module-level like this so it is never pickled through
# emcee's MPIPool args; passing a live Generator through pickled per-task
# args would re-send the same unadvanced snapshot to a worker on every
# call, making every "random" draw on that worker identical.
rng = np.random.default_rng()


def _dtype(names):
    return [(name, np.float64) for name in names]


def _draw_record(names, exprs):
    """Eval one expression per name and pack the results into a structured scalar."""
    arr = np.zeros(1, dtype=_dtype(names))
    for name, expr in zip(names, exprs):
        arr[name] = eval(expr, {"rng": rng, "np": np}, {})
    return arr[0]


def draw_moon_params(runprops):
    """Draw a fresh moon-like hyperparameter record from runprops' moon_<name> expressions."""
    names = runprops["moon_param_names"]
    exprs = [runprops[f"moon_{name}"] for name in names]
    return _draw_record(names, exprs)


def draw_wide_params(runprops):
    """
    Draw a fresh wide-binary hyperparameter record from runprops' wide_<name>
    expressions. Reuses moon_param_names for the field set (fb, ka, ae, ke,
    ai, ki, mdm, sdm) since wide binaries share the same parameter shape.
    """
    names = runprops["moon_param_names"]
    exprs = [runprops[f"wide_{name}"] for name in names]
    return _draw_record(names, exprs)


def draw_orbit_params(moon_params, runprops):
    """
    Draw one fresh set of orbital elements (a, e, i, w, Om, mu) for a single
    TNO, from runprops' orbital-element lambda expressions -- each is
    eval'd with `params` bound to `moon_params` (the walker's current
    moon-level hyperparameters, e.g. so `a`'s expression can reference
    params['ka']), then immediately called to draw one sample.
    """
    names = runprops["orb_param_names"]
    namespace = {"rng": rng, "np": np, "params": moon_params}
    arr = np.zeros(1, dtype=_dtype(names))
    for name in names:
        draw_fn = eval(runprops[name], namespace)
        arr[name] = draw_fn()
    return arr[0]


def moon_params_to_array(moon_params):
    """Structured moon-params record -> flat float64 array (emcee's theta), in the record's own field order."""
    return rfn.structured_to_unstructured(np.array([moon_params]))[0].astype(np.float64)


def moon_params_from_array(theta, runprops):
    """Inverse of moon_params_to_array: flat array -> structured record keyed by moon_param_names."""
    names = runprops["moon_param_names"]
    return rfn.unstructured_to_structured(np.array([theta], dtype=np.float64), dtype=_dtype(names))[0]


def save_walker_positions(positions, runprops, results_folder, filename="burnin_end.csv"):
    """
    Write a (nwalkers, ndim) array of flat moon-like parameter vectors --
    e.g. an emcee State's .coords right after burn-in -- to
    results_folder/filename, one row per walker, columns in
    moon_param_names order. Lets a later run seed its own walkers from
    here (see load_walker_positions()) instead of drawing fresh from the
    moon_<name> expressions, e.g. to resume from where burn-in left off
    without re-paying for it.

    Returns the path written.
    """
    names = runprops["moon_param_names"]
    path = os.path.join(results_folder, filename)
    pd.DataFrame(np.asarray(positions), columns=names).to_csv(path, index=False)
    return path


def load_walker_positions(path, runprops, nwalkers):
    """
    Inverse of save_walker_positions(): load a (nwalkers, ndim) array of
    flat moon-like parameter vectors from a CSV it wrote (or any CSV with
    the same one-row-per-walker, moon_param_names-ordered-columns shape),
    for seeding emcee_walker()'s initial walker positions instead of
    drawing them fresh from the moon_<name> expressions.

    Raises ValueError if the file's columns don't match moon_param_names
    (order included), or its row count doesn't match `nwalkers` -- there
    would otherwise be no sane way to pick which rows become which
    walkers' starting positions.
    """
    names = runprops["moon_param_names"]
    df = pd.read_csv(path)
    if list(df.columns) != list(names):
        raise ValueError(
            f"{path}: columns {list(df.columns)} don't match "
            f"moon_param_names {list(names)}"
        )
    if len(df) != nwalkers:
        raise ValueError(
            f"{path}: has {len(df)} rows but nwalkers is {nwalkers} -- "
            "can't seed one walker position per row otherwise."
        )
    return df.to_numpy(dtype=np.float64)


def param_bounds(runprops):
    """runprops' param_bounds (JSON [lo, hi] pairs, null = unbounded) -> {name: (lo, hi)} with +-inf."""
    bounds = {}
    for name, bound in runprops["param_bounds"].items():
        lo, hi = bound
        bounds[name] = (-np.inf if lo is None else lo, np.inf if hi is None else hi)
    return bounds


def log_prior(theta, runprops):
    """
    Uniform prior over runprops' param_bounds, in moon_param_names order.
    Returns 0.0 if every parameter is within bounds, -inf otherwise (e.g. a
    negative ka/ae/ke/ai/ki, which would crash rng.power/rng.beta).
    """
    params = moon_params_from_array(theta, runprops)
    bounds = param_bounds(runprops)
    for name in runprops["moon_param_names"]:
        lo, hi = bounds[name]
        if not (lo <= params[name] <= hi):
            return -np.inf
    return 0.0
