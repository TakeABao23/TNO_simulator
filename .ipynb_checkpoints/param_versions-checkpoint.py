"""
Dynamic loader for versioned parameter/prior modules under Parameters/.

A runprops.txt's "params_version" field (e.g. "params_1.0") names a
Parameters/<version>.py snapshot to lock a run to, independent of whatever
TNO_simulator/params.py currently contains -- so a run stays reproducible
even as params.py keeps evolving with "swap these functions as science
evolves" changes.
"""
import importlib.util
import os

PARAMETERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Parameters")

# Names params.py exposes that a versioned snapshot may override.
PARAM_EXPORT_NAMES = [
    "MOON_PARAM_NAMES", "InitMoonParams", "moon_params_from_array",
    "InitWideParams", "OrbitParamDist", "PARAM_BOUNDS", "log_prior", "det_prob",
]


def load_params_version(version_name):
    """Import Parameters/<version_name>.py and return it as a module."""
    path = os.path.join(PARAMETERS_DIR, version_name + ".py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such params version: {path}")
    spec = importlib.util.spec_from_file_location(f"Parameters.{version_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_params_version(version_name, target_namespaces):
    """
    Load Parameters/<version_name>.py and, for each name in
    PARAM_EXPORT_NAMES it defines, overwrite that name in every namespace
    given (each a dict, e.g. a module's globals() or vars(some_module)).

    population_class.py resolves OrbitParamDist/InitWideParams through its
    own module globals (it did `from params import *` at its own import
    time), so the caller needs to pass vars(population_class) alongside its
    own globals() for the swap to actually take effect inside Population().
    """
    module = load_params_version(version_name)
    for name in PARAM_EXPORT_NAMES:
        if hasattr(module, name):
            value = getattr(module, name)
            for namespace in target_namespaces:
                namespace[name] = value
    return module
