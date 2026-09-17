import commentjson as json
import os
import datetime
import shutil
import numpy as np


class ReadJson(object):
    def __init__(self, filename):
        print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data


def load_runprops(run_dir=".", filename="runprops.txt"):
    """
    Load runprops.txt from `run_dir` and set up a results folder for this run.

    `run_dir` must be a runs/<objectname>/<run_file> directory (a fresh run)
    or a results/<objectname>/<run> directory (resuming a prior run). Every
    path this stores in the returned dict (results_folder, runs_file,
    reference_pop_file, chain_file) is absolute, and the caller's working
    directory is never touched -- this is safe to call from a notebook whose
    cwd is the TNO_simulator directory itself, by passing a relative
    `run_dir` such as "runs/Pluto_test/000".

    Returns the runprops dict, or None if `run_dir` doesn't match that
    runs/... or results/... layout, or doesn't contain `filename`. Callers
    that want to support running without a runprops.txt (e.g. a notebook
    with its own hardcoded defaults) should treat a None return as "fall
    back to hardcoded config" rather than an error.
    """
    run_dir = os.path.abspath(run_dir)
    filepath = os.path.join(run_dir, filename)
    parts = run_dir.rstrip(os.sep).split(os.sep)
    kind = parts[-3] if len(parts) >= 3 else None

    if kind not in ("runs", "results") or not os.path.exists(filepath):
        return None

    runprops = ReadJson(filepath).outProps()
    runprops['chain_file'] = None
    runprops['first_run'] = True
    runprops['runs_file'] = run_dir

    objname = runprops.get("objectname")
    sim_root = os.path.dirname(os.path.dirname(os.path.dirname(run_dir)))

    try:
        import git
        repo = git.Repo(run_dir, search_parent_directories=True)
        sha = repo.head.object.hexsha
    except Exception:
        sha = "N/A (not a git repository)"
    runprops['TNO_simulator commit hash'] = sha

    x = datetime.datetime.now()
    date = (str(x.strftime("%Y")) + "-" + str(x.strftime("%m")) + "-" + str(x.strftime("%d"))
            + "_" + str(x.strftime("%H")) + "." + str(x.strftime("%M")) + "." + str(x.strftime("%S")))
    runprops["date"] = date
    runprops["time"] = x.strftime("%H:%M:%S")
    runprops['first_run'] = False
    if runprops['run_file'] == 'debug':
        newpath = os.path.join(sim_root, "results", "debug", objname,
                        objname + "_" + date + "_" + str(runprops.get("run_file")))
    else:
        newpath = os.path.join(sim_root, "results", objname,
                                objname + "_" + date + "_" + str(runprops.get("run_file")))
    os.makedirs(newpath, exist_ok=True)
    runprops['results_folder'] = newpath

    if isinstance(runprops.get('numpy_seed'), int):
        np.random.seed(seed=runprops.get('numpy_seed'))
    else:
        seed = x.microsecond * x.second
        np.random.seed(seed=seed)
        runprops['numpy_seed'] = seed

    shutil.copy(filepath, os.path.join(newpath, 'runprops.txt'))

    if kind == "results":
        chain = os.path.join(run_dir, 'chain.h5')
        if os.path.exists(chain):
            shutil.copy(chain, os.path.join(newpath, 'chain.h5'))
            runprops['chain_file'] = os.path.join(newpath, 'chain.h5')

    ref_pop = runprops.get('reference_pop')
    if ref_pop and ref_pop != "Pluto":
        ref_pop_path = os.path.join(run_dir, ref_pop)
        if os.path.exists(ref_pop_path):
            runprops['reference_pop_file'] = ref_pop_path
            shutil.copy(ref_pop_path, os.path.join(newpath, ref_pop))

    return runprops


# Used only when emcee_walker()/run_once() are called with no real runprops
# (the "hardcoded defaults" path) -- there's no more params.py to fall back
# on, so the moon/wide/orbital parameter definitions have to live somewhere
# even without a runprops.txt. Mirrors runs/Pluto_test/000/runprops.txt.
FALLBACK_RUNPROPS = {
    "moon_param_names": ["fb", "ka", "ae", "ke", "ai", "ki", "mdm", "sdm"],
    "moon_fb": "rng.uniform(0.0, 1.0)",
    "moon_ka": "rng.uniform(0.1, 5.0)",
    "moon_ae": "rng.uniform(0.5, 5.0)",
    "moon_ke": "rng.uniform(0.5, 5.0)",
    "moon_ai": "rng.uniform(1.9, 2.1)",
    "moon_ki": "rng.uniform(4.9, 5.1)",
    "moon_mdm": "rng.uniform(0.0, 4.0)",
    "moon_sdm": "rng.uniform(0.1, 2.0)",
    "wide_fb": "0.0",
    "wide_ka": "np.nan",
    "wide_ae": "np.nan",
    "wide_ke": "np.nan",
    "wide_ai": "np.nan",
    "wide_ki": "np.nan",
    "wide_mdm": "np.nan",
    "wide_sdm": "np.nan",
    "orb_param_names": ["a", "e", "i", "w", "Om", "mu"],
    "a": "lambda: rng.power(params['ka']) * 10000",
    "e": "lambda: rng.beta(params['ae'], params['ke'])",
    "i": "lambda: 180.0 * rng.beta(params['ai'], params['ki'])",
    "w": "lambda: rng.uniform(0.0, 360.0)",
    "Om": "lambda: rng.uniform(0.0, 360.0)",
    "mu": "lambda: rng.uniform(0.0, 360.0)",
    "param_bounds": {
        "fb": [0.0, 1.0], "ka": [0.1, None], "ae": [0.5, None], "ke": [0.5, None],
        "ai": [0.5, None], "ki": [0.5, None], "mdm": [None, None], "sdm": [0.1, None],
    },
}


def fallback_det_prob(sep, dm):
    if sep < 0.1:
        return False
    elif sep < 0.5:
        if dm > 12.5 * sep - 1.25:
            return False
    elif dm > 5:
        return False
    return True