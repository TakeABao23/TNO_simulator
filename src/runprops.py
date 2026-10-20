import commentjson as json
import os
import datetime
import shutil
import socket
import traceback
import numpy as np


class ReadJson(object):
    def __init__(self, filename):
        print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data


# Original runprops.txt text per results folder, kept in memory (on the
# process that created the folder) so ensure_results_folder() can restore
# the folder's copy if the folder is deleted mid-run -- not stored in the
# runprops dict itself, which is pickled into every emcee task.
_RUNPROPS_TEXT = {}

RUNNING_MARKER = "RUNNING"
CRASHED_MARKER = "CRASHED"


def load_runprops(run_dir=".", filename="runprops.txt",
                  make_results_folder=False):
    """
    Load runprops.txt from `run_dir`, and optionally set up a results
    folder for this run.

    Only a real run should pass make_results_folder=True (and, under MPI,
    only its driver rank): every call that creates one leaves a new
    timestamped folder behind, which used to make results/ fill up with
    near-empty folders from tests, checks, and worker ranks -- easy to
    mistake for junk and delete out from under a live run. With the
    default False, results_folder is None and nothing is written to disk.

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
    runprops['results_folder'] = newpath if make_results_folder else None

    if isinstance(runprops.get('numpy_seed'), int):
        np.random.seed(seed=runprops.get('numpy_seed'))
    else:
        seed = x.microsecond * x.second
        np.random.seed(seed=seed)
        runprops['numpy_seed'] = seed

    if not make_results_folder:
        return runprops

    os.makedirs(newpath, exist_ok=True)
    with open(filepath) as f:
        _RUNPROPS_TEXT[newpath] = f.read()
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


def is_mpi_driver():
    """
    True on the MPI rank that drives emcee (rank 0, which schwimmbad's
    MPIPool makes the master), or when not running under MPI at all. Only
    this rank should create a results folder: the worker ranks just
    evaluate log_posterior, and each one creating its own timestamped
    folder left stray near-empty copies whenever ranks straddled a
    second boundary at startup.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return True
    return MPI.COMM_WORLD.Get_rank() == 0


def mark_running(results_folder, run_dir=None):
    """
    Write a RUNNING marker into `results_folder`, so a live run's folder
    is distinguishable from an abandoned one. mark_finished() removes it
    and mark_crashed() swaps it for CRASHED, so a RUNNING left behind on
    a run that isn't running means it was killed outright (e.g. SIGKILL,
    or mpiexec aborting the driver).
    """
    with open(os.path.join(results_folder, RUNNING_MARKER), "w") as f:
        f.write("This run is in progress -- do not delete this folder.\n"
                f"started: {datetime.datetime.now().isoformat()}\n"
                f"host: {socket.gethostname()}\n"
                f"pid: {os.getpid()}\n"
                f"run_dir: {run_dir}\n")


def mark_finished(results_folder):
    """Remove `results_folder`'s RUNNING marker once the run completes."""
    path = os.path.join(results_folder, RUNNING_MARKER)
    if os.path.exists(path):
        os.remove(path)


def mark_crashed(results_folder):
    """
    Replace `results_folder`'s RUNNING marker with a CRASHED file holding
    the current exception's traceback. Call from inside an except block.
    """
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, CRASHED_MARKER), "w") as f:
        f.write(f"crashed: {datetime.datetime.now().isoformat()}\n\n")
        f.write(traceback.format_exc())
    mark_finished(results_folder)


def ensure_results_folder(results_folder):
    """
    Make sure `results_folder` still exists before writing into it,
    recreating it (with its runprops.txt copy, if this process created
    the folder, and a RUNNING marker) and warning if it was deleted
    mid-run -- so a deleted folder costs the files already written there,
    not the whole in-memory chain.
    """
    if os.path.isdir(results_folder):
        return
    print(f"WARNING: results folder {results_folder} was deleted mid-run; "
          "recreating it. Files already written there (e.g. "
          "burnin_end.csv) are lost.")
    os.makedirs(results_folder, exist_ok=True)
    text = _RUNPROPS_TEXT.get(results_folder)
    if text is not None:
        with open(os.path.join(results_folder, "runprops.txt"), "w") as f:
            f.write(text)
    mark_running(results_folder)


# Used only when emcee_walker()/run_once() are called with no real runprops
# (the "hardcoded defaults" path) -- there's no more params.py to fall back
# on, so the moon/wide/orbital parameter definitions have to live somewhere
# even without a runprops.txt. Mirrors runs/Pluto_test/000/runprops.txt.
FALLBACK_RUNPROPS = {
    "albedo": 0.1,
    "density": 1.0,
    "n_realizations": 1,
    "voxel_prior_alpha": 0.5,
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
    "a": "lambda n: rng.power(params['ka'], n) * 20000",  # scale raised from 10000 km, see runs/Pluto_test/000/runprops.txt's "a" comment
    "e": "lambda n: rng.beta(params['ae'], params['ke'], n)",
    "i": "lambda n: 180.0 * rng.beta(params['ai'], params['ki'], n)",
    "w": "lambda n: rng.uniform(0.0, 360.0, n)",
    "Om": "lambda n: rng.uniform(0.0, 360.0, n)",
    "mu": "lambda n: rng.uniform(0.0, 360.0, n)",
    "param_bounds": {
        "fb": [0.0, 1.0], "ka": [0.1, 1.0e4], "ae": [0.5, 1.0e4], "ke": [0.5, 1.0e4],
        "ai": [0.5, 1.0e4], "ki": [0.5, 1.0e4], "mdm": [None, None], "sdm": [0.1, None],
    },
}


def fallback_det_prob(sep, dm):
    """Elementwise over arrays (or scalars): True where detectable."""
    sep = np.asarray(sep)
    dm = np.asarray(dm)
    return ((sep >= 0.1) & ((sep >= 0.5) | (dm <= 12.5 * sep - 1.25))
            & ((sep < 0.5) | (dm <= 5)))