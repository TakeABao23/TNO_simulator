"""
Tests for the voxel-grid likelihood and the synthetic Pluto reference.

Run from TNO_simulator/src/:

    python -m unittest discover tests -v

or, for the noise test's full table of log-likelihood mean/std vs K:

    python tests/test_likelihood.py --report
"""
import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import voxel_grid  # noqa: E402
from runprops import FALLBACK_RUNPROPS, fallback_det_prob  # noqa: E402
from emcee_walker import Pluto, get_log_likelihood  # noqa: E402
from TNO_sim_lib import moon_params_from_array  # noqa: E402

# theta order: fb, ka, ae, ke, ai, ki, mdm, sdm
# Run 002's best fit, from the old (fixed-orientation, no-detection,
# 1e-300-floor) setup.
THETA_002_BEST = [0.977, 5.51, 4.01, 2.06, 2.09, 4.85, 1.84, 0.267]
# Close to Pluto()'s truth: fb 0.9; a -> 20000 km (vs 19591); e ~ 0;
# i = 180 * 1069 / 2000 = 96.2 deg, tightly; dm ~ |N(1.9, 0.2)|.
THETA_TRUTH = [0.9, 200.0, 0.5, 500.0, 1069.0, 931.0, 1.9, 0.2]

N_REPEATS = 10
SEED = 1


def _setup(n_realizations):
    """
    FALLBACK_RUNPROPS (a scale 20000 km) with K overridden, plus the
    reference -- fixed here rather than read from a runs/ directory, so
    editing a run's runprops.txt can't silently move THETA_TRUTH.
    """
    cfg = dict(FALLBACK_RUNPROPS)
    cfg["n_realizations"] = n_realizations
    det = fallback_det_prob
    np.random.seed(SEED)
    ref = Pluto(runprops=cfg, detection_prob=det)
    ref.voxel_n_obs = voxel_grid.voxel_counts(ref.popu)
    return cfg, det, ref


def _lls(theta, cfg, det, ref, n_repeats=N_REPEATS):
    """Log-likelihood at fixed theta, re-evaluated n_repeats times."""
    params = moon_params_from_array(np.array(theta), cfg)
    return np.array([get_log_likelihood(params, ref, det, cfg)
                     for _ in range(n_repeats)])


class TestLikelihoodNoise(unittest.TestCase):
    """
    The likelihood is stochastic (each call simulates a fresh population).
    Its spread at fixed parameters must be small compared with the
    differences emcee needs to resolve, or walkers lock onto lucky draws
    (run 002: std ~1400, acceptance 0.1%).
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.det, cls.ref = _setup(n_realizations=10)
        cls.ll_best = _lls(THETA_002_BEST, cls.cfg, cls.det, cls.ref)
        cls.ll_truth = _lls(THETA_TRUTH, cls.cfg, cls.det, cls.ref)

    def test_noise_small_at_fixed_params(self):
        for name, lls in [("002 best", self.ll_best),
                          ("truth", self.ll_truth)]:
            with self.subTest(point=name):
                self.assertLess(np.std(lls), 10.0)

    def test_truth_beats_002_best(self):
        gap = np.mean(self.ll_truth) - np.mean(self.ll_best)
        noise = np.hypot(np.std(self.ll_truth), np.std(self.ll_best))
        self.assertGreater(gap, 5.0 * noise)


class TestVoxelLikelihood(unittest.TestCase):

    def _popu(self, n, sep):
        return pd.DataFrame({"H": [0.0] * n, "dm": [1.5] * n,
                             "sep": [sep] * n, "pa": [10.0] * n})

    def test_empty_voxel_penalty_is_bounded(self):
        """
        Moving one observed point from a populated voxel into one the
        simulation left empty costs a few units, not the old floor's ~691.
        (Absolute values include a parameter-independent alpha term from
        every voxel, so compare differences.)
        """
        sim = self._popu(10, 0.2)
        ll_in = voxel_grid.voxel_grid_log_likelihood(
            sim, self._popu(1, 0.2), n_realizations=10)
        ll_out = voxel_grid.voxel_grid_log_likelihood(
            sim, self._popu(1, 0.5), n_realizations=10)
        self.assertTrue(np.isfinite(ll_out))
        self.assertLess(ll_out, ll_in)
        self.assertGreater(ll_out - ll_in, -10.0)

    def test_matching_beats_mismatched(self):
        obs = self._popu(5, 0.5)
        match = self._popu(50, 0.5)
        miss = self._popu(50, 0.2)
        ll_match = voxel_grid.voxel_grid_log_likelihood(
            match, obs, n_realizations=10)
        ll_miss = voxel_grid.voxel_grid_log_likelihood(
            miss, obs, n_realizations=10)
        self.assertGreater(ll_match, ll_miss)


class TestPlutoReference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.det, cls.ref = _setup(n_realizations=1)
        cls.binaries = cls.ref.popu[cls.ref.popu["binary_type"] != "single"]

    def test_inclination_is_fixed(self):
        incs = np.array([p[2] for p in self.binaries["params"]])
        np.testing.assert_allclose(incs, 96.2)

    def test_orientation_is_randomized(self):
        lans = np.array([p[4] for p in self.binaries["params"]])
        self.assertGreater(np.std(lans), 50.0)

    def test_undetected_blanked(self):
        observed = self.ref.popu.dropna(subset=["sep"])
        self.assertTrue(observed["detected"].all())
        self.assertGreaterEqual(observed["sep"].min(), 0.1)

    def test_every_row_has_position(self):
        cols = ["x", "y", "z", "H"]
        self.assertFalse(self.ref.popu[cols].isna().any().any())


def report(ks=(1, 10, 50)):
    """Print log-likelihood mean/std/time per evaluation vs K."""
    points = [("002 best", THETA_002_BEST), ("truth", THETA_TRUTH)]
    print(f"{'K':>4} {'point':10} {'mean':>9} {'std':>7} {'s/eval':>7}")
    for k in ks:
        cfg, det, ref = _setup(n_realizations=k)
        for name, theta in points:
            t0 = time.time()
            lls = _lls(theta, cfg, det, ref)
            dt = (time.time() - t0) / N_REPEATS
            print(f"{k:4d} {name:10} {np.mean(lls):9.1f} "
                  f"{np.std(lls):7.2f} {dt:7.2f}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        unittest.main()
