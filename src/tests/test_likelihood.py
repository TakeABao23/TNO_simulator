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

import spiceypy as spice  # noqa: E402

import voxel_grid  # noqa: E402
from runprops import FALLBACK_RUNPROPS, fallback_det_prob  # noqa: E402
from emcee_walker import Pluto, get_log_likelihood  # noqa: E402
from population_class import Population, PlutoSpread  # noqa: E402
from TNO_sim_lib import (moon_params_from_array, draw_orbit_params,  # noqa: E402
                         load_det_prob)

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


# PlutoSpread.TRUTH as theta (fb, ka, ae, ke, ai, ki, mdm, sdm); FALLBACK's
# "a" scale is 20000 km, matching TRUTH's a_max.
THETA_SPREAD_TRUTH = [0.9, 8.0, 2.0, 10.0, 15.0, 13.0, 1.9, 0.2]
# Same e/i means, near-zero a/e/i spread -- the zero-spread Pluto's
# runaway fit. (Mean a moves 17800 -> 20000 km with ka at the cap.)
THETA_SPREAD_NARROW = [0.9, 1.0e4, 1667.0, 8333.0, 5357.0, 4643.0, 1.9, 0.2]


class TestPlutoSpread(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = dict(FALLBACK_RUNPROPS)
        cls.cfg["n_realizations"] = 10
        np.random.seed(SEED)
        cls.ref = PlutoSpread(runprops=cls.cfg,
                              detection_prob=fallback_det_prob)
        cls.ref.voxel_n_obs = voxel_grid.voxel_counts(cls.ref.popu)
        binaries = cls.ref.popu[cls.ref.popu["binary_type"] != "single"]
        cls.orb = np.array(binaries["params"].tolist())

    def test_aei_have_spread_near_pluto(self):
        a, e, i = self.orb[:, 0], self.orb[:, 1], self.orb[:, 2]
        self.assertAlmostEqual(np.mean(a), 20000 * 8 / 9, delta=300)
        self.assertGreater(np.std(a), 1500)
        self.assertAlmostEqual(np.mean(e), 2 / 12, delta=0.02)
        self.assertAlmostEqual(np.mean(i), 96.4, delta=3.0)
        self.assertGreater(np.std(i), 12.0)

    def test_runprops_override(self):
        cfg = dict(self.cfg, pluto_spread={"ka": 1000.0})
        ref = PlutoSpread(runprops=cfg)
        self.assertEqual(ref.truth["ka"], 1000.0)
        self.assertEqual(ref.truth["ae"], PlutoSpread.TRUTH["ae"])

    def test_truth_beats_zero_spread(self):
        ll_truth = _lls(THETA_SPREAD_TRUTH, self.cfg, fallback_det_prob,
                        self.ref)
        ll_narrow = _lls(THETA_SPREAD_NARROW, self.cfg, fallback_det_prob,
                         self.ref)
        gap = np.mean(ll_truth) - np.mean(ll_narrow)
        noise = np.hypot(np.std(ll_truth), np.std(ll_narrow))
        self.assertGreater(gap, 5.0 * noise)


def _sep_pa_spice(xyz, orb):
    """Scalar reference: spice.conics + sky projection, one system."""
    a, e, i, w, Om, mu = orb
    elts = [a * (1.0 - e), e, np.radians(i), np.radians(Om), np.radians(w),
            np.radians(mu), 0.0, 1.0]
    off = np.array(spice.conics(elts, 0.0)[:3])
    dist = np.linalg.norm(xyz)
    los = xyz / dist
    sky = off - los * np.dot(off, los)
    sep = np.degrees(np.linalg.norm(sky) / dist) * 3600.0
    north = np.array([0.0, 0.0, 1.0]) - los * los[2]
    north /= np.linalg.norm(north)
    east = np.cross(north, los)
    east /= np.linalg.norm(east)
    pa = np.degrees(np.arctan2(np.dot(sky, east), np.dot(sky, north))) % 360.0
    return sep, pa


class TestComputeSeparation(unittest.TestCase):
    """The vectorized Kepler/sky projection must match spice.conics."""

    def test_matches_spice(self):
        r = np.random.default_rng(0)
        n = 2000
        xyz = r.normal(0.0, 1.0, (n, 3)) * 5e9
        orb = np.column_stack([
            r.uniform(1e3, 1e5, n),
            np.concatenate([r.uniform(0.0, 0.999, n - 2), [0.0, 0.9999]]),
            r.uniform(0.0, 180.0, n), r.uniform(0.0, 360.0, n),
            r.uniform(0.0, 360.0, n), r.uniform(0.0, 360.0, n)])
        sep, pa = Population.compute_separation(None, xyz, orb)
        ref = np.array([_sep_pa_spice(x, o) for x, o in zip(xyz, orb)])
        np.testing.assert_allclose(sep, ref[:, 0], rtol=1e-8, atol=1e-12)
        dpa = (pa - ref[:, 1] + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(dpa, 0.0, atol=1e-6)


class TestVectorizedInputs(unittest.TestCase):

    def test_det_prob_matches_scalar(self):
        r = np.random.default_rng(1)
        sep = r.uniform(0.0, 1.0, 5000)
        dm = r.uniform(0.0, 8.0, 5000)

        def scalar(s, d):
            return not (s < 0.1 or (s < 0.5 and d > 12.5 * s - 1.25)
                        or (s >= 0.5 and d > 5))

        expected = np.array([scalar(s, d) for s, d in zip(sep, dm)])
        np.testing.assert_array_equal(fallback_det_prob(sep, dm), expected)
        # Legacy scalar-only runprops expression still works (looped).
        legacy = load_det_prob({"det_prob": "lambda sep, dm: not (sep < 0.1 "
                                "or (sep < 0.5 and dm > 12.5 * sep - 1.25) "
                                "or (sep >= 0.5 and dm > 5))"})
        np.testing.assert_array_equal(legacy(sep, dm), expected)

    def test_legacy_zero_arg_orbit_lambda(self):
        cfg = dict(FALLBACK_RUNPROPS)
        cfg["w"] = "lambda: rng.uniform(0.0, 360.0)"
        cfg["e"] = "lambda n: 0.1"
        params = moon_params_from_array(np.array(THETA_TRUTH), cfg)
        orb = draw_orbit_params(params, cfg, 1000)
        self.assertEqual(orb["w"].shape, (1000,))
        self.assertGreater(np.std(orb["w"]), 50.0)
        np.testing.assert_array_equal(orb["e"], 0.1)
        self.assertEqual(orb["a"].shape, (1000,))


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
