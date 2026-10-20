import math

import numpy as np
import pandas as pd
import TNO_sim_lib as TNO_sim_lib


def _solve_kepler(M, e, tol=1e-12, max_iter=50):
    """
    Eccentric anomaly E [rad] solving E - e sin E = M, elementwise, for
    elliptical orbits (0 <= e < 1). Newton's method from Danby's starting
    guess E0 = M + 0.85 e sign(sin M), which converges for every e < 1.
    """
    M = np.remainder(M, 2.0 * np.pi)
    E = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def _orbit_offset(a, e, i, w, Om, mu):
    """
    Secondary-minus-primary position (km), shape (n, 3), at the epoch where
    the mean anomaly is `mu`, in the same frame as the elements (J2000
    ecliptic). Angles in degrees.

    Vectorized replacement for one spice.conics(elts, 0.0) call per binary
    -- at t = T0 the position depends only on the elements, not on the
    system's GM (which only sets the period, i.e. propagation away from
    T0). Agrees with spice.conics to ~1e-9 relative (see
    tests/test_likelihood.py's TestComputeSeparation).
    """
    i, w, Om, mu = (np.radians(x) for x in (i, w, Om, mu))
    E = _solve_kepler(mu, e)
    # Perifocal coordinates
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(1.0 - e * e) * np.sin(E)
    cw, sw = np.cos(w), np.sin(w)
    cO, sO = np.cos(Om), np.sin(Om)
    ci, si = np.cos(i), np.sin(i)
    # Rotate perifocal -> ecliptic: R_z(Om) R_x(i) R_z(w)
    out = np.empty((len(xp), 3))
    out[:, 0] = (cO * cw - sO * sw * ci) * xp - (cO * sw + sO * cw * ci) * yp
    out[:, 1] = (sO * cw + cO * sw * ci) * xp - (sO * sw - cO * cw * ci) * yp
    out[:, 2] = (sw * si) * xp + (cw * si) * yp
    return out


def _rowdot(a, b):
    return np.einsum("ij,ij->i", a, b)


_G = 6.674e-20  # km^3 kg^-1 s^-2


def _gm_h0(runprops):
    """
    GM [km^3/s^2] of an H = 0 sphere, from runprops' albedo and density.

    D [km] = 1329 / sqrt(albedo) * 10^(-H/5), so
    GM(H) = G * rho * (pi/6) * D^3 = _gm_h0(runprops) * 10^(-0.6 H).
    Falls back to albedo 0.1, density 1 g/cm^3 if runprops is None or
    predates these keys.

    Not needed for sep/pa at the epoch (see _orbit_offset); kept for
    propagating orbits away from it.
    """
    runprops = runprops or {}
    albedo = runprops.get("albedo", 0.1)
    density = runprops.get("density", 1.0) * 1.0e12  # g/cm^3 -> kg/km^3
    return _G * density * (math.pi / 6.0) * (1329.0 / math.sqrt(albedo)) ** 3

class Population():
    def __init__(self, reference_pop, init_params=None, detection_prob = None, runprops=None):
        # Initial Parameters -- init_params/self.wide are numpy structured
        # scalars (see TNO_sim_lib.draw_moon_params/draw_wide_params), not
        # class instances; runprops carries the moon/wide/orbital draw
        # expressions population_simulator() needs per TNO.
        self.moonlike = init_params
        self.runprops = runprops
        self.gm_h0 = _gm_h0(runprops)
        self.wide = TNO_sim_lib.draw_wide_params(runprops)
        self.ref_binary_frac = self.moonlike['fb'] + self.wide['fb']

        # Create Population
        self.popu = self.population_simulator(reference_pop)
        self.type = "Simulated"
        if detection_prob is not None:
            self.apply_detection(detection_prob)
        self.ll_count = "DNE; missing detection probability"
        # Log population characteristics
        self._count_types()
        self.sim_binary_frac = (self.moonlike_pop + self.wide_pop) / (self.moonlike_pop + self.wide_pop + self.single_pop)

    def __str__(self):
        return f'{self.type} population\n Singles: {self.single_pop}\n Moonlike: {self.moonlike_pop}\n Wide: {self.wide_pop}\n Total: {self.single_pop + self.moonlike_pop + self.wide_pop}\n Binary fraction: {self.sim_binary_frac}\n Log likelihood: {self.ll_count}\n'

    def population_simulator(self, reference_pop=None):
        """
        Simulate the full TNO population over every object in the data table.

        Parameters
        ----------
        reference_pop : pd.DataFrame
            One row per TNO. Required columns: Name, x, y, z (geocentric), H.
        moon_params : dict
            Population-level parameters from emcee.
            Keys: fb, ka, ae, ke, ai, ki, mdm, sdm

        Returns
        -------
        pd.DataFrame
            Simulated population. Columns: Name, H, binary_type, sep, pa, dm,
            orbital_params. H is the primary's own absolute magnitude,
            carried straight through from `reference_pop` (not drawn/
            simulated); sep/pa/dm/orbital_params are NaN for single
            (non-binary) objects. H is still reported for singles, since
            it's an observed property of the primary regardless of
            binary_type, unlike sep/pa/dm which describe a companion that
            doesn't exist for a single.
        """

        n = len(reference_pop)
        xyz = reference_pop[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        Hs = reference_pop['H'].to_numpy()

        # Same draw thresholds as before: single above ref_binary_frac,
        # moonlike between wide fb and ref_binary_frac, wide below wide fb.
        binary_check = np.random.rand(n)
        is_single = binary_check > self.ref_binary_frac
        is_wide = ~is_single & (binary_check <= self.wide['fb'])
        binary_type = np.where(is_single, 'single',
                               np.where(is_wide, 'wide', 'moonlike'))

        # Wide binaries reuse the moonlike orbital-element distribution
        # (not one derived from self.wide, which is normally inert -- see
        # draw_wide_params), matching the original code's
        # self.worb = self.morb. Only dm differs between the two types.
        binary = np.flatnonzero(~is_single)
        nb = len(binary)
        orb = TNO_sim_lib.draw_orbit_params(self.moonlike, self.runprops, nb)
        orbital_params = np.column_stack(
            [orb[name] for name in self.runprops["orb_param_names"]])
        wide_b = is_wide[binary]
        dm_b = np.abs(np.random.normal(
            np.where(wide_b, 0.0, self.moonlike['mdm']),
            np.where(wide_b, self.wide['sdm'], self.moonlike['sdm'])))
        sep_b, pa_b = self.compute_separation(xyz[binary], orbital_params)

        return self._build_popu(reference_pop['Name'].to_numpy(), None, Hs,
                                binary_type, binary, sep_b, pa_b, dm_b,
                                orbital_params)

    def _build_popu(self, names, xyz, Hs, binary_type, binary, sep_b, pa_b,
                    dm_b, orbital_params):
        """
        Assemble the population DataFrame from per-row arrays plus
        per-binary arrays indexed by `binary` (row indices of the
        non-singles). sep/pa/dm/params are NaN for singles. x/y/z columns
        are included only if `xyz` is given.
        """
        n = len(names)
        sep = np.full(n, np.nan)
        pa = np.full(n, np.nan)
        dm = np.full(n, np.nan)
        sep[binary] = sep_b
        pa[binary] = pa_b
        dm[binary] = dm_b
        params = np.full(n, np.nan, dtype=object)
        for k, row in zip(binary, orbital_params.tolist()):
            params[k] = row
        cols = {'Name': names}
        if xyz is not None:
            cols.update(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
        cols.update(H=Hs, binary_type=binary_type, sep=sep, pa=pa, dm=dm,
                    params=params)
        return pd.DataFrame(cols)

    def _count_types(self):
        counts = self.popu['binary_type'].value_counts()
        self.single_pop = int(counts.get('single', 0))
        self.moonlike_pop = int(counts.get('moonlike', 0))
        self.wide_pop = int(counts.get('wide', 0))

    def compute_separation(self, xyz_earth, orbital_params):
        """
        Compute the sky-plane separation and position angle of TNO
        binaries, vectorized over n systems.

        Propagates orbital elements to a Cartesian offset (dx, dy, dz) at
        the reference epoch (see _orbit_offset), then projects onto the
        sky plane. Sky-projection logic adapted from multimoon mm_relast.

        Parameters
        ----------
        xyz_earth : array-like, shape (n, 3)
            Geocentric J2000 ecliptic position of each primary TNO in km.
        orbital_params : array-like, shape (n, 6), columns (a, e, i, w, Om, mu)
            a   [km]   semi-major axis
            e          eccentricity
            i   [deg]  inclination
            w   [deg]  argument of periapse
            Om  [deg]  longitude of ascending node
            mu  [deg]  mean anomaly at epoch

        Returns
        -------
        sep : ndarray, shape (n,)  sky-plane separation in arcsec
        pa  : ndarray, shape (n,)  position angle in degrees, North through East
        """
        obs_to_prim = np.asarray(xyz_earth, dtype=np.float64).reshape(-1, 3)
        orbital_params = np.asarray(orbital_params, dtype=np.float64).reshape(-1, 6)
        prim_to_sat = _orbit_offset(*orbital_params.T)

        dist = np.linalg.norm(obs_to_prim, axis=1)

        # Line-of-sight unit vector (Earth -> TNO)
        los = obs_to_prim / dist[:, None]

        # Project satellite offset onto sky plane
        prim_to_sat_sky = prim_to_sat - los * _rowdot(prim_to_sat, los)[:, None]

        # Angular separation (arcsec)
        sep = np.degrees(np.linalg.norm(prim_to_sat_sky, axis=1) / dist) * 3600.0

        # Sky-plane North: ecliptic north pole (0,0,1) projected
        # perpendicular to LOS; if LOS is nearly parallel to the ecliptic
        # pole, fall back to the x-axis as north.
        north = -los * los[:, 2:3]
        north[:, 2] += 1.0
        north_norm = np.linalg.norm(north, axis=1)
        polar = north_norm < 1e-10
        if np.any(polar):
            alt = -los[polar] * los[polar, 0:1]
            alt[:, 0] += 1.0
            north[polar] = alt
            north_norm[polar] = np.linalg.norm(alt, axis=1)
        north /= north_norm[:, None]

        # Sky-plane East: North x LOS gives increasing-ecliptic-longitude direction
        east = np.cross(north, los)
        east /= np.linalg.norm(east, axis=1)[:, None]

        delta_n = _rowdot(prim_to_sat_sky, north)
        delta_e = _rowdot(prim_to_sat_sky, east)
        pa = np.degrees(np.arctan2(delta_e, delta_n)) % 360.0

        return sep, pa

    def apply_detection(self, detection_prob_func):
        """
        Determine which simulated binaries are detectable.

        Calls an externally-supplied detection probability function on
        every binary at once and does a Bernoulli draw for each to decide
        if it would be detected. Singles are always marked undetected.

        Parameters
        ----------
        detection_prob_func : callable
            Signature: f(sep, dm) -> float in [0, 1] (or bool), with sep
            [arcsec] and dm [mag]. Called with arrays; a scalar-only
            function is looped over instead (see
            TNO_sim_lib.call_det_prob).

        Adds two columns to self.popu:
              detected      bool   True if the binary would be detected
              detect_prob   float  Probability returned by detection_prob_func
                                   (NaN for singles)
        """
        result = self.popu
        seps = result['sep'].to_numpy()
        dms = result['dm'].to_numpy()

        candidate = (result['binary_type'].to_numpy() != 'single') & ~np.isnan(seps)
        probs = np.full(len(result), np.nan)
        probs[candidate] = TNO_sim_lib.call_det_prob(
            detection_prob_func, seps[candidate], dms[candidate])
        detected = np.zeros(len(result), dtype=bool)
        detected[candidate] = np.random.random(candidate.sum()) < probs[candidate]

        result['detect_prob'] = probs
        result['detected'] = detected

class Pluto(Population):
    def __init__(self, reference_pop=None, runprops=None, detection_prob=None):
        self.gm_h0 = _gm_h0(runprops)
        self.moonlike = 0.9
        self.wide = 0.0
        self.ref_binary_frac = self.moonlike + self.wide
        self.popu = self.population_simulator(reference_pop)
        if detection_prob is not None:
            # Observe the reference through the same detection function
            # the model is scored with, then blank sep/pa/dm for the
            # binaries it misses -- the same convention as a real
            # observed csv, where an undetected companion is
            # indistinguishable from a single (voxel_counts drops NaN
            # rows). Without this, binaries the model can never detect
            # (e.g. sep < 0.1") counted as observed.
            self.apply_detection(detection_prob)
            missed = ~self.popu['detected']
            self.popu.loc[missed, ['sep', 'pa', 'dm']] = np.nan

        # Log population characteristics
        self.type = "Reference"
        self._count_types()
        self.sim_binary_frac = (self.moonlike_pop + self.wide_pop) / (self.moonlike_pop + self.wide_pop + self.single_pop)
        self.ll_count = "N/A"

    def population_simulator(self, reference_pop=None):
        AU_km = 1.496e8
        d_km = 34 * AU_km
        lon_rad = np.radians(298.0)
        lat_rad = np.radians(16.0)
        H = -0.44
        dm_mean = 1.9
        dm_std = 0.2
        scatter_km = 0.1 * AU_km
        x0 = d_km * np.cos(lat_rad) * np.cos(lon_rad)
        y0 = d_km * np.cos(lat_rad) * np.sin(lon_rad)
        z0 = d_km * np.sin(lat_rad)


        n = 500
        names = np.array([f'Pluto_{k}' for k in range(n)], dtype=object)
        xyz = np.array([x0, y0, z0]) + np.random.normal(0, scatter_km, (n, 3))
        Hs = np.full(n, H)

        # Check binary type
        binary_check = np.random.rand(n)
        is_single = binary_check > self.ref_binary_frac
        is_wide = ~is_single & (binary_check <= self.wide)
        binary_type = np.where(is_single, 'single',
                               np.where(is_wide, 'wide', 'moonlike'))

        # Random orientation per system -- lan/aop uniform, like the
        # model's own Om/w draws -- so this test population is reachable by
        # the model family (a single fixed lan gave every system the same
        # sky orientation, a PA structure the model's uniform Om can never
        # reproduce). Wide and moonlike share the same orbit and dm here.
        binary = np.flatnonzero(~is_single)
        nb = len(binary)
        lan = np.random.uniform(0.0, 360.0, nb)
        aop = np.random.uniform(0.0, 360.0, nb)
        mea = np.random.uniform(0.0, 360.0, nb)
        a, e, i = self._draw_aei(nb)
        orbital_params = np.column_stack([a, e, i, aop, lan, mea])
        dm_b = np.abs(np.random.normal(dm_mean, dm_std, nb))
        sep_b, pa_b = self.compute_separation(xyz[binary], orbital_params)

        # x/y/z/H recorded for singles too: Population() re-simulates
        # every reference row, so a single with NaN position could never
        # come out as a binary there.
        return self._build_popu(names, xyz, Hs, binary_type, binary, sep_b,
                                pa_b, dm_b, orbital_params)

    def _draw_aei(self, nb):
        """
        Semi-major axis [km], eccentricity, inclination [deg] for `nb`
        binaries: Pluto-Charon's values for every system, no spread.
        """
        return (np.full(nb, 19591.0), np.full(nb, 0.0002),
                np.full(nb, 96.2))


class PlutoSpread(Pluto):
    """
    Pluto reference with a real spread in a, e and i, drawn from the
    model's own distribution family (see the a/e/i expressions in
    runprops) with known true hyperparameters -- so a fit should recover a
    peaked posterior near them, instead of the zero-spread Pluto
    reference's runaway toward the param_bounds cap.

    Defaults (TRUTH below) keep the means near Pluto-Charon's:
      a = a_max * U^(1/ka):     mean ~17800 km, std ~2000 km
      e ~ Beta(ae, ke):         mean 0.17, std 0.10
      i = 180 * Beta(ai, ki):   mean 96.4 deg, std ~17 deg
    Only the spread in a is clearly detectable by the voxel likelihood:
    dropping it costs ~25 in log-likelihood at n_realizations=10, while
    dropping the e or i spread costs less than the noise (~3), even for
    much wider e/i spreads. Expect ka to give a peaked posterior here and
    ae/ke, ai/ki to still pin down only their means.
    Override any of them with a runprops "pluto_spread" dict, e.g.
    "pluto_spread": {"ka": 3, "a_max": 30000}. a_max should match the
    scale in runprops' "a" expression, or the model can't reach the truth.
    Everything else (positions, fb, dm, orientation, detection) is the
    same as Pluto.
    """
    TRUTH = {"ka": 8.0, "a_max": 20000.0, "ae": 2.0, "ke": 10.0,
             "ai": 15.0, "ki": 13.0}

    def __init__(self, reference_pop=None, runprops=None, detection_prob=None):
        self.truth = dict(self.TRUTH)
        self.truth.update((runprops or {}).get("pluto_spread", {}))
        super().__init__(reference_pop, runprops, detection_prob)
        self.type = "Reference (Pluto with a/e/i spread)"

    def _draw_aei(self, nb):
        t = self.truth
        a = t["a_max"] * np.random.power(t["ka"], nb)
        e = np.random.beta(t["ae"], t["ke"], nb)
        i = 180.0 * np.random.beta(t["ai"], t["ki"], nb)
        return a, e, i


# runprops "reference_pop" names for the built-in synthetic references;
# any other value is a csv filename.
SYNTHETIC_REFERENCES = {"Pluto": Pluto, "Pluto_spread": PlutoSpread}
