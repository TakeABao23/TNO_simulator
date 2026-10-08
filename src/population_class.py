import math

import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.stats import gaussian_kde, poisson
import TNO_sim_lib as TNO_sim_lib


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm3(a):
    return math.sqrt(_dot3(a, a))


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale3(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


_G = 6.674e-20  # km^3 kg^-1 s^-2


def _gm_h0(runprops):
    """
    GM [km^3/s^2] of an H = 0 sphere, from runprops' albedo and density.

    D [km] = 1329 / sqrt(albedo) * 10^(-H/5), so
    GM(H) = G * rho * (pi/6) * D^3 = _gm_h0(runprops) * 10^(-0.6 H).
    Falls back to albedo 0.1, density 1 g/cm^3 if runprops is None or
    predates these keys.
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
        self.single_pop = self.popu.query("binary_type == \'single\'").shape[0]
        self.moonlike_pop = self.popu.query("binary_type == \'moonlike\'").shape[0]
        self.wide_pop = self.popu.query("binary_type == \'wide\'").shape[0]
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

        # Plain positional array indexing instead of .iterrows(): iterrows()
        # builds a full pandas Series (with its own dtype-unification and
        # index-alignment overhead) for every single row, which profiling
        # showed as a dominant cost when this loop runs on every emcee
        # likelihood evaluation, for every reference TNO.
        names = reference_pop['Name'].to_numpy()
        xs = reference_pop['x'].to_numpy()
        ys = reference_pop['y'].to_numpy()
        zs = reference_pop['z'].to_numpy()
        Hs = reference_pop['H'].to_numpy()

        results = []
        for idx in range(len(reference_pop)):
            name = names[idx]
            H = Hs[idx]
            xyz_earth  = (xs[idx], ys[idx], zs[idx])

            # Check binary type
            binary_check = np.random.rand()
            if binary_check > (self.ref_binary_frac):
                binary_type = 'single'
                results.append({'Name': name, 'H': H, 'binary_type': binary_type, 'sep': np.nan, 'pa': np.nan, 'dm': np.nan, 'params': np.nan})
                continue
            elif binary_check > self.wide['fb']:
                binary_type = 'moonlike'
                orb_names = self.runprops["orb_param_names"]
                orb = TNO_sim_lib.draw_orbit_params(self.moonlike, self.runprops)
                orbital_params = [orb[name] for name in orb_names]
                dm = abs(np.random.normal(self.moonlike['mdm'], self.moonlike['sdm']))
                sep, pa = self.compute_separation(xyz_earth, orbital_params, H, dm)
            else:
                binary_type = 'wide'
                # Wide binaries reuse the moonlike orbital-element
                # distribution (not one derived from self.wide, which is
                # normally inert -- see draw_wide_params), matching the
                # original code's self.worb = self.morb.
                orb_names = self.runprops["orb_param_names"]
                orb = TNO_sim_lib.draw_orbit_params(self.moonlike, self.runprops)
                orbital_params = [orb[name] for name in orb_names]
                dm = abs(np.random.normal(0.0, self.wide['sdm']))
                sep, pa = self.compute_separation(xyz_earth, orbital_params, H, dm)

            results.append({'Name': name, 'H': H, 'binary_type': binary_type, 'sep': sep, 'pa': pa, 'dm': dm, 'params': orbital_params})
        return pd.DataFrame(results)

    def compute_separation(self, xyz_earth, orbital_params, H, dm):
        """
        Compute the sky-plane separation and position angle of a TNO binary.

        Uses spiceypy conics to propagate orbital elements to a Cartesian offset
        vector (dx, dy, dz) at the reference epoch, then projects onto the sky
        plane. Sky-projection logic adapted from multimoon mm_relast.

        Parameters
        ----------
        orbital_params : tuple (a, e, i, w, Om, mu)
            a   [km]   semi-major axis
            e          eccentricity
            i   [deg]  inclination
            w   [deg]  argument of periapse
            Om  [deg]  longitude of ascending node
            mu  [deg]  mean anomaly at epoch
        xyz_earth : array-like, shape (3,)
            Geocentric J2000 ecliptic position of the primary TNO in km.
        H : float
            Absolute magnitude of the primary.
        dm : float
            Secondary-minus-primary magnitude difference. With H, sets the
            system mass (primary + secondary, both assumed to share
            runprops' albedo and density; see _gm_h0).

        Returns
        -------
        sep : float  sky-plane separation in arcsec
        pa  : float  position angle in degrees, measured North through East
        """

        a, e, i, w, Om, mu = orbital_params

        i_rad   = math.radians(i)
        w_rad   = math.radians(w)
        Om_rad  = math.radians(Om)
        mu_rad = math.radians(mu)

        # mu_grav only affects period, not position at t=T0
        # Secondary has H + dm, so its mass is the primary's * 10^(-0.6 dm)
        mu_grav = self.gm_h0 * 10.0 ** (-0.6 * H) * (1.0 + 10.0 ** (-0.6 * dm))

        rp   = a * (1.0 - e)   # perifocal distance (km)
        elts = [rp, e, i_rad, Om_rad, w_rad, mu_rad, 0.0, mu_grav]
        state = spice.conics(elts, 0.0)
        prim_to_sat = (state[0], state[1], state[2])  # (dx, dy, dz) in km

        # Plain-tuple/math arithmetic below instead of numpy calls (np.dot/
        # np.cross/np.linalg.norm on 3-element vectors): numpy's ufunc
        # dispatch overhead vastly exceeds the ~3-9 FLOPs these actually
        # need, and this runs once per binary on every likelihood
        # evaluation. Verified bit-for-bit (to float round-off) against the
        # numpy version across 200k random orbital elements/positions,
        # including the near-ecliptic-pole fallback branch below.
        obs_to_prim = (float(xyz_earth[0]), float(xyz_earth[1]), float(xyz_earth[2]))
        dist = _norm3(obs_to_prim)

        # Line-of-sight unit vector (Earth → TNO)
        los = _scale3(obs_to_prim, 1.0 / dist)

        # Project satellite offset onto sky plane
        prim_to_sat_sky = _sub3(prim_to_sat, _scale3(los, _dot3(prim_to_sat, los)))

        # Angular separation (arcsec)
        sep_rad = _norm3(prim_to_sat_sky) / dist
        sep = math.degrees(sep_rad) * 3600.0

        # Sky-plane North: ecliptic north pole (0,0,1) projected perpendicular to LOS
        ecliptic_north = (0.0, 0.0, 1.0)
        north = _sub3(ecliptic_north, _scale3(los, _dot3(ecliptic_north, los)))
        north_norm = _norm3(north)
        if north_norm < 1e-10:
            # LOS nearly parallel to ecliptic pole; fall back to x-axis as north
            north = _sub3((1.0, 0.0, 0.0), _scale3(los, los[0]))
            north = _scale3(north, 1.0 / _norm3(north))
        else:
            north = _scale3(north, 1.0 / north_norm)

        # Sky-plane East: North × LOS gives increasing-ecliptic-longitude direction
        east = _cross3(north, los)
        east = _scale3(east, 1.0 / _norm3(east))

        delta_n = _dot3(prim_to_sat_sky, north)
        delta_e = _dot3(prim_to_sat_sky, east)
        pa = math.degrees(math.atan2(delta_e, delta_n)) % 360.0

        return sep, pa

    def apply_detection(self, detection_prob_func):
        """
        WORKING
        Determine which simulated binaries are detectable.

        For each binary, calls an externally-supplied detection probability
        function and does a Bernoulli draw to decide if it would be detected.
        Singles are always marked undetected.

        Parameters
        ----------
        simulated : pd.DataFrame
            Output of simulate_population(). Must have columns:
            binary_type, sep [arcsec], dm [mag].
        detection_prob_func : callable
            Signature: f(sep, dm) -> float in [0, 1].
            Returns the probability of detecting a binary at this separation
            and delta magnitude. Defined externally (e.g. from HST survey limits).

        Returns
        -------
        pd.DataFrame
            Copy of `simulated` with two added columns:
              detected      bool   True if the binary would be detected
              detect_prob   float  Probability returned by detection_prob_func
                                   (NaN for singles)
        """
        result = self.popu

        # Plain positional array indexing instead of .iterrows() -- see
        # population_simulator()'s equivalent change for why.
        binary_types = result['binary_type'].to_numpy()
        seps = result['sep'].to_numpy()
        dms = result['dm'].to_numpy()

        probs    = []
        detected = []

        for i in range(len(result)):
            if binary_types[i] == 'single' or np.isnan(seps[i]):
                probs.append(np.nan)
                detected.append(False)
            else:
                p = detection_prob_func(seps[i], dms[i])
                probs.append(p)
                detected.append(np.random.random() < p)

        result['detect_prob'] = probs
        result['detected']    = detected

class Pluto(Population):
    def __init__(self, reference_pop=None, runprops=None):
        self.gm_h0 = _gm_h0(runprops)
        self.moonlike = 0.9
        self.wide = 0.0
        self.ref_binary_frac = self.moonlike + self.wide
        self.popu = self.population_simulator(reference_pop)

        # Log population characteristics
        self.type = "Reference"
        self.single_pop = self.popu.query("binary_type == \'single\'").shape[0]
        self.moonlike_pop = self.popu.query("binary_type == \'moonlike\'").shape[0]
        self.wide_pop = self.popu.query("binary_type == \'wide\'").shape[0]
        self.sim_binary_frac = (self.moonlike_pop + self.wide_pop) / (self.moonlike_pop + self.wide_pop + self.single_pop)
        self.ll_count = "N/A"

    def population_simulator(self, reference_pop=None):
        AU_km = 1.496e8
        d_km = 34 * AU_km
        lon_rad = np.radians(298.0)
        lat_rad = np.radians(16.0)
        H = -0.44
        a = 19591.0 #km
        e = 0.0002
        i = 96.2 # deg
        lan = 223.1 # deg
        aop = 0.0 # deg
        dm_mean = 1.9
        dm_std = 0.2
        scatter_km = 0.1 * AU_km
        x0 = d_km * np.cos(lat_rad) * np.cos(lon_rad)
        y0 = d_km * np.cos(lat_rad) * np.sin(lon_rad)
        z0 = d_km * np.sin(lat_rad)


        results = []
        for i in range(0, 500):
            name = f'Pluto_{i}'

            x = x0 + np.random.normal(0, scatter_km)
            y = y0 + np.random.normal(0, scatter_km)
            z = z0 + np.random.normal(0, scatter_km)
            xyz_earth  = (x, y, z)
            scatter_km = 0.1 * AU_km

            # Check binary type
            binary_check = np.random.rand()
            if binary_check > (self.ref_binary_frac):
                binary_type = 'single'
                results.append({'Name': name, 'binary_type': binary_type, 'sep': np.nan, 'pa': np.nan, 'dm': np.nan, 'params': np.nan})
                continue
            elif binary_check > self.wide:
                binary_type = 'moonlike'
                mea = np.random.uniform(0.0, 360.0)
                orbital_params = (a, e, i, aop, lan, mea)
                dm = abs(np.random.normal(dm_mean, dm_std))
                sep, pa = self.compute_separation(xyz_earth, orbital_params, H, dm)
            else:
                binary_type = 'wide'
                mea = np.random.uniform(0.0, 360.0)
                orbital_params = (a, e, i, aop, lan, mea)
                dm = abs(np.random.normal(dm_mean, dm_std))
                sep, pa = self.compute_separation(xyz_earth, orbital_params, H, dm)

            results.append({'Name': name, 'x': x, 'y': y, 'z': z, 'H': H, 'binary_type': binary_type, 'sep': sep, 'pa': pa, 'dm': dm, 'params': orbital_params})
        return pd.DataFrame(results)
