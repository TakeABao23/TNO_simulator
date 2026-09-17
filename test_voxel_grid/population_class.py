import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.stats import gaussian_kde, poisson
import TNO_sim_lib

class Population():
    def __init__(self, reference_pop, init_params=None, detection_prob = None, runprops=None):
        # Initial Parameters -- init_params/self.wide are numpy structured
        # scalars (see TNO_sim_lib.draw_moon_params/draw_wide_params), not
        # class instances; runprops carries the moon/wide/orbital draw
        # expressions population_simulator() needs per TNO.
        self.moonlike = init_params
        self.runprops = runprops
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
            Simulated population. Columns: Name, binary_type, sep, pa, dm, orbital_params.
            sep/pa/dm/orbital_params are NaN for single (non-binary) objects.
        """

        results = []
        for _, row in reference_pop.iterrows():
            name = row['Name']
            xyz_earth  = (row['x'], row['y'], row['z'])

            # Check binary type
            binary_check = np.random.rand()
            if binary_check > (self.ref_binary_frac):
                binary_type = 'single'
                results.append({'Name': name, 'binary_type': binary_type, 'sep': np.nan, 'pa': np.nan, 'dm': np.nan, 'H': row['H'], 'params': np.nan})
                continue
            elif binary_check > self.wide['fb']:
                binary_type = 'moonlike'
                orb_names = self.runprops["orb_param_names"]
                orb = TNO_sim_lib.draw_orbit_params(self.moonlike, self.runprops)
                orbital_params = [orb[name] for name in orb_names]
                sep, pa = self.compute_separation(xyz_earth, orbital_params)
                dm = abs(np.random.normal(self.moonlike['mdm'], self.moonlike['sdm']))
            else:
                binary_type = 'wide'
                # Wide binaries reuse the moonlike orbital-element
                # distribution (not one derived from self.wide, which is
                # normally inert -- see draw_wide_params), matching the
                # original code's self.worb = self.morb.
                orb_names = self.runprops["orb_param_names"]
                orb = TNO_sim_lib.draw_orbit_params(self.moonlike, self.runprops)
                orbital_params = [orb[name] for name in orb_names]
                sep, pa = self.compute_separation(xyz_earth, orbital_params)
                dm = abs(np.random.normal(0.0, self.wide['sdm']))

            results.append({'Name': name, 'binary_type': binary_type, 'sep': sep, 'pa': pa, 'dm': dm, 'H': row['H'], 'params': orbital_params})
        return pd.DataFrame(results)

    def compute_separation(self, xyz_earth, orbital_params):
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

        Returns
        -------
        sep : float  sky-plane separation in arcsec
        pa  : float  position angle in degrees, measured North through East
        """

        a, e, i, w, Om, mu = orbital_params

        i_rad   = np.radians(i)
        w_rad   = np.radians(w)
        Om_rad  = np.radians(Om)
        mu_rad = np.radians(mu)

        # mu_grav only affects period, not position at t=T0, so value is arbitrary here
        G        = 6.674e-20   # km^3 kg^-1 s^-2
        mu_grav  = G * 1.0e18  # representative TNO system mass in kg

        rp   = a * (1.0 - e)   # perifocal distance (km)
        elts = [rp, e, i_rad, Om_rad, w_rad, mu_rad, 0.0, mu_grav]
        state = spice.conics(elts, 0.0)
        prim_to_sat = np.array(state[:3])  # (dx, dy, dz) in km

        obs_to_prim = np.array(xyz_earth, dtype=float)
        dist = np.linalg.norm(obs_to_prim)

        # Line-of-sight unit vector (Earth → TNO)
        los = obs_to_prim / dist

        # Project satellite offset onto sky plane
        prim_to_sat_sky = prim_to_sat - np.dot(prim_to_sat, los) * los

        # Angular separation (arcsec)
        sep_rad = np.linalg.norm(prim_to_sat_sky) / dist
        sep = np.degrees(sep_rad) * 3600.0

        # Sky-plane North: ecliptic north pole (0,0,1) projected perpendicular to LOS
        ecliptic_north = np.array([0.0, 0.0, 1.0])
        north = ecliptic_north - np.dot(ecliptic_north, los) * los
        north_norm = np.linalg.norm(north)
        if north_norm < 1e-10:
            # LOS nearly parallel to ecliptic pole; fall back to x-axis as north
            north = np.array([1.0, 0.0, 0.0]) - los[0] * los
            north /= np.linalg.norm(north)
        else:
            north /= north_norm

        # Sky-plane East: North × LOS gives increasing-ecliptic-longitude direction
        east = np.cross(north, los)
        east /= np.linalg.norm(east)

        delta_n = np.dot(prim_to_sat_sky, north)
        delta_e = np.dot(prim_to_sat_sky, east)
        pa = np.degrees(np.arctan2(delta_e, delta_n)) % 360.0

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

        probs    = []
        detected = []

        for _, row in result.iterrows():
            if row['binary_type'] == 'single' or np.isnan(row['sep']):
                probs.append(np.nan)
                detected.append(False)
            else:
                p = detection_prob_func(row['sep'], row['dm'])
                probs.append(p)
                detected.append(np.random.random() < p)

        result['detect_prob'] = probs
        result['detected']    = detected

class Pluto(Population):
    def __init__(self, reference_pop=None):
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
        dm = 1.9
        dm_std = 0.05
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
                results.append({'Name': name, 'binary_type': binary_type, 'sep': np.nan, 'pa': np.nan, 'dm': np.nan, 'H': H, 'params': np.nan})
                continue
            elif binary_check > self.wide:
                binary_type = 'moonlike'
                mea = np.random.uniform(0.0, 360.0)
                orbital_params = (a, e, i, aop, lan, mea)
                sep, pa = self.compute_separation(xyz_earth, orbital_params)
                dm = abs(np.random.normal(dm, dm_std))
            else:
                binary_type = 'wide'
                mea = np.random.uniform(0.0, 360.0)
                orbital_params = (a, e, i, aop, lan, mea)
                sep, pa = self.compute_separation(xyz_earth, orbital_params)
                dm = abs(np.random.normal(dm, dm_std))

            results.append({'Name': name, 'x': x, 'y': y, 'z': z, 'H': H, 'binary_type': binary_type, 'sep': sep, 'pa': pa, 'dm': dm, 'params': orbital_params})
        return pd.DataFrame(results)
