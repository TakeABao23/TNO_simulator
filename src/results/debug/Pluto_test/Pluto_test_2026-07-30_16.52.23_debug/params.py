"""
Frozen parameter snapshot -- InitMoonParams below is fixed to the
last emcee posterior sample from this run, instead of drawing a
random walker start. Generated from Parameters/params_1.0.py.
"""

import numpy as np
rng = np.random.default_rng()


def det_prob(sep, dm):
    if sep < 0.1:
        return False
    elif sep < 0.5:
        if dm > 12.5 * sep - 1.25:
            return False
    elif dm > 5:
        return False
    return True

    
# Canonical order of the free moonlike parameters. This order is what
# InitMoonParams.to_array()/moon_params_from_array() use to translate between
# the dict form (used by Population/OrbitParamDist) and the flat numpy vector
# that emcee requires (walker positions, theta passed to the log-prob function).
MOON_PARAM_NAMES = ['fb', 'ka', 'ae', 'ke', 'ai', 'ki', 'mdm', 'sdm']


class InitMoonParams():
    """
    Set of population-level parameters to seed emcee walkers for moonlike binaries.
    Each instantiation draws a fresh random set (needed so that every emcee
    walker starts from a distinct position).

    | Function | Parameter | Range | Description |
|---|---|---|---|
| `get_fb()` | `fb` | U(0, 1) | binary fraction |
| `get_ka()` | `ka` | U(0.1, 5) | power-law shape for semi-major axis |
| `get_ae()` | `ae` | U(0.5, 5) | Beta α for eccentricity |
| `get_ke()` | `ke` | U(0.5, 5) | Beta β for eccentricity |
| `get_ai()` | `ai` | U(0.5, 5) | Beta α for inclination |
| `get_ki()` | `ki` | U(0.5, 5) | Beta β for inclination |
| `get_mdm()` | `mdm` | U(0, 4) | mean delta-magnitude |
| `get_sdm()` | `sdm` | U(0.1, 2) | std of delta-magnitude distribution |
    """
    def __init__(self):
        self.fb  = 0.9784680112074126
        self.ka  = 3.12736716180973
        self.ae  = 5.05298096052564
        self.ke  = 4.374845153944438
        self.ai  = 1.9487814535952794
        self.ki  = 4.939591845499558
        self.mdm = 0.7451855321263925
        self.sdm = 0.38365537696343677
        self.moon_params = {'fb': self.fb, 'ka': self.ka, 'ae': self.ae, 'ke': self.ke,
                             'ai': self.ai, 'ki': self.ki, 'mdm': self.mdm, 'sdm': self.sdm}

    def to_array(self):
        """Flat numpy array in MOON_PARAM_NAMES order, e.g. for an emcee walker position."""
        return np.array([self.moon_params[name] for name in MOON_PARAM_NAMES])


def moon_params_from_array(theta):
    """Inverse of InitMoonParams.to_array(): flat array/sequence -> dict keyed by MOON_PARAM_NAMES."""
    return dict(zip(MOON_PARAM_NAMES, theta))


# Support of the uniform prior on each moonlike parameter. Only the bounds
# that are numerically required are enforced: fb is a fraction (must stay in
# [0, 1]); ka/ae/ke/ai/ki feed rng.power/rng.beta, which raise ValueError on
# a non-positive shape parameter; sdm is a stddev, so must stay positive.
# There is deliberately no upper cap on the shape parameters or on mdm -- an
# earlier version capped ka/ae/ke/ai/ki at 5.0 to match InitMoonParams'
# walker-seeding range, but ki is *seeded* from U(4.9, 5.1), so ~half of all
# walkers started outside that cap and were rejected from step 0 on. A hard
# upper bound also risks truncating the real posterior if the true best-fit
# value happens to exceed it.
PARAM_BOUNDS = {
    'fb':  (0.0, 1.0),
    'ka':  (0.1, np.inf),
    'ae':  (0.5, np.inf),
    'ke':  (0.5, np.inf),
    'ai':  (0.5, np.inf),
    'ki':  (0.5, np.inf),
    'mdm': (-np.inf, np.inf),
    'sdm': (0.1, np.inf),
}


def log_prior(theta):
    """
    Uniform prior over PARAM_BOUNDS, in MOON_PARAM_NAMES order.
    Returns 0.0 if every parameter is within bounds, -inf otherwise.
    """
    params = moon_params_from_array(theta)
    for name, value in params.items():
        lo, hi = PARAM_BOUNDS[name]
        if not (lo <= value <= hi):
            return -np.inf
    return 0.0


class InitWideParams():
    """
    Set of population-level parameters to seed emcee walkers for wide binaries.
    Currently doing nothing
    """
    def __init__(self):
        self.use = None
        self.fb  = 0
        self.ka  = None
        self.ae  = None
        self.ke  = None
        self.ai  = None
        self.ki  = None
        self.mdm = None
        self.sdm = None
        self.wide_params = {'use': self.use, 'fb': self.fb, 'ka': self.ka, 'ae': self.ae,
                             'ke': self.ke, 'ai': self.ai, 'ki': self.ki, 'mdm': self.mdm, 'sdm': self.sdm}

class OrbitParamDist():
    """
    Probability distributions of Orbital Parameters
    (a, e, i, w, Om, mu)
      a   [km, 0–a_scale_km]  semi-major axis,            Power(ka) · a_scale_km
      e   [0, 1)              eccentricity,                Beta(ae, ke)
      i   [deg, 0–180)        inclination,                 180 · Beta(ai, ki)
      w   [deg, 0–360)        argument of periapse,        U(0, 360)
      Om  [deg, 0–360)        longitude of ascending node, U(0, 360)
      mu  [deg, 0–360)        mean anomaly at epoch,       U(0, 360)

      input: params (dict)
    """
    def __init__(self, params):
        self.a_fn  = lambda: rng.power(params['ka']) * 10000
        self.e_fn  = lambda: rng.beta(params['ae'], params['ke'])
        self.i_fn  = lambda: 180.0 * rng.beta(params['ai'], params['ki'])
        self.w_fn  = lambda: rng.uniform(0.0, 360.0)
        self.Om_fn = lambda: rng.uniform(0.0, 360.0)
        self.mu_fn = lambda: rng.uniform(0.0, 360.0)

    @property
    def a(self):  return self.a_fn()

    @property
    def e(self):  return self.e_fn()

    @property
    def i(self):  return self.i_fn()

    @property
    def w(self):  return self.w_fn()

    @property
    def Om(self): return self.Om_fn()

    @property
    def mu(self): return self.mu_fn()