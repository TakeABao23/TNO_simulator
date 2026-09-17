"""
Parameters version 1.0
"""

import numpy as np
rng = np.random.default_rng()

class InitMoonParams():
    """
    Set of population-level parameters to seed emcee walkers for moonlike binaries.

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
    fb  = rng.uniform(0.0, 1.0)
    ka  = rng.uniform(0.1, 5.0)
    ae  = rng.uniform(0.5, 5.0)
    ke  = rng.uniform(0.5, 5.0)
    ai  = 2
    ki  = 5
    mdm = rng.uniform(0.0, 4.0)
    sdm = rng.uniform(0.1, 2.0)


class InitWideParams():
    """
    Set of population-level parameters to seed emcee walkers for wide binaries.
    Currently doing nothing
    """
    use = None
    fb  = 0
    ka  = None
    ae  = None
    ke  = None
    ai  = None
    ki  = None
    mdm = None
    sdm = None


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
    """
    def __init__(self, params):
        self.a  = rng.power(params.ka) * 10000
        e  = rng.beta(params.ae, params.ke)
        i  = 180.0 * rng.beta(params.ai, params.ki)
        w  = rng.uniform(0.0, 360.0)
        Om = rng.uniform(0.0, 360.0)
        mu = rng.uniform(0.0, 360.0)