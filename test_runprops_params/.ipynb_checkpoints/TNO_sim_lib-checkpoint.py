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

def moon_params_from_array(theta):
    """Inverse of InitMoonParams.to_array(): flat array/sequence -> dict keyed by MOON_PARAM_NAMES."""
    return dict(zip(MOON_PARAM_NAMES, theta))