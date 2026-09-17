import numpy as np
import matplotlib.pyplot as plt

PARAM_UNITS = {
    'a': 'km',
    'e': '',
    'i': 'deg',
    'w': 'deg',
    'Om': 'deg',
    'mu': 'deg',
}

def plot_sep_vs_dm(simpop, observed=None):
    """
    Plot sky-plane separation vs delta magnitude for the simulated population.

    If `simulated` contains a 'detected' column (from apply_detection),
    detected and undetected binaries are shown in different colors.

    Parameters
    ----------
    simulated : pd.DataFrame
        Output of simulate_population() or apply_detection().
        Must have columns: sep, dm. Optionally: detected.
    observed : pd.DataFrame, optional
        Real observed binaries to overlay. Must have columns: sep, dm.
    """
    simulated = simpop.popu
    binaries = simulated.dropna(subset=['sep', 'dm'])

    fig, ax = plt.subplots(figsize=(7, 5))

    # shades detected binaries in different color from undetected
    if 'detected' in binaries.columns:
        show_detection(binaries, ax)
        
    # if detection not applied, just graph them
    else:
        ax.scatter(binaries['sep'], binaries['dm'],
                   s=20, alpha=0.6, label='simulated')
        
    # actual data to compare against
    if observed is not None:
        ax.scatter(observed['sep'], observed['dm'],
                   s=40, marker='*', color='red', zorder=5, label='observed')

    ax.set_xlabel('Separation (arcsec)')
    ax.set_ylabel(r'$\Delta m$ (mag)')
    ax.set_title('Separation vs. Delta Magnitude')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_sep_vs_dm_comparison(named_pops, observed=None):
    """
    Plot separation vs. delta magnitude for several simulated populations
    side by side, sharing axes so they're easy to compare (e.g. the first
    vs. last posterior sample of an emcee chain).

    Parameters
    ----------
    named_pops : sequence of (label, simpop)
        Each simpop is a Population-like object (has a `.popu` DataFrame
        with sep, dm, and optionally detected). label is used as that
        subplot's title.
    observed : pd.DataFrame, optional
        Real observed binaries to overlay on every subplot. Must have
        columns: sep, dm.
    """
    fig, axes = plt.subplots(1, len(named_pops), figsize=(7 * len(named_pops), 5),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (label, simpop) in zip(axes, named_pops):
        binaries = simpop.popu.dropna(subset=['sep', 'dm'])

        if 'detected' in binaries.columns:
            show_detection(binaries, ax)
        else:
            ax.scatter(binaries['sep'], binaries['dm'],
                       s=20, alpha=0.6, label='simulated')

        if observed is not None:
            ax.scatter(observed['sep'], observed['dm'],
                       s=40, marker='*', color='red', zorder=5, label='observed')

        ax.set_xlabel('Separation (arcsec)')
        ax.set_title(label)
        ax.legend()

    axes[0].set_ylabel(r'$\Delta m$ (mag)')
    fig.suptitle('Separation vs. Delta Magnitude')
    plt.tight_layout()
    plt.show()
    return fig

def plot_pa_vs_sep_comparison(named_pops, observed=None):
    """
    Plot position angle vs. separation on polar projections for several
    simulated populations side by side, sharing a color scale so they're
    easy to compare (e.g. the first vs. last posterior sample of an emcee
    chain).

    Parameters
    ----------
    named_pops : sequence of (label, simpop)
        Each simpop is a Population-like object (has a `.popu` DataFrame
        with pa, sep, dm). label is used as that subplot's title.
    observed : pd.DataFrame, optional
        Real observed binaries to overlay on every subplot. Must have
        columns: pa, sep, dm.
    """
    binaries_per_pop = [simpop.popu.dropna(subset=['pa', 'sep', 'dm'])
                        for _, simpop in named_pops]

    dm_values = [binaries['dm'] for binaries in binaries_per_pop]
    if observed is not None:
        observed = observed.dropna(subset=['pa', 'sep', 'dm'])
        dm_values.append(observed['dm'])
    vmin = min(dm.min() for dm in dm_values)
    vmax = max(dm.max() for dm in dm_values)

    fig = plt.figure(figsize=(7 * len(named_pops), 7))
    sc = None
    for idx, ((label, _), binaries) in enumerate(zip(named_pops, binaries_per_pop)):
        ax = fig.add_subplot(1, len(named_pops), idx + 1, projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        theta = np.radians(binaries['pa'])
        r = binaries['sep']
        sc = ax.scatter(theta, r, c=binaries['dm'], cmap='viridis',
                         vmin=vmin, vmax=vmax, s=30, alpha=0.8, label='simulated')

        if observed is not None:
            obs_theta = np.radians(observed['pa'])
            obs_r = observed['sep']
            ax.scatter(obs_theta, obs_r, c=observed['dm'], cmap='viridis',
                       vmin=vmin, vmax=vmax, s=80, marker='*',
                       edgecolors='red', linewidths=1.2, zorder=5,
                       label='observed')
            ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.1))

        ax.set_title(label)

    fig.colorbar(sc, ax=fig.get_axes(), label=r'$\Delta m$ (mag)')
    fig.suptitle('Position Angle vs. Separation')
    plt.show()
    return fig

def plot_pa_vs_sep(simpop, observed=None):
    """
    Plot position angle vs. separation on a polar projection, colored
    by delta magnitude.

    Parameters
    ----------
    simpop : Population
        Object with a `.popu` DataFrame containing columns: pa, sep, dm, orbit_params.
    observed : pd.DataFrame, optional
        Real observed binaries to overlay. Must have columns: pa, sep, dm.
    """
    simulated = simpop.popu
    binaries = simulated.dropna(subset=['pa', 'sep', 'dm'])

    theta = np.radians(binaries['pa'])
    r = binaries['sep']

    # shared color scale so both scatters map dm the same way
    vmin, vmax = binaries['dm'].min(), binaries['dm'].max()
    if observed is not None:
        observed = observed.dropna(subset=['pa', 'sep', 'dm'])
        vmin = min(vmin, observed['dm'].min())
        vmax = max(vmax, observed['dm'].max())

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    sc = ax.scatter(theta, r, c=binaries['dm'], cmap='viridis',
                     vmin=vmin, vmax=vmax, s=30, alpha=0.8,
                     label='simulated')

    if observed is not None:
        obs_theta = np.radians(observed['pa'])
        obs_r = observed['sep']
        ax.scatter(obs_theta, obs_r, c=observed['dm'], cmap='viridis',
                   vmin=vmin, vmax=vmax, s=80, marker='*',
                   edgecolors='red', linewidths=1.2, zorder=5,
                   label='observed')
        ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.1))

    fig.colorbar(sc, ax=ax, label=r'$\Delta m$ (mag)')
    ax.set_title('Position Angle vs. Separation')
    plt.tight_layout()
    plt.show()

def show_detection(binaries, ax):
    det   = binaries[binaries['detected']]
    undet = binaries[~binaries['detected']]
    ax.scatter(undet['sep'], undet['dm'],
               s=20, alpha=0.4, color='steelblue', label='undetected')
    ax.scatter(det['sep'], det['dm'],
               s=20, alpha=0.7, color='darkorange', label='detected')
    
def plot_element(distribution, data, name, n_samples=500):
    """
    Plot the analytic parameter distribution (params.py) against the
    simulated population's distribution for the same parameter.

    Parameters
    ----------
    distribution : callable
        Zero-argument sampler for the parameter, e.g. OrbitParamDist.a_fn.
        Called `n_samples` times to build the analytic distribution.
    data : array-like (1D)
        Simulated values of the matching parameter, one per object.
    name : str
        Parameter name, used as the plot title.
    n_samples : int, optional
        Number of draws taken from `distribution` (default 10000).
    """
    dist_samples = np.array([distribution() for _ in range(n_samples)])

    sim_samples = np.asarray(data, dtype=float)
    sim_samples = sim_samples[~np.isnan(sim_samples)]

    units = PARAM_UNITS.get(name, '')
    xlabel = f'{name} ({units})' if units else name

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dist_samples, bins=40, density=True, alpha=0.5, label='distribution')
    ax.hist(sim_samples, bins=40, density=True, alpha=0.5, label='simulation')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('probability density')
    ax.set_title(name)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_orbits(population):
    param_names = ['a', 'e', 'i', 'w', 'Om', 'mu']
    orbital_rows = [row for row in population.popu['params']
                     if isinstance(row, (list, tuple))]

    for idx, name in enumerate(param_names):
        distribution = getattr(population.morb, f'{name}_fn')
        data = [row[idx] for row in orbital_rows]
        plot_element(distribution, data, name)
