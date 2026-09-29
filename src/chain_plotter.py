"""
emcee chain diagnostics: trace, corner, and parameter-vs-likelihood plots.

Adapts the diagnostic plots multimoon's mm_plots_multi.plots() makes for
its emcee chains (walker traces, a corner.corner posterior plot, and
parameter-vs-log-likelihood panels) to TNO_simulator's much smaller
8-parameter moon-like model, which has no coordinate transforms or
derived-parameter bookkeeping to undo.

Two ways to get a chain in:
  - sampler_to_chain(sampler): straight off a freshly-run emcee sampler
    (e.g. what emcee_walker.emcee_walker() returns), before posteriors.csv
    is even written.
  - load_chain_from_csv(posteriors_path): reshapes a posteriors.csv
    written by run_emcee_walker.py back into (nsteps, nwalkers, ndim), for
    post-hoc plotting (e.g. from plot_posterior_result.py).

plot_trace(), plot_corner(), and plot_likelihood() all take the resulting
chain array (plus a per-step/per-walker metric array for the latter two),
so the same plotting code runs either way.
"""
import os

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_chain_from_csv(posteriors_path):
    """
    Load a posteriors.csv written by run_emcee_walker.py (one row per
    (step, walker) of the post-burn-in chain, columns: step, ll_count,
    then one column per moon_param_names entry) and reshape it back into
    the (nsteps, nwalkers, ndim) layout emcee's sampler.get_chain() uses.

    Recovers nwalkers from how many rows share the first step -- plot_and_write()
    in run_emcee_walker.py writes rows in step-major order (chain.reshape(-1,
    ndim) on a (nsteps, nwalkers, ndim) array), so this is the exact
    inverse of that reshape.

    Returns (chain, ll_chain, param_names):
        chain : ndarray, shape (nsteps, nwalkers, ndim)
        ll_chain : ndarray, shape (nsteps, nwalkers) -- the ll_count blob
            recorded for each sample. Note this is only the likelihood
            term, not the full log-posterior (log_prior isn't recorded in
            posteriors.csv) -- prefer sampler_to_chain()'s log_prob_chain
            when a live sampler is available.
        param_names : list of str, in posteriors.csv column order
    """
    df = pd.read_csv(posteriors_path)
    param_names = [c for c in df.columns if c not in ("step", "ll_count")]

    nwalkers = int((df["step"] == df["step"].iloc[0]).sum())
    nrows = len(df)
    if nwalkers == 0 or nrows % nwalkers != 0:
        raise ValueError(
            f"{posteriors_path}: {nrows} rows doesn't divide evenly by "
            f"the inferred {nwalkers} walkers (rows sharing the first "
            "step) -- posteriors.csv may be truncated or malformed."
        )
    nsteps = nrows // nwalkers

    chain = df[param_names].to_numpy().reshape(nsteps, nwalkers, len(param_names))
    ll_chain = df["ll_count"].to_numpy().reshape(nsteps, nwalkers)
    return chain, ll_chain, param_names


def sampler_to_chain(sampler):
    """
    Pull the (nsteps, nwalkers, ndim) chain and (nsteps, nwalkers)
    log-probability array straight off a freshly-run emcee sampler (e.g.
    the one emcee_walker.emcee_walker() returns), for plotting before
    posteriors.csv is even written.

    Unlike load_chain_from_csv()'s ll_chain (the ll_count blob only), this
    uses get_log_prob(), the actual log_prior + log-likelihood value emcee
    sampled on -- prefer this path when a live sampler is available.

    Returns (chain, log_prob_chain).
    """
    return sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)


def _save(fig, results_folder, filename, label):
    if results_folder is not None:
        path = os.path.join(results_folder, filename)
        fig.savefig(path, dpi=150)
        print(f"{label} saved to {path}")


def plot_trace(chain, param_names, metric=None, metric_label="log-likelihood",
                results_folder=None, filename="trace.png"):
    """
    Plot every walker's value of each parameter across steps (and, if
    given, `metric`, e.g. log-likelihood or log-posterior), one subplot
    per parameter -- lets you eyeball whether the chain has settled into a
    flat, well-mixed band (converged) or is still drifting/stuck.

    Parameters
    ----------
    chain : ndarray, shape (nsteps, nwalkers, ndim)
    param_names : list of str, length ndim
    metric : ndarray, shape (nsteps, nwalkers), optional
        Extra per-step/per-walker scalar (e.g. the ll_chain from
        load_chain_from_csv, or log_prob_chain from sampler_to_chain) to
        plot in its own subplot alongside the parameters.
    results_folder : str, optional
        If given, saves the figure there as `filename`.

    Returns the Figure.
    """
    nsteps, nwalkers, ndim = chain.shape
    panel_names = list(param_names) + ([metric_label] if metric is not None else [])

    ncols = 2
    nrows = int(np.ceil(len(panel_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.5 * nrows),
                              sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, name in enumerate(param_names):
        ax = axes_flat[idx]
        for w in range(nwalkers):
            ax.plot(chain[:, w, idx], alpha=0.3, linewidth=0.7)
        ax.set_ylabel(name)

    if metric is not None:
        ax = axes_flat[len(param_names)]
        for w in range(nwalkers):
            ax.plot(metric[:, w], alpha=0.3, linewidth=0.7)
        ax.set_ylabel(metric_label)

    for ax in axes_flat[len(panel_names):]:
        ax.axis("off")
    for idx in range(max(0, len(panel_names) - ncols), len(panel_names)):
        axes_flat[idx].set_xlabel("step")

    fig.suptitle("Walker traces")
    plt.tight_layout()

    _save(fig, results_folder, filename, "Trace plot")
    plt.show()
    return fig


def plot_corner(chain, param_names, burnin=0, truths=None,
                 results_folder=None, filename="corner.png"):
    """
    Corner plot (1D marginal + 2D pairwise posteriors) of the flattened
    chain, via the `corner` package -- same approach as multimoon's
    mm_plots_multi.plots().

    Parameters
    ----------
    chain : ndarray, shape (nsteps, nwalkers, ndim)
    param_names : list of str, length ndim
    burnin : int, optional
        Extra leading steps to discard before flattening, on top of
        whatever burn-in emcee_walker() already discarded before writing
        the chain out (posteriors.csv only contains post-burn-in steps).
    truths : array-like, shape (ndim,), optional
        Values to mark on every panel, e.g. the maximum-likelihood sample.
    results_folder : str, optional
        If given, saves the figure there as `filename`.

    Returns the Figure.
    """
    flat = chain[burnin:].reshape(-1, chain.shape[-1])
    fig = corner.corner(flat, labels=list(param_names), bins=40, show_titles=True,
                         plot_datapoints=False, color="steelblue",
                         fill_contours=True, title_fmt=".3f", truths=truths,
                         label_kwargs=dict(fontsize=12))
    fig.suptitle("Posterior distributions", y=1.02)

    _save(fig, results_folder, filename, "Corner plot")
    plt.show()
    return fig


def plot_likelihood(chain, metric, param_names, metric_label="log-likelihood",
                     results_folder=None, filename="likelihood.png"):
    """
    For each parameter, scatter its flattened samples against `metric`
    (e.g. the ll_chain from load_chain_from_csv, or log_prob_chain from
    sampler_to_chain), colored by walker index, with marginal histograms
    of the parameter and of `metric` -- same panel layout as multimoon's
    mm_plots_multi.plots() likelihood plots. Useful for spotting which
    parameters drive the fit and whether any walkers are stuck in a
    low-likelihood mode.

    Parameters
    ----------
    chain : ndarray, shape (nsteps, nwalkers, ndim)
    metric : ndarray, shape (nsteps, nwalkers)
    param_names : list of str, length ndim
    results_folder : str, optional
        If given, saves the figure there as `filename`.

    Returns the Figure.
    """
    nsteps, nwalkers, ndim = chain.shape
    flat_metric = metric.flatten()
    walker_idx = np.tile(np.arange(nwalkers), nsteps)
    finite = np.isfinite(flat_metric)

    ylim = (np.nanpercentile(flat_metric[finite], 1),
             np.nanmax(flat_metric[finite]) + 1)

    ncols = 2
    nrows = int(np.ceil(ndim / ncols))
    fig = plt.figure(figsize=(6 * ncols, 4.2 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(2 * nrows, 2 * ncols, height_ratios=[1, 3] * nrows,
                           width_ratios=[3, 1] * ncols)

    for idx, name in enumerate(param_names):
        flat_param = chain[:, :, idx].flatten()
        row, col = divmod(idx, ncols)

        ax_hist = fig.add_subplot(gs[2 * row, 2 * col])
        ax_scatter = fig.add_subplot(gs[2 * row + 1, 2 * col], sharex=ax_hist)
        ax_llhist = fig.add_subplot(gs[2 * row + 1, 2 * col + 1], sharey=ax_scatter)

        ax_hist.hist(flat_param, bins=40, histtype="step", color="black")
        ax_hist.set_yticks([])
        plt.setp(ax_hist.get_xticklabels(), visible=False)

        ax_scatter.scatter(flat_param[finite], flat_metric[finite],
                            c=walker_idx[finite], cmap="nipy_spectral",
                            s=6, alpha=0.3, edgecolors="none", rasterized=True)
        ax_scatter.set_xlabel(name)
        ax_scatter.set_ylabel(metric_label)
        ax_scatter.set_ylim(*ylim)

        ax_llhist.hist(flat_metric[finite], bins=40, orientation="horizontal",
                        histtype="step", color="black")
        ax_llhist.set_xticks([])
        plt.setp(ax_llhist.get_yticklabels(), visible=False)
        ax_llhist.set_ylim(*ylim)

    fig.suptitle(f"Parameters vs. {metric_label}")

    _save(fig, results_folder, filename, "Likelihood plot")
    plt.show()
    return fig


def plot_diagnostics(chain, metric, param_names, metric_label="log-likelihood",
                      burnin=0, truths=None, results_folder=None):
    """
    Convenience wrapper: makes the trace, corner, and likelihood plots
    together from one chain, saving each to `results_folder` if given.

    Returns (trace_fig, corner_fig, likelihood_fig).
    """
    trace_fig = plot_trace(chain, param_names, metric=metric, metric_label=metric_label,
                            results_folder=results_folder)
    corner_fig = plot_corner(chain, param_names, burnin=burnin, truths=truths,
                              results_folder=results_folder)
    likelihood_fig = plot_likelihood(chain, metric, param_names, metric_label=metric_label,
                                      results_folder=results_folder)
    return trace_fig, corner_fig, likelihood_fig


def plot_diagnostics_from_sampler(sampler, param_names, results_folder=None):
    """
    plot_diagnostics(), pulling the chain and full log-posterior straight
    off a freshly-run emcee sampler (see sampler_to_chain()).
    """
    chain, log_prob_chain = sampler_to_chain(sampler)
    return plot_diagnostics(chain, log_prob_chain, param_names,
                             metric_label="log-posterior", results_folder=results_folder)


def plot_diagnostics_from_csv(posteriors_path, results_folder=None):
    """
    plot_diagnostics(), loading the chain from a posteriors.csv written by
    run_emcee_walker.py (see load_chain_from_csv()). Defaults
    `results_folder` to posteriors_path's own directory, matching where
    run_emcee_walker.py writes its other plots.
    """
    chain, ll_chain, param_names = load_chain_from_csv(posteriors_path)
    if results_folder is None:
        results_folder = os.path.dirname(posteriors_path)
    return plot_diagnostics(chain, ll_chain, param_names,
                             metric_label="ll_count", results_folder=results_folder)
