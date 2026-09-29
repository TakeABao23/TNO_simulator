"""
emcee chain diagnostics: trace, corner, parameter-vs-likelihood, and
burn-in-included trace plots, plus a percentile/best-fit summary table.

Adapts the diagnostic plots multimoon's mm_plots_multi.plots() makes for
its emcee chains (walker traces incl. its walkers_full.pdf burn-in view,
a corner.corner posterior plot, parameter-vs-log-likelihood panels, and
its sigsdf.csv summary table) to TNO_simulator's much smaller 8-parameter
moon-like model, which has no coordinate transforms or derived-parameter
bookkeeping to undo.

Two ways to get a chain in:
  - sampler_to_chain(sampler)/sampler_to_full_chain(sampler): straight off
    a freshly-run emcee sampler (e.g. what emcee_walker.emcee_walker()
    returns), before posteriors.csv is even written. sampler_to_chain()
    discards burn-in (sampler.burnin_steps); sampler_to_full_chain() keeps
    it, for plot_trace_full().
  - load_chain_from_csv(posteriors_path): reshapes a posteriors.csv
    written by run_emcee_walker.py back into (nsteps, nwalkers, ndim), for
    post-hoc plotting (e.g. from plot_posterior_result.py). Only ever
    post-burn-in -- posteriors.csv doesn't record burn-in steps.

plot_trace(), plot_corner(), plot_likelihood(), and summarize_chain() all
take the resulting chain array (plus a per-step/per-walker metric array
for most of them), so the same code runs either way.
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
    Pull the post-burn-in (nsteps, nwalkers, ndim) chain and (nsteps,
    nwalkers) log-probability array off a freshly-run emcee sampler (e.g.
    the one emcee_walker.emcee_walker() returns), for plotting before
    posteriors.csv is even written.

    Discards sampler.burnin_steps leading steps (0 if the sampler doesn't
    have that attribute -- e.g. a sampler that isn't from emcee_walker()),
    matching what ends up in posteriors.csv. Unlike load_chain_from_csv()'s
    ll_chain (the ll_count blob only), this uses get_log_prob(), the actual
    log_prior + log-likelihood value emcee sampled on -- prefer this path
    when a live sampler is available.

    Returns (chain, log_prob_chain).
    """
    burnin_steps = getattr(sampler, "burnin_steps", 0)
    return (sampler.get_chain(flat=False, discard=burnin_steps),
            sampler.get_log_prob(flat=False, discard=burnin_steps))


def sampler_to_full_chain(sampler):
    """
    Like sampler_to_chain(), but keeps every step -- burn-in included --
    since emcee_walker.emcee_walker() no longer reset()s the sampler after
    burn-in. Feeds plot_trace_full(), the walkers_full-style plot that
    shows walkers actually converging rather than only the already-
    converged post-burn-in tail sampler_to_chain()/plot_trace() show.

    Returns (chain, log_prob_chain, burnin_steps) -- burnin_steps (0 if the
    sampler doesn't report one) is where the burn-in/sampling boundary
    falls in `chain`.
    """
    burnin_steps = getattr(sampler, "burnin_steps", 0)
    return sampler.get_chain(flat=False), sampler.get_log_prob(flat=False), burnin_steps


def _save(fig, results_folder, filename, label):
    if results_folder is not None:
        path = os.path.join(results_folder, filename)
        fig.savefig(path, dpi=150)
        print(f"{label} saved to {path}")


def plot_trace(chain, param_names, metric=None, metric_label="log-likelihood",
                burnin=None, results_folder=None, filename="trace.png"):
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
    burnin : int, optional
        If given, draws a vertical line at this step on every panel --
        e.g. the burn-in/sampling boundary, when `chain` includes burn-in
        (see plot_trace_full()).
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
        if burnin is not None:
            ax.axvline(x=burnin, color="black", linestyle="--", linewidth=0.8)

    if metric is not None:
        ax = axes_flat[len(param_names)]
        for w in range(nwalkers):
            ax.plot(metric[:, w], alpha=0.3, linewidth=0.7)
        ax.set_ylabel(metric_label)
        if burnin is not None:
            ax.axvline(x=burnin, color="black", linestyle="--", linewidth=0.8)

    for ax in axes_flat[len(panel_names):]:
        ax.axis("off")
    for idx in range(max(0, len(panel_names) - ncols), len(panel_names)):
        axes_flat[idx].set_xlabel("step")

    fig.suptitle("Walker traces" if burnin is None else
                 "Walker traces (dashed line: burn-in ends)")
    plt.tight_layout()

    _save(fig, results_folder, filename, "Trace plot")
    plt.show()
    return fig


def plot_trace_full(sampler, param_names, metric_label="log-posterior",
                     results_folder=None, filename="trace_full.png"):
    """
    walkers_full-style trace plot: every walker's full chain, burn-in
    included, with a vertical line marking where burn-in ends -- lets you
    see the walkers actually converge, not just the already-converged
    tail plot_trace()/plot_diagnostics() show.

    Only meaningful with a live sampler that still has its burn-in steps
    (see sampler_to_full_chain()) -- not available from posteriors.csv,
    which only ever records post-burn-in samples.

    Returns the Figure.
    """
    chain, log_prob_chain, burnin_steps = sampler_to_full_chain(sampler)
    return plot_trace(chain, param_names, metric=log_prob_chain, metric_label=metric_label,
                       burnin=burnin_steps, results_folder=results_folder, filename=filename)


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


def summarize_chain(chain, param_names, metric=None, burnin=0,
                     results_folder=None, filename="sigsdf.csv"):
    """
    Per-parameter percentile/mean/best-fit summary table of the flattened
    chain -- same layout as multimoon's mm_plots_multi.plots() sigsdf.csv:
    for each parameter, the -3/-2/-1 sigma, median, +1/+2/+3 sigma
    percentiles (the sigma columns as offsets *from* the median, matching
    multimoon's convention -- read a row as "median (+1sigma/-1sigma)"),
    the mean, and the "best fit" value: this parameter's value in whichever
    sample has the highest `metric`.

    Parameters
    ----------
    chain : ndarray, shape (nsteps, nwalkers, ndim)
    param_names : list of str, length ndim
    metric : ndarray, shape (nsteps, nwalkers), optional
        e.g. the ll_chain from load_chain_from_csv, or log_prob_chain from
        sampler_to_chain. If omitted, the 'best fit' column is left NaN.
    burnin : int, optional
        Extra leading steps to discard before flattening.
    results_folder : str, optional
        If given, saves the table there as `filename`.

    Returns a pandas DataFrame indexed by param_names.
    """
    flat = chain[burnin:].reshape(-1, chain.shape[-1])

    best_row = None
    if metric is not None:
        flat_metric = metric[burnin:].flatten()
        finite = np.isfinite(flat_metric)
        if finite.any():
            best_row = flat[finite][np.argmax(flat_metric[finite])]

    sigma_cols = ["-3sigma", "-2sigma", "-1sigma", "median", "1sigma", "2sigma", "3sigma"]
    percentiles = [0.37, 2.275, 15.866, 50, 84.134, 97.724, 99.63]

    rows = []
    for i, name in enumerate(param_names):
        values = flat[:, i]
        pcts = dict(zip(sigma_cols, np.percentile(values, percentiles)))
        median = pcts["median"]
        row = {col: (val if col == "median" else val - median) for col, val in pcts.items()}
        row["mean"] = np.mean(values)
        row["best fit"] = best_row[i] if best_row is not None else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows, index=list(param_names),
                            columns=sigma_cols + ["mean", "best fit"])

    if results_folder is not None:
        path = os.path.join(results_folder, filename)
        summary.to_csv(path)
        print(f"Parameter summary saved to {path}")

    return summary


def plot_diagnostics(chain, metric, param_names, metric_label="log-likelihood",
                      burnin=0, truths=None, results_folder=None):
    """
    Convenience wrapper: makes the trace, corner, and likelihood plots plus
    the percentile/best-fit summary table, together from one chain, saving
    each to `results_folder` if given.

    Returns (trace_fig, corner_fig, likelihood_fig, summary_df).
    """
    trace_fig = plot_trace(chain, param_names, metric=metric, metric_label=metric_label,
                            results_folder=results_folder)
    corner_fig = plot_corner(chain, param_names, burnin=burnin, truths=truths,
                              results_folder=results_folder)
    likelihood_fig = plot_likelihood(chain, metric, param_names, metric_label=metric_label,
                                      results_folder=results_folder)
    summary_df = summarize_chain(chain, param_names, metric=metric, burnin=burnin,
                                  results_folder=results_folder)
    return trace_fig, corner_fig, likelihood_fig, summary_df


def plot_diagnostics_from_sampler(sampler, param_names, results_folder=None):
    """
    plot_diagnostics(), pulling the post-burn-in chain and full
    log-posterior off a freshly-run emcee sampler (see sampler_to_chain()),
    plus plot_trace_full() -- the burn-in-included trace plot, only
    available with a live sampler.

    Returns (trace_fig, corner_fig, likelihood_fig, summary_df, full_trace_fig).
    """
    chain, log_prob_chain = sampler_to_chain(sampler)
    trace_fig, corner_fig, likelihood_fig, summary_df = plot_diagnostics(
        chain, log_prob_chain, param_names, metric_label="log-posterior",
        results_folder=results_folder
    )
    full_trace_fig = plot_trace_full(sampler, param_names, metric_label="log-posterior",
                                      results_folder=results_folder)
    return trace_fig, corner_fig, likelihood_fig, summary_df, full_trace_fig


def plot_diagnostics_from_csv(posteriors_path, results_folder=None):
    """
    plot_diagnostics(), loading the chain from a posteriors.csv written by
    run_emcee_walker.py (see load_chain_from_csv()). Defaults
    `results_folder` to posteriors_path's own directory, matching where
    run_emcee_walker.py writes its other plots.

    No plot_trace_full() here -- posteriors.csv only ever records
    post-burn-in samples, so there's no burn-in left to plot; use
    plot_diagnostics_from_sampler() right after a run instead.
    """
    chain, ll_chain, param_names = load_chain_from_csv(posteriors_path)
    if results_folder is None:
        results_folder = os.path.dirname(posteriors_path)
    return plot_diagnostics(chain, ll_chain, param_names,
                             metric_label="ll_count", results_folder=results_folder)
