"""Helpers for summarising experiment runs: recall / precision curves and recall AUC.

Typical usage (see experiment_summary_template.ipynb):

    results = load_experiments(EXPERIMENTS, n_points=101, reference="Baseline")
    show_summary_tables(results, highlight_x=[0.05, 0.25], recall_targets=[0.9], reference="Baseline")
    plot_mean_curves(results, kind="recall", zoom_x_max=0.4)

The x-axis of every curve is ``x = 1 - quantile`` on the grid ``np.linspace(0, 1, n_points)``.
Deltas are in percentage points (pp).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from email_outreach.experiment_utils.metric_loader import load_experiment_metrics
from email_outreach.sae_based_mab.metrics import calculate_recall_precision_at_quantiles, calculate_auc

Results = Mapping[str, "ExperimentResult"]


# --------------------------------------------------------------------------- basics
def make_grid(n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, quantiles)`` with ``x = linspace(0, 1, n_points)`` and ``quantiles = 1 - x``."""
    x = np.linspace(0, 1, n_points)
    return x, 1 - x


def delta_pct(lhs, rhs, decimals: int = 2) -> np.ndarray:
    """Difference ``lhs - rhs`` in percentage points."""
    return ((np.asarray(lhs) - np.asarray(rhs)) * 100).round(decimals)


def nearest_index(x: np.ndarray, value: float) -> int:
    """Index of the grid point closest to ``value``."""
    return int(np.argmin(np.abs(x - value)))


# --------------------------------------------------------------------------- loading
@dataclass
class ExperimentResult:
    """All runs of one experiment, evaluated on a common grid."""

    name: str
    path: Path
    x: np.ndarray            # (n_points,) grid, x = 1 - quantile
    recalls: np.ndarray      # (n_runs, n_points)
    precisions: np.ndarray   # (n_runs, n_points)
    aucs: np.ndarray         # (n_runs,)
    std_ddof: int = 0        # 0 = population std, 1 = sample std

    @property
    def n_runs(self) -> int:
        return len(self.aucs)

    def runs(self, kind: str) -> np.ndarray:
        """Per-run curves for ``kind`` in {"recall", "precision"}."""
        return {"recall": self.recalls, "precision": self.precisions}[kind]

    def mean(self, kind: str) -> np.ndarray:
        return self.runs(kind).mean(axis=0)

    def std(self, kind: str) -> np.ndarray:
        return self.runs(kind).std(axis=0, ddof=self.std_ddof)

    @property
    def mean_recall(self) -> np.ndarray:
        return self.mean("recall")

    @property
    def std_recall(self) -> np.ndarray:
        return self.std("recall")


def load_experiment(name: str, path: Path, quantiles: np.ndarray, x: np.ndarray, std_ddof: int = 0) -> ExperimentResult:
    """Load the runs of one experiment once and compute recall, precision and AUC in a single pass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} does not exist")
    recalls, precisions, aucs = [], [], []
    for run in load_experiment_metrics(path):
        recall, precision = calculate_recall_precision_at_quantiles(run, quantiles=quantiles)[1:3]
        recalls.append(np.asarray(recall, dtype=float))
        precisions.append(np.asarray(precision, dtype=float))
        aucs.append(float(calculate_auc(run)))
    return ExperimentResult(name, path, x, np.array(recalls), np.array(precisions), np.array(aucs), std_ddof)


def load_experiments(
    experiments: Mapping[str, Path],
    n_points: int = 101,
    std_ddof: int = 0,
    reference: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, ExperimentResult]:
    """Load every experiment in ``experiments`` (display name -> folder), keeping the given order."""
    if reference is not None and reference not in experiments:
        raise ValueError(f"reference={reference!r} is not one of the experiments: {list(experiments)}")
    x, quantiles = make_grid(n_points)
    results: dict[str, ExperimentResult] = {}
    for name, path in experiments.items():
        t0 = time.time()
        results[name] = load_experiment(name, path, quantiles, x, std_ddof)
        if verbose:
            print(f"{name}: {results[name].n_runs} runs loaded in {time.time() - t0:.0f}s  ({path})")
    return results


# --------------------------------------------------------------------------- tables
def auc_table(results: Results, reference: Optional[str] = None) -> pd.DataFrame:
    """Per-experiment AUC statistics, plus the mean difference to ``reference`` in pp."""
    rows = []
    for name, r in results.items():
        row = {
            "runs": r.n_runs,
            "AUC mean": r.aucs.mean(),
            "AUC std": r.aucs.std(ddof=r.std_ddof),
            "AUC min": r.aucs.min(),
            "AUC max": r.aucs.max(),
        }
        if reference is not None:
            row["Δ AUC vs ref (pp)"] = (
                np.nan if name == reference else delta_pct(r.aucs.mean(), results[reference].aucs.mean())
            )
        rows.append(pd.Series(row, name=name))
    return pd.DataFrame(rows)


def point_tables(
    results: Results, kind: str, highlight_x: Sequence[float], reference: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Mean, std and (optionally) delta-vs-reference tables of ``kind`` at the highlighted x values.

    Rows are experiments, columns are the grid x values closest to ``highlight_x``.
    """
    x = _grid(results)
    idx = [nearest_index(x, v) for v in highlight_x]
    cols = [f"{x[i]:.2f}" for i in idx]
    mean = pd.DataFrame({n: r.mean(kind)[idx] for n, r in results.items()}, index=cols).T
    std = pd.DataFrame({n: r.std(kind)[idx] for n, r in results.items()}, index=cols).T
    delta = None
    if reference is not None:
        others = [n for n in results if n != reference]
        delta = pd.DataFrame(
            delta_pct(mean.loc[others].values, mean.loc[reference].values), index=others, columns=cols
        )
    return mean, std, delta


def format_mean_std(mean: pd.DataFrame, std: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Combine two tables of the same shape into 'mean ± std' strings."""
    return pd.DataFrame(
        {c: [f"{mean.loc[n, c]:.{decimals}f} ± {std.loc[n, c]:.{decimals}f}" for n in mean.index] for c in mean.columns},
        index=mean.index,
    )


def reach_table(results: Results, targets: Sequence[float]) -> pd.DataFrame:
    """Smallest x at which the mean recall curve reaches each target (lower is better; NaN = never)."""

    def first_x(curve: np.ndarray, x: np.ndarray, target: float) -> float:
        hit = np.where(curve >= target)[0]
        return float(x[hit[0]]) if len(hit) else np.nan

    x = _grid(results)
    return pd.DataFrame(
        {f"recall ≥ {t:.0%}": {n: first_x(r.mean_recall, x, t) for n, r in results.items()} for t in targets}
    )


def show_summary_tables(
    results: Results,
    highlight_x: Sequence[float],
    recall_targets: Sequence[float] = (),
    reference: Optional[str] = None,
    decimals: int = 4,
) -> None:
    """Display the AUC, recall, precision and recall-target tables."""
    display(Markdown("### Recall AUC"))
    display(auc_table(results, reference).round(decimals))

    for kind in ("recall", "precision"):
        mean, std, delta = point_tables(results, kind, highlight_x, reference)
        display(Markdown(f"### {kind.capitalize()} at selected points (mean ± std over runs; columns = x = 1 - quantile)"))
        display(format_mean_std(mean, std, decimals))
        if delta is not None and len(delta):
            display(Markdown(f"**Δ {kind} vs `{reference}` (pp)**"))
            display(delta)

    if len(recall_targets):
        display(Markdown("### Smallest x at which the mean recall reaches the target (lower is better)"))
        display(reach_table(results, recall_targets).round(3))


# --------------------------------------------------------------------------- plots
def get_colors(results: Results) -> dict[str, str]:
    """One stable matplotlib colour per experiment."""
    return {name: f"C{i}" for i, name in enumerate(results)}


def _grid(results: Results) -> np.ndarray:
    return next(iter(results.values())).x


def _finish(fig, filename: str, save_dir: Optional[Path]) -> None:
    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f"{filename}.png", dpi=150)
    plt.show()


def _fit_ylim(ax, x: np.ndarray, curves: Sequence[np.ndarray], xmax: float) -> None:
    """Rescale the y-axis to the part of the curves that is visible for x <= xmax."""
    mask = x <= xmax
    lo = min(c[mask].min() for c in curves)
    hi = max(c[mask].max() for c in curves)
    pad = 0.03 * (hi - lo + 1e-12)
    ax.set_ylim(lo - pad, hi + pad)


def _panels(zoom_x_max: Optional[float], title: str):
    """Figure with a full-range panel and, if ``zoom_x_max`` is given, a zoomed one."""
    if zoom_x_max is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        return fig, [(ax, 1.0, title)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    return fig, [(axes[0], 1.0, title), (axes[1], zoom_x_max, f"{title} (zoom)")]


def plot_mean_curves(
    results: Results,
    kind: str = "recall",
    highlight_x: Sequence[float] = (),
    zoom_x_max: Optional[float] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """Mean ± std curve of ``kind`` ("recall" or "precision") per experiment."""
    x, colors = _grid(results), get_colors(results)
    fig, panels = _panels(zoom_x_max, f"Mean {kind} ± std")
    for ax, xmax, title in panels:
        bands = []
        for name, r in results.items():
            m, s = r.mean(kind), r.std(kind)
            ax.plot(x, m, color=colors[name], label=f"{name} (n={r.n_runs})")
            ax.fill_between(x, m - s, m + s, color=colors[name], alpha=0.2)
            bands += [m - s, m + s]
        for v in highlight_x:
            if v <= xmax:
                ax.axvline(v, color="gray", lw=0.5, ls=":")
        ax.set_xlim(0, xmax)
        if xmax < 1:
            _fit_ylim(ax, x, bands, xmax)
        ax.set(title=title, xlabel="x = 1 - quantile", ylabel=kind)
        ax.grid(alpha=0.3)
        ax.legend()
    _finish(fig, f"{kind}_mean_std", save_dir)


def plot_recall_delta(results: Results, reference: str, save_dir: Optional[Path] = None) -> None:
    """Mean recall of every experiment minus the mean recall of ``reference`` (pp)."""
    x, colors = _grid(results), get_colors(results)
    others = [n for n in results if n != reference]
    if not others:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in others:
        ax.plot(x, (results[name].mean_recall - results[reference].mean_recall) * 100, color=colors[name], label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title=f"Mean recall difference vs {reference}", xlabel="x = 1 - quantile", ylabel="Δ recall (pp)")
    ax.grid(alpha=0.3)
    ax.legend()
    _finish(fig, "recall_delta_vs_reference", save_dir)


def plot_individual_runs(
    results: Results,
    kind: str = "recall",
    zoom_x_max: Optional[float] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """Every individual run (thin lines) overlaid per experiment, with the mean as a dashed line."""
    x, colors = _grid(results), get_colors(results)
    fig, panels = _panels(zoom_x_max, "All runs")
    for ax, xmax, title in panels:
        curves = []
        for name, r in results.items():
            for i, run in enumerate(r.runs(kind)):
                ax.plot(x, run, color=colors[name], lw=0.6, alpha=0.6, label=f"{name} runs" if i == 0 else None)
            ax.plot(x, r.mean(kind), color=colors[name], lw=2.2, ls="--", label=f"{name} mean")
            curves += list(r.runs(kind))
        ax.set_xlim(0, xmax)
        if xmax < 1:
            _fit_ylim(ax, x, curves, xmax)
        ax.set(title=title, xlabel="x = 1 - quantile", ylabel=kind)
        ax.grid(alpha=0.3)
        ax.legend()
    _finish(fig, f"{kind}_individual_runs", save_dir)


def plot_auc(results: Results, save_dir: Optional[Path] = None) -> None:
    """Box plot of the per-run recall AUC with the individual runs as dots."""
    colors = get_colors(results)
    names = list(results)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(names)), 5))
    ax.boxplot([results[n].aucs for n in names], showfliers=False, widths=0.5)
    for i, n in enumerate(names, start=1):
        a = results[n].aucs
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(a)), a, color=colors[n], alpha=0.8, zorder=3)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.set(title="Recall AUC per run", ylabel="AUC")
    ax.grid(alpha=0.3)
    _finish(fig, "auc_per_run", save_dir)
