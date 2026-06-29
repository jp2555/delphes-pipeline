"""CMS-style overlay plotting for the Level-0 checks.

Uses ``mplhep`` for CMS cosmetics when available and degrades silently to plain
matplotlib otherwise (mplhep is an optional dependency). All functions write a
PNG and return its path; they never raise on cosmetic failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display on Perlmutter / CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:  # cosmetic only
    import mplhep as hep

    _HAVE_MPLHEP = True
except Exception:  # pragma: no cover - mplhep optional
    _HAVE_MPLHEP = False


def cms_style() -> None:
    if _HAVE_MPLHEP:
        try:
            plt.style.use(hep.style.CMS)
        except Exception:  # pragma: no cover
            pass


def hist_overlay(
    series: "Sequence[tuple[str, Sequence[float], Optional[Sequence[float]]]]",
    *,
    bins,
    outpath,
    xlabel: str = "",
    ylabel: str = "events (norm.)",
    title: str = "",
    density: bool = True,
    logy: bool = False,
    vlines: "Optional[Sequence[tuple[float, str]]]" = None,
) -> str:
    """Overlay 1-D histograms of several labelled samples.

    ``series`` is a list of ``(label, values, weights)`` (weights may be None).
    ``vlines`` is a list of ``(x, label)`` vertical markers (e.g. analysis cuts).
    """
    cms_style()
    import numpy as _np

    edges = _np.asarray(bins, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for label, values, weights in series:
        v = _np.asarray(values, dtype=float)
        v = v[_np.isfinite(v)]
        if v.size == 0:  # nothing to draw (e.g. no gen Higgs on a fixture)
            ax.plot([], [], label=f"{label} (empty)")
            continue
        w = None if weights is None else _np.asarray(weights, dtype=float)[: v.size]
        ax.hist(v, bins=edges, weights=w, density=density, histtype="step", linewidth=1.6, label=label)
    for x, lbl in (vlines or []):
        ax.axvline(x, ls="--", color="0.4", lw=1.2)
        ax.text(x, ax.get_ylim()[1] * 0.92, f" {lbl}", rotation=90, va="top", ha="left", fontsize=8, color="0.4")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _cms_label(ax)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return str(outpath)


def _cms_label(ax) -> None:
    if _HAVE_MPLHEP:
        try:
            hep.cms.label("Simulation Preliminary", data=False, ax=ax, fontsize=12)
        except Exception:  # pragma: no cover
            pass


def efficiency_overlay(
    centers: Sequence[float],
    measured: Sequence[float],
    measured_err: Optional[Sequence[float]],
    expected: Sequence[float],
    *,
    outpath: str | Path,
    xlabel: str = "pT [GeV]",
    ylabel: str = "efficiency",
    title: str = "",
    measured_label: str = "Delphes (measured)",
    expected_label: str = "card formula",
    ref_centers: Optional[Sequence[float]] = None,
    ref_values: Optional[Sequence[float]] = None,
    ref_errors: Optional[Sequence[float]] = None,
    ref_label: str = "POG (digitised)",
) -> str:
    """Overlay measured efficiency, the card-formula closure target, and an
    optional digitised reference curve. Returns the output path as a string."""
    cms_style()
    centers = np.asarray(centers, dtype=float)
    measured = np.asarray(measured, dtype=float)
    expected = np.asarray(expected, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    yerr = np.asarray(measured_err, dtype=float) if measured_err is not None else None
    ax.errorbar(centers, measured, yerr=yerr, fmt="o", color="black",
                label=measured_label, capsize=2, markersize=4)
    ax.plot(centers, expected, "-", color="C0", label=expected_label)
    if ref_centers is not None and ref_values is not None:
        rc = np.asarray(ref_centers, dtype=float)
        rv = np.asarray(ref_values, dtype=float)
        re = np.asarray(ref_errors, dtype=float) if ref_errors is not None else None
        ax.errorbar(rc, rv, yerr=re, fmt="s", color="C3", label=ref_label,
                    capsize=2, markersize=4, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _cms_label(ax)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return str(outpath)


def resolution_plot(
    centers: Sequence[float],
    resolution: Sequence[float],
    *,
    outpath: str | Path,
    xlabel: str = "sum E_T [GeV]",
    ylabel: str = "MET resolution [GeV]",
    title: str = "",
    label: str = "Delphes",
) -> str:
    """Plot a 1-D resolution curve vs a binning variable."""
    cms_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.asarray(centers, dtype=float), np.asarray(resolution, dtype=float),
            "o-", color="black", markersize=4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _cms_label(ax)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return str(outpath)
