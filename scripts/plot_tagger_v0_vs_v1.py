"""Plot the v0 (stock, Run-1 form) vs v1 (Run-3 anchor fit) tagger formulas
against the 2024 NanoAODv15 anchor measurement, one panel per quantity.

    python scripts/plot_tagger_v0_vs_v1.py   # -> docs/figures/tagger_v0_vs_v1.png

The black points are the measured anchor (the physics content); the curves are
just smooth interpolants through them. See docs/card_v1_summary.md.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PT = np.linspace(15, 300, 500)
CENTERS = np.array([25, 35, 45, 60, 85, 125, 175, 250.0])

STOCK = {  # cms_card_v0.tcl (arXiv:1211.4462, Run-1 CSV-era)
    "b": lambda p: 0.85 * np.tanh(0.0025 * p) * (25.0 / (1 + 0.063 * p)),
    "c": lambda p: 0.25 * np.tanh(0.018 * p) * (1 / (1 + 0.0013 * p)),
    "light": lambda p: 0.01 + 0.000038 * p,
    "tau": lambda p: 0.6 * np.ones_like(p),
}
V1 = {  # cms_card_v1.tcl (PATCH-6/7, fit to the 2024 anchor)
    "b": lambda p: np.where(p > 4, 0.904 - 3.53 / p, 0),
    "c": lambda p: 0.094 + 2.22 / p,
    "light": lambda p: 0.019 + 0.32 * np.exp(-p / 30.0),
    "tau": lambda p: np.where(p > 12.5, 0.776 - 9.7 / p, 0),
}
ANCHOR = {
    "b": [0.763, 0.807, 0.829, 0.855, 0.868, 0.881, 0.892, 0.890],
    "c": [0.183, 0.153, 0.157, 0.117, 0.102, 0.100, 0.110, 0.103],
    "light": [0.159, 0.132, 0.094, 0.058, 0.033, 0.023, 0.017, 0.019],
    "tau": [0.388, 0.519, 0.579, 0.608, 0.637, 0.674, 0.714, 0.737],
}
TITLES = {
    "b": "b-jet EFF {5}  (b-tag efficiency)",
    "c": "c-jet {4}  (c→b mistag)",
    "light": "light {0}  (udsg→b mistag)",
    "tau": "τ-jet EFF {15}  (τ_h efficiency)",
}
VERDICT = {
    "b": "stock: right shape, too LOW\n→ could re-fit coeffs",
    "c": "stock RISES, anchor FALLS\n→ form must change",
    "light": "stock RISES, anchor FALLS\n→ form must change",
    "tau": "stock FLAT 0.6, anchor RISES\n→ form must change",
}


def main() -> str:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, k in zip(axes.flat, ["b", "c", "light", "tau"]):
        ax.plot(PT, STOCK[k](PT), "--", color="tab:red", lw=2, label="v0 stock (Run-1 form)")
        ax.plot(PT, V1[k](PT), "-", color="tab:blue", lw=2.2, label="v1 (Run-3 fit)")
        ax.plot(CENTERS, ANCHOR[k], "ko", ms=7, zorder=5, label="2024 anchor (measured)")
        ax.set_title(TITLES[k], fontsize=12, fontweight="bold")
        ax.set_xlabel("jet pT [GeV]"); ax.set_ylabel("efficiency / rate")
        ax.set_xlim(15, 300); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
        ax.text(0.97, 0.05, VERDICT[k], transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
    fig.suptitle("Delphes tagger formulas: v0 stock (Run-1) vs v1 (Run-3 anchor fit)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "tagger_v0_vs_v1.png")
    out = os.path.abspath(out)
    fig.savefig(out, dpi=115); plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
