"""Entry point: produce the validation + signal-baseline figures.

    python -m delphes_pipeline.plots.make_plots --config <config.yml> [--klambda-base DIR]

Object/patch + single-sample baseline figures run on the primary sample
(``input.delphes_root``); the lepton-floor formula figure needs no data; the
gen m_HH overlay runs across the six kappa_lambda sample directories under
``--klambda-base`` (skipped if that base is unavailable, e.g. off Perlmutter).
Figures are written to ``<output.dir>/figures``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import glob

from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.validation.run_validation import load_config
from . import figures

# kappa_lambda label -> directory tag (c2=0, kt=1 GluGluHHto2B2Tau samples).
# kl=2.45 excluded (not needed); missing points are skipped automatically.
KL_TAGS = {"-1.00": "m1p00", "0.00": "0p00", "1.00": "1p00",
           "3.00": "3p00", "5.00": "5p00"}


def run(config: dict, *, klambda_base: str | None, max_events: int | None) -> list[str]:
    outdir = Path(config.get("output", {}).get("dir", "outputs/validation")) / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    treename = config.get("input", {}).get("treename", "Delphes")
    written: list[str] = []

    # formula figure (no data)
    written.append(figures.lepton_eff_floor_curves(outdir))

    # single-sample figures on the primary sample
    primary = config["input"]["delphes_root"]
    ev = DelphesEvents(primary, treename=treename, entry_stop=max_events)
    for fn in (figures.jet_pt_spectrum, figures.tauh_pt_spectrum, figures.jet_eta_spectrum,
               figures.mbb_peak_figure, figures.lepton_pt_spectra, figures.mtautau_figure,
               figures.multiplicity_figure):
        written.append(fn(ev, outdir))

    # gen m_HH overlay across the six kappa_lambda points
    samples = _klambda_samples(klambda_base, treename, max_events)
    if len(samples) >= 2:
        written.append(figures.mhh_klambda_overlay(samples, outdir))
    else:
        print(f"[plots] kappa_lambda base unavailable ({klambda_base!r}); skipping m_HH overlay")

    print(f"[plots] wrote {len(written)} figures -> {outdir}")
    return written


def _klambda_samples(base, treename, max_events):
    """Match each kappa_lambda point by glob ``<base>/*kl-<tag>*`` (exact name agnostic)."""
    if not base or not os.path.isdir(base):
        return {}
    out = {}
    for kl, tag in KL_TAGS.items():
        if not glob.glob(os.path.join(base, f"*kl-{tag}*")):
            continue
        try:
            out[f"kl={kl}"] = DelphesEvents(os.path.join(base, f"*kl-{tag}*"),
                                            treename=treename, entry_stop=max_events)
        except Exception as exc:  # a missing/broken point should not kill the overlay
            print(f"[plots] skip kl={kl}: {exc}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Delphes validation / baseline plots")
    ap.add_argument("--config", required=True)
    ap.add_argument("--delphes-root", default=None, help="override the primary input.delphes_root")
    ap.add_argument("--klambda-base",
                    default=os.environ.get("DELPHES_BASE", "/ceph/jpan/cms_nanoaod_2024_hh2b2tau/delphes"),
                    help="base dir holding the kappa_lambda Delphes sample directories")
    ap.add_argument("--max-events", type=int, default=None, help="cap events per sample (shape plots)")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    if args.delphes_root:
        config.setdefault("input", {})["delphes_root"] = args.delphes_root
    run(config, klambda_base=args.klambda_base, max_events=args.max_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
