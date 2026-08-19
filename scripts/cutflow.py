"""Selection cutflow for the Delphes ntuples: events remaining after each cut.

    pixi run python scripts/cutflow.py --ntuple /ceph/jpan/ntuples_untuned/merged \
        --sample signal [--cms-selection] [--max-events 500000] [--tex]

The counts are read off the SAME masks ``select`` / ``cms_select`` apply -- the cutflow
is instrumentation, never a reimplementation, so it cannot drift away from what the
converter actually does.

Two things to know when reading the output:

  * The CMS-selection rows are PER CHANNEL. Each channel starts from the full input
    sample, so the channel columns are alternatives, not a sequence: an event that
    becomes tau_mu tau_h was never a candidate for tau_e tau_h (Sec. 5 priority).
  * The denominator is events READ, which for a merged signal sample is the generated
    count per kappa_lambda point. Efficiencies are quoted against it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delphes_to_sbi import COLUMNS, features  # noqa: E402

from delphes_pipeline.core.io import NtupleEvents, available_kl, resolve_ntuple_paths  # noqa: E402


def collect(path, *, kl=None, max_events=None, sel_kw=None):
    """(cutflow rows, n_read, n_final). Rows are (channel, label, n_remaining)."""
    sel_kw = dict(sel_kw or {})
    rows: list[tuple[str, str, int]] = []
    n_read = n_final = 0
    acc: dict[tuple[str, str], int] = {}
    order: list[tuple[str, str]] = []
    for f in resolve_ntuple_paths(path):
        if max_events is not None and n_read >= max_events:
            break
        try:
            ev = NtupleEvents(f, kl=kl, columns=COLUMNS,
                              entry_stop=None if max_events is None else max_events - n_read)
        except ValueError:
            continue
        n_read += ev.n
        cf: list[tuple[str, str, int]] = []
        d, _, _ = features(ev, cutflow=cf, **sel_kw)
        n_final += len(d["m_hh"])
        for ch, label, n in cf:
            key = (ch, label)
            if key not in acc:
                acc[key] = 0
                order.append(key)
            acc[key] += n
    rows = [(ch, label, acc[(ch, label)]) for ch, label in order]
    return rows, n_read, n_final


def _fmt(rows, n_read, n_final, title, tex=False):
    by_ch: dict[str, list] = {}
    for ch, label, n in rows:
        by_ch.setdefault(ch, []).append((label, n))

    if tex:
        out = ["\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
               "channel & selection & events & rel. & abs. \\\\", "\\midrule"]
        for ch, items in by_ch.items():
            prev = n_read
            out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{ch}}}}} \\\\")
            out.append(f" & events read & {n_read:,} & --- & 100\\% \\\\")
            for label, n in items:
                rel = 100 * n / prev if prev else 0.0
                out.append(f" & {label} & {n:,} & {rel:.1f}\\% & "
                           f"{100 * n / n_read:.3f}\\% \\\\")
                prev = n
            out.append("\\midrule")
        out += [f"\\multicolumn{{2}}{{@{{}}l}}{{final (all channels)}} & {n_final:,} & "
                f"--- & {100 * n_final / max(n_read, 1):.3f}\\% \\\\",
                "\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)

    w = max((len(l) for _, l, _ in rows), default=20) + 2
    out = [f"\n{title}", "=" * (w + 34),
           f"{'selection':{w}s} {'events':>12s} {'rel':>8s} {'abs':>9s}"]
    for ch, items in by_ch.items():
        out.append(f"\n[{ch}]")
        prev = n_read
        out.append(f"  {'events read':{w - 2}s} {n_read:>12,} {'':>8s} {100.0:>8.3f}%")
        for label, n in items:
            rel = 100 * n / prev if prev else 0.0
            out.append(f"  {label:{w - 2}s} {n:>12,} {rel:>7.1f}% "
                       f"{100 * n / max(n_read, 1):>8.3f}%")
            prev = n
    out.append("-" * (w + 34))
    out.append(f"{'final (all channels, finite)':{w}s} {n_final:>12,} "
               f"{'':>8s} {100 * n_final / max(n_read, 1):>8.3f}%")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--sample", default="signal")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--cms-selection", action="store_true")
    ap.add_argument("--channels", default="mt,et")
    ap.add_argument("--btag-min", type=int, default=0)
    ap.add_argument("--lep-veto", action="store_true")
    ap.add_argument("--no-ellipse", action="store_true")
    ap.add_argument("--tex", action="store_true", help="emit a LaTeX tabular")
    args = ap.parse_args(argv)

    if args.cms_selection:
        sel_kw = {"cms": True, "btag_min": max(args.btag_min, 1),
                  "ellipse": not args.no_ellipse,
                  "channels": tuple(c.strip() for c in args.channels.split(","))}
        mode = "CMS HIG-25-008 resolved"
    else:
        sel_kw = {"btag_min": args.btag_min, "lep_veto": args.lep_veto}
        mode = "preselection (CROWN-baseline counterpart)"

    points = available_kl(args.ntuple) if args.sample == "signal" else [None]
    for kl in points or [None]:
        rows, n_read, n_final = collect(args.ntuple, kl=kl,
                                        max_events=args.max_events, sel_kw=sel_kw)
        tag = f"{args.sample}" + (f", kappa_lambda = {kl:g}" if kl is not None else "")
        print(_fmt(rows, n_read, n_final, f"{tag}  --  {mode}", tex=args.tex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
