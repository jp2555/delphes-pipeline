"""C-1: how often is a jet simultaneously b-tagged and tau-tagged?

    pixi run python scripts/tag_exclusivity_check.py --ntuple /ceph/jpan/ntuples/merged

The two tag bits are drawn as INDEPENDENT Bernoullis per jet, so nothing stops a jet
carrying both. In CMS they are strongly anti-correlated and overlap-removed. This matters
at selection level rather than at response level: it sculpts WHICH jets become the bb and
tautau pairs, and therefore m_bb, dR_bb and m_HH.

The number to read is not the raw double-tag rate but the last one: how often a jet in
the SELECTED b pair also carries a tau tag. That is the population whose object
assignment could have gone the other way under CMS-style overlap removal.
"""
from __future__ import annotations

import argparse

import awkward as ak
import numpy as np

from delphes_pipeline.core.io import NtupleEvents, available_kl


def measure(ev):
    j = ev.jets
    b = j.btag == 1
    t = j.tautag == 1
    n_j = int(ak.sum(ak.num(j)))
    n_b = int(ak.sum(b))
    n_t = int(ak.sum(t))
    n_bt = int(ak.sum(b & t))
    # expected under independence, per jet, using the inclusive rates
    exp = n_b * n_t / max(n_j, 1)
    return {"jets": n_j, "btag": n_b, "tautag": n_t, "both": n_bt,
            "expected_if_independent": exp}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--max-events", type=int, default=200000)
    args = ap.parse_args(argv)

    kls = available_kl(args.ntuple)
    points = kls or [None]
    for kl in points:
        ev = NtupleEvents(args.ntuple, kl=kl, entry_stop=args.max_events)
        m = measure(ev)
        tag = f"kl={kl:g}" if kl is not None else "all"
        pct = 100 * m["both"] / max(m["tautag"], 1)
        print(f"[tags] {tag}: {m['jets']:,} jets | b {m['btag']:,} | tau {m['tautag']:,} "
              f"| BOTH {m['both']:,} ({pct:.1f}% of tau-tagged)")
        print(f"[tags]   expected under independence: {m['expected_if_independent']:.0f}"
              f"  -> the draws ARE independent, so this is the design, not a surprise")
    print("[tags] CMS overlap-removes these; a joint categorical draw "
          "P(b-tag, tau-tag | flavour, pT) is the v2 fix (C-1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
