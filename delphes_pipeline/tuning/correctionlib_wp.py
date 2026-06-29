"""Resolve b-tag working-point thresholds from CMS jsonpog-integration (CVMFS).

The UParT-AK4 Medium discriminant cut is a physics number that should come from
the official BTV JSON, not be hand-entered. With CVMFS the b-tagging JSON lives at
e.g. ``/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/<era>/btagging.json.gz``;
its ``<tagger>_wp_values`` correction maps a working-point name (L/M/T/…) to the
discriminant threshold. ``resolve_btag_wp`` returns that threshold so the anchor
reader can threshold ``Jet_btagUParTAK4B`` consistently with CMS.

``correctionlib`` is an optional dependency (``pip/pixi`` extra ``anchor``); a
clear error is raised if it is needed but missing. Use ``python -m
delphes_pipeline.tuning.correctionlib_wp <json>`` to list the corrections and find
the right name for your era.
"""

from __future__ import annotations

from typing import Optional


def _cset(json_path):
    try:
        from correctionlib import CorrectionSet
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "correctionlib is required to resolve b-tag WPs from jsonpog-integration; "
            "install it (pixi/pip extra 'anchor') or set anchor.wp.btag_medium explicitly."
        ) from exc
    return CorrectionSet.from_file(str(json_path))


def list_corrections(json_path) -> list[str]:
    """All correction names in the JSON (to find the right ``*_wp_values`` name)."""
    return list(_cset(json_path).keys())


def find_wp_correction(json_path, tagger: str = "UParT") -> Optional[str]:
    """Best-effort: the ``*_wp_values`` correction whose name contains ``tagger``."""
    kw = tagger.lower()
    cands = [n for n in list_corrections(json_path) if "wp_value" in n.lower() and kw in n.lower()]
    return cands[0] if cands else None


def load_wp(json_path, correction: str, wp: str = "M") -> float:
    """Evaluate ``correction`` at working point ``wp`` -> discriminant threshold."""
    return float(_cset(json_path)[correction].evaluate(wp))


def resolve_btag_wp(block: dict) -> float:
    """Resolve the b-tag WP from an ``anchor.wp.btag_correctionlib`` config block.

    block keys: ``json`` (path), ``wp`` (default "M"), and either ``correction``
    (explicit name) or ``tagger`` (keyword to auto-find the ``*_wp_values`` name).
    """
    json_path = block["json"]
    correction = block.get("correction") or find_wp_correction(json_path, block.get("tagger", "UParT"))
    if not correction:
        raise ValueError(
            f"no '*_wp_values' correction matching tagger {block.get('tagger', 'UParT')!r} in "
            f"{json_path}; available: {list_corrections(json_path)}"
        )
    return load_wp(json_path, correction, block.get("wp", "M"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect/evaluate b-tag WP thresholds from a BTV JSON")
    ap.add_argument("json", help="path to btagging.json(.gz) on CVMFS")
    ap.add_argument("--correction", default=None, help="explicit *_wp_values correction name")
    ap.add_argument("--tagger", default="UParT", help="tagger keyword for auto-find")
    ap.add_argument("--wp", default="M", help="working point (L/M/T/...)")
    args = ap.parse_args(argv)

    if args.correction is None:
        print("corrections in", args.json, ":")
        for n in list_corrections(args.json):
            print("  ", n)
        guess = find_wp_correction(args.json, args.tagger)
        print("auto-found wp_values correction:", guess)
        if guess:
            print(f"{args.tagger} {args.wp} threshold = {load_wp(args.json, guess, args.wp)}")
    else:
        print(f"{args.correction} {args.wp} = {load_wp(args.json, args.correction, args.wp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
