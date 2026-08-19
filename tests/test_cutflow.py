"""The cutflow must be instrumentation, never a reimplementation.

A cutflow that recomputes the selection drifts away from it the first time a cut
changes, and then quietly reports a selection nobody runs. These tests pin that the
numbers come from the same masks and that the last row equals the converter's own
output.
"""
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cutflow as C  # noqa: E402
import delphes_to_sbi as S  # noqa: E402

from delphes_pipeline.core.io import NtupleEvents  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_delphes_to_sbi import _cms_event, _write  # noqa: E402


def test_the_final_row_equals_the_converters_own_output(tmp_path):
    p = _write(tmp_path / "a.parquet", n=6)
    rows, n_read, n_final = C.collect(p)
    d, _, _ = S.features(NtupleEvents(p))
    assert n_read == 6
    assert n_final == len(d["m_hh"]), "the cutflow must end where the converter ends"


def test_counts_are_monotonically_non_increasing(tmp_path):
    p = _cms_event(tmp_path, n=8)
    rows, n_read, _ = C.collect(p, sel_kw={"cms": True, "ellipse": False})
    for ch in {r[0] for r in rows}:
        seq = [n for c, _, n in rows if c == ch]
        assert seq == sorted(seq, reverse=True), f"{ch}: {seq}"
        assert seq[0] <= n_read


def test_a_tightened_cut_shows_up_in_the_cutflow(tmp_path):
    """The instrumentation follows the selection, so changing a cut moves the rows."""
    p = _cms_event(tmp_path, n=6)
    loose, _, n_loose = C.collect(p, sel_kw={"cms": True, "ellipse": False,
                                             "btag_min": 1})
    tight, _, n_tight = C.collect(p, sel_kw={"cms": True, "ellipse": False,
                                             "btag_min": 2})
    assert n_tight <= n_loose
    assert any("b-tag" in lbl for _, lbl, _ in tight)


def test_the_preselection_rows_name_their_thresholds(tmp_path):
    p = _write(tmp_path / "a.parquet", n=4)
    rows, _, _ = C.collect(p)
    labels = " ".join(lbl for _, lbl, _ in rows)
    assert "lepton" in labels and "tau_h" in labels and "jets" in labels


def test_the_ellipse_appears_only_when_it_is_applied(tmp_path):
    p = _cms_event(tmp_path, n=6)
    off, _, _ = C.collect(p, sel_kw={"cms": True, "ellipse": False})
    on, _, _ = C.collect(p, sel_kw={"cms": True, "ellipse": True})
    assert not any("ellip" in l for _, l, _ in off)
    assert any("ellip" in l for _, l, _ in on)


def test_rows_accumulate_across_files(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    for i in range(3):
        _write(d / f"s.{i:04d}.parquet", n=4)
    rows, n_read, n_final = C.collect(d)
    assert n_read == 12
    one, _, _ = C.collect(_write(tmp_path / "one.parquet", n=4))
    assert rows[0][2] == 3 * one[0][2], "per-file counts must sum"


def test_the_latex_table_is_emitted_on_request(tmp_path):
    p = _write(tmp_path / "a.parquet", n=4)
    rows, n_read, n_final = C.collect(p)
    tex = C._fmt(rows, n_read, n_final, "t", tex=True)
    assert tex.startswith("\\begin{tabular}") and tex.rstrip().endswith("\\end{tabular}")
    assert "\\toprule" in tex and "events read" in tex


# --------------------------------------------------------------------------- #
# LaTeX text mode: `<` and `>` are MATH symbols (OT1 renders them as inverted
# punctuation), and a bare `_` -- as in "tau_h" -- is a hard error.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,want", [
    (">=1 tau_h", "$>$=1 tau\\_h"),
    ("|eta|<2.5", "$|$eta$|$$<$2.5"),
    ("100%", "100\\%"),
    ("a&b", "a\\&b"),
    ("x^2", "x\\textasciicircum{}2"),
])
def test_tex_escaping(raw, want):
    assert C.texesc(raw) == want


def test_the_emitted_table_has_no_bare_math_characters(tmp_path):
    p = _write(tmp_path / "a.parquet", n=4)
    rows, n_read, n_final = C.collect(p)
    tex = C._fmt(rows, n_read, n_final, "t", tex=True)
    body = [ln for ln in tex.splitlines() if ln.startswith(" & ")]
    assert body
    for ln in body:
        label = ln.split("&")[1] if ln.count("&") > 1 else ""
        # every < > | _ in a label must be wrapped or escaped, never bare
        for i, c in enumerate(label):
            if c in "<>|":
                assert label[i - 1] == "$" or label[i + 1] == "$", ln
            if c == "_":
                assert label[i - 1] == "\\", ln


def test_the_emitted_table_compiles(tmp_path):
    """The only test that actually proves it: run pdflatex over it."""
    import shutil
    import subprocess
    if not shutil.which("pdflatex"):
        pytest.skip("no pdflatex")
    p = _write(tmp_path / "a.parquet", n=4)
    rows, n_read, n_final = C.collect(p)
    doc = ("\\documentclass{article}\\usepackage{booktabs}\\begin{document}\n"
           + C._fmt(rows, n_read, n_final, "t", tex=True)
           + "\n\\end{document}\n")
    (tmp_path / "t.tex").write_text(doc)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert (tmp_path / "t.pdf").exists(), r.stdout[-1500:]


# --------------------------------------------------------------------------- #
# The CROWN baseline is a DIFFERENT STAGE from the paper's selection: pT>20
# everywhere, no b-tag requirement, no elliptical SR. Matching it is what makes
# the Delphes sample comparable to the ntuples the NSBI test actually reads.
# --------------------------------------------------------------------------- #
def test_crown_thresholds_match_the_crown_config():
    import delphes_to_sbi as D
    assert D.CROWN_SEL["mt"] == {"lep_pt": 20.0, "lep_eta": 2.4, "tau_pt": 20.0}
    assert D.CROWN_SEL["et"] == {"lep_pt": 20.0, "lep_eta": 2.5, "tau_pt": 20.0}
    assert D.CROWN_SEL["tt"]["tau_pt"] == 20.0


def test_crown_keeps_events_the_paper_thresholds_reject(tmp_path):
    """A 25 GeV muon with a 25 GeV tau_h is in CROWN's ntuple, out of the paper's SR."""
    import delphes_to_sbi as D
    p = _cms_event(tmp_path, mu_pt=25.0, tau_pt=25.0, n=5)
    paper = S.features(NtupleEvents(p), cms=True, ellipse=False)[0]["m_hh"]
    crown = S.features(NtupleEvents(p), cms=True, ellipse=False,
                       btag_min=0, thresholds=D.CROWN_SEL)[0]["m_hh"]
    assert len(paper) == 0 and len(crown) == 5


def test_the_crown_cutflow_labels_show_the_looser_thresholds(tmp_path):
    import delphes_to_sbi as D
    p = _cms_event(tmp_path, n=4)
    rows, _, _ = C.collect(p, sel_kw={"cms": True, "ellipse": False, "btag_min": 0,
                                      "thresholds": D.CROWN_SEL})
    labels = " ".join(l for _, l, _ in rows)
    assert "pT>20" in labels and "pT>22" not in labels
    assert not any("b-tag" in l for _, l, _ in rows), "CROWN has no b-tag requirement"
