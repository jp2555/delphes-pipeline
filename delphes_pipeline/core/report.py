"""Aggregate ``CheckResult``s into report.json + summary.md and the gate code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .result import CheckResult, Severity


class Report:
    """Collects results across levels and renders the verdict."""

    def __init__(self, provenance: dict[str, Any] | None = None):
        self.provenance = provenance or {}
        self.results: list[CheckResult] = []

    def add(self, results: Iterable[CheckResult]) -> None:
        self.results.extend(results)

    @property
    def gate_failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.is_gate_failure]

    @property
    def passed(self) -> bool:
        return not self.gate_failures

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    # ----- rendering ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "passed": self.passed,
            "n_checks": len(self.results),
            "n_gate_failures": len(self.gate_failures),
            "results": [r.to_dict() for r in self.results],
        }

    def _summary_md(self) -> str:
        verdict = "PASS ✅" if self.passed else "FAIL ❌"
        lines = [
            "# Delphes card-validation report",
            "",
            f"**Verdict: {verdict}**  "
            f"({len(self.results)} checks, {len(self.gate_failures)} gate failures)",
            "",
        ]
        prov = self.provenance
        if prov:
            lines += [
                "## Provenance",
                "",
                f"- input: `{prov.get('input_path', '?')}` "
                f"({prov.get('n_events', '?')} events)",
                f"- card: `{prov.get('card_path', '?')}` "
                f"(sha256 `{str(prov.get('card_sha256'))[:12]}…`)",
                f"- tuning set: {prov.get('tuning_set', '?')}",
                f"- git: `{str(prov.get('git_commit'))[:12]}`  ·  "
                f"{prov.get('timestamp_utc', '')}",
                "",
            ]

        # group by level, preserving first-seen order
        levels: list[str] = []
        for r in self.results:
            if r.level not in levels:
                levels.append(r.level)

        for level in levels:
            lines += [f"## {level}", "",
                      "| check | result | measured | target | detail |",
                      "|---|---|---|---|---|"]
            for r in [x for x in self.results if x.level == level]:
                mark = {Severity.GATE: "", Severity.WARN: " (warn)", Severity.INFO: " (info)"}[r.severity]
                status = ("✅" if r.passed else "❌") + mark
                meas = "—" if r.measured is None else f"{r.measured:.4g}{(' ' + r.units) if r.units else ''}"
                tgt = "—" if r.target is None else f"{r.target:.4g}"
                lines.append(f"| {r.name} | {status} | {meas} | {tgt} | {r.detail} |")
            lines.append("")
        return "\n".join(lines)

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "report.json"
        md_path = output_dir / "summary.md"
        with open(json_path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        with open(md_path, "w") as fh:
            fh.write(self._summary_md())
        return json_path, md_path
