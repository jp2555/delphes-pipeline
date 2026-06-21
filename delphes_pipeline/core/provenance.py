"""Run provenance stamped into every ``report.json`` (note §8.3).

Records the card hash, git commit, generator/container versions (from an
optional ``versions.json`` next to the card), the input file, and event count,
so a report can always be traced back to the exact configuration that produced
it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _sha256(path: str | Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions(card_path: str | Path) -> dict[str, Any]:
    vpath = Path(card_path).resolve().parent / "versions.json"
    if vpath.exists():
        try:
            with open(vpath) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def collect(
    card_path: str | Path,
    input_path: str | Path,
    n_events: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the provenance block for a validation run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "card_path": str(card_path),
        "card_sha256": _sha256(card_path),
        "tuning_set": config.get("tuning_set", "v0"),
        "generator_versions": _versions(card_path),
        "git_commit": _git_commit(),
        "input_path": str(input_path),
        "n_events": int(n_events),
    }
