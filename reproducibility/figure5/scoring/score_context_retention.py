#!/usr/bin/env python3
"""Reproduce the automatic regex-based Figure 5 context-retention scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def last_assistant(path: Path) -> str:
    final = ""
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "assistant":
                final = str(event.get("content", "") or "")
    return final


def session_metadata(path: Path) -> dict:
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "session_start":
                return event
    return {}


def score_final(text: str) -> tuple[int, str]:
    """Score context retention using the final 16k/32k rule."""
    low = text.lower()
    has_gbm = bool(re.search(
        r"glioblastoma|\bgbm\b|tumou?r|cancer|malignan|neoplasm", low
    ))
    has_immune = bool(re.search(
        r"inflamm|immune|microgl|macrophage|myeloid|t cell|cytotoxic|antigen|hla|interferon|apc",
        low,
    ))
    if has_gbm and has_immune:
        return 2, "GBM/cancer and immune-context terms detected."
    if has_immune:
        return 1, "Immune-context terms detected without a GBM/cancer term."
    return 0, "Neither the required GBM/cancer and immune-context combination nor immune context was detected."


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, default=root / "trajectories")
    parser.add_argument("--output", type=Path, default=root / "recomputed" / "context_scores.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.trajectory_dir.rglob("*.jsonl")):
        relative = path.relative_to(args.trajectory_dir)
        context_group, model_group = relative.parts[:2]
        final = last_assistant(path)
        score, reason = score_final(final)
        metadata = session_metadata(path)
        rows.append({
            "context_group": context_group,
            "model_group": model_group,
            "source_file": path.name,
            "model": metadata.get("model"),
            "context_score": score,
            "context_reason": reason,
            "final_answer_chars": len(final),
        })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = result.groupby(["context_group", "model_group"], as_index=False).agg(
        n=("source_file", "count"), context=("context_score", "mean")
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
