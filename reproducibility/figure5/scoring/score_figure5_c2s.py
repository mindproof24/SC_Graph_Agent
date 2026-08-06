#!/usr/bin/env python3
"""Recompute Figure 5 strict and lineage-compatible scores from JSONL logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


CLUSTERS = ("0", "3", "12", "18")
SEPARATOR = "\n\n<<<EVENT>>>\n\n"


def read_events(path: Path, mode: str) -> tuple[str, dict]:
    parts = []
    metadata = {}
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "session_start":
                metadata = event
            if event.get("event") == "assistant":
                parts.append(str(event.get("content", "")))
            elif mode == "visible" and event.get("event") == "thinking":
                parts.append(str(event.get("content", "")))
    return SEPARATOR.join(parts), metadata


def cluster_window(text: str, cluster: str) -> str:
    others = "|".join(value for value in CLUSTERS if value != cluster)
    section = re.compile(
        rf"(?is)(?:^|\n)\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:\*\*)?"
        rf"cluster\s*{cluster}\b.*?"
        rf"(?=\n\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:\*\*)?"
        rf"cluster\s*(?:{others})\b|\Z)"
    )
    table_row = re.compile(rf"(?im)^\s*\|\s*(?:\*\*)?{cluster}(?:\*\*)?\s*\|.*$")
    sections, fallbacks = [], []
    for event_text in text.split(SEPARATOR):
        matches = [match.group(0) for match in section.finditer(event_text)]
        if matches:
            sections.extend(matches)
            continue
        rows = table_row.findall(event_text)
        if rows:
            sections.extend(rows)
            continue
        for match in re.finditer(rf"(?i)cluster\s*{cluster}\b", event_text):
            fallbacks.append(event_text[max(0, match.start() - 120):match.end() + 500])
    return "\n".join(sections or fallbacks)


def first_match(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return match.start() if match else None


def classify(snippet: str) -> str:
    text = snippet.lower()
    head = text[:900]
    microglia = r"microgl"
    t_cell = r"t\s*cell|\bt-cells?\b|\bcd3d\b|\bcd3e\b|\btrac\b|cytotoxic|lymphocyte|lymphoid|\bnk\b|nkg7|klrd1"
    microglia_pos = first_match(microglia, head)
    t_cell_pos = first_match(t_cell, head)
    if microglia_pos is not None or t_cell_pos is not None:
        if t_cell_pos is not None and (microglia_pos is None or t_cell_pos < microglia_pos):
            return "T cell"
        return "microglial cell"
    if re.search(t_cell, text):
        return "T cell"
    if re.search(microglia, text):
        return "microglial cell"
    patterns = (
        (r"dendritic|\bdc\b|cdc1|cdc2|clec9a", "dendritic cell"),
        (r"macrophage", "macrophage"),
        (r"monocyte", "monocyte"),
        (r"myeloid|cd14|cd68|cd163|fcgr3a|cst3|lyz|antigen-presenting cells?|\bapc\b", "myeloid cell"),
        (r"eryth|hbb|stress", "erythroid/stress"),
    )
    for pattern, label in patterns:
        if re.search(pattern, text):
            return label
    return "unassigned"


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--trajectory-dir", type=Path, default=root / "trajectories")
    parser.add_argument("--truth", type=Path, default=root / "results" / "figure5_truth_by_cluster.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "recomputed")
    args = parser.parse_args()

    truth_df = pd.read_csv(args.truth, dtype={"cluster": str})
    truth = dict(zip(truth_df["cluster"], truth_df["majority_label"]))
    strict_accept = {"microglial cell": {"microglial cell"}, "T cell": {"T cell"}}
    lineage_accept = {
        "microglial cell": {"microglial cell", "myeloid cell", "macrophage", "monocyte"},
        "T cell": {"T cell"},
    }

    rows = []
    for path in sorted(args.trajectory_dir.rglob("*.jsonl")):
        relative = path.relative_to(args.trajectory_dir)
        context_group, model_group = relative.parts[:2]
        for mode in ("final", "visible"):
            text, metadata = read_events(path, mode)
            for cluster in CLUSTERS:
                predicted = classify(cluster_window(text, cluster))
                expected = truth[cluster]
                rows.append({
                    "context_group": context_group,
                    "model_group": model_group,
                    "run_id": path.stem,
                    "model": metadata.get("model"),
                    "num_ctx": metadata.get("num_ctx"),
                    "mode": mode,
                    "cluster": cluster,
                    "truth": expected,
                    "prediction": predicted,
                    "strict": int(predicted in strict_accept[expected]),
                    "lineage": int(predicted in lineage_accept[expected]),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_scores = pd.DataFrame(rows)
    cluster_scores.to_csv(args.output_dir / "cluster_scores.csv", index=False)
    run_scores = cluster_scores.groupby(
        ["context_group", "model_group", "run_id", "model", "num_ctx", "mode"],
        as_index=False,
    ).agg(strict=("strict", "mean"), lineage=("lineage", "mean"))
    run_scores.to_csv(args.output_dir / "run_scores.csv", index=False)
    summary = run_scores.groupby(
        ["context_group", "model_group", "mode"], as_index=False
    ).agg(n=("run_id", "count"), strict=("strict", "mean"), lineage=("lineage", "mean"))
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
