"""Reproduce the strict scores reported in Figure 1.

The original evaluation harness required a final ``{"labels": [...]}`` array.
This script reads the resulting strict ``score`` field without recovering
alternative JSON structures or classifications from free text. Stored scores
are retained exactly, including tool-call penalties and timeout scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = [
    ("T cell", "base", ROOT / "experiments/unseen_eval_20260617/results/tcell/base_unseen_r16.jsonl"),
    ("T cell", "step9", ROOT / "experiments/unseen_eval_20260617/results/tcell/step9_unseen_r16.jsonl"),
    ("B cell", "base", ROOT / "experiments/unseen_eval_20260617/results/bcell/base_bcell_r16.jsonl"),
    ("B cell", "step9", ROOT / "experiments/unseen_eval_20260617/results/bcell/step9_bcell_r16.jsonl"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_result_arg(value: str) -> tuple[str, str, Path]:
    """Parse GROUP:MODEL=/path/to/results.jsonl."""
    try:
        label, raw_path = value.split("=", 1)
        group, model = label.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use GROUP:MODEL=/path/to/results.jsonl"
        ) from exc
    return group, model, Path(raw_path)


def collect_strict_rows(
    result_sets: list[tuple[str, str, Path]] = DEFAULT_RESULTS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, model, path in result_sets:
        for result in read_jsonl(path):
            sampleid = result["id"].rsplit("__rep", 1)[0]
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "sampleid": sampleid,
                    "repeat": result.get("repeat"),
                    "score": float(result.get("score", 0) or 0),
                    "score_raw": float(result.get("score_raw", result.get("score", 0)) or 0),
                    "penalty_applied": bool(result.get("penalty_applied", False)),
                    "timed_out": bool(result.get("timed_out", False)),
                    "n_correct": result.get("n_correct", 0),
                    "n_total": result.get("n_total", 4),
                    "n_tool_calls": result.get("n_tool_calls", 0),
                    "result_file": display_path(path),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["model"], row["sampleid"])].append(row["score"])

    summary = []
    for (group, model, sampleid), scores in sorted(grouped.items()):
        summary.append(
            {
                "group": group,
                "model": model,
                "sampleid": sampleid,
                "n_runs": len(scores),
                "mean_strict_score": sum(scores) / len(scores),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        action="append",
        type=parse_result_arg,
        metavar="GROUP:MODEL=JSONL",
        help="Result JSONL; repeatable. Defaults to the Figure 1 T- and B-cell runs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).with_name("figure1_strict_scores"),
    )
    args = parser.parse_args()

    rows = collect_strict_rows(args.results or DEFAULT_RESULTS)
    summary = summarize(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows_jsonl = args.output_prefix.with_suffix(".jsonl")
    rows_csv = args.output_prefix.with_suffix(".csv")
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")

    with rows_jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    row_fields = [
        "group", "model", "sampleid", "repeat", "score", "score_raw",
        "penalty_applied", "timed_out", "n_correct", "n_total",
        "n_tool_calls", "result_file",
    ]
    summary_fields = ["group", "model", "sampleid", "n_runs", "mean_strict_score"]
    write_csv(rows_csv, rows, row_fields)
    write_csv(summary_csv, summary, summary_fields)

    print(f"Wrote {len(rows)} strict-scored runs to {rows_jsonl} and {rows_csv}")
    print(f"Wrote {len(summary)} task-level means to {summary_csv}")


if __name__ == "__main__":
    main()
