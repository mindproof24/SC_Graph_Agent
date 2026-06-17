#!/usr/bin/env python3
"""Build summary tables for the 2026-06-17 UNSEEN evaluation snapshot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


TCELL_NAMES = {
    "imm_atlas_TT_p01_CD8-positive_alpha-b_vs_effector_memory_CD8-_r000_anon": "CD8 memory vs eff.mem CD8",
    "imm_atlas_TT_p01_CD4-positive_helper__vs_effector_memory_CD4-_r000_anon": "CD4 helper vs eff.mem CD4",
    "imm_atlas_TT_p01_gamma-delta_T_cell_vs_regulatory_T_cell_r000_anon": "gamma-delta T vs Treg",
}

BCELL_NAMES = {
    "imm_atlas_BB_p01_c15c27_cell_r001_anon": "memory B vs naive B",
    "imm_atlas_BB_p01_c24c48_cell_r001_anon": "memory B vs plasmablast",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sampleid_from_result(row: dict) -> str:
    return row["id"].rsplit("__rep", 1)[0]


def mean_by_sample(rows: list[dict], score_key: str, sample_key: str = "sampleid") -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        sid = row.get(sample_key) or sampleid_from_result(row)
        grouped[sid].append(float(row.get(score_key, 0) or 0))
    return {sid: mean(vals) for sid, vals in grouped.items()}


def strict_means(path: Path) -> dict[str, float]:
    return mean_by_sample(read_jsonl(path), "score")


def adjusted_means(path: Path, model: str) -> dict[str, float]:
    rows = [row for row in read_jsonl(path) if row["model"] == model]
    return mean_by_sample(rows, "adjusted_score")


def overall(means: dict[str, float]) -> float:
    return mean(means.values())


def fmt(x: float) -> str:
    return f"{x:.3f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_tables(root: Path) -> str:
    tcell_dir = root / "results" / "tcell"
    bcell_dir = root / "results" / "bcell"

    tcell_strict = {
        "base": strict_means(tcell_dir / "base_unseen_r16.jsonl"),
        "step9": strict_means(tcell_dir / "step9_unseen_r16.jsonl"),
    }
    tcell_adjusted = {
        "base": adjusted_means(tcell_dir / "format_corrected_regrade_audited_unseen.jsonl", "base"),
        "step9": adjusted_means(tcell_dir / "format_corrected_regrade_audited_unseen.jsonl", "step9"),
    }
    bcell_strict = {
        "base": strict_means(bcell_dir / "base_bcell_r16.jsonl"),
        "step9": strict_means(bcell_dir / "step9_bcell_r16.jsonl"),
    }
    bcell_adjusted = {
        "base": adjusted_means(bcell_dir / "bcell_format_corrected_regrade_audited.jsonl", "base"),
        "step9": adjusted_means(bcell_dir / "bcell_format_corrected_regrade_audited.jsonl", "step9"),
    }

    def delta_row(values: dict[str, dict[str, float]], keys: list[str]) -> list[str]:
        return ["delta"] + [fmt(values["step9"][key] - values["base"][key]) for key in keys]

    tkeys = list(TCELL_NAMES)
    bkeys = list(BCELL_NAMES)

    sections = [
        "# UNSEEN Evaluation Summary",
        "",
        "Models:",
        "- `base`: `qwen35-base-8k` for the T-cell run, `qwen35-base-16k` for the B-cell run.",
        "- `step9`: `qwen35-grpo-step9-16k`.",
        "",
        "Scoring:",
        "- Strict uses the original automatic `{\"labels\": [...]}` parser.",
        "- Format-corrected audited scoring rescues explicit cell-wise classifications from malformed JSON/tables.",
        "- One correct cell contributes 0.25.",
        "",
        "## T-cell UNSEEN, Strict",
        "",
        markdown_table(
            ["Model", "UNSEEN overall"] + [TCELL_NAMES[k] for k in tkeys],
            [
                ["base", fmt(overall(tcell_strict["base"]))] + [fmt(tcell_strict["base"][k]) for k in tkeys],
                ["step9", fmt(overall(tcell_strict["step9"]))] + [fmt(tcell_strict["step9"][k]) for k in tkeys],
                ["delta", fmt(overall(tcell_strict["step9"]) - overall(tcell_strict["base"]))]
                + [fmt(tcell_strict["step9"][k] - tcell_strict["base"][k]) for k in tkeys],
            ],
        ),
        "",
        "## T-cell UNSEEN, Format-Corrected Audited",
        "",
        markdown_table(
            ["Model", "UNSEEN overall"] + [TCELL_NAMES[k] for k in tkeys],
            [
                ["base", fmt(overall(tcell_adjusted["base"]))] + [fmt(tcell_adjusted["base"][k]) for k in tkeys],
                ["step9", fmt(overall(tcell_adjusted["step9"]))] + [fmt(tcell_adjusted["step9"][k]) for k in tkeys],
                ["delta", fmt(overall(tcell_adjusted["step9"]) - overall(tcell_adjusted["base"]))]
                + [fmt(tcell_adjusted["step9"][k] - tcell_adjusted["base"][k]) for k in tkeys],
            ],
        ),
        "",
        "## B-cell UNSEEN, Strict",
        "",
        markdown_table(
            ["Model", "B-cell overall"] + [BCELL_NAMES[k] for k in bkeys],
            [
                ["base", fmt(overall(bcell_strict["base"]))] + [fmt(bcell_strict["base"][k]) for k in bkeys],
                ["step9", fmt(overall(bcell_strict["step9"]))] + [fmt(bcell_strict["step9"][k]) for k in bkeys],
                ["delta", fmt(overall(bcell_strict["step9"]) - overall(bcell_strict["base"]))]
                + [fmt(bcell_strict["step9"][k] - bcell_strict["base"][k]) for k in bkeys],
            ],
        ),
        "",
        "## B-cell UNSEEN, Format-Corrected Audited",
        "",
        markdown_table(
            ["Model", "B-cell overall"] + [BCELL_NAMES[k] for k in bkeys],
            [
                ["base", fmt(overall(bcell_adjusted["base"]))] + [fmt(bcell_adjusted["base"][k]) for k in bkeys],
                ["step9", fmt(overall(bcell_adjusted["step9"]))] + [fmt(bcell_adjusted["step9"][k]) for k in bkeys],
                ["delta", fmt(overall(bcell_adjusted["step9"]) - overall(bcell_adjusted["base"]))]
                + [fmt(bcell_adjusted["step9"][k] - bcell_adjusted["base"][k]) for k in bkeys],
            ],
        ),
        "",
        "## Combined UNSEEN",
        "",
    ]

    strict_base_total = mean(list(tcell_strict["base"].values()) + list(bcell_strict["base"].values()))
    strict_step_total = mean(list(tcell_strict["step9"].values()) + list(bcell_strict["step9"].values()))
    adj_base_total = mean(list(tcell_adjusted["base"].values()) + list(bcell_adjusted["base"].values()))
    adj_step_total = mean(list(tcell_adjusted["step9"].values()) + list(bcell_adjusted["step9"].values()))

    sections += [
        markdown_table(
            ["Scoring", "base", "step9", "delta"],
            [
                ["Strict", fmt(strict_base_total), fmt(strict_step_total), fmt(strict_step_total - strict_base_total)],
                ["Format-corrected audited", fmt(adj_base_total), fmt(adj_step_total), fmt(adj_step_total - adj_base_total)],
            ],
        ),
        "",
        "## Data Files",
        "",
        "Compressed h5ad files are stored in `data/h5ad_gz/`.",
        "Decompress with `gzip -dk data/h5ad_gz/*.h5ad.gz` if local h5ad files are needed.",
        "",
    ]

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    text = build_tables(args.root)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
