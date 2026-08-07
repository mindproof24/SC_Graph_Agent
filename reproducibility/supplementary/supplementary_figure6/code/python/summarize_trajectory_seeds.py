#!/usr/bin/env python3
"""Summarize repeated trajectory path-association runs across random seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/elicer/SC_Graph_Agent/benchmarks/trajectory_seed_comparison"),
        help="Directory containing seed_<N>/trajectory_cardiac_metrics.csv.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--per-seed-name", default="trajectory_5seed_per_seed.csv")
    parser.add_argument("--summary-name", default="trajectory_5seed_summary.csv")
    return parser.parse_args()


def collect_seed_results(root: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = root / f"seed_{seed}" / "trajectory_cardiac_metrics.csv"
        data = pd.read_csv(path)
        for tf in sorted(data["tf"].unique()):
            table = data[data["tf"] == tf].set_index("method")
            astar = float(table.loc["astar_path", "coherence"])
            random_paths = float(table.loc["random_paths", "coherence"])
            shuffle = table.loc["ASTAR_vs_SHUFFLE"]
            program_null = float(shuffle["coh_shuffle_mean"])
            rows.append({
                "seed": seed,
                "tf": tf,
                "astar_coherence": astar,
                "program_shuffle_mean": program_null,
                "random_paths_coherence": random_paths,
                "delta_vs_program": astar - program_null,
                "delta_vs_random": astar - random_paths,
                "within_seed_z": float(shuffle["coh_z"]),
            })
    return pd.DataFrame(rows)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "astar_coherence",
        "program_shuffle_mean",
        "random_paths_coherence",
        "delta_vs_program",
        "delta_vs_random",
        "within_seed_z",
    ]
    rows = []
    for tf, group in per_seed.groupby("tf", sort=True):
        row = {"tf": tf, "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        row["program_delta_positive_seeds"] = int((group["delta_vs_program"] > 0).sum())
        row["random_delta_positive_seeds"] = int((group["delta_vs_random"] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    per_seed = collect_seed_results(args.root, seeds)
    summary = summarize(per_seed)
    per_seed.to_csv(args.root / args.per_seed_name, index=False)
    summary.to_csv(args.root / args.summary_name, index=False)
    print("PER-SEED RESULTS")
    print(per_seed.round(3).to_string(index=False))
    print("\nSUMMARY (mean and sample SD across seeds)")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
