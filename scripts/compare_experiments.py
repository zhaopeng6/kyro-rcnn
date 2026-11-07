#!/usr/bin/env python3
"""Create a concise Phase-4 comparison from two scale benchmark JSON reports."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment-a", required=True); parser.add_argument("--experiment-b", required=True); parser.add_argument("--output", default="outputs/phase4_comparison.md")
    args = parser.parse_args()
    reports = {"Experiment A": json.loads(Path(args.experiment_a).read_text()), "Experiment B": json.loads(Path(args.experiment_b).read_text())}
    lines = ["# Phase 4 experiment comparison", "", "All comparisons require identical held-out data, scale factors, score threshold, and checkpoint-selection policy.", "", "| Metric | Experiment A | Experiment B |", "|---|---:|---:|"]
    for metric in ("robustness_score_mean_mask_ap50_proxy",):
        lines.append(f"| {metric} | {reports['Experiment A'].get(metric)} | {reports['Experiment B'].get(metric)} |")
    lines += ["", "## Per-scale results", ""]
    for factor_a, factor_b in zip(reports["Experiment A"]["results"], reports["Experiment B"]["results"]):
        lines.append(f"### Scale {factor_a['factor']}")
        lines += ["", f"- Mask AP50 proxy: A={factor_a['mask_ap50_proxy']:.4f}, B={factor_b['mask_ap50_proxy']:.4f}", f"- Instance Dice: A={factor_a['instance_mean_dice']:.4f}, B={factor_b['instance_mean_dice']:.4f}", f"- Recall@0.5: A={factor_a['instance_recall50']:.4f}, B={factor_b['instance_recall50']:.4f}", f"- Split candidate rate: A={factor_a['split_candidate_rate']:.4f}, B={factor_b['split_candidate_rate']:.4f}", f"- Merge candidate rate: A={factor_a['merge_candidate_rate']:.4f}, B={factor_b['merge_candidate_rate']:.4f}", ""]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__": main()
