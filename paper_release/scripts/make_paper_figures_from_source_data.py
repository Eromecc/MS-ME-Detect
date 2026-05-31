#!/usr/bin/env python3
"""Create lightweight audit figures from paper_release source tables."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simple paper audit figures from source-data CSVs."
    )
    parser.add_argument(
        "--release-dir",
        default="paper_release",
        help="Path to the paper_release directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_release/generated_figures",
        help="Directory for generated figure files.",
    )
    return parser.parse_args()


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rank_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    ranked = sorted(rows, key=lambda row: float(row["roc_auc"]), reverse=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "model", "roc_auc", "f1"])
        writer.writeheader()
        for idx, row in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "model": row["model"],
                    "roc_auc": row["roc_auc"],
                    "f1": row["f1"],
                }
            )


def make_bar_plot(rows: list[dict[str, str]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local env
        print(f"WARNING: matplotlib unavailable; skipped PNG plot: {exc}", file=sys.stderr)
        return False

    ranked = sorted(rows, key=lambda row: float(row["roc_auc"]), reverse=True)
    names = [row["model"] for row in ranked]
    values = [float(row["roc_auc"]) for row in ranked]

    fig_height = max(4.0, 0.35 * len(names))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(names[::-1], values[::-1], color="#2f6f73")
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Figure 3 audit sketch: external benchmark ROC-AUC")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def warn_if_missing_or_todo(path: Path, figure_name: str) -> None:
    if not path.exists():
        print(f"WARNING: {figure_name} source data missing: {path}", file=sys.stderr)
        return
    text = path.read_text(errors="replace")
    if "TODO" in text:
        print(f"WARNING: {figure_name} source data contains TODO rows: {path}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure3_path = release_dir / "tables" / "table1_external_metrics.csv"
    if not figure3_path.exists():
        raise SystemExit(f"Missing Figure 3 source table: {figure3_path}")
    rows = read_table(figure3_path)
    write_rank_csv(rows, output_dir / "figure3_external_benchmark_ranked.csv")
    made_plot = make_bar_plot(rows, output_dir / "figure3_external_benchmark_roc_auc.png")

    warn_if_missing_or_todo(release_dir / "figure_data" / "figure2_source_data.csv", "Figure 2")
    warn_if_missing_or_todo(release_dir / "figure_data" / "figure4_source_data.csv", "Figure 4")

    print(f"Wrote {output_dir / 'figure3_external_benchmark_ranked.csv'}")
    if made_plot:
        print(f"Wrote {output_dir / 'figure3_external_benchmark_roc_auc.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
