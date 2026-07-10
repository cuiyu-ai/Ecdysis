#!/usr/bin/env python3
"""Aggregate experiment results into paper-ready tables.

Reads from data/experiments/ and generates:
- CSV tables for the paper
- Summary statistics
- LaTeX table snippets

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --domain airline
    python scripts/aggregate_results.py --format latex
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
TABLES_DIR = PROJECT_ROOT / "data" / "paper_tables"


def load_experiment_results(exp_name: str, domain: str) -> list:
    """Load all results for an experiment/domain combination."""
    exp_dir = EXPERIMENTS_DIR / exp_name / domain
    if not exp_dir.exists():
        return []

    results = []
    for f in sorted(exp_dir.glob("*.json")):
        data = json.loads(f.read_text())
        results.append(data)

    return results


def get_best_result(results: list) -> dict:
    """Get the best (latest) result from a list."""
    if not results:
        return None
    return results[-1]


def build_baseline_table(domains: list) -> dict:
    """Build Table 1: Baseline comparison (E1 vs E2)."""
    table = {}

    for domain in domains:
        e1_results = load_experiment_results("E1_baseline", domain)
        e2_results = load_experiment_results("E2_frozen_harness", domain)

        e1 = get_best_result(e1_results)
        e2 = get_best_result(e2_results)

        if e1 and e2:
            e1_pass = e1["summary"].get("pass@k", 0)
            e2_pass = e2["summary"].get("pass@k", 0)
            improvement = (e2_pass - e1_pass) / e1_pass * 100 if e1_pass > 0 else 0

            table[domain] = {
                "no_harness": e1_pass,
                "frozen_harness": e2_pass,
                "improvement": improvement,
                "no_harness_tokens": e1["summary"].get("total_token_count", 0),
                "frozen_harness_tokens": e2["summary"].get("total_token_count", 0),
            }

    return table


def build_evolution_table(domains: list) -> dict:
    """Build Table 2: Evolution comparison (E3 vs E4 vs E5)."""
    table = {}

    for domain in domains:
        table[domain] = {}

        for exp in ["E3_original_evolution", "E4_cross_instance_evolution", "E5_debate"]:
            results = load_experiment_results(exp, domain)
            best = get_best_result(results)

            if best:
                table[domain][exp] = {
                    "pass@k": best["summary"].get("pass@k", 0),
                    "avg_reward": best["summary"].get("average_reward", 0),
                    "total_tokens": best["summary"].get("total_token_count", 0),
                    "rounds": len(results),
                }

    return table


def format_csv_baseline(table: dict) -> str:
    """Format baseline table as CSV."""
    lines = ["Domain,No Harness,Frozen Harness,Improvement (%),No Harness Tokens,Frozen Harness Tokens"]

    for domain, data in sorted(table.items()):
        lines.append(f"{domain},{data['no_harness']:.3f},{data['frozen_harness']:.3f},"
                     f"{data['improvement']:.1f},{data['no_harness_tokens']},{data['frozen_harness_tokens']}")

    # Add average
    if table:
        avg_no_h = sum(d['no_harness'] for d in table.values()) / len(table)
        avg_frozen = sum(d['frozen_harness'] for d in table.values()) / len(table)
        avg_imp = (avg_frozen - avg_no_h) / avg_no_h * 100 if avg_no_h > 0 else 0
        lines.append(f"Average,{avg_no_h:.3f},{avg_frozen:.3f},{avg_imp:.1f},,")

    return "\n".join(lines)


def format_csv_evolution(table: dict) -> str:
    """Format evolution table as CSV."""
    lines = ["Domain,E3 Original,E4 Cross-Instance Learning,E5 Debate,E3 Tokens,E4 Tokens,E5 Tokens"]

    for domain, data in sorted(table.items()):
        row = [domain]
        tokens = []
        for exp in ["E3_original_evolution", "E4_cross_instance_evolution", "E5_debate"]:
            if exp in data:
                row.append(f"{data[exp]['pass@k']:.3f}")
                tokens.append(str(data[exp]['total_tokens']))
            else:
                row.append("N/A")
                tokens.append("N/A")
        lines.append(",".join(row + tokens))

    return "\n".join(lines)


def format_latex_baseline(table: dict) -> str:
    """Format baseline table as LaTeX."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Baseline Comparison (Qwen3-8B)}",
        "\\label{tab:baseline}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Domain & No Harness & Frozen Harness & Improvement \\\\",
        "\\midrule",
    ]

    for domain, data in sorted(table.items()):
        lines.append(f"{domain} & {data['no_harness']:.1f}\\% & {data['frozen_harness']:.1f}\\% & "
                     f"+{data['improvement']:.1f}\\% \\\\")

    if table:
        avg_no_h = sum(d['no_harness'] for d in table.values()) / len(table)
        avg_frozen = sum(d['frozen_harness'] for d in table.values()) / len(table)
        avg_imp = (avg_frozen - avg_no_h) / avg_no_h * 100 if avg_no_h > 0 else 0
        lines.append("\\midrule")
        lines.append(f"Average & {avg_no_h:.1f}\\% & {avg_frozen:.1f}\\% & +{avg_imp:.1f}\\% \\\\")

    lines.extend(["\\Cross-Instance Learningtomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--domain", help="Specific domain (default: all)")
    parser.add_argument("--format", choices=["csv", "latex"], default="csv", help="Output format")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    domains = [args.domain] if args.domain else ["airline", "retail", "telecom"]

    # Table 1: Baseline
    baseline_table = build_baseline_table(domains)
    if baseline_table:
        if args.format == "csv":
            content = format_csv_baseline(baseline_table)
            outfile = TABLES_DIR / "table1_baseline.csv"
        else:
            content = format_latex_baseline(baseline_table)
            outfile = TABLES_DIR / "table1_baseline.tex"

        outfile.write_text(content)
        print(f"Table 1 saved ->{outfile}")

    # Table 2: Evolution
    evolution_table = build_evolution_table(domains)
    if evolution_table:
        content = format_csv_evolution(evolution_table)
        outfile = TABLES_DIR / "table2_evolution.csv"
        outfile.write_text(content)
        print(f"Table 2 saved ->{outfile}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    for domain in domains:
        print(f"\n{domain.upper()}:")
        if domain in baseline_table:
            d = baseline_table[domain]
            print(f"  No harness:     {d['no_harness']:.1%}")
            print(f"  Frozen harness: {d['frozen_harness']:.1%}")
            print(f"  Improvement:    +{d['improvement']:.1f}%")

        if domain in evolution_table:
            for exp, data in evolution_table[domain].items():
                print(f"  {exp}: {data['pass@k']:.1%}")


if __name__ == "__main__":
    main()
