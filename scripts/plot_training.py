# re-plot training run from training_metrics.csv and episode_metrics.csv without re-training.

# usage: python scripts/plot_training.py --run-dir results/training/ppo/20260217_110800_pmf-check
#       python scripts/plot_training.py --run-dir results/training/lstm/20260217_110800_pmf-check


from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_utils import (
    plot_decision_time_histogram,
    plot_decision_time_over_time,
    plot_latency,
    plot_network_distribution,
    plot_ratio,
)
from scripts.train_utils import (
    enrich_episode_rows_with_scenarios,
    plot_action_by_scenario,
    plot_action_stats,
    plot_action_trajectory,
    plot_convergence_dashboard,
    plot_iter_time,
    plot_learner_diagnostics,
    plot_observation_age,
    plot_path_quality_scatter,
    plot_plr_by_traffic_class,
    plot_plr_conditions,
    plot_reward,
    plot_reward_breakdown_stacked,
    plot_reward_by_traffic_class,
    plot_reward_components,
    plot_reward_heatmap,
    plot_rtt_by_traffic_class,
    plot_rtt_conditions,
    plot_scenario_distribution,
    plot_total_reward_per_episode,
    plot_network_quality_dashboard,
    plot_actual_rtt_plr,
    plot_cost_breakdown_stacked,
)


def load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_training_records(run_dir: Path) -> List[Dict[str, Any]]:
    csv_path = run_dir / "training_step_records.csv"
    if csv_path.exists():
        return load_csv(csv_path)
    json_path = run_dir / "training_records.json"
    if json_path.exists():
        with json_path.open("r") as f:
            return json.load(f)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-plot training run from training_metrics.csv")
    parser.add_argument("--run-dir", required=True, help="Path to training run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        return 1
    if not run_dir.is_dir():
        print(f"Error: Not a directory: {run_dir}")
        return 1

    print(f"[plot_training] Run dir: {run_dir}")

    rows = load_csv(run_dir / "training_metrics.csv")
    episode_rows = load_csv(run_dir / "episode_metrics.csv")
    training_records = load_training_records(run_dir)

    if not rows:
        print("[plot_training] No training_metrics.csv found. Nothing to plot.")
        return 1

    for r in rows:
        for k, v in r.items():
            if v != "" and k != "iteration":
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
            elif k == "iteration":
                r[k] = int(float(v))

    # coerce string fields from CSV back to numeric types
    _str_fields = {"scenario_label", "traffic_class", "user_data_mode"}
    for r in episode_rows:
        for k, v in r.items():
            if v == "" or k in _str_fields:
                continue
            if k in ("episode_index", "iteration"):
                try:
                    r[k] = int(float(v))
                except (ValueError, TypeError):
                    pass
            else:
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass

    for r in training_records:
        for k, v in r.items():
            if v == "" or k in _str_fields:
                continue
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass

    if episode_rows and training_records:
        if not episode_rows[0].get("scenario_label"):
            enrich_episode_rows_with_scenarios(episode_rows, training_records)

    # detect whether this run used scenarios
    has_scenarios = any(r.get("scenario_label") for r in episode_rows) if episode_rows else False

    plot_reward(rows, run_dir / "reward_curve.png")
    plot_iter_time(rows, run_dir / "iteration_time.png")
    plot_learner_diagnostics(rows, run_dir / "training_diagnostics.png")
    plot_rtt_conditions(rows, run_dir / "rtt_conditions.png")
    plot_plr_conditions(rows, run_dir / "plr_conditions.png")
    plot_action_stats(rows, run_dir / "action_stats.png")
    plot_reward_components(rows, run_dir / "reward_components.png")
    plot_observation_age(rows, run_dir / "observation_age.png")
    plot_convergence_dashboard(rows, run_dir / "convergence_dashboard.png")
    plot_reward_breakdown_stacked(rows, run_dir / "reward_breakdown_stacked.png")
    plot_actual_rtt_plr(rows, run_dir / "actual_rtt_plr.png")
    plot_cost_breakdown_stacked(rows, run_dir / "cost_breakdown_stacked.png")
    print("[plot_training] Saved iteration-level plots (12)")

    if episode_rows:
        plot_total_reward_per_episode(episode_rows, run_dir / "total_reward_per_episode.png")
        print("[plot_training] Saved episode-level plots (1)")

    if training_records:
        plot_latency(
            training_records,
            run_dir / "latency_over_time.png",
            title="Training Step Latency Over Time",
        )
        plot_ratio(training_records, run_dir / "ratio_over_time.png")
        plot_network_distribution(training_records, run_dir / "network_distribution.png")
        plot_action_trajectory(training_records, run_dir / "action_trajectory.png")
        plot_path_quality_scatter(training_records, run_dir / "path_quality_scatter.png")
        plot_decision_time_over_time(
            training_records, run_dir / "decision_time_over_time.png",
            title="Training Decision Time Over Time",
        )
        plot_decision_time_histogram(
            training_records, run_dir / "decision_time_histogram.png",
            title="Training Decision Time Distribution",
        )
        plot_network_quality_dashboard(
            rows, run_dir / "network_quality_dashboard.png",
            step_records=training_records,
        )
        print("[plot_training] Saved step-level plots (8)")

    # scenario-aware plots
    if has_scenarios and episode_rows:
        plot_reward_by_traffic_class(episode_rows, run_dir / "reward_by_traffic_class.png")
        plot_scenario_distribution(episode_rows, run_dir / "scenario_distribution.png")
        plot_action_by_scenario(episode_rows, run_dir / "action_by_scenario.png")
        plot_reward_heatmap(episode_rows, run_dir / "reward_heatmap.png")
        print("[plot_training] Saved scenario episode plots (4)")
    if has_scenarios and training_records:
        plot_rtt_by_traffic_class(training_records, run_dir / "rtt_by_traffic_class.png")
        plot_plr_by_traffic_class(training_records, run_dir / "plr_by_traffic_class.png")
        print("[plot_training] Saved scenario step plots (2)")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print("\n[plot_training] Summary:")
        print(json.dumps({k: v for k, v in summary.items() if k != "artifacts"}, indent=2))

    print(f"\n[plot_training] Done. Plots saved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
