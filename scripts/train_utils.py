# shared utilities for train.py and plot_training.py.

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def create_run_dir(base: Path, algo: str, tag: str) -> Path:
    run_dir = base / "results" / "training" / algo / f"{utc_ts()}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def resolve_run_dir(base: Path, algo: str, tag: str, run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser()
        if not run_dir.is_absolute():
            run_dir = (base / run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        return run_dir
    return create_run_dir(base, algo, tag)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def default_config_path(repo_root: Path, algo: str) -> Path:
    return repo_root / "configs" / "training" / f"{algo}.yaml"


def normalize_checkpoint_path(checkpoint_obj: Any) -> str:
    if isinstance(checkpoint_obj, str):
        return checkpoint_obj
    checkpoint = getattr(checkpoint_obj, "checkpoint", None)
    if checkpoint is not None:
        path = getattr(checkpoint, "path", None)
        if path:
            return str(path)
    return str(checkpoint_obj)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


STEP_CSV_COLUMNS = [
    "elapsed_s", "step", "episode_idx", "step_in_episode", "latency_ms", "decision_time_ms",
    "wifi_rtt_ms", "fiveg_rtt_ms", "wifi_plr", "fiveg_plr",
    "wifi_rtt_min_ms", "wifi_rtt_max_ms", "wifi_rtt_std_ms",
    "fiveg_rtt_min_ms", "fiveg_rtt_max_ms", "fiveg_rtt_std_ms",
    "observation_age_ms",
    "wifi_ratio", "fiveg_ratio",
    "weighted_rtt_ms", "weighted_plr", "delta_action",
    "rtt_term", "plr_term", "stability_term", "reward_total",
    "scenario_label",
    "traffic_class",
    "rtt_preset_idx",
    "loss_pair_idx",
    "capacity_ratio_idx",
    "cross_traffic_level_idx",
    "wifi_extra_delay_ms",
    "fiveg_extra_delay_ms",
    "wifi_loss_pct",
    "fiveg_loss_pct",
    "wifi_rate_limit_mbps",
    "fiveg_rate_limit_mbps",
    "cross_traffic_mbps",
    "cross_traffic_on_wifi",
    "user_data_mode",
    "user_data_mbps",
    "bursty_mbit_per_period",
    "bursty_period_s",
    "bytes_sent_wifi", "bytes_sent_fiveg",
    "timestamp_decision_sent",
    "timestamp_decision_applied",
    "timestamp_observation",
]


def save_step_csv(records: List[Dict[str, Any]], path: Path) -> None:
    """Write per-step training records to CSV."""
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STEP_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _safe_mean(values: List) -> float | None:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 6) if nums else None


def _safe_min(values: List) -> float | None:
    nums = [v for v in values if v is not None]
    return round(min(nums), 6) if nums else None


def _safe_max(values: List) -> float | None:
    nums = [v for v in values if v is not None]
    return round(max(nums), 6) if nums else None


def _safe_std(values: List) -> float | None:
    nums = [v for v in values if v is not None]
    if len(nums) < 2:
        return None
    mean = sum(nums) / len(nums)
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    return round(var ** 0.5, 6)


def _aggregate_iter_records(iter_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-iteration aggregates from step records."""
    if not iter_records:
        return {}
    return {
        "wifi_rtt_ms_mean": _safe_mean([r.get("wifi_rtt_ms") for r in iter_records]),
        "wifi_rtt_ms_min": _safe_min([r.get("wifi_rtt_ms") for r in iter_records]),
        "wifi_rtt_ms_max": _safe_max([r.get("wifi_rtt_ms") for r in iter_records]),
        "fiveg_rtt_ms_mean": _safe_mean([r.get("fiveg_rtt_ms") for r in iter_records]),
        "fiveg_rtt_ms_min": _safe_min([r.get("fiveg_rtt_ms") for r in iter_records]),
        "fiveg_rtt_ms_max": _safe_max([r.get("fiveg_rtt_ms") for r in iter_records]),
        "wifi_rtt_std_ms_mean": _safe_mean([r.get("wifi_rtt_std_ms") for r in iter_records]),
        "fiveg_rtt_std_ms_mean": _safe_mean([r.get("fiveg_rtt_std_ms") for r in iter_records]),
        "wifi_plr_mean": _safe_mean([r.get("wifi_plr") for r in iter_records]),
        "fiveg_plr_mean": _safe_mean([r.get("fiveg_plr") for r in iter_records]),
        "wifi_ratio_mean": _safe_mean([r.get("wifi_ratio") for r in iter_records]),
        "wifi_ratio_std": _safe_std([r.get("wifi_ratio") for r in iter_records]),
        "weighted_rtt_ms_mean": _safe_mean([r.get("weighted_rtt_ms") for r in iter_records]),
        "weighted_plr_mean": _safe_mean([r.get("weighted_plr") for r in iter_records]),
        "rtt_term_mean": _safe_mean([r.get("rtt_term") for r in iter_records]),
        "plr_term_mean": _safe_mean([r.get("plr_term") for r in iter_records]),
        "stability_term_mean": _safe_mean([r.get("stability_term") for r in iter_records]),
        "observation_age_ms_mean": _safe_mean([r.get("observation_age_ms") for r in iter_records]),
        "observation_age_ms_max": _safe_max([r.get("observation_age_ms") for r in iter_records]),
    }


def extract_metrics(
    result: Dict[str, Any],
    iteration_idx: int,
    iter_s: float,
    elapsed_s: float,
    success_reward_threshold: float | None,
    iter_records: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    env_runners = result.get("env_runners", {}) if isinstance(result, dict) else {}
    reward_mean = result.get("episode_reward_mean")
    reward_min = result.get("episode_reward_min")
    reward_max = result.get("episode_reward_max")
    episode_len_mean = result.get("episode_len_mean")
    if episode_len_mean is None:
        episode_len_mean = env_runners.get("episode_len_mean")

    episodes_this_iter = env_runners.get("episodes_this_iter")
    hist_stats = env_runners.get("hist_stats", {}) if isinstance(env_runners, dict) else {}
    reward_hist = hist_stats.get("episode_reward", []) if isinstance(hist_stats, dict) else []

    custom_metrics = result.get("custom_metrics", {}) if isinstance(result, dict) else {}
    if not custom_metrics and isinstance(env_runners, dict):
        custom_metrics = env_runners.get("custom_metrics", {}) or {}
    success_rate_iter = custom_metrics.get("success_rate_mean")
    if success_rate_iter is None:
        success_rate_iter = custom_metrics.get("success_rate")
    if success_rate_iter is None and success_reward_threshold is not None and reward_hist:
        successes = sum(1 for r in reward_hist if r >= success_reward_threshold)
        success_rate_iter = successes / len(reward_hist)

    if reward_mean is None:
        reward_mean = env_runners.get("episode_reward_mean")
    if reward_min is None:
        reward_min = env_runners.get("episode_reward_min")
    if reward_max is None:
        reward_max = env_runners.get("episode_reward_max")

    row = {
        "iteration": iteration_idx,
        "iteration_time_s": round(iter_s, 4),
        "elapsed_time_s": round(elapsed_s, 4),
        "episode_reward_mean": reward_mean,
        "episode_reward_min": reward_min,
        "episode_reward_max": reward_max,
        "episode_len_mean": episode_len_mean,
        "episodes_total": result.get("episodes_total") or env_runners.get("num_episodes"),
        "episodes_this_iter": episodes_this_iter,
        "success_rate_iter": success_rate_iter,
        "timesteps_total": result.get("timesteps_total"),
        "num_env_steps_sampled": result.get("num_env_steps_sampled"),
        "num_env_steps_trained": result.get("num_env_steps_trained"),
        "learner_kl": result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("kl"),
        "learner_entropy": result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("entropy"),
        "learner_policy_loss": result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("policy_loss"),
        "learner_vf_loss": result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("vf_loss"),
    }

    if iter_records:
        row.update(_aggregate_iter_records(iter_records))

    return row


def extract_episode_rows(
    result: Dict[str, Any],
    iteration_idx: int,
    start_episode_idx: int,
    success_reward_threshold: float | None,
) -> Tuple[List[Dict[str, Any]], int]:
    env_runners = result.get("env_runners", {}) if isinstance(result, dict) else {}
    hist_stats = env_runners.get("hist_stats", {}) if isinstance(env_runners, dict) else {}
    rewards = hist_stats.get("episode_reward", []) if isinstance(hist_stats, dict) else []
    lengths = hist_stats.get("episode_lengths", []) if isinstance(hist_stats, dict) else []

    count = min(len(rewards), len(lengths))
    rows: List[Dict[str, Any]] = []
    ep_idx = start_episode_idx
    for i in range(count):
        ep_idx += 1
        reward = rewards[i]
        length = lengths[i]
        success = None
        if success_reward_threshold is not None:
            success = 1 if reward >= success_reward_threshold else 0
        rows.append(
            {
                "episode_index": ep_idx,
                "iteration": iteration_idx,
                "episode_reward_total": reward,
                "episode_steps": length,
                "success": success,
            }
        )
    return rows, ep_idx


def enrich_episode_rows_with_scenarios(
    episode_rows: List[Dict[str, Any]],
    step_records: List[Dict[str, Any]],
) -> None:
    #add scenario metadata to episode rows by looking up step records.

    #each episode row gets scenario_label, traffic_class, and per-episode
    #aggregates (mean wifi_ratio, mean reward) derived from step-level data.
    #modifies episode_rows in place.
    if not step_records:
        return

    episodes: Dict[int, List[Dict[str, Any]]] = {}
    for r in step_records:
        eidx = r.get("episode_idx")
        if eidx is not None:
            episodes.setdefault(eidx, []).append(r)

    for ep_row in episode_rows:
        ep_idx = ep_row.get("episode_index")
        steps = episodes.get(ep_idx, [])
        if not steps:
            continue

        ep_row["scenario_label"] = steps[0].get("scenario_label")
        ep_row["traffic_class"] = steps[0].get("traffic_class")
        ep_row["capacity_ratio_idx"] = steps[0].get("capacity_ratio_idx")
        wifi_ratios = [s["wifi_ratio"] for s in steps if s.get("wifi_ratio") is not None]
        rewards = [s["reward_total"] for s in steps if s.get("reward_total") is not None]
        ep_row["wifi_ratio_mean"] = round(sum(wifi_ratios) / len(wifi_ratios), 6) if wifi_ratios else None
        ep_row["reward_mean_per_step"] = round(sum(rewards) / len(rewards), 6) if rewards else None


def save_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_episode_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_reward(rows: List[Dict[str, Any]], out_path: Path) -> None:
    xs = [r["iteration"] for r in rows]
    ys = [r["episode_reward_mean"] for r in rows]
    ymin = [r["episode_reward_min"] for r in rows]
    ymax = [r["episode_reward_max"] for r in rows]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, label="reward mean", linewidth=1.2)
    if all(v is not None for v in ymin) and all(v is not None for v in ymax):
        plt.fill_between(xs, ymin, ymax, alpha=0.15, label="min/max range")
    plt.title("Training Reward Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Episode Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_iter_time(rows: List[Dict[str, Any]], out_path: Path) -> None:
    xs = [r["iteration"] for r in rows]
    ys = [r["iteration_time_s"] for r in rows]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, linewidth=1.2)
    plt.title("Iteration Time")
    plt.xlabel("Iteration")
    plt.ylabel("Seconds")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_total_reward_per_episode(episode_rows: List[Dict[str, Any]], out_path: Path) -> None:
    xs = [r["episode_index"] for r in episode_rows]
    ys = [r["episode_reward_total"] for r in episode_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, linewidth=1.0)
    plt.title("Total Reward Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


WIFI_COLOR = "#3498db"
FIVEG_COLOR = "#e74c3c"


def plot_rtt_conditions(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """WiFi/5G mean RTT per iteration with min/max shading."""
    xs = [r["iteration"] for r in rows if r.get("wifi_rtt_ms_mean") is not None]
    if not xs:
        return
    filtered = [r for r in rows if r.get("wifi_rtt_ms_mean") is not None]
    wifi_mean = [r["wifi_rtt_ms_mean"] for r in filtered]
    wifi_min = [r.get("wifi_rtt_ms_min") for r in filtered]
    wifi_max = [r.get("wifi_rtt_ms_max") for r in filtered]
    fiveg_mean = [r.get("fiveg_rtt_ms_mean") for r in filtered]
    fiveg_min = [r.get("fiveg_rtt_ms_min") for r in filtered]
    fiveg_max = [r.get("fiveg_rtt_ms_max") for r in filtered]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, wifi_mean, label="WiFi RTT mean", linewidth=1.2, color=WIFI_COLOR)
    if all(v is not None for v in wifi_min) and all(v is not None for v in wifi_max):
        plt.fill_between(xs, wifi_min, wifi_max, alpha=0.12, color=WIFI_COLOR)
    if all(v is not None for v in fiveg_mean):
        plt.plot(xs, fiveg_mean, label="5G RTT mean", linewidth=1.2, color=FIVEG_COLOR)
        if all(v is not None for v in fiveg_min) and all(v is not None for v in fiveg_max):
            plt.fill_between(xs, fiveg_min, fiveg_max, alpha=0.12, color=FIVEG_COLOR)
    plt.title("Network RTT Conditions Per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("RTT (ms)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_plr_conditions(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """WiFi/5G PLR per iteration."""
    xs = [r["iteration"] for r in rows if r.get("wifi_plr_mean") is not None]
    if not xs:
        return
    filtered = [r for r in rows if r.get("wifi_plr_mean") is not None]
    wifi_plr = [r["wifi_plr_mean"] for r in filtered]
    fiveg_plr = [r.get("fiveg_plr_mean") for r in filtered]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, wifi_plr, label="WiFi PLR", linewidth=1.2, color=WIFI_COLOR)
    if all(v is not None for v in fiveg_plr):
        plt.plot(xs, fiveg_plr, label="5G PLR", linewidth=1.2, color=FIVEG_COLOR)
    plt.title("Packet Loss Rate Per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("PLR")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_action_stats(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """WiFi ratio mean with +/- std band per iteration."""
    xs = [r["iteration"] for r in rows if r.get("wifi_ratio_mean") is not None]
    if not xs:
        return
    filtered = [r for r in rows if r.get("wifi_ratio_mean") is not None]
    means = [r["wifi_ratio_mean"] for r in filtered]
    stds = [r.get("wifi_ratio_std", 0.0) or 0.0 for r in filtered]
    lo = [m - s for m, s in zip(means, stds)]
    hi = [m + s for m, s in zip(means, stds)]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, means, label="wifi_ratio mean", linewidth=1.2)
    plt.fill_between(xs, lo, hi, alpha=0.15, label="+/- std")
    plt.ylim(-0.05, 1.05)
    plt.title("Action Distribution Per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("WiFi ratio")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_reward_components(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Per-iteration reward component means (rtt_term, plr_term, stability_term)."""
    xs = [r["iteration"] for r in rows if r.get("rtt_term_mean") is not None]
    if not xs:
        return
    filtered = [r for r in rows if r.get("rtt_term_mean") is not None]
    rtt = [r["rtt_term_mean"] for r in filtered]
    plr = [r.get("plr_term_mean") for r in filtered]
    stab = [r.get("stability_term_mean") for r in filtered]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, rtt, label="rtt_term", linewidth=1.2)
    if all(v is not None for v in plr):
        plt.plot(xs, plr, label="plr_term", linewidth=1.2)
    if all(v is not None for v in stab):
        plt.plot(xs, stab, label="stability_term", linewidth=1.2)
    plt.title("Reward Components Per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Mean term value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_observation_age(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Observation age (mean/max) per iteration -- data freshness indicator."""
    xs = [r["iteration"] for r in rows if r.get("observation_age_ms_mean") is not None]
    if not xs:
        return
    filtered = [r for r in rows if r.get("observation_age_ms_mean") is not None]
    means = [r["observation_age_ms_mean"] for r in filtered]
    maxes = [r.get("observation_age_ms_max") for r in filtered]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, means, label="mean obs age", linewidth=1.2)
    if all(v is not None for v in maxes):
        plt.plot(xs, maxes, label="max obs age", linewidth=1.0, alpha=0.6)
    plt.title("Observation Age Per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Observation age (ms)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# scenario-aware plots (only useful with --with-scenarios)


def plot_reward_by_traffic_class(episode_rows: List[Dict[str, Any]], out_path: Path) -> None:
    labeled = [r for r in episode_rows if r.get("traffic_class")]
    if not labeled:
        return

    groups: Dict[str, List[float]] = {}
    for r in labeled:
        tc = r["traffic_class"]
        rew = r.get("episode_reward_total")
        if rew is not None:
            groups.setdefault(tc, []).append(rew)
    if not groups:
        return

    order = [c for c in ("control", "bulk", "bursty") if c in groups]
    order += sorted(set(groups.keys()) - set(order))
    data = [groups[c] for c in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"control": "#2ecc71", "bulk": "#e67e22", "bursty": "#9b59b6"}
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanline=True)
    for patch, label in zip(bp["boxes"], order):
        patch.set_facecolor(colors.get(label, "#aec7e8"))
    ax.set_xticklabels(order, fontsize=10)
    ax.set_title("Episode Reward by Traffic Class")
    ax.set_ylabel("Episode reward (total)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scenario_distribution(episode_rows: List[Dict[str, Any]], out_path: Path) -> None:
    labeled = [r for r in episode_rows if r.get("scenario_label")]
    if not labeled:
        return

    counts: Dict[str, int] = {}
    for r in labeled:
        lbl = r["scenario_label"]
        counts[lbl] = counts.get(lbl, 0) + 1

    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 5))
    ax.bar(range(len(labels)), values, color="#5dade2", edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_title("Scenario Sampling Distribution")
    ax.set_ylabel("Episode count")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reward_by_scenario(episode_rows: List[Dict[str, Any]], out_path: Path) -> None:
    labeled = [r for r in episode_rows if r.get("scenario_label") and r.get("episode_reward_total") is not None]
    if not labeled:
        return

    groups_total: Dict[str, List[float]] = {}
    groups_per_step: Dict[str, List[float]] = {}
    for r in labeled:
        lbl = r["scenario_label"]
        groups_total.setdefault(lbl, []).append(r["episode_reward_total"])
        steps = r.get("steps_ok") or r.get("steps_total") or 1
        groups_per_step.setdefault(lbl, []).append(r["episode_reward_total"] / steps)

    labels = sorted(groups_total.keys())

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(max(18, len(labels) * 0.75), 6))

    # linear panel — total episode reward
    data_total = [groups_total[l] for l in labels]
    bp = ax_lin.boxplot(data_total, patch_artist=True, showmeans=True, meanline=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a9dfbf")
    ax_lin.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax_lin.set_ylabel("Total episode reward")
    ax_lin.set_title("Episode Reward by Scenario (linear)")
    ax_lin.grid(alpha=0.3, axis="y")

    # symlog panel 
    data_per_step = [groups_per_step[l] for l in labels]
    bp2 = ax_log.boxplot(data_per_step, patch_artist=True, showmeans=True, meanline=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#a9dfbf")
    ax_log.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax_log.set_ylabel("Reward per step")
    ax_log.set_title("Reward/step by Scenario (symlog)")
    ax_log.set_yscale("symlog", linthresh=0.01)
    ax_log.axhline(-0.1, color="crimson", linestyle="--", linewidth=1.2, label="target: −0.1/step")
    ax_log.legend(fontsize=8)
    ax_log.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_action_by_scenario(episode_rows: List[Dict[str, Any]], out_path: Path) -> None:
    labeled = [r for r in episode_rows if r.get("scenario_label") and r.get("wifi_ratio_mean") is not None]
    if not labeled:
        return

    groups: Dict[str, List[float]] = {}
    for r in labeled:
        groups.setdefault(r["scenario_label"], []).append(r["wifi_ratio_mean"])

    labels = sorted(groups.keys())
    data = [groups[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanline=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#f9e79f")
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("WiFi Ratio by Scenario")
    ax.set_ylabel("Mean wifi_ratio per episode")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reward_heatmap(episode_rows: List[Dict[str, Any]], out_path: Path) -> None:

    labeled = [r for r in episode_rows if r.get("scenario_label") and r.get("iteration") is not None]
    if not labeled:
        return

    scenarios = sorted(set(r["scenario_label"] for r in labeled))
    max_iter = max(r["iteration"] for r in labeled)
    n_bins = min(10, max_iter)
    if n_bins < 2:
        return
    bin_size = max_iter / n_bins
    grid = np.full((len(scenarios), n_bins), np.nan)
    counts = np.zeros((len(scenarios), n_bins), dtype=int)
    scen_idx = {s: i for i, s in enumerate(scenarios)}

    for r in labeled:
        si = scen_idx[r["scenario_label"]]
        bi = min(int((r["iteration"] - 1) / bin_size), n_bins - 1)
        rew = r.get("episode_reward_total")
        if rew is not None:
            if np.isnan(grid[si, bi]):
                grid[si, bi] = 0.0
            grid[si, bi] += rew
            counts[si, bi] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)

    fig, ax = plt.subplots(figsize=(max(10, n_bins), max(8, len(scenarios) * 0.35)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=7)
    bin_labels = [f"{int(i * bin_size + 1)}-{int((i + 1) * bin_size)}" for i in range(n_bins)]
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_xlabel("Iteration range")
    ax.set_title("Mean Reward by Scenario Over Training")
    fig.colorbar(im, ax=ax, label="Mean episode reward")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rtt_by_traffic_class(step_records: List[Dict[str, Any]], out_path: Path) -> None:

    labeled = [r for r in step_records if r.get("traffic_class") and r.get("wifi_rtt_ms") is not None]
    if not labeled:
        return

    order = [c for c in ("control", "bulk", "bursty") if any(r["traffic_class"] == c for r in labeled)]
    if not order:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (path_label, key, color) in zip(
        axes, [("WiFi RTT", "wifi_rtt_ms", WIFI_COLOR), ("5G RTT", "fiveg_rtt_ms", FIVEG_COLOR)]
    ):
        data = []
        for tc in order:
            vals = [r[key] for r in labeled if r["traffic_class"] == tc and r.get(key) is not None]
            data.append(vals)
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanline=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xticklabels(order)
        ax.set_title(path_label)
        ax.set_ylabel("RTT (ms)")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("RTT Distribution by Traffic Class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_plr_by_traffic_class(step_records: List[Dict[str, Any]], out_path: Path) -> None:
    labeled = [r for r in step_records if r.get("traffic_class") and r.get("wifi_plr") is not None]
    if not labeled:
        return

    order = [c for c in ("control", "bulk", "bursty") if any(r["traffic_class"] == c for r in labeled)]
    if not order:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (path_label, key, color) in zip(
        axes, [("WiFi PLR", "wifi_plr", WIFI_COLOR), ("5G PLR", "fiveg_plr", FIVEG_COLOR)]
    ):
        data = []
        for tc in order:
            vals = [r[key] for r in labeled if r["traffic_class"] == tc and r.get(key) is not None]
            data.append(vals)
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanline=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xticklabels(order)
        ax.set_title(path_label)
        ax.set_ylabel("PLR")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Packet Loss Distribution by Traffic Class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_action_trajectory(step_records: List[Dict[str, Any]], out_path: Path,
                           max_episodes: int = 12) -> None:

    if not step_records:
        return

    ep_indices = sorted(set(r.get("episode_idx") for r in step_records if r.get("episode_idx") is not None))
    if not ep_indices:
        return
    selected = ep_indices[-max_episodes:]
    episodes: Dict[int, List[Dict[str, Any]]] = {}
    for r in step_records:
        eidx = r.get("episode_idx")
        if eidx in selected:
            episodes.setdefault(eidx, []).append(r)

    ncols = min(4, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, ep_idx in enumerate(selected):
        ax = axes[idx // ncols][idx % ncols]
        steps = sorted(episodes[ep_idx], key=lambda r: r.get("step_in_episode", 0))
        xs = [s.get("step_in_episode", i) for i, s in enumerate(steps)]
        wr = [s.get("wifi_ratio", 0.5) for s in steps]
        label = steps[0].get("scenario_label", "")
        ax.plot(xs, wr, linewidth=1.0, color=WIFI_COLOR)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"ep{ep_idx}: {label[:25]}", fontsize=7)
        ax.set_xlabel("step", fontsize=7)
        ax.grid(alpha=0.3)

    for idx in range(len(selected), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("WiFi Ratio Trajectory (Recent Episodes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_network_quality_dashboard(
    rows: List[Dict[str, Any]],
    out_path: Path,
    step_records: List[Dict[str, Any]] = None,
) -> None:
    filtered = [r for r in rows if r.get("wifi_rtt_ms_mean") is not None]
    if not filtered:
        return

    xs = [r["iteration"] for r in filtered]

    def _f(key):
        return [r.get(key) for r in filtered]

    wifi_rtt_mean  = _f("wifi_rtt_ms_mean")
    wifi_rtt_min   = _f("wifi_rtt_ms_min")
    wifi_rtt_max   = _f("wifi_rtt_ms_max")
    fiveg_rtt_mean = _f("fiveg_rtt_ms_mean")
    fiveg_rtt_min  = _f("fiveg_rtt_ms_min")
    fiveg_rtt_max  = _f("fiveg_rtt_ms_max")
    wifi_plr       = [v * 100 if v is not None else None for v in _f("wifi_plr_mean")]
    fiveg_plr      = [v * 100 if v is not None else None for v in _f("fiveg_plr_mean")]
    weighted_rtt   = _f("weighted_rtt_ms_mean")
    weighted_plr   = [v * 100 if v is not None else None for v in _f("weighted_plr_mean")]
    wifi_ratio     = _f("wifi_ratio_mean")
    wifi_ratio_std = _f("wifi_ratio_std")

    WIFI  = "#2196F3"
    G5    = "#FF5722"
    WGT   = "#4CAF50"
    ACT   = "#9C27B0"

    fig, axs = plt.subplots(4, 1, figsize=(13, 14), sharex=True)

    # --- Panel 1: RTT ---
    ax = axs[0]
    ax.plot(xs, wifi_rtt_mean, color=WIFI, linewidth=1.5, label="WiFi RTT mean")
    if all(v is not None for v in wifi_rtt_min + wifi_rtt_max):
        ax.fill_between(xs, wifi_rtt_min, wifi_rtt_max, alpha=0.15, color=WIFI,
                        label="WiFi RTT min–max")
    ax.plot(xs, fiveg_rtt_mean, color=G5, linewidth=1.5, label="5G RTT mean")
    if all(v is not None for v in fiveg_rtt_min + fiveg_rtt_max):
        ax.fill_between(xs, fiveg_rtt_min, fiveg_rtt_max, alpha=0.15, color=G5,
                        label="5G RTT min–max")
    ax.set_ylabel("RTT (ms)")
    ax.set_title("Actual RTT per Iteration")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # --- Panel 2: PLR ---
    ax = axs[1]
    wifi_plr_clean  = [v for v in wifi_plr  if v is not None]
    fiveg_plr_clean = [v for v in fiveg_plr if v is not None]
    xs_plr_w = [x for x, v in zip(xs, wifi_plr)  if v is not None]
    xs_plr_f = [x for x, v in zip(xs, fiveg_plr) if v is not None]
    ax.plot(xs_plr_w, wifi_plr_clean,  color=WIFI, linewidth=1.5, label="WiFi PLR")
    ax.plot(xs_plr_f, fiveg_plr_clean, color=G5,   linewidth=1.5, label="5G PLR")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("PLR (%)")
    ax.set_title("Actual Packet Loss Rate per Iteration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.0f%%" % v))

    # --- Panel 3: Weighted (experienced) quality ---
    ax = axs[2]
    wrt_clean = [v for v in weighted_rtt if v is not None]
    wpl_clean = [v for v in weighted_plr if v is not None]
    xs_wrt = [x for x, v in zip(xs, weighted_rtt) if v is not None]
    xs_wpl = [x for x, v in zip(xs, weighted_plr) if v is not None]
    ax.plot(xs_wrt, wrt_clean, color=WGT, linewidth=1.5, label="Weighted RTT (ms)")
    ax.set_ylabel("Weighted RTT (ms)", color=WGT)
    ax.tick_params(axis="y", labelcolor=WGT)
    if wpl_clean:
        ax2 = ax.twinx()
        ax2.plot(xs_wpl, wpl_clean, color="#FF9800", linewidth=1.5,
                 linestyle="--", label="Weighted PLR (%)")
        ax2.set_ylabel("Weighted PLR (%)", color="#FF9800")
        ax2.tick_params(axis="y", labelcolor="#FF9800")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.0f%%" % v))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.set_title("Experienced Quality (weighted by agent action)")
    ax.grid(alpha=0.3)

    # --- Panel 4: Agent action ---
    ax = axs[3]
    ratio_clean = [v for v in wifi_ratio if v is not None]
    std_clean   = [v for v in wifi_ratio_std if v is not None]
    xs_ratio    = [x for x, v in zip(xs, wifi_ratio) if v is not None]
    ax.plot(xs_ratio, ratio_clean, color=ACT, linewidth=1.5, label="WiFi ratio (mean)")
    if len(std_clean) == len(ratio_clean):
        lo = [max(0.0, r - s) for r, s in zip(ratio_clean, std_clean)]
        hi = [min(1.0, r + s) for r, s in zip(ratio_clean, std_clean)]
        ax.fill_between(xs_ratio, lo, hi, alpha=0.15, color=ACT, label="± std")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="50/50 split")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("WiFi ratio")
    ax.set_xlabel("Iteration")
    ax.set_title("Agent Action (WiFi steering ratio)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Network Quality & Agent Behaviour Dashboard", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_actual_rtt_plr(rows: List[Dict[str, Any]], out_path: Path) -> None:
    filtered = [r for r in rows if r.get("wifi_rtt_ms_mean") is not None]
    if not filtered:
        return

    xs = [r["iteration"] for r in filtered]
    wifi_rtt_mean  = [r.get("wifi_rtt_ms_mean") for r in filtered]
    wifi_rtt_min   = [r.get("wifi_rtt_ms_min")  for r in filtered]
    wifi_rtt_max   = [r.get("wifi_rtt_ms_max")  for r in filtered]
    fiveg_rtt_mean = [r.get("fiveg_rtt_ms_mean") for r in filtered]
    fiveg_rtt_min  = [r.get("fiveg_rtt_ms_min")  for r in filtered]
    fiveg_rtt_max  = [r.get("fiveg_rtt_ms_max")  for r in filtered]
    wifi_plr  = [v * 100 if v is not None else None for v in [r.get("wifi_plr_mean")  for r in filtered]]
    fiveg_plr = [v * 100 if v is not None else None for v in [r.get("fiveg_plr_mean") for r in filtered]]

    WIFI = "#2196F3"
    G5   = "#FF5722"

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # --- Panel 1: RTT ---
    ax = axs[0]
    ax.plot(xs, wifi_rtt_mean, color=WIFI, linewidth=1.8, label="WiFi RTT (mean)", zorder=3)
    if all(v is not None for v in (wifi_rtt_min + wifi_rtt_max)):
        ax.fill_between(xs, wifi_rtt_min, wifi_rtt_max, alpha=0.18, color=WIFI, label="WiFi min–max")
    ax.plot(xs, fiveg_rtt_mean, color=G5, linewidth=1.8, label="5G RTT (mean)", zorder=3)
    if all(v is not None for v in (fiveg_rtt_min + fiveg_rtt_max)):
        ax.fill_between(xs, fiveg_rtt_min, fiveg_rtt_max, alpha=0.18, color=G5, label="5G min–max")
    ax.set_ylabel("RTT (ms)")
    ax.set_title("Actual Path RTT per Iteration (WiFi vs 5G)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    # --- Panel 2: PLR ---
    ax = axs[1]
    xs_pw = [x for x, v in zip(xs, wifi_plr)  if v is not None]
    xs_pf = [x for x, v in zip(xs, fiveg_plr) if v is not None]
    ys_pw = [v for v in wifi_plr  if v is not None]
    ys_pf = [v for v in fiveg_plr if v is not None]
    ax.plot(xs_pw, ys_pw, color=WIFI, linewidth=1.8, label="WiFi PLR")
    ax.plot(xs_pf, ys_pf, color=G5,   linewidth=1.8, label="5G PLR")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Packet Loss Rate (%)")
    ax.set_xlabel("Iteration")
    ax.set_title("Actual Packet Loss Rate per Iteration (WiFi vs 5G)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.0f%%" % v))

    fig.suptitle("Raw Network Conditions  (no weighting by agent action)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cost_breakdown_single(
    xs, rtt_cost, plr_cost, stab_cost, ax, use_smooth: bool, subtitle: str
) -> None:
    import statistics

    def smooth(ys, w=5):
        out = []
        for i in range(len(ys)):
            window = ys[max(0, i - w // 2): i + w // 2 + 1]
            out.append(statistics.mean(window))
        return out

    RTT_COLOR  = "#1565C0"
    PLR_COLOR  = "#FB8C00"
    STAB_COLOR = "#C62828"

    rtt_  = smooth(rtt_cost)  if use_smooth else rtt_cost
    plr_  = smooth(plr_cost)  if use_smooth else plr_cost
    stab_ = smooth(stab_cost) if use_smooth else stab_cost

    bottom_rtt   = rtt_
    bottom_plr   = [r + p for r, p in zip(rtt_, plr_)]
    bottom_total = [r + p + s for r, p, s in zip(rtt_, plr_, stab_)]

    ax.fill_between(xs, 0,           bottom_rtt,   alpha=0.75, color=RTT_COLOR,  label="RTT penalty")
    ax.fill_between(xs, bottom_rtt,  bottom_plr,   alpha=0.75, color=PLR_COLOR,  label="PLR penalty")
    ax.fill_between(xs, bottom_plr,  bottom_total, alpha=0.75, color=STAB_COLOR, label="Stability penalty")
    ax.plot(xs, bottom_total, color="black", linewidth=1.2, linestyle="--", label="Total cost")
    ax.annotate("start: %.0f" % bottom_total[0],
                xy=(xs[0], bottom_total[0]), xytext=(xs[0] + 2, bottom_total[0] * 1.03),
                fontsize=8, color="black")
    ax.annotate("end: %.0f" % bottom_total[-1],
                xy=(xs[-1], bottom_total[-1]), xytext=(xs[-1] - 12, bottom_total[-1] * 1.03),
                fontsize=8, color="black")
    ax.axhline(0, color="green", linewidth=1.0, linestyle=":", label="Optimal (0)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost per episode (penalty × steps)")
    ax.set_title(subtitle)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(bottom=0)


def plot_cost_breakdown_stacked(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Stacked area chart: total cost = |RTT penalty| + |PLR penalty| + |Stability penalty|.

    Shows how each penalty component contributes to the total negative reward and
    how the total cost shrinks as the agent learns — matching the hand-drawn sketch.

    Saves two files: out_path (smoothed) and out_path.stem + '_raw' (unsmoothed).
    """
    filtered = [
        r for r in rows
        if r.get("rtt_term_mean") is not None
        and r.get("plr_term_mean") is not None
        and r.get("stability_term_mean") is not None
    ]
    if not filtered:
        return

    xs = [r["iteration"] for r in filtered]
    steps = [r.get("episode_len_mean") or 200.0 for r in filtered]
    rtt_cost  = [abs(r["rtt_term_mean"])       * s for r, s in zip(filtered, steps)]
    plr_cost  = [abs(r["plr_term_mean"])       * s for r, s in zip(filtered, steps)]
    stab_cost = [abs(r["stability_term_mean"]) * s for r, s in zip(filtered, steps)]

    suptitle = "Total Cost Breakdown: How Each Penalty Shrinks Over Training"

    for use_smooth, suffix, subtitle in [
        (True,  "",     "Smoothed (window=5)"),
        (False, "_raw", "Raw (unsmoothed)"),
    ]:
        save_path = out_path.parent / (out_path.stem + suffix + out_path.suffix)
        fig, ax = plt.subplots(figsize=(13, 5))
        _plot_cost_breakdown_single(xs, rtt_cost, plr_cost, stab_cost, ax, use_smooth, subtitle)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_convergence_dashboard(rows: List[Dict[str, Any]], out_path: Path) -> None:
    xs = [r["iteration"] for r in rows]

    def _y(key):
        return [r.get(key) for r in rows]

    reward   = _y("episode_reward_mean")
    entropy  = _y("learner_entropy")
    vf_loss  = _y("learner_vf_loss")
    ratio    = _y("wifi_ratio_mean")

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    def _plot(ax, ys, label, color, ylabel):
        xs_ = [x for x, y in zip(xs, ys) if y is not None]
        ys_ = [y for y in ys if y is not None]
        if xs_:
            ax.plot(xs_, ys_, color=color, linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    _plot(axs[0][0], reward,  "Episode Reward (mean)",   "#2196F3", "Reward")
    _plot(axs[0][1], entropy, "Policy Entropy",           "#FF9800", "Entropy")
    _plot(axs[1][0], vf_loss, "Value Function Loss",      "#F44336", "VF Loss")
    _plot(axs[1][1], ratio,   "WiFi Ratio (mean action)", "#4CAF50", "WiFi Ratio")

    ax_ent = axs[0][1]
    ent_vals = [y for y in entropy if y is not None]
    if ent_vals:
        ax_ent.axhline(ent_vals[0], color="grey", linestyle="--", linewidth=0.8,
                       label="initial entropy")
        ax_ent.legend(fontsize=8)

    fig.suptitle("Convergence Dashboard", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reward_breakdown_stacked(rows: List[Dict[str, Any]], out_path: Path) -> None:

    xs      = [r["iteration"] for r in rows]
    rtt     = np.array([r.get("rtt_term_mean", 0.0) or 0.0 for r in rows])
    plr     = np.array([r.get("plr_term_mean", 0.0) or 0.0 for r in rows])
    stab    = np.array([r.get("stability_term_mean", 0.0) or 0.0 for r in rows])
    stab_neg = np.abs(stab)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(xs, np.abs(rtt), np.abs(plr), stab_neg,
                 labels=["RTT penalty", "PLR penalty", "Stability penalty"],
                 colors=["#42A5F5", "#EF5350", "#FFA726"],
                 alpha=0.75)
    total = [r.get("episode_reward_mean") for r in rows]
    total_xs = [x for x, y in zip(xs, total) if y is not None]
    total_ys = [y for y in total if y is not None]
    if total_xs:
        ax2 = ax.twinx()
        ax2.plot(total_xs, total_ys, color="black", linewidth=1.5,
                 linestyle="--", label="Total reward")
        ax2.set_ylabel("Total reward (mean)")
        ax2.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Component magnitude (abs)")
    ax.set_title("Reward Component Breakdown Over Training")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_path_quality_scatter_rtt(records: List[Dict[str, Any]], out_path: Path) -> None:
    wifi_rtt = [r.get("wifi_rtt_ms") for r in records]
    fiveg_rtt = [r.get("fiveg_rtt_ms") for r in records]
    ratio = [r.get("wifi_ratio") for r in records]

    pts = [(w, f, a) for w, f, a in zip(wifi_rtt, fiveg_rtt, ratio)
           if w is not None and f is not None and a is not None]
    if not pts:
        return

    ws, fs, rs = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(ws, fs, c=rs, cmap="RdYlGn", alpha=0.4, s=8,
                    vmin=0.0, vmax=1.0, edgecolor='none')
    cbar = plt.colorbar(sc, ax=ax, label="WiFi ratio (0=5G only, 1=WiFi only)")
    cbar.solids.set(alpha=1)

    lim = max(max(ws, default=1), max(fs, default=1)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="Equal RTT")
    ax.set_xlabel("WiFi RTT (ms)")
    ax.set_ylabel("5G RTT (ms)")
    ax.set_title("Path Quality vs Agent Decision\n(green = WiFi preferred, red = 5G preferred)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_path_quality_scatter_plr(records: List[Dict[str, Any]], out_path: Path) -> None:
    wifi_plr = [r.get("wifi_plr") for r in records]
    fiveg_plr = [r.get("fiveg_plr") for r in records]
    ratio = [r.get("wifi_ratio") for r in records]

    pts = [(w, f, a) for w, f, a in zip(wifi_plr, fiveg_plr, ratio)
           if w is not None and f is not None and a is not None]
    if not pts:
        return

    ws, fs, rs = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(ws, fs, c=rs, cmap="RdYlGn", alpha=0.4, s=8,
                    vmin=0.0, vmax=1.0, edgecolor='none')
    cbar = plt.colorbar(sc, ax=ax, label="WiFi ratio (0=5G only, 1=WiFi only)")
    cbar.solids.set(alpha=1)

    lim = max(max(ws, default=1), max(fs, default=1)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="Equal PLR")
    ax.set_xlabel("WiFi PLR")
    ax.set_ylabel("5G PLR")
    ax.set_title("Path Quality vs Agent Decision\n(green = WiFi preferred, red = 5G preferred)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_path_quality_scatter_rtt_vs_plr(records: List[Dict[str, Any]], out_path: Path) -> None:
    wifi_rtt = [r.get("wifi_rtt_ms") for r in records]
    wifi_plr = [r.get("wifi_plr") for r in records]
    ratio = [r.get("wifi_ratio") for r in records]

    pts = [(w, f, a) for w, f, a in zip(wifi_rtt, wifi_plr, ratio)
           if w is not None and f is not None and a is not None]

    if not pts:
        return

    ws, fs, rs = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(ws, fs, c=rs, cmap="RdYlGn", alpha=0.4, s=8,
                    vmin=0.0, vmax=1.0, edgecolor='none')
    cbar = plt.colorbar(sc, ax=ax, label="WiFi ratio (0=5G only, 1=WiFi only)")
    cbar.solids.set(alpha=1)

    ax.set_xlabel("WiFi RTT (ms)")
    ax.set_ylabel("WiFi PLR")
    ax.set_title("Path Quality vs Agent Decision\n(green = WiFi preferred, red = 5G preferred)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learner_diagnostics(rows: List[Dict[str, Any]], out_path: Path) -> None:
    xs = [r["iteration"] for r in rows]
    kl = [r.get("learner_kl") for r in rows]
    ent = [r.get("learner_entropy") for r in rows]
    pol = [r.get("learner_policy_loss") for r in rows]
    vf = [r.get("learner_vf_loss") for r in rows]

    if not any(v is not None for v in kl + ent + pol + vf):
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    series = [
        ("Learner KL", kl, axs[0][0]),
        ("Entropy", ent, axs[0][1]),
        ("Policy Loss", pol, axs[1][0]),
        ("Value Loss", vf, axs[1][1]),
    ]
    for title, ys, ax in series:
        plot_x = [x for x, y in zip(xs, ys) if y is not None]
        plot_y = [y for y in ys if y is not None]
        if plot_x:
            ax.plot(plot_x, plot_y, linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)

    fig.suptitle("Training Diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
