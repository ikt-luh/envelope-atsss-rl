#!/usr/bin/env python3
"""Train RL agents with PMF. Saves checkpoints, metrics, and generates plots.

Outputs to results/training/<algo>/<timestamp>_<tag>/:
  - params_used.json
  - training_metrics.csv
  - episode_metrics.csv
  - training_step_records.csv
  - summary.json
  - checkpoints/
  - model/final_checkpoint.txt
  - reward_curve.png, iteration_time.png, rtt_conditions.png, etc.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl_agent.sac_agent import MPQUICSACAgent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path), override=False)
except Exception:
    pass

import config  
from aue_client.client import AUEClient 
from pmf_client.client import PMFClientManager  
from rl_agent.ppo_agent import MPQUICPPOAgent 
from rl_agent.lstm_agent import MPQUICLSTMAgent  
from scripts.plot_utils import (  
    plot_bytes_sent,
    plot_decision_time_histogram,
    plot_decision_time_over_time,
    plot_latency,
    plot_network_distribution,
    plot_ratio,
)
from scripts.train_utils import (  
    create_run_dir,
    deep_merge,
    default_config_path,
    enrich_episode_rows_with_scenarios,
    extract_episode_rows,
    extract_metrics,
    load_yaml,
    normalize_checkpoint_path,
    plot_action_by_scenario,
    plot_action_stats,
    plot_action_trajectory,
    plot_actual_rtt_plr,
    plot_convergence_dashboard,
    plot_cost_breakdown_stacked,
    plot_iter_time,
    plot_learner_diagnostics,
    plot_network_quality_dashboard,
    plot_observation_age,
    plot_path_quality_scatter_plr,
    plot_path_quality_scatter_rtt,
    plot_path_quality_scatter_rtt_vs_plr,
    plot_plr_conditions,
    plot_reward,
    plot_plr_by_traffic_class,
    plot_reward_breakdown_stacked,
    plot_reward_by_traffic_class,
    plot_reward_components,
    plot_reward_heatmap,
    plot_rtt_by_traffic_class,
    plot_rtt_conditions,
    plot_scenario_distribution,
    plot_total_reward_per_episode,
    resolve_run_dir,
    save_episode_csv,
    save_metrics_csv,
    save_step_csv,
    to_jsonable,
)


def build_agent(
    algo: str,
    decision_interval: Optional[float],
    sequence_len: int,
    config_override: Dict[str, Any],
    session_id: str,
    pmf_client: PMFClientManager,
    training_records: List[Dict[str, Any]],
    training_start_time: float,
    early_termination_reward_threshold=None,
    early_termination_window: int = 50,
    reward_params: Dict[str, Any] | None = None,
    max_steps: int = 1000,
    augment_observations: bool = False,
    augment_prob: float = 0.3,
    scenario_manager=None,
    episodes_per_config: int = 1,
    aue_client=None,
):
    if algo == "ppo":
        agent = MPQUICPPOAgent()
        agent.config = deep_merge(agent._default_config(), config_override or {})
    elif algo == "sac":
        agent = MPQUICSACAgent()
        agent.config = deep_merge(agent._default_config(), config_override or {})
    elif algo == "lstm":
        agent = MPQUICLSTMAgent(sequence_len=sequence_len)
        merged = deep_merge(agent._default_config(), config_override or {})
        if "model" in merged and isinstance(merged["model"], dict):
            merged["model"]["max_seq_len"] = sequence_len
        agent.config = merged
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    agent.initialize(
        decision_interval,
        pmf_client,
        session_id=session_id,
        training_records=training_records,
        training_start_time=training_start_time,
        early_termination_reward_threshold=early_termination_reward_threshold,
        early_termination_window=early_termination_window,
        reward_params=reward_params or {},
        max_steps=max_steps,
        augment_observations=augment_observations,
        augment_prob=augment_prob,
        scenario_manager=scenario_manager,
        episodes_per_config=episodes_per_config,
        aue_client=aue_client,
    )
    return agent


def _scenario_env_overrides() -> Dict[str, Any]:
    o: Dict[str, Any] = {}
    if getattr(config, "SCENARIO_ANE_IP", ""):
        o["ane_ip"] = config.SCENARIO_ANE_IP
    if getattr(config, "SCENARIO_DRY_RUN", False):
        o["dry_run"] = True
    if getattr(config, "SCENARIO_WIFI_BIND_IP", ""):
        o["wifi_bind_ip"] = config.SCENARIO_WIFI_BIND_IP
    if getattr(config, "SCENARIO_FIVEG_BIND_IP", ""):
        o["fiveg_bind_ip"] = config.SCENARIO_FIVEG_BIND_IP
    return o


def build_scenario_manager(repo_root: Path, training_cfg: Dict[str, Any], dry_run:bool, pmf_client=None):
    from scenarios.applier import ScenarioManager
    from scenarios.config_space import load_scenario_settings

    rel = training_cfg.get("scenario_config_file", "configs/training/scenarios.yaml")
    sp = Path(rel)
    if not sp.is_absolute():
        sp = repo_root / sp
    sdata = load_yaml(sp)
    sdata = deep_merge(sdata, _scenario_env_overrides())
    sdata = deep_merge(sdata, config.scenario_iperf_port_overrides())
    settings = load_scenario_settings(sdata)
    settings.wifi_iface = config.PMF_UE_WIFI_IFACE
    settings.fiveg_iface = config.PMF_UE_5G_IFACE
    if not settings.wifi_bind_ip:
        settings.wifi_bind_ip = config.PMF_UE_WIFI_IP or None
    if not settings.fiveg_bind_ip:
        settings.fiveg_bind_ip = config.PMF_UE_5G_IP or None
    if getattr(config, "SCENARIO_ANE_IP", ""):
        settings.ane_ip = config.SCENARIO_ANE_IP
    settings.dry_run = settings.dry_run or getattr(config, "SCENARIO_DRY_RUN", False) or dry_run
    ep = int(sdata.get("episodes_per_config", 1))
    seed = sdata.get("seed", None)
    rng = random.Random(seed) if seed is not None else random.Random()
    if seed is not None:
        print(f"[train] Scenario seed: {seed}")
    print(
        f"[train] Scenario iperf ports: cross_traffic={settings.cross_traffic_port} "
        "(override YAML: TESTBED_SCENARIO_IPERF_USER_PORT / TESTBED_SCENARIO_IPERF_CROSS_PORT in .env)"
    )
    return ScenarioManager(settings, rng=rng, pmf_client=pmf_client), ep


def _configure_training_logging(verbose: bool) -> None:
    # cleaner training console: Ray RLlib deprecations, gymnasium Box dtype hints.
    # Set RL_VERBOSE=1 to restore full verbose logging.
    import logging
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(
        "ignore",
        message=".*precision lowered.*",
        category=UserWarning,
        module="gymnasium.spaces.box",
    )

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        return

    logging.getLogger().setLevel(logging.WARNING)
    # suppress scenario applier / iperf / traffic_gen noise — only show warnings and errors
    # for noisy in ("scenarios", "scenarios.applier", "scenarios.config_space",
    #               "pmf_client", "rl_agent", "ray", "urllib3", "requests"):
    #     logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RL agent with PMF")
    parser.add_argument("--agent", choices=["ppo", "lstm", "sac"], default=None)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--sequence-len", type=int, default=None)
    parser.add_argument("--dry-run", dest='dry_run', action='store_true')
    parser.set_defaults(dry_run=False)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--success-reward-threshold", type=float, default=None)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--run-dir", default=None, help="Output directory (default: auto-generated)")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint for warm-start (resume training)")
    parser.add_argument(
        "--with-scenarios",
        action="store_true",
        help="Enable tc/netem + iperf3 scenario manager (see configs/training/scenarios.yaml)",
    )
    parser.add_argument(
        "--scenario-config",
        default="configs/training/scenarios.yaml",
        help="Path to scenario YAML (default: configs/training/scenarios.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose outputs",
    )
    args = parser.parse_args()

    _configure_training_logging(os.environ.get("RL_VERBOSE", "").lower() in ("1", "true", "yes") or args.verbose)

    repo_root = Path(__file__).resolve().parents[1]
    algo = args.agent
    cfg_path = Path(args.config) if args.config else default_config_path(repo_root, algo)
    cfg = load_yaml(cfg_path)

    iterations = args.iterations or cfg.get("iterations", 50)
    checkpoint_freq = args.checkpoint_freq or cfg.get("checkpoint_freq", 10)
    sequence_len = args.sequence_len or cfg.get("sequence_len", 10)
    session_id = args.session_id or cfg.get("session_id", "train-session-001")
    agent_override = cfg.get("agent_config", {})

    run_dir = resolve_run_dir(repo_root, algo, args.tag, args.run_dir)
    ray_logs_dir = run_dir / "ray_logs"
    ray_logs_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RL_TRAIN_RESULTS_DIR"] = str(ray_logs_dir)

    print(f"[train] Run dir: {run_dir}")
    print(f"[train] Config: {cfg_path}")
    print(f"[train] Algo={algo} iterations={iterations} checkpoint_freq={checkpoint_freq}")
    print(f"[train] Agent info: decision interval {config.DECISION_INTERVAL}")

    scenario_config_file = cfg.get("scenario_config_file", args.scenario_config)
    cfg["scenario_config_file"] = scenario_config_file

    params_used = {
        "algo": algo,
        "iterations": iterations,
        "checkpoint_freq": checkpoint_freq,
        "sequence_len": sequence_len if algo == "lstm" else None,
        "session_id": session_id,
        "success_reward_threshold": args.success_reward_threshold,
        "agent_config": agent_override,
        "config_file": str(cfg_path),
        "ray_logs_dir": str(ray_logs_dir),
        "with_scenarios": bool(args.with_scenarios or cfg.get("with_scenarios", False)),
        "scenario_config_file": scenario_config_file,
    }
    (run_dir / "params_used.json").write_text(json.dumps(params_used, indent=2))

    training_records: List[Dict[str, Any]] = []
    train_start_time = time.time()

    pmf_client = PMFClientManager(
        dry_run=args.dry_run,
        pmf_ue_url=config.PMF_UE_URL,
        pmf_upf_url=config.PMF_UPF_URL,
        use_polling=config.USE_PMF_POLLING,
        poll_interval=config.PMF_POLL_INTERVAL,
        job_timeout=config.PMF_JOB_TIMEOUT,
        request_timeout=config.PMF_REQUEST_TIMEOUT,
        probe_count=config.PMF_PROBE_COUNT,
        probe_timeout_ms=config.PMF_PROBE_TIMEOUT_MS,
        cache_max_age=config.PMF_CACHE_MAX_AGE,
        upf_ip=config.PMF_UPF_IP,
        ue_wifi_ip=config.PMF_UE_WIFI_IP,
        ue_5g_ip=config.PMF_UE_5G_IP,
        ue_wifi_iface=config.PMF_UE_WIFI_IFACE,
        ue_5g_iface=config.PMF_UE_5G_IFACE,
        use_uplink=config.PMF_USE_UPLINK,
        asynchronous=config.PMF_ASYNCHRONOUS,
    )
    print(f"[train] PMF: UE={config.PMF_UE_URL} UPF={config.PMF_UPF_URL}")

    aue_client = None
    if not args.dry_run and config.AUE_URL:
        aue_client = AUEClient(config.AUE_URL)
        print(f"[train] AUE client: {config.AUE_URL} (online steering enabled)")
    else:
        reason = ""
        if args.dry_run:
            reason = "dry run"
        elif config.AUE_URL is None:
            reason = "AUE URL not set"
        print(f"[train] WARNING: AUE not set (reason: {reason}). RATIOS WILL NOT BE APPLIED.")
    wait_s = max(60.0, config.PMF_POLL_INTERVAL * 15)
    print(f"[train] Waiting up to {wait_s:.0f}s for PMF cache to warm up...")
    pmf_ready = False
    for elapsed in range(0, int(wait_s) + 1, 2):
        if pmf_client.get_metrics(session_id) is not None:
            print(f"[train] PMF metrics ready (after {elapsed}s)")
            pmf_ready = True
            break
        if elapsed % 10 == 0 and elapsed > 0:
            print(f"[train] Still waiting for PMF... ({elapsed}s / {wait_s:.0f}s)")
        time.sleep(2)
    if not pmf_ready:
        print(f"[train] WARNING: No PMF metrics after {wait_s:.0f}s. Training will likely fail -- check PMF connectivity.")

    early_term_threshold = cfg.get("early_termination_reward_threshold")
    early_term_window = cfg.get("early_termination_window", 50)
    reward_params = cfg.get("reward_params", {})
    max_steps = cfg.get("max_steps", 1000)
    augment_observations = cfg.get("augment_observations", False)
    augment_prob = cfg.get("augment_prob", 0.3)

    with_scenarios = bool(args.with_scenarios or cfg.get("with_scenarios", False))
    scenario_manager = None
    episodes_per_config = 1
    if with_scenarios:
        scenario_manager, episodes_per_config = build_scenario_manager(repo_root, cfg, dry_run=args.dry_run, pmf_client=pmf_client)
        atexit.register(scenario_manager.shutdown)
        print(
            f"[train] Scenarios: enabled (episodes_per_config={episodes_per_config}, "
            f"file={scenario_config_file})"
        )
        if early_term_threshold is not None:
            print(
                "[train] WARNING: early_termination_reward_threshold is set with "
                "--with-scenarios. Easy scenarios will terminate early, biasing "
                "training away from hard scenarios. Set to null to disable."
            )
    else:
        print("[train] Scenarios: disabled")

    agent = build_agent(
        algo=algo,
        decision_interval=config.DECISION_INTERVAL,
        sequence_len=sequence_len,
        config_override=agent_override,
        session_id=session_id,
        pmf_client=pmf_client,
        training_records=training_records,
        training_start_time=train_start_time,
        early_termination_reward_threshold=early_term_threshold,
        early_termination_window=early_term_window,
        reward_params=reward_params,
        max_steps=max_steps,
        augment_observations=augment_observations,
        augment_prob=augment_prob,
        scenario_manager=scenario_manager,
        episodes_per_config=episodes_per_config,
        aue_client=aue_client,
    )

    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser().resolve()
        if ckpt.exists():
            agent.load_checkpoint(str(ckpt))
            print(f"[train] Warm-start: loaded checkpoint from {ckpt}")
        else:
            print(f"[train] WARNING: Checkpoint not found: {ckpt}. Starting from scratch.")

    rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    checkpoints: List[str] = []
    train_start = time.time()
    failed = False
    error_msg = ""
    episode_idx = 0

    try:
        for i in range(1, iterations + 1):
            records_before = len(training_records)
            iter_t0 = time.time()
            result = agent.agent.train()
            iter_s = time.time() - iter_t0
            elapsed_s = time.time() - train_start

            iter_records = training_records[records_before:]
            row = extract_metrics(
                result, i, iter_s, elapsed_s,
                args.success_reward_threshold,
                iter_records=iter_records,
            )
            rows.append(row)
            new_ep_rows, episode_idx = extract_episode_rows(
                result, i, episode_idx, args.success_reward_threshold
            )
            episode_rows.extend(new_ep_rows)

            reward = row.get("episode_reward_mean")
            print(f"[train] iter={i:03d} reward_mean={reward if reward is not None else 'n/a'} iter_s={iter_s:.2f}")

            if i % checkpoint_freq == 0:
                ckpt_dir = run_dir / "checkpoints" / f"iter_{i}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_raw = agent.agent.save(str(ckpt_dir))
                ckpt_path = normalize_checkpoint_path(ckpt_raw)
                checkpoints.append(str(ckpt_path))
                print(f"[train] checkpoint @ iter {i}: {ckpt_path}")

        # always save a final checkpoint (even if checkpoint_freq > iterations)
        if not checkpoints or (iterations % checkpoint_freq != 0):
            try:
                ckpt_dir = run_dir / "checkpoints" / f"iter_{iterations}_final"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_raw = agent.agent.save(str(ckpt_dir))
                ckpt_path = normalize_checkpoint_path(ckpt_raw)
                checkpoints.append(str(ckpt_path))
                print(f"[train] final checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[train] WARNING: Could not save final checkpoint: {e}")
    except Exception as exc:
        failed = True
        import traceback
        traceback.print_exc()
        error_msg = str(exc)
        print(f"[train] ERROR: {error_msg}")
    finally:
        try:
            agent.shutdown()
        except Exception:
            pass
        try:
            pmf_client.shutdown()
        except Exception:
            pass

    train_total_s = time.time() - train_start

    if episode_rows and training_records:
        enrich_episode_rows_with_scenarios(episode_rows, training_records)

    save_metrics_csv(rows, run_dir / "training_metrics.csv")
    if episode_rows:
        save_episode_csv(episode_rows, run_dir / "episode_metrics.csv")

    save_step_csv(training_records, run_dir / "training_step_records.csv")

    best_reward = None
    final_reward = None
    if rows:
        rewards = [r["episode_reward_mean"] for r in rows if r["episode_reward_mean"] is not None]
        if rewards:
            best_reward = max(rewards)
            final_reward = rewards[-1]

    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = checkpoints[-1] if checkpoints else None
    (model_dir / "final_checkpoint.txt").write_text(final_checkpoint or "")

    summary = {
        "status": "failed" if failed else "completed",
        "error": error_msg if failed else None,
        "algo": algo,
        "iterations_requested": iterations,
        "iterations_completed": len(rows),
        "checkpoint_count": len(checkpoints),
        "episodes_logged": len(episode_rows),
        "checkpoints": checkpoints,
        "training_time_s_total": round(train_total_s, 3),
        "training_time_s_per_iter_mean": round(train_total_s / len(rows), 3) if rows else None,
        "reward_final": final_reward,
        "reward_best": best_reward,
        "episode_steps_mean": (
            round(sum(r["episode_steps"] for r in episode_rows) / len(episode_rows), 3)
            if episode_rows else None
        ),
        "episode_reward_mean": (
            round(sum(r["episode_reward_total"] for r in episode_rows) / len(episode_rows), 3)
            if episode_rows else None
        ),
        "success_rate_overall": (
            round(
                sum(r["success"] for r in episode_rows if r.get("success") is not None)
                / len([r for r in episode_rows if r.get("success") is not None]),
                4,
            )
            if any(r.get("success") is not None for r in episode_rows)
            else None
        ),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    print(json.dumps(to_jsonable(summary), indent=2))

    if rows:
        print("[train] Generating plots...")
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
        if episode_rows:
            plot_total_reward_per_episode(episode_rows, run_dir / "total_reward_per_episode.png")
        if training_records:
            plot_latency(
                training_records,
                run_dir / "latency_over_time.png",
                title="Training Step Latency Over Time",
            )
            if any(r.get("decision_time_ms") is not None for r in training_records):
                plot_decision_time_over_time(
                    training_records,
                    run_dir / "decision_time_over_time.png",
                    title="Training Decision Time Over Time",
                )
                plot_decision_time_histogram(
                    training_records,
                    run_dir / "decision_time_histogram.png",
                    title="Training Decision Time Distribution",
                )
            plot_ratio(training_records, run_dir / "ratio_over_time.png")
            plot_network_distribution(training_records, run_dir / "network_distribution.png")
            plot_action_trajectory(training_records, run_dir / "action_trajectory.png")
            plot_path_quality_scatter_rtt(training_records, run_dir / "path_quality_scatter_rtt.png")
            plot_path_quality_scatter_plr(training_records, run_dir / "path_quality_scatter_plr.png")
            plot_path_quality_scatter_rtt_vs_plr(training_records, run_dir / "path_quality_scatter_rtt_vs_plr.png")
            plot_network_quality_dashboard(
                rows, run_dir / "network_quality_dashboard.png",
                step_records=training_records,
            )
            plot_bytes_sent(training_records, run_dir / "bytes_sent.png")

        # scenario-aware plots (only meaningful with --with-scenarios)
        if with_scenarios and episode_rows:
            plot_reward_by_traffic_class(episode_rows, run_dir / "reward_by_traffic_class.png")
            plot_scenario_distribution(episode_rows, run_dir / "scenario_distribution.png")
            plot_action_by_scenario(episode_rows, run_dir / "action_by_scenario.png")
            plot_reward_heatmap(episode_rows, run_dir / "reward_heatmap.png")
        if with_scenarios and training_records:
            plot_rtt_by_traffic_class(training_records, run_dir / "rtt_by_traffic_class.png")
            plot_plr_by_traffic_class(training_records, run_dir / "plr_by_traffic_class.png")

        print(f"[train] Plots saved to {run_dir}")

    print(f"\n[train] Done.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
