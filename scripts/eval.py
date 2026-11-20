# evaluate trained ppo or lstm policy with pmf.
#usage: python scripts/eval.py --checkpoint results/training/ppo/xxx/checkpoints/iter_50 --steps 100
#       python scripts/eval.py --checkpoint results/training/lstm/xxx/checkpoints/iter_50 --steps 100
#       python scripts/eval.py --checkpoint results/training/ppo/xxx/checkpoints/iter_50 --steps 50 --with-scenarios
#or
# ./run_rl_evaluation.sh --checkpoint results/training/ppo/xxx/checkpoints/iter_50 --steps 100
# ./run_rl_evaluation.sh --checkpoint results/training/lstm/xxx/checkpoints/iter_50 --steps 100

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scenarios.applier import ScenarioManager, TrafficWatcher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path), override=False)
except Exception:
    pass

from aue_client.client import AUEClient
import config
from pmf_client.client import PMFClientManager, PMFMetrics
from rl_agent.heuristic_agent import HeuristicAgent
from rl_agent.lstm_agent import MPQUICLSTMAgent
from rl_agent.ppo_agent import MPQUICPPOAgent
from rl_agent.rl_env import RLEnv, StepPacer, build_observation, get_timestamp_str
from scripts.plot_utils import (
    plot_bitrate_sent,
    plot_bytes_sent,
    plot_decision_time_histogram,
    plot_decision_time_over_time,
    plot_eval_network_conditions,
    plot_eval_timeline_reward_action,
    plot_eval_timeline_conditions_action,
    plot_latency,
    plot_network_distribution,
    plot_ratio,
)
from scripts.train_utils import (
    deep_merge,
    load_yaml,
    plot_action_by_scenario,
    plot_reward_by_scenario,
    plot_reward_by_traffic_class,
    plot_rtt_by_traffic_class,
    plot_plr_by_traffic_class,
    plot_scenario_distribution,
)

logger = logging.getLogger(__name__)


def _detect_agent(checkpoint_path: Optional[Path]) -> str:
    # auto-detect agent from checkpoint path (ppo or lstm). heuristic has no checkpoint.
    if checkpoint_path is None:
        return "heuristic"
    path_str = str(checkpoint_path).lower()
    if "lstm" in path_str:
        return "lstm"
    return "ppo"


def _create_eval_step_record(
    step: int,
    experiment_start_monotonic: float,
    episode_start_monotonic: float,
    step_start_monotonic: float,
    inference_time_s: Optional[float],
    pacer: StepPacer,
    wifi_ratio: float,
    metrics: PMFMetrics,
    reward: Optional[float],
    reward_metrics: Optional[dict],
    delta_bytes_sent_dict: dict,
    timestamp_ratio_sent: str,
    timestamp_ratio_applied: str,
    timestamp_observation: str,
    scenario_label: Optional[str],
    traffic_class: Optional[str],
):
    now = time.monotonic()
    return {
        "step": step,
        "experiment_elapsed_s": now - experiment_start_monotonic,
        "episode_elapsed_s": now - episode_start_monotonic,
        "step_time_ms": (now - step_start_monotonic) * 1000,
        "inference_time_ms": None if inference_time_s is None else inference_time_s * 1000,
        "pacer_obs_time_ms": None if pacer.duration_observation is None else pacer.duration_observation * 1000,
        "pacer_sleep_time_ms": None if pacer.duration_sleep is None else pacer.duration_sleep * 1000,
        "pacer_outside_time_ms": None if pacer.duration_outside is None else pacer.duration_outside * 1000,
        "wifi_ratio": wifi_ratio,
        "fiveg_ratio": 1 - wifi_ratio,
        "reward": None if reward is None else reward,
        "wifi_rtt_ms": metrics.wifi_rtt_ms,
        "fiveg_rtt_ms": metrics.fiveg_rtt_ms,
        "wifi_plr": metrics.wifi_plr,
        "fiveg_plr": metrics.fiveg_plr,
        "wifi_rtt_min_ms": metrics.wifi_rtt_min_ms,
        "wifi_rtt_max_ms": metrics.wifi_rtt_max_ms,
        "wifi_rtt_std_ms": metrics.wifi_rtt_std_ms,
        "fiveg_rtt_min_ms": metrics.fiveg_rtt_min_ms,
        "fiveg_rtt_max_ms": metrics.fiveg_rtt_max_ms,
        "fiveg_rtt_std_ms": metrics.fiveg_rtt_std_ms,
        "wifi_valid": metrics.wifi_valid,
        "fiveg_valid": metrics.fiveg_valid,
        "timestamp_ratio_sent": timestamp_ratio_sent,
        "timestamp_ratio_applied": timestamp_ratio_applied,
        "timestamp_observation": timestamp_observation,
        "observation_age_ms": getattr(metrics, "observation_age_s", 0.0) * 1000.0,
        "scenario_label": scenario_label,
        "traffic_class": traffic_class,
        **({} if reward_metrics is None else reward_metrics),
        **delta_bytes_sent_dict
    }


def _run_eval_steps(
    agent,
    agent_type: str,
    pmf_client,
    aue_client,
    session_id: str,
    num_steps: int,
    reward_params: Dict[str, Any],
    experiment_start_monotonic: float,
    pacer: StepPacer,
    *,
    scenario_label: Optional[str] = None,
    traffic_class: Optional[str] = None,
    watcher: Optional[TrafficWatcher] = None,
) -> List[Dict[str, Any]]:
    #run num_steps of evaluation, returning per-step records.
    records: List[Dict[str, Any]] = []
    prev_wifi_ratio: Optional[float] = None

    episode_start_monotonic = time.monotonic()
    step_start_monotonic = time.monotonic()

    wifi_ratio = RLEnv.EVAL_INITIAL_WIFI_RATIO
    timestamp_ratio_sent = get_timestamp_str()
    if aue_client is not None:
        try:
            aue_client.send_ratio(wifi_ratio)
        except Exception as e:
            print("[eval] ERROR: AUE send_ratio failed: %s", e)

    if watcher is not None:
        watcher.reset_bytes_sent()

    bytes_sent_dict = TrafficWatcher.get_bytes_sent_dict(watcher)

    with pacer(reset=True):
        metrics = pmf_client.get_metrics(session_id)
    timestamp_observation = get_timestamp_str()

    new_bytes_sent_dict = TrafficWatcher.get_bytes_sent_dict(watcher)
    delta_bytes_sent = TrafficWatcher.delta(bytes_sent_dict, new_bytes_sent_dict)
    bytes_sent_dict = new_bytes_sent_dict

    if aue_client is not None:
        timestamp_ratio_applied = aue_client.get_timestamp()
    else:
        timestamp_ratio_applied = timestamp_ratio_sent

    next_obs = build_observation(
        metrics,
        wifi_ratio,
        delta_bytes_wifi=delta_bytes_sent['delta_bytes_sent_wifi'],
        delta_bytes_fiveg=delta_bytes_sent['delta_bytes_sent_fiveg'],
        delta_time_s=delta_bytes_sent['delta_time_monotonic'],
    )

    records.append(
        _create_eval_step_record(
            step=0,
            experiment_start_monotonic=experiment_start_monotonic,
            episode_start_monotonic=episode_start_monotonic,
            step_start_monotonic=step_start_monotonic,
            inference_time_s=None,
            pacer=pacer,
            wifi_ratio=wifi_ratio,
            metrics=metrics,
            reward=None,
            reward_metrics=None,
            delta_bytes_sent_dict=delta_bytes_sent,
            timestamp_ratio_sent=timestamp_ratio_sent,
            timestamp_ratio_applied=timestamp_ratio_applied,
            timestamp_observation=timestamp_observation,
            scenario_label=scenario_label,
            traffic_class=traffic_class
        )
    )

    logger.info(f"0: initial wifi_ratio {wifi_ratio:.3f}")

    for step in range(num_steps):
        prev_wifi_ratio = wifi_ratio
        obs = next_obs

        step_start_monotonic = time.monotonic()
        wifi_ratio = float(np.clip(agent.select_action(obs).item(), 0.0, 1.0))
        inference_time_s = time.monotonic() - step_start_monotonic

        timestamp_ratio_sent = get_timestamp_str()
        if aue_client is not None:
            try:
                aue_client.send_ratio(wifi_ratio)
            except Exception as e:
                print("[eval] ERROR: AUE send_ratio failed: %s", e)

        # get metrics of the next step that depend on the agent's current action
        with pacer:
            metrics = pmf_client.get_metrics(session_id)
        timestamp_observation = get_timestamp_str()

        new_bytes_sent_dict = TrafficWatcher.get_bytes_sent_dict(watcher)
        delta_bytes_sent = TrafficWatcher.delta(bytes_sent_dict, new_bytes_sent_dict)
        bytes_sent_dict = new_bytes_sent_dict

        next_obs = build_observation(
            metrics,
            wifi_ratio,
            delta_bytes_wifi=delta_bytes_sent['delta_bytes_sent_wifi'],
            delta_bytes_fiveg=delta_bytes_sent['delta_bytes_sent_fiveg'],
            delta_time_s=delta_bytes_sent['delta_time_monotonic'],
        )

        reward, reward_metrics = RLEnv.get_reward_and_metrics(next_obs, wifi_ratio, prev_wifi_ratio, **reward_params)

        # check when action was applied
        if aue_client is not None:
            timestamp_ratio_applied = aue_client.get_timestamp()
        else:
            timestamp_ratio_applied = timestamp_ratio_sent

        logger.info(f"{step + 1}: wifi_ratio {wifi_ratio:.3f}, sent {timestamp_ratio_sent[11:]}, applied {timestamp_ratio_applied[11:]}, obs {timestamp_observation[11:]}")

        records.append(
            _create_eval_step_record(
                step=step + 1,
                experiment_start_monotonic=experiment_start_monotonic,
                episode_start_monotonic=episode_start_monotonic,
                step_start_monotonic=step_start_monotonic,
                inference_time_s=inference_time_s,
                pacer=pacer,
                wifi_ratio=wifi_ratio,
                metrics=metrics,
                reward=reward,
                reward_metrics=reward_metrics,
                delta_bytes_sent_dict=delta_bytes_sent,
                timestamp_ratio_sent=timestamp_ratio_sent,
                timestamp_ratio_applied=timestamp_ratio_applied,
                timestamp_observation=timestamp_observation,
                scenario_label=scenario_label,
                traffic_class=traffic_class
            )
        )

        if (step + 1) % 10 == 0:
            print(f"  step {step + 1}/{num_steps} wifi={wifi_ratio:.3f} reward={reward:.3f}")

    return records


def np_arr_summary_dict(arr: np.ndarray):
    return {
        "mean": np.mean(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "std": np.std(arr)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PPO, LSTM, or heuristic policy with live PMF")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (required for ppo/lstm, ignored for heuristic)")
    parser.add_argument(
        "--agent",
        choices=["ppo", "lstm", "heuristic"],
        default=None,
        help="Agent type (default: auto-detect from path or heuristic if no checkpoint)",
    )
    parser.add_argument(
        "--no-exploration",
        action="store_true",
        help="Disables exploration in the agent's policy",
    )
    parser.add_argument("--heuristic", default="lb-rtt-min", help="Heuristic type when agent=heuristic: lb-rtt-min or lb-plr-min (default: lb-rtt-min)")
    parser.add_argument("--heuristic-alpha", type=float, default=0.95, help="EMA alpha (default: 0.95)")
    parser.add_argument("--sequence-len", type=int, default=10, help="LSTM sequence length (default: 10)")
    parser.add_argument("--steps", type=int, default=100, help="Number of evaluation steps")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes (per config, scenarios only)")
    parser.add_argument("--interval-s", type=float, default=None, help="Seconds between steps (default: DECISION_INTERVAL)")
    parser.add_argument("--session-id", default=None, help="Session ID (default: from config)")
    parser.add_argument("--tag", default="policy-eval", help="Tag for output directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results/evaluation/<agent>/<timestamp>_<tag>)")
    parser.add_argument("--discrete-n", type=int, default=None)
    parser.add_argument("--reward-type", default=None, help="Reward type: logarithmic or linear (default: from config)")
    parser.add_argument("--reward-r-max", type=float, default=None, help="Reward r_max for logarithmic (default: from config)")
    parser.add_argument("--reward-epsilon", type=float, default=None, help="Reward epsilon for logarithmic (default: from config)")
    parser.add_argument("--reward-alpha", type=float, default=None, help="Reward alpha for logarithmic (default: from config)")
    parser.add_argument("--reward-rtt-weight", type=float, default=None, help="Reward rtt weight scales importance of rtt over plr (default: from config)")
    parser.add_argument(
        "--with-scenarios",
        action="store_true",
        help="Iterate all 30 scenario configs, applying tc/netem + iperf3 per scenario",
    )
    parser.add_argument(
        "--scenario-config",
        default="configs/training/scenarios.yaml",
        help="Path to scenario YAML (default: configs/training/scenarios.yaml)",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=5.0,
        help="Seconds to wait after applying a scenario before collecting data (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose outputs",
    )
    parser.add_argument("--dry-run", dest='dry_run', action='store_true')
    parser.set_defaults(dry_run=False)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    assert args.episodes >= 1, "There must be at least one evaluation episode"

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    agent_type = args.agent or _detect_agent(checkpoint_path)

    if agent_type in ("ppo", "lstm", "sac"):
        if not args.checkpoint:
            print("Error: --checkpoint is required for model-based agents")
            return 1
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return 1

    session_id = args.session_id or config.SESSION_ID
    interval_s = args.interval_s if args.interval_s is not None else config.DECISION_INTERVAL
    reward_params = {
        "r_max": args.reward_r_max if args.reward_r_max is not None else config.REWARD_R_MAX,
        "epsilon": args.reward_epsilon if args.reward_epsilon is not None else config.REWARD_EPSILON,
        "alpha": args.reward_alpha if args.reward_alpha is not None else config.REWARD_ALPHA,
    }

    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = repo_root / "results" / "evaluation" / agent_type / f"{ts}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] Agent: {agent_type}" + (f" | Checkpoint: {checkpoint_path}" if checkpoint_path else f" | Heuristic: {args.heuristic}"))
    print(f"[eval] Steps: {args.steps} | Decision Interval: {interval_s}s | Session: {session_id}")
    if agent_type == "lstm":
        print(f"[eval] Sequence length: plot_bitrate_sent{args.sequence_len}")
    if agent_type == "heuristic":
        print(f"[eval] Heuristic: {args.heuristic} (alpha={args.heuristic_alpha})")
    if args.with_scenarios:
        print(f"[eval] Scenarios: ON (config={args.scenario_config}, settle={args.settle_s}s)")
    print(f"[eval] PMF probe count: {config.PMF_PROBE_COUNT}, probe timeout {config.PMF_PROBE_TIMEOUT_MS}, poll interval {config.PMF_POLL_INTERVAL}")
    print(f"[eval] Output: {run_dir}")

    pmf_client = PMFClientManager(
        dry_run=args.dry_run,
        pmf_ue_url=config.PMF_UE_URL,
        pmf_upf_url=config.PMF_UPF_URL,
        use_polling=True,
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
        asynchronous=config.PMF_ASYNCHRONOUS
    )
    # first poll cycle can take job_timeout (30s) or more; allow enough time
    wait_s = max(40.0, config.PMF_JOB_TIMEOUT + 10)
    print(f"[eval] Waiting up to {wait_s:.0f}s for PMF metrics (first poll cycle)...")
    for _ in range(int(wait_s / 2) + 1):
        metrics = pmf_client.get_metrics(session_id)
        if metrics is not None:
            break
        time.sleep(2)
    else:
        pmf_client.shutdown()
        print("Error: No PMF metrics after wait. Ensure PMF is running and reachable.")
        return 1
    print("[eval] PMF metrics available")

    if args.dry_run:
        aue_client = None
        print("[eval] WARNING: dry run, actions will not be applied")
    else:
        aue_client = AUEClient(config.AUE_URL, timeout=config.AUE_TIMEOUT)

    if agent_type == "ppo":
        agent = MPQUICPPOAgent()
        agent.initialize(decision_interval=config.DECISION_INTERVAL, pmf_client=pmf_client, session_id=session_id, window_size=config.PPO_OBS_WINDOW_SIZE, discrete_n=args.discrete_n)
        agent.load_checkpoint(str(checkpoint_path))
        if args.no_exploration:
            agent.agent.get_policy().config["explore"] = False
    elif agent_type == "lstm":
        agent = MPQUICLSTMAgent(sequence_len=args.sequence_len)
        agent.initialize(decision_interval=config.DECISION_INTERVAL, pmf_client=pmf_client, session_id=session_id, discrete_n=args.discrete_n)
        agent.load_checkpoint(str(checkpoint_path))
        if args.no_exploration:
            agent.agent.get_policy().config["explore"] = False
    else:
        agent = HeuristicAgent(
            heuristic_type=args.heuristic,
            alpha=args.heuristic_alpha,
        )
        agent.initialize(reward_params=reward_params, pmf_client=pmf_client, session_id=session_id)

    # build scenario list if --with-scenarios
    scenario_applier = None
    traffic_watcher = None
    scenario_configs = None
    if args.with_scenarios:
        from scenarios.applier import ScenarioApplier
        from scenarios.config_space import load_scenario_settings
        from scenarios.rollout import experiment_config_list

        sp = Path(args.scenario_config)
        if not sp.is_absolute():
            sp = repo_root / sp
        sdata = load_yaml(sp)
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
        settings.dry_run = settings.dry_run or getattr(config, "SCENARIO_DRY_RUN", False) or args.dry_run

        scenario_applier = ScenarioApplier(settings, pmf_client=pmf_client)
        traffic_watcher = TrafficWatcher(settings)
        scenario_configs = experiment_config_list(settings)
        print(f"[eval] Loaded {len(scenario_configs)} scenario configs")

    all_records: List[Dict[str, Any]] = []
    episode_summaries: List[Dict[str, Any]] = []
    experiment_start_monotonic = time.monotonic()
    pacer = StepPacer(interval_s=interval_s, ema_alpha=RLEnv.EMA_ALPHA)

    try:
        if args.with_scenarios and scenario_configs and scenario_applier:
            episode_index = 0
            for sc_idx, sc_cfg in enumerate(scenario_configs):
                label = sc_cfg.label or f"scenario_{sc_idx}"
                tc = sc_cfg.traffic_class.value
                print(f"\n[eval] Scenario {sc_idx + 1}/{len(scenario_configs)}: {label}")

                for _ in range(args.episodes):
                    episode_index += 1

                    # apply tc rules + iperf3
                    scenario_applier.apply(sc_cfg)
                    # flush PMF cache and wait for metrics to settle
                    if hasattr(pmf_client, "flush_cache"):
                        pmf_client.flush_cache()
                    time.sleep(args.settle_s)

                    if args.dry_run:
                        pmf_client.dry_run_reset_metrics_to_base()

                    if hasattr(agent, "reset"):
                        agent.reset()

                    sc_records = _run_eval_steps(
                        agent, agent_type, pmf_client, aue_client, session_id,
                        args.steps, reward_params, experiment_start_monotonic, pacer,
                        scenario_label=label,
                        traffic_class=tc,
                        watcher=traffic_watcher
                    )
                    all_records.extend(sc_records)

                    # per-scenario summary
                    ok = [r for r in sc_records if r["ok"]]
                    rewards = [r["reward"] for r in ok if r.get("reward") is not None]
                    wifi_ratios = [r["wifi_ratio"] for r in ok if r.get("wifi_ratio") is not None]
                    episode_summaries.append({
                        "episode_index": episode_index,
                        "scenario_label": label,
                        "traffic_class": tc,
                        "steps_ok": len(ok),
                        "steps_total": len(sc_records),
                        "episode_reward_total": round(sum(rewards), 4) if rewards else None,
                        "reward_mean_per_step": round(sum(rewards) / len(rewards), 4) if rewards else None,
                        "wifi_ratio_mean": round(sum(wifi_ratios) / len(wifi_ratios), 4) if wifi_ratios else None,
                    })
                    if rewards:
                        print(f"  -> reward_mean={sum(rewards)/len(rewards):.3f} wifi_ratio={sum(wifi_ratios)/len(wifi_ratios):.3f}")

            # clean up tc rules
            scenario_applier.cleanup()
        else:
            if args.episodes > 1:
                raise NotImplementedError()
            # standard eval (no scenarios, single episode)
            sc_records = _run_eval_steps(
                agent, agent_type, pmf_client, aue_client, session_id,
                args.steps, reward_params, experiment_start_monotonic, pacer
            )
            all_records.extend(sc_records)
    finally:
        agent.shutdown()
        pmf_client.shutdown()
        if scenario_applier is not None:
            try:
                scenario_applier.cleanup()
            except Exception:
                pass

    # save CSV
    csv_path = run_dir / "decisions.csv"
    if all_records:
        with csv_path.open("w", newline="") as f:
            # important: get field names from first step, not from initial observation (entry/step 0)
            fieldnames = list(all_records[1].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    # save episode/scenario summary CSV
    if episode_summaries:
        ep_csv = run_dir / "scenario_summary.csv"
        with ep_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(episode_summaries)

    step_time_ms = np.array([r["step_time_ms"] for r in all_records if r["step_time_ms"] is not None])
    inference_time_ms = np.array([r.get("inference_time_ms") for r in all_records if r.get("inference_time_ms") is not None])
    pacer_obs_time_ms = np.array([r.get("pacer_obs_time_ms") for r in all_records if r.get("pacer_obs_time_ms") is not None])
    pacer_sleep_time_ms = np.array([r.get("pacer_sleep_time_ms") for r in all_records if r.get("pacer_sleep_time_ms") is not None])
    pacer_outside_time_ms = np.array([r.get("pacer_outside_time_ms") for r in all_records if r.get("pacer_outside_time_ms") is not None])
    rewards = np.array([r["reward"] for r in all_records if r["reward"] is not None])
    wifi_ratios = np.array([r["wifi_ratio"] for r in all_records if r["wifi_ratio"] is not None])
    fiveg_ratios = np.array([r["fiveg_ratio"] for r in all_records if r["fiveg_ratio"] is not None])
    summary = {
        "agent": agent_type,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "with_scenarios": args.with_scenarios,
        "scenarios_evaluated": len(scenario_configs) if scenario_configs else 0,
        "steps_per_scenario": args.steps,
        "total_num_records": len(all_records),
        "success_rate_pct": round(len(all_records) / len(all_records) * 100, 2) if all_records else 0,
        "step_time_ms": np_arr_summary_dict(step_time_ms),
        "inference_time_ms": np_arr_summary_dict(inference_time_ms),
        "pacer_obs_time_ms": np_arr_summary_dict(pacer_obs_time_ms),
        "pacer_sleep_time_ms": np_arr_summary_dict(pacer_sleep_time_ms),
        "pacer_outside_time_ms": np_arr_summary_dict(pacer_outside_time_ms),
        "reward": np_arr_summary_dict(rewards),
        "wifi_ratio": np_arr_summary_dict(wifi_ratios),
        "fiveg_ratio": np_arr_summary_dict(fiveg_ratios),
        "sequence_len": args.sequence_len if agent_type == "lstm" else None,
        "heuristic_type": args.heuristic if agent_type == "heuristic" else None,
    }

    # per-scenario reward breakdown in summary
    if episode_summaries:
        summary["per_scenario"] = episode_summaries

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # plots
    if all_records:
        plot_latency(all_records, run_dir / "latency_over_time.png", title="Policy Step Latency Over Time")
        if any(r.get("inference_time_ms") is not None for r in all_records):
            plot_decision_time_over_time(
                all_records,
                run_dir / "inference_time_ms.png",
                "inference_time_ms",
                "Inference time (ms)",
                title="Inference Time",
            )
            plot_decision_time_histogram(
                all_records,
                run_dir / "inference_time_ms.png",
                "inference_time_ms",
                "Inference time (ms)",
                title="Inference Time Distribution",
            )

        plot_ratio(all_records, run_dir / "ratio_over_time.png")
        plot_network_distribution(all_records, run_dir / "network_distribution.png")
        plot_eval_timeline_reward_action(all_records, run_dir / "eval_timeline_reward_actions.png")
        plot_eval_network_conditions(all_records, run_dir / "eval_network_conditions.png")
        plot_eval_timeline_conditions_action(all_records, run_dir / "eval_timeline_conditions_actions.png")
        plot_bytes_sent(all_records, run_dir / "bytes_sent.png")
        plot_bitrate_sent(all_records, run_dir / "bitrate_sent.png")

    if episode_summaries:
        plot_reward_by_scenario(episode_summaries, run_dir / "reward_by_scenario.png")
        plot_reward_by_traffic_class(episode_summaries, run_dir / "reward_by_traffic_class.png")
        plot_scenario_distribution(episode_summaries, run_dir / "scenario_distribution.png")
        plot_action_by_scenario(episode_summaries, run_dir / "action_by_scenario.png")

    if args.with_scenarios and all_records:
        plot_rtt_by_traffic_class(all_records, run_dir / "rtt_by_traffic_class.png")
        plot_plr_by_traffic_class(all_records, run_dir / "plr_by_traffic_class.png")

    print("\n[eval] Done")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
