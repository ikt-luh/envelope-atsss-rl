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
from pmf_client.client import PMFClientManager
from rl_agent.heuristic_agent import HeuristicAgent
from rl_agent.lstm_agent import MPQUICLSTMAgent
from rl_agent.ppo_agent import MPQUICPPOAgent
from rl_agent.rl_env import RLEnv, build_observation, get_timestamp_str
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

# import logging
# logging.basicConfig(level=logging.INFO)


def _detect_agent(checkpoint_path: Optional[Path]) -> str:
    # auto-detect agent from checkpoint path (ppo or lstm). heuristic has no checkpoint.
    if checkpoint_path is None:
        return "heuristic"
    path_str = str(checkpoint_path).lower()
    if "lstm" in path_str:
        return "lstm"
    return "ppo"


def _run_eval_steps(
    agent,
    agent_type: str,
    pmf_client,
    aue_client,
    session_id: str,
    num_steps: int,
    interval_s: float,
    reward_params: Dict[str, Any],
    start_t: float,
    *,
    scenario_label: Optional[str] = None,
    traffic_class: Optional[str] = None,
    watcher: Optional[TrafficWatcher] = None,
) -> List[Dict[str, Any]]:
    #run num_steps of evaluation, returning per-step records.
    records: List[Dict[str, Any]] = []
    prev_wifi_ratio: Optional[float] = None

    _null_record = {
        "ok": False, "wifi_ratio": None, "fiveg_ratio": None, "reward": None,
        "decision_time_ms": None,
        "wifi_rtt_ms": None, "fiveg_rtt_ms": None, "wifi_plr": None, "fiveg_plr": None,
        "wifi_rtt_min_ms": None, "wifi_rtt_max_ms": None, "wifi_rtt_std_ms": None,
        "fiveg_rtt_min_ms": None, "fiveg_rtt_max_ms": None, "fiveg_rtt_std_ms": None,
        "observation_age_ms": None, "weighted_rtt_ms": None, "weighted_plr": None,
        "delta_action": None, "rtt_term": None, "plr_term": None,
        "stability_term": None, "reward_total": None,
    }

    def get_metrics_and_time():
        metrics_requested_t = time.time()
        metrics = pmf_client.get_metrics(session_id)
        metrics_time = (time.time() - metrics_requested_t)
        return metrics, metrics_time

    if aue_client is not None:
        try:
            aue_client.send_ratio(RLEnv.INITIAL_WIFI_RATIO)
        except Exception as e:
            print("[eval] ERROR: AUE send_ratio failed: %s", e)
    last_action = RLEnv.INITIAL_WIFI_RATIO

    if watcher is not None:
        watcher.reset_bytes_sent()

    bytes_sent_dict = TrafficWatcher.get_bytes_sent_dict(watcher)  # to compute the diff later

    # get initial metrics
    next_metrics, next_metrics_time = get_metrics_and_time()
    next_delta_bytes_wifi, next_delta_bytes_fiveg = 0, 0
    # print(f"[eval] Getting the initial metrics for step 0 took {(next_metrics_time) * 1000} ms")
    smooth_metrics_time = next_metrics_time
    smooth_metrics_time_alpha = 0.5
    for step in range(num_steps):
        metrics, metrics_time = next_metrics, next_metrics_time
        delta_bytes_wifi, delta_bytes_fiveg = next_delta_bytes_wifi, next_delta_bytes_fiveg
        step_t0 = time.time()

        obs = build_observation(metrics, last_action, delta_bytes_wifi=delta_bytes_wifi, delta_bytes_fiveg=delta_bytes_fiveg)
        t0 = time.time()
        action = agent.select_action(obs)
        decision_time_ms = (time.time() - t0) * 1000.0
        wifi_ratio = float(np.clip(action.item(), 0.0, 1.0))
        fiveg_ratio = 1.0 - wifi_ratio

        timestamp_decision_sent = get_timestamp_str()
        if aue_client is not None:
            try:
                aue_client.send_ratio(wifi_ratio)
            except Exception as e:
                print("[eval] ERROR: AUE send_ratio failed: %s", e)

        if interval_s is not None:
            # Add wait time so that we take a step approximately every interval_s seconds
            wait_time = max(0, interval_s - smooth_metrics_time)
            # print(f"[eval] Waiting for {(wait_time) * 1000} ms")
            if smooth_metrics_time > interval_s:
                print(f"[eval] WARNING: PMF metrics time {smooth_metrics_time} exceeds decision interval {interval_s}")
            time.sleep(wait_time)

        # check when action was applied
        if aue_client is not None:
            timestamp_decision_applied = aue_client.get_timestamp()
        else:
            timestamp_decision_applied = timestamp_decision_sent

        # get metrics of the next step that depend on the agent's current action
        next_metrics, next_metrics_time = get_metrics_and_time()
        timestamp_observation = get_timestamp_str()
        # print(f"[eval] Getting the next metrics took {(next_metrics_time) * 1000} ms")
        smooth_metrics_time = smooth_metrics_time_alpha * smooth_metrics_time + (1 - smooth_metrics_time_alpha) * next_metrics_time

        new_bytes_sent_dict = TrafficWatcher.get_bytes_sent_dict(watcher)
        bytes_sent_delta = TrafficWatcher.delta(bytes_sent_dict, new_bytes_sent_dict)
        next_delta_bytes_wifi = bytes_sent_delta['bytes_sent_wifi']
        next_delta_bytes_fiveg = bytes_sent_delta['bytes_sent_fiveg']
        bytes_sent_dict = new_bytes_sent_dict

        # print(bytes_sent_dict, new_bytes_sent_dict, bytes_sent_delta)

        reward, reward_metrics = RLEnv.get_reward_and_metrics(obs, wifi_ratio, prev_wifi_ratio, **reward_params)

        prev_wifi_ratio = wifi_ratio
        last_action = action.item()

        latency_ms = (time.time() - step_t0) * 1000
        rec = {
            "step": step,
            "elapsed_s": round(time.time() - start_t, 3),
            "latency_ms": round(latency_ms, 3),
            "decision_time_ms": round(float(decision_time_ms), 3),
            "metrics_time_ms": round(float(next_metrics_time * 1000), 3),
            "ok": True,
            "wifi_ratio": wifi_ratio,
            "fiveg_ratio": fiveg_ratio,
            "reward": round(reward, 4),
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
            "timestamp_decision_sent": timestamp_decision_sent,
            "timestamp_decision_applied": timestamp_decision_applied,
            "timestamp_observation": timestamp_observation,
            "observation_age_ms": round(getattr(metrics, "observation_age_s", 0.0) * 1000.0, 3),
            **reward_metrics,
            **new_bytes_sent_dict
        }
        if scenario_label is not None:
            rec["scenario_label"] = scenario_label
            rec["traffic_class"] = traffic_class
        records.append(rec)

        if (step + 1) % 10 == 0:
            print(f"  step {step + 1}/{num_steps} wifi={wifi_ratio:.3f} reward={reward:.3f}")

    return records


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
    start_t = time.time()

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

                    # reset LSTM hidden state at each scenario boundary so the
                    # agent starts each scenario without stale temporal context
                    if agent_type == "lstm" and hasattr(agent, "reset_state"):
                        agent.reset_state()

                    if hasattr(agent, "reset"):
                        agent.reset()

                    sc_records = _run_eval_steps(
                        agent, agent_type, pmf_client, aue_client, session_id,
                        args.steps, interval_s, reward_params, start_t,
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
                args.steps, interval_s, reward_params, start_t,
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
    ok_records = [r for r in all_records if r["ok"]]
    csv_path = run_dir / "decisions.csv"
    if all_records:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)

    # save episode/scenario summary CSV
    if episode_summaries:
        ep_csv = run_dir / "scenario_summary.csv"
        with ep_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(episode_summaries)

    latencies = [r["latency_ms"] for r in all_records if r["latency_ms"] is not None]
    decision_times = [r.get("decision_time_ms") for r in all_records if r.get("decision_time_ms") is not None]
    metrics_time_ms = [r.get("metrics_time_ms") for r in all_records if r.get("metrics_time_ms") is not None]
    rewards = [r["reward"] for r in ok_records if r["reward"] is not None]
    summary = {
        "agent": agent_type,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "with_scenarios": args.with_scenarios,
        "scenarios_evaluated": len(scenario_configs) if scenario_configs else 0,
        "steps_per_scenario": args.steps,
        "steps_total": len(all_records),
        "steps_ok": len(ok_records),
        "success_rate_pct": round(len(ok_records) / len(all_records) * 100, 2) if all_records else 0,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "decision_time_ms": {
            "mean": round(sum(decision_times) / len(decision_times), 3) if decision_times else None,
            "max": max(decision_times) if decision_times else None,
        },
        "metrics_time_ms": {
            "mean": round(sum(metrics_time_ms) / len(metrics_time_ms), 3) if metrics_time_ms else None,
            "max": max(metrics_time_ms) if metrics_time_ms else None,
        },
        "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "ratio_mean": {
            "wifi": round(sum(r["wifi_ratio"] for r in ok_records) / len(ok_records), 6) if ok_records else None,
            "fiveg": round(sum(r["fiveg_ratio"] for r in ok_records) / len(ok_records), 6) if ok_records else None,
        },
        "ratio_std": {
            "wifi": round(np.std([r["wifi_ratio"] for r in ok_records]), 6) if ok_records else None,
            "fiveg": round(np.std([r["fiveg_ratio"] for r in ok_records]), 6) if ok_records else None,
        },
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
        if any(r.get("decision_time_ms") is not None for r in all_records):
            plot_decision_time_over_time(
                all_records,
                run_dir / "decision_time_over_time.png",
                title="RL Decision Time (per decision)",
            )
            plot_decision_time_histogram(
                all_records,
                run_dir / "decision_time_histogram.png",
                title="Decision Time Distribution",
            )
        if ok_records:
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
