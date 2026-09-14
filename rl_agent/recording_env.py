# gym wrapper that records (observation, action, timing) for training plots.

from __future__ import annotations

import time
from typing import Any, Dict, List

import gymnasium as gym

from rl_agent.rl_env import OBS_DIM


class RecordingEnv(gym.Wrapper):

    def __init__(self, env: gym.Env, records: List[Dict[str, Any]], start_time: float):
        super().__init__(env)
        self._records = records
        self._start_time = start_time
        self._episode_start = start_time
        self._obs_ready_time: float | None = None

    def reset(self, *, seed=None, options=None):
        self._episode_start = time.time()
        obs, info = self.env.reset(seed=seed, options=options)
        # Timestamp when a fresh observation becomes available to the policy.
        self._obs_ready_time = time.time()
        return obs, info

    def step(self, action):
        action_received_t = time.time()
        decision_time_ms = None
        if self._obs_ready_time is not None:
            decision_time_ms = max(0.0, (action_received_t - self._obs_ready_time) * 1000.0)
        t0 = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        step_duration_ms = (time.time() - t0) * 1000.0
        elapsed_s = time.time() - self._start_time
        # the next observation is now available to the policy.
        self._obs_ready_time = time.time()

        flat_obs = obs[-1] if (hasattr(obs, "ndim") and obs.ndim == 2) else obs

        record: Dict[str, Any] = {
            "elapsed_s": round(elapsed_s, 3),
            "step": info.get("step"),
            "episode_idx": info.get("episode_idx"),
            "step_in_episode": info.get("step_in_episode", info.get("step")),
            "latency_ms": round(step_duration_ms, 3),
            "decision_time_ms": round(decision_time_ms, 3) if decision_time_ms is not None else None,
        }

        if flat_obs is not None and hasattr(flat_obs, "__len__") and len(flat_obs) >= OBS_DIM:
            record["wifi_rtt_ms"] = round(float(flat_obs[0]) * 1000.0, 4)
            record["fiveg_rtt_ms"] = round(float(flat_obs[1]) * 1000.0, 4)
            record["wifi_plr"] = round(float(flat_obs[2]), 6)
            record["fiveg_plr"] = round(float(flat_obs[3]), 6)
            record["wifi_rtt_min_ms"] = round(float(flat_obs[4]) * 1000.0, 4)
            record["wifi_rtt_max_ms"] = round(float(flat_obs[5]) * 1000.0, 4)
            record["wifi_rtt_std_ms"] = round(float(flat_obs[6]) * 1000.0, 4)
            record["fiveg_rtt_min_ms"] = round(float(flat_obs[7]) * 1000.0, 4)
            record["fiveg_rtt_max_ms"] = round(float(flat_obs[8]) * 1000.0, 4)
            record["fiveg_rtt_std_ms"] = round(float(flat_obs[9]) * 1000.0, 4)
            record["observation_age_ms"] = round(float(flat_obs[10]) * 1000.0, 3)
        else:
            for k in ("wifi_rtt_ms", "fiveg_rtt_ms", "wifi_plr", "fiveg_plr",
                       "wifi_rtt_min_ms", "wifi_rtt_max_ms", "wifi_rtt_std_ms",
                       "fiveg_rtt_min_ms", "fiveg_rtt_max_ms", "fiveg_rtt_std_ms",
                       "observation_age_ms"):
                record[k] = None

        record["wifi_ratio"] = info.get("wifi_ratio")
        record["fiveg_ratio"] = info.get("fiveg_ratio")

        record["weighted_rtt_ms"] = info.get("weighted_rtt_ms")
        record["weighted_plr"] = info.get("weighted_plr")
        record["delta_action"] = info.get("delta_action")

        record["rtt_term"] = info.get("rtt_term")
        record["plr_term"] = info.get("plr_term")
        record["stability_term"] = info.get("stability_term")
        record["reward_total"] = info.get("reward_total")

        record["bytes_sent_fifo"] = info.get("bytes_sent_fifo")
        record["bytes_sent_wifi"] = info.get("bytes_sent_wifi")
        record["bytes_sent_fiveg"] = info.get("bytes_sent_fiveg")

        for k, v in info.items():
            if k.startswith("scenario_") or k in (
                # scenarios
                "traffic_class",
                "cross_traffic_mbps",
                "cross_traffic_on_wifi",
                "user_data_mode",
                "user_data_mbps",
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
                "bursty_mbit_per_period",
                "bursty_period_s",
                # timestamp strings
                "timestamp_decision_sent",
                "timestamp_decision_applied",
                "timestamp_observation",
            ):
                record[k] = v

        self._records.append(record)
        return obs, reward, terminated, truncated, info
