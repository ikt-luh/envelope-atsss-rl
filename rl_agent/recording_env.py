# gym wrapper that records (observation, action, timing) for training plots.

from __future__ import annotations

import time
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np

from rl_agent.rl_env import OBS_DIM


class RecordingEnv(gym.Wrapper):

    def __init__(self, env: gym.Env, records: List[Dict[str, Any]]):
        super().__init__(env)
        self._records = records
        now_monotonic = time.monotonic()
        self._start_time = now_monotonic
        self._episode_start_monotonic = now_monotonic
        self._obs_ready_time: float | None = None

    def reset(self, *, seed=None, options=None):
        self._episode_start_monotonic = time.monotonic()
        obs, info = self.env.reset(seed=seed, options=options)
        # Timestamp when a fresh observation becomes available to the policy.
        self._obs_ready_time = time.monotonic()
        return obs, info

    def step(self, action):
        action_received_t = time.monotonic()
        inter_step_time_ms = None
        if self._obs_ready_time is not None:
            inter_step_time_ms = max(0.0, (action_received_t - self._obs_ready_time) * 1000.0)

        before_env_step_monotonic = time.monotonic()
        obs, reward, terminated, truncated, info = self.env.step(action)
        after_env_step_monotonic = time.monotonic()

        step_time_ms = (after_env_step_monotonic - before_env_step_monotonic) * 1000.0

        self._obs_ready_time = after_env_step_monotonic
        record: Dict[str, Any] = {
            "experiment_elapsed_s": after_env_step_monotonic - self._start_time,
            "episode_elapsed_s": after_env_step_monotonic - self._episode_start_monotonic,
            "step": info.get("step"),
            "episode_idx": info.get("episode_idx"),
            "step_time_ms": step_time_ms,
            "inter_step_time_ms": inter_step_time_ms if inter_step_time_ms is not None else None,
        }

        # handle both, windows of observations (latest observations at the end), and single observations
        flat_obs = np.array(obs).reshape(-1)[-OBS_DIM:]
        record["wifi_rtt_ms"] = float(flat_obs[0]) * 1000.0
        record["fiveg_rtt_ms"] = float(flat_obs[1]) * 1000.0
        record["wifi_plr"] = float(flat_obs[2])
        record["fiveg_plr"] = float(flat_obs[3])
        record["wifi_rtt_min_ms"] = float(flat_obs[4]) * 1000.0
        record["wifi_rtt_max_ms"] = float(flat_obs[5]) * 1000.0
        record["wifi_rtt_std_ms"] = float(flat_obs[6]) * 1000.0
        record["fiveg_rtt_min_ms"] = float(flat_obs[7]) * 1000.0
        record["fiveg_rtt_max_ms"] = float(flat_obs[8]) * 1000.0
        record["fiveg_rtt_std_ms"] = float(flat_obs[9]) * 1000.0

        for k, v in info.items():
            if k.startswith("scenario_") or k in (
                # general
                "wifi_ratio",
                "fiveg_ratio",
                # whether measurements are valid
                "wifi_valid",
                "fiveg_valid",
                # reward metrics
                "weighted_rtt_ms",
                "weighted_plr",
                "delta_action",
                "rtt_term",
                "plr_term",
                "stability_term",
                "reward_total",
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
                "timestamp_ratio_sent",
                "timestamp_ratio_applied",
                "timestamp_observation",
                # traffic watcher stats
                "delta_bytes_sent_fifo",
                "delta_bytes_sent_wifi",
                "delta_bytes_sent_fiveg",
                "delta_time_monotonic",
                # additional obs and pacer stats
                "pacer_sleep_time_ms",
                "pacer_outside_time_ms",
                "pacer_obs_time_ms",
                "observation_age_ms",
            ):
                record[k] = v

        self._records.append(record)
        return obs, reward, terminated, truncated, info
