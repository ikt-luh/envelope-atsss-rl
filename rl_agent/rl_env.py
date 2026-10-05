# Shared RL environment for MPQUIC. RLEnv holds common logic; PPO and LSTM differ by observation shape.
# Logarithmic reward for spike control and unit-invariance.

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import time

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from pmf_client.client import PMFMetrics
from scenarios.applier import ScenarioManager, TrafficWatcher
from scenarios.sampler import ScenarioConfig

from datetime import datetime

try:
    from datetime import UTC
except ImportError:  # fix for Python < 3.11
    from datetime import timezone
    UTC = timezone.utc

logger = logging.getLogger(__name__)

OBS_DIM = 16
OBS_FLAG_VALID = 1
OBS_FLAG_INVALID = 0

OBS_LOW = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, OBS_FLAG_INVALID, OBS_FLAG_INVALID, OBS_FLAG_INVALID],
    dtype=np.float32,
)
OBS_HIGH = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, float('inf'), float('inf'), 1.0, OBS_FLAG_VALID, OBS_FLAG_VALID, OBS_FLAG_VALID],
    dtype=np.float32,
)
OBS_PLACEHOLDER = OBS_LOW


def build_observation(metrics: PMFMetrics, previous_action, delta_bytes_wifi, delta_bytes_fiveg) -> np.ndarray:
    """
    Build an observation from a PMFMetrics instance and additional info.
    """
    obs = np.array([
        metrics.wifi_rtt_ms / 1000.0,
        metrics.fiveg_rtt_ms / 1000.0,
        metrics.wifi_plr,
        metrics.fiveg_plr,
        metrics.wifi_rtt_min_ms / 1000.0,
        metrics.wifi_rtt_max_ms / 1000.0,
        metrics.wifi_rtt_std_ms / 1000.0,
        metrics.fiveg_rtt_min_ms / 1000.0,
        metrics.fiveg_rtt_max_ms / 1000.0,
        metrics.fiveg_rtt_std_ms / 1000.0,
        delta_bytes_wifi / 1_000_000.0,
        delta_bytes_fiveg / 1_000_000.0,
        previous_action,
        OBS_FLAG_VALID if metrics.wifi_valid else OBS_FLAG_INVALID,
        OBS_FLAG_VALID if metrics.fiveg_valid else OBS_FLAG_INVALID,
        OBS_FLAG_VALID], dtype=np.float32)
    return np.clip(obs, OBS_LOW, OBS_HIGH)


class LogRLEnvMetricsCallback(RLlibCallback):
    metric_keys = [
        "wifi_ratio",
        "fiveg_ratio",
        "weighted_rtt_ms",
        "weighted_plr",
        "delta_action",
        "rtt_term",
        "plr_term",
        "stability_term",
        "reward_total",
        # interface metrics
        "bytes_sent_fifo",
        "bytes_sent_wifi",
        "bytes_sent_fiveg",
    ]

    def on_episode_created(self, *, episode, **kwargs):
        for key in self.metric_keys:
            episode.user_data[key] = []

    def on_episode_step(self, *, episode: EpisodeV2, **kwargs):
        info = episode.last_info_for()
        for key in self.metric_keys:
            episode.user_data[key].append(info[key])

    def on_episode_end(self, *, episode: EpisodeV2, **kwargs):
        for key in self.metric_keys:
            vals = np.array(episode.user_data[key])
            episode.custom_metrics[f"episode/{key}_mean"] = vals.mean()
            episode.custom_metrics[f"episode/{key}_min"] = vals.min()
            episode.custom_metrics[f"episode/{key}_max"] = vals.max()
            episode.custom_metrics[f"episode/{key}_std"] = vals.std()


def get_timestamp_str() -> str:
    timestamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    return timestamp.replace("+00:00", "Z")


class RLEnv(gym.Env):
    """Base environment with shared step, reset, reward, and early termination logic."""

    # Initial ratio applied upon episode reset
    INITIAL_WIFI_RATIO: float = 0.5

    def __init__(
        self,
        decision_interval: Optional[float],
        pmf_client=None,
        session_id: str = "default",
        reward_params: Optional[Dict[str, Any]] = None,
        early_termination_reward_threshold: Optional[float] = None,
        early_termination_window: int = 50,
        max_steps: int = 1000,
        scenario_manager: Optional[ScenarioManager] = None,
        episodes_per_config: int = 1,
        aue_client=None,
    ):
        super().__init__()
        self.decision_interval = decision_interval
        self.pmf_client = pmf_client
        self.session_id = session_id
        self.aue_client = aue_client
        self.reward_params = reward_params or {}
        self.early_termination_reward_threshold = early_termination_reward_threshold
        self.early_termination_window = early_termination_window
        self.current_step = 0
        self.max_steps = max_steps
        self.current_observation = OBS_PLACEHOLDER.copy()
        self.prev_wifi_ratio: Optional[float] = None
        self._reward_history: list = []
        self._last_reward_components: Dict[str, float] = {}
        self._episode_idx: int = 0
        self.scenario_manager = scenario_manager
        self.episodes_per_config = max(1, int(episodes_per_config))
        self._scenario_info: Dict[str, Any] = {}

        self.observation_space = spaces.Box(
            low=OBS_LOW.copy(),
            high=OBS_HIGH.copy(),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32,
        )
        self.episode_start_time = None
        self.get_metrics_time_smooth = None
        self.get_metrics_time_alpha = 0.5

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_wifi_ratio = RLEnv.INITIAL_WIFI_RATIO
        self._reward_history = []
        self._last_reward_components = {}
        self._episode_idx += 1

        # first apply the action in the environment
        if self.aue_client is not None:
            try:
                self.aue_client.send_ratio(RLEnv.INITIAL_WIFI_RATIO)
            except Exception as e:
                logger.warning("AUE send_ratio failed: %s", e)

        if self.scenario_manager is not None:
            if (self._episode_idx - 1) % self.episodes_per_config == 0:
                cfg: ScenarioConfig = self.scenario_manager.sample_and_apply()
                # print(f"[train] Applying scenario {cfg.label}, rtt {cfg.wifi_extra_delay_ms}, {cfg.fiveg_extra_delay_ms}, loss {cfg.wifi_loss_pct}, {cfg.fiveg_loss_pct}")
                self._scenario_info = cfg.to_log_dict()
        else:
            self._scenario_info = {}

        if self.pmf_client is not None and self.scenario_manager is None:
            self.pmf_client.dry_run_sample_random_conditions()

        self.episode_start_time = time.time()
        self.bytes_sent_dict = self.get_bytes_sent_dict()  # to compute the diff later
        self.current_observation = self._get_observation(RLEnv.INITIAL_WIFI_RATIO, 0, 0)
        self._update_obs_for_step(self.current_observation)
        return self._get_obs_for_step(), dict(self._scenario_info)

    def get_bytes_sent_dict(self):
        return TrafficWatcher.get_bytes_sent_dict(self.scenario_manager.watcher if self.scenario_manager is not None else None)

    def step(self, action):
        self.current_step += 1
        wifi_ratio = float(np.clip(action[0], 0.0, 1.0))

        # first apply the action in the environment
        timestamp_decision_sent = get_timestamp_str()
        if self.aue_client is not None:
            try:
                self.aue_client.send_ratio(wifi_ratio)
            except Exception as e:
                logger.warning("AUE send_ratio failed: %s", e)

        # wait a bit until we have (approximately) reached our decision interval
        if self.decision_interval is not None:
            sleep_time = self.decision_interval - self.get_metrics_time_smooth
            if sleep_time > 0:
                time.sleep(sleep_time)
                logging.debug(f"Sleeping for {int(sleep_time * 1000)} ms to approximate {self.decision_interval} s decision interval")
            else:
                logging.warning(f"The decision interval ({self.decision_interval} s) might be too short. Getting the observation took {self.get_metrics_time_smooth} s.")

        # check when action was applied
        if self.aue_client is not None:
            timestamp_decision_applied = self.aue_client.get_timestamp()
        else:
            timestamp_decision_applied = timestamp_decision_sent

        # update traffic stats
        new_bytes_sent_dict = self.get_bytes_sent_dict()
        delta_bytes_sent = TrafficWatcher.delta(self.bytes_sent_dict, new_bytes_sent_dict)
        self.bytes_sent_dict = new_bytes_sent_dict

        # get a new observation that reflects the network conditions after applying our action
        next_observation = self._get_observation(
            self.prev_wifi_ratio,
            delta_bytes_wifi=delta_bytes_sent['bytes_sent_wifi'],
            delta_bytes_fiveg=delta_bytes_sent['bytes_sent_fiveg']
        )
        self._update_obs_for_step(next_observation)
        timestamp_observation = get_timestamp_str()

        # get the reward based this observation
        reward, self._last_reward_components = RLEnv.get_reward_and_metrics(
            next_observation, wifi_ratio, self.prev_wifi_ratio, **self.reward_params)
        self.prev_wifi_ratio = wifi_ratio

        # packet routing is an infinite-horizon task, meaning that episodes
        # are never done, only truncated
        done = False
        truncated = self.current_step >= self.max_steps
        if not truncated and self.early_termination_reward_threshold is not None:
            self._reward_history.append(reward)
            if len(self._reward_history) > self.early_termination_window:
                self._reward_history.pop(0)
            if len(self._reward_history) >= self.early_termination_window:
                mean_reward = sum(self._reward_history) / len(self._reward_history)
                if mean_reward >= self.early_termination_reward_threshold:
                    truncated = True

        info = {
            "step": self.current_step,
            "episode_idx": self._episode_idx,
            "step_in_episode": self.current_step,  # TODO: remove duplicate info
            "timestamp_decision_sent": timestamp_decision_sent,
            "timestamp_decision_applied": timestamp_decision_applied,
            "timestamp_observation": timestamp_observation,
            **self._last_reward_components,
            **self._scenario_info,
            **new_bytes_sent_dict
        }
        if hasattr(self, "observation_history"):
            info["sequence_len"] = len(self.observation_history)
        self.current_observation = next_observation
        return self._get_obs_for_step(), reward, done, truncated, info

    def _get_observation(self, previous_action, delta_bytes_wifi, delta_bytes_fiveg) -> np.ndarray:
        if self.pmf_client is None:
            raise RuntimeError(
                "get_observation failed. PMF is not connected. Training requires a PMF client. "
            )
        t0 = time.time()
        metrics = self.pmf_client.get_metrics(self.session_id)
        get_metrics_time = time.time() - t0

        if self.get_metrics_time_smooth is None:
            self.get_metrics_time_smooth = get_metrics_time
        else:
            self.get_metrics_time_smooth = self.get_metrics_time_alpha * self.get_metrics_time_smooth + (1 - self.get_metrics_time_alpha) * get_metrics_time

        logging.debug(f"{self.current_step}: t = {time.time() - self.episode_start_time}, delta_t_obs = {get_metrics_time} s, delta_t_obs_avg = {self.get_metrics_time_smooth} s")

        if metrics is None:
            raise RuntimeError(
                "No metrics available. "
                "Ensure PMF is running, reachable, and polling has produced cached metrics."
            )
        return build_observation(metrics, previous_action=previous_action, delta_bytes_wifi=delta_bytes_wifi, delta_bytes_fiveg=delta_bytes_fiveg)

    @staticmethod
    def get_reward_and_metrics(observation: np.ndarray, wifi_ratio: float, prev_wifi_ratio: float, r_max=1.0, epsilon=1e-6, alpha=25) -> float:
        wifi_rtt = float(observation[0])
        fiveg_rtt = float(observation[1])
        wifi_plr_val = float(observation[2])
        fiveg_plr_val = float(observation[3])

        weighted_rtt = wifi_ratio * wifi_rtt + (1 - wifi_ratio) * fiveg_rtt
        weighted_plr = wifi_ratio * wifi_plr_val + (1 - wifi_ratio) * fiveg_plr_val
        change = abs(wifi_ratio - prev_wifi_ratio) if prev_wifi_ratio is not None else 0.0

        # TODO: Maybe relative weighted rtt/plr targets based on observed metrics per link?

        if weighted_rtt > r_max:
            print(f"[train] WARNING: Weighted RTT of exceeds r_max: {weighted_rtt:.4f}s > {r_max:.4f}s")
            rtt_norm = 1.0
        else:
            rtt_norm = weighted_rtt / r_max
        plr_norm = min(weighted_plr, 1.0)
        rtt_term = math.log(1.0 - rtt_norm + epsilon)
        plr_term = math.log(1.0 - plr_norm + epsilon)
        stability_term = math.log(1.0 + alpha * change)
        reward = rtt_term + plr_term - stability_term

        last_reward_components = {
            "wifi_ratio": wifi_ratio,
            "fiveg_ratio": 1.0 - wifi_ratio,
            "weighted_rtt_ms": weighted_rtt * 1000.0,
            "weighted_plr": weighted_plr,
            "delta_action": change,
            "rtt_term": rtt_term,
            "plr_term": plr_term,
            "stability_term": -stability_term,
            "reward_total": reward,
        }
        return reward, last_reward_components

    def _get_obs_for_step(self):
        """Return observation for step/reset. Override in LSTM."""
        return self.current_observation

    def _update_obs_for_step(self, next_observation: np.ndarray) -> None:
        """Update state after step. Override in LSTM for observation_history."""
        pass


class MPQUICPPOEnv(RLEnv):

    def __init__(
        self,
        decision_interval,
        pmf_client=None,
        session_id: str = "default",
        reward_params: Optional[Dict[str, Any]] = None,
        early_termination_reward_threshold: Optional[float] = None,
        early_termination_window: int = 50,
        max_steps: int = 1000,
        scenario_manager: Any = None,
        episodes_per_config: int = 1,
        window_size: int = 10,
        aue_client=None,
    ):
        super().__init__(
            decision_interval=decision_interval,
            pmf_client=pmf_client,
            session_id=session_id,
            reward_params=reward_params,
            early_termination_reward_threshold=early_termination_reward_threshold,
            early_termination_window=early_termination_window,
            max_steps=max_steps,
            scenario_manager=scenario_manager,
            episodes_per_config=episodes_per_config,
            aue_client=aue_client,
        )
        self.window_size = max(1, int(window_size))
        self.observation_history: list = []
        flat_low = np.tile(OBS_LOW, self.window_size)
        flat_high = np.tile(OBS_HIGH, self.window_size)
        self.observation_space = spaces.Box(
            low=flat_low, high=flat_high, dtype=np.float32
        )

    def _get_obs_for_step(self) -> np.ndarray:
        # remove old observations outside the history size
        while len(self.observation_history) > self.window_size:
            self.observation_history.pop(0)
        # print(np.concatenate(self.observation_history))
        return np.concatenate(self.observation_history)

    def _update_obs_for_step(self, next_observation: np.ndarray) -> None:
        """Append latest observation to the history buffer."""
        self.observation_history.append(next_observation.copy())

    def reset(self, *, seed=None, options=None):
        self.observation_history = []
        # reset internally adds the observation to the history
        _, info = super().reset(seed=seed, options=options)
        # add initial placeholders to observation history
        placeholder = self.current_observation.copy()
        placeholder[-1] = OBS_FLAG_VALID
        while len(self.observation_history) < self.window_size:
            self.observation_history.insert(0, placeholder)
        return self._get_obs_for_step(), info


class MPQUICLSTMEnv(RLEnv):
    """LSTM env: flat observation (OBS_DIM,).

    The LSTM hidden state is carried across steps by the agent, not the
    environment.  RLlib handles BPTT internally during training using
    max_seq_len; during inference the agent threads (h, c) between calls.
    """
    pass
