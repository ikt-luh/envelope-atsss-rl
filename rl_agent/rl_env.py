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


def _fmt_opt_float(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}"


class StepPacer:
    """
    Paces steps in the ATSSS RL environment according to a given time interval.

    For example, assume the goal is to apply an action each 1s. After applying an action, 
    the agent should wait for approximately 1 second, then get a new observation and 
    apply an action for the new observation.

    The problem is that the time it takes to get an observation and the time it takes to make
    a decision (including pauses due to training steps being performed) is unknown.

    This class keeps an exponential moving average of the time it takes to get an observation,
    and measures the time until the next iteration, and aims to set the wait time accordingly.

    This class is designed to be used as follows:

    ```
    # init pacer at beginning of your code
    pacer = StepPacer(...)

    for episode in range(100):
        with pacer(reset=True):
            initial_observation = get_observation()

        for step in range(100):
            ...
            with pacer:
                next_observation = get_observation()
    ```
    """
    def __init__(self, interval_s: float, ema_alpha: float):
        """
        Args:
            interval_s (float): the step interval in seconds  
            ema_alpha (float): alpha for exponential moving average (alpha * new + (1-alpha) * old)
        """
        self.interval_s = interval_s
        self.alpha = ema_alpha
        assert self.alpha >= 0 and self.alpha <= 1

        self.sleep = True

        self.time_enter = None
        self.time_exit = None
        self.duration_observation = None
        self.duration_observation_smooth = None
        self.duration_outside = None
        self.time_start = time.monotonic()

    def ema(self, old_value, new_value):
        if old_value is None:
            return new_value

        return self.alpha * new_value + (1 - self.alpha) * old_value

    def __call__(self, reset=False):
        if reset:
            self.sleep = False
            self.time_enter = None
            self.time_exit = None
            self.duration_outside = None
            self.time_start = time.monotonic()
        return self

    def _timing_str(self):
        return (
            f"observation: {_fmt_opt_float(self.duration_observation)} (ema {_fmt_opt_float(self.duration_observation_smooth)}), "
            f"outside: {_fmt_opt_float(self.duration_outside)}"
        )

    def __enter__(self):
        now = time.monotonic()
        if self.time_exit is not None:
            # time between two subsequent steps outside the with block,
            # including the agent's decision-making
            self.duration_outside = now - self.time_exit

        if self.sleep and self.interval_s is not None:
            sleep_time = self.interval_s
            if self.duration_outside is not None:
                sleep_time -= self.duration_outside
            if self.duration_observation_smooth is not None:
                sleep_time -= self.duration_observation_smooth

            if sleep_time >= 0:
                time.sleep(sleep_time)
            else:
                logger.warning(
                    f"Decision interval of {self.interval_s} s may be too short. " + self._timing_str()
                )
        else:
            sleep_time = None

        logger.info(f"t = {now - self.time_start:.2f}s: {self._timing_str()}, sleep {_fmt_opt_float(sleep_time)}s")

        self.time_enter = time.monotonic()
        self.sleep = True
        return self

    def __exit__(self, type, value, tb):
        self.time_exit = time.monotonic()
        # time for getting the observation
        self.duration_observation = self.time_exit - self.time_enter
        self.duration_observation_smooth = self.ema(self.duration_observation_smooth, self.duration_observation)


class RLEnv(gym.Env):
    """Base environment with shared step, reset, reward, and early termination logic."""

    # Initial ratio applied upon episode reset
    INITIAL_WIFI_RATIO: float = 0.5
    EMA_ALPHA: float = 0.1

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
        discrete_n: Optional[bool] = None,
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
        self.discrete_n = discrete_n
        self.action_is_discrete = discrete_n is not None
        if self.action_is_discrete:
            assert discrete_n >= 2
            self.action_space = spaces.Discrete(discrete_n)
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0]),
                high=np.array([1.0]),
                dtype=np.float32,
            )

        self.pacer = StepPacer(self.decision_interval, RLEnv.EMA_ALPHA)

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

        with self.pacer(reset=True):
            metrics = self.pmf_client.get_metrics(self.session_id)

        self.current_observation = build_observation(metrics, RLEnv.INITIAL_WIFI_RATIO, 0, 0)
        self.bytes_sent_dict = self.get_bytes_sent_dict()  # to compute the diff later

        self._update_obs_for_step(self.current_observation)
        return self._get_obs_for_step(), dict(self._scenario_info)

    def get_bytes_sent_dict(self):
        return TrafficWatcher.get_bytes_sent_dict(self.scenario_manager.watcher if self.scenario_manager is not None else None)

    def step(self, action):
        self.current_step += 1
        if self.action_is_discrete:
            wifi_ratio = self.discrete_to_continuous_action(action.item(), self.discrete_n)
        else:
            wifi_ratio = float(np.clip(action.item(), 0.0, 1.0))

        # first apply the action in the environment
        timestamp_decision_sent = get_timestamp_str()
        if self.aue_client is not None:
            try:
                self.aue_client.send_ratio(wifi_ratio)
            except Exception as e:
                logger.warning("AUE send_ratio failed: %s", e)

        # the metrics reflect the network conditions after applying the action
        # the pacer might sleep some time before we get the metrics
        with self.pacer:
            metrics = self.pmf_client.get_metrics(self.session_id)

        # update traffic stats
        new_bytes_sent_dict = self.get_bytes_sent_dict()
        delta_bytes_sent = TrafficWatcher.delta(self.bytes_sent_dict, new_bytes_sent_dict)
        self.bytes_sent_dict = new_bytes_sent_dict

        next_observation = build_observation(
            metrics,
            wifi_ratio,
            delta_bytes_wifi=delta_bytes_sent['bytes_sent_wifi'],
            delta_bytes_fiveg=delta_bytes_sent['bytes_sent_fiveg']
        )
        self._update_obs_for_step(next_observation)
        timestamp_observation = get_timestamp_str()

        # record when action was applied
        if self.aue_client is not None:
            timestamp_decision_applied = self.aue_client.get_timestamp()
        else:
            timestamp_decision_applied = timestamp_decision_sent

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

    @staticmethod
    def discrete_to_continuous_action(action: int, n: int) -> float:
        assert n >= 2
        return float(action) / (n - 1)

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
        discrete_n: Optional[bool] = None,
        aue_client=None,
        window_size: int = 10,
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
            discrete_n=discrete_n,
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
