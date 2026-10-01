# heuristic baseline agents.
# implements lb-rtt-min and lb-plr-min (argmin over smoothed RTT/PLR).

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from rl_agent.rl_env import RLEnv

logger = logging.getLogger(__name__)

HEURISTIC_TYPES = ("lb-rtt-min", "lb-plr-min", "lb-utility", "lb-random", "lb-wifi", "lb-fiveg", "lb-equal")


class HeuristicAgent:

    def __init__(
        self,
        heuristic_type: str = "lb-rtt-min",
        alpha: float = 0.95,
    ):
        if heuristic_type not in HEURISTIC_TYPES:
            raise ValueError(
                f"Unknown heuristic_type: {heuristic_type}. "
                f"Supported: {list(HEURISTIC_TYPES)}"
            )
        self.heuristic_type = heuristic_type
        self.alpha = alpha
        self.session_id: Optional[str] = None

        # EMA state
        self._smooth_wifi_rtt: Optional[float] = None
        self._smooth_fiveg_rtt: Optional[float] = None
        self._smooth_wifi_plr: Optional[float] = None
        self._smooth_fiveg_plr: Optional[float] = None

    def initialize(
        self,
        reward_params,
        pmf_client=None,
        session_id: str = "default",
        training_records=None,
        training_start_time=0.0,
        early_termination_reward_threshold=None,
        early_termination_window: int = 50,
    ):
        self.session_id = session_id
        logger.info(f"Heuristic agent initialized: {self.heuristic_type} (alpha={self.alpha})")
        self.last_action = None
        self.reward_params = reward_params

    def reset(self):
        self.last_action = None
        self._smooth_wifi_rtt = None
        self._smooth_fiveg_rtt = None
        self._smooth_wifi_pl = None
        self._smooth_fiveg_plr = None

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        if observation.ndim != 1 or observation.shape[0] < 4:
            raise ValueError(f"Expected 1D observation with >= 4 elements, got shape {observation.shape}")

        if self.heuristic_type == "lb-rtt-min":
            wifi_ratio = self._lb_rtt_min(observation)
        elif self.heuristic_type == "lb-plr-min":
            wifi_ratio = self._lb_plr_min(observation)
        elif self.heuristic_type == "lb-utility":
            wifi_ratio = self._lb_utility(observation)
        elif self.heuristic_type == "lb-random":
            wifi_ratio = self._lb_random_uniform()
        elif self.heuristic_type == "lb-wifi":
            wifi_ratio = 1.0
        elif self.heuristic_type == "lb-fiveg":
            wifi_ratio = 0.0
        elif self.heuristic_type == "lb-equal":
            wifi_ratio = 0.5
        else:
            raise RuntimeError(f"Unhandled heuristic_type: {self.heuristic_type}")

        action = float(np.clip(wifi_ratio, 0.0, 1.0))
        self.last_action = action
        return np.array([action], dtype=np.float32)

    def _ema(self, prev: Optional[float], current: float) -> float:
        # exponential moving average: x_{t+1} = alpha * x_t + (1 - alpha) * v
        if prev is None:
            return current
        return self.alpha * prev + (1.0 - self.alpha) * current

    def _lb_rtt_min(self, observation) -> float:
        self._smooth_wifi_rtt = self._ema(self._smooth_wifi_rtt, observation[0])
        self._smooth_fiveg_rtt = self._ema(self._smooth_fiveg_rtt, observation[1])
        if self._smooth_wifi_rtt < self._smooth_fiveg_rtt:
            return 1.0
        if self._smooth_wifi_rtt > self._smooth_fiveg_rtt:
            return 0.0
        return 0.5

    def _lb_plr_min(self, observation) -> float:
        self._smooth_wifi_plr = self._ema(self._smooth_wifi_plr, observation[2])
        self._smooth_fiveg_plr = self._ema(self._smooth_fiveg_plr, observation[3])
        if self._smooth_wifi_plr < self._smooth_fiveg_plr:
            return 1.0
        if self._smooth_wifi_plr > self._smooth_fiveg_plr:
            return 0.0
        return 0.5

    def _lb_utility(self, observation) -> float:
        actions = np.arange(0, 1.01, 0.01)
        rewards = np.empty_like(actions)
        self._smooth_wifi_rtt = self._ema(self._smooth_wifi_rtt, observation[0])
        self._smooth_fiveg_rtt = self._ema(self._smooth_fiveg_rtt, observation[1])
        self._smooth_wifi_plr = self._ema(self._smooth_wifi_plr, observation[2])
        self._smooth_fiveg_plr = self._ema(self._smooth_fiveg_plr, observation[3])
        smoothed_obs = [
            self._smooth_wifi_rtt, self._smooth_fiveg_rtt,
            self._smooth_wifi_plr, self._smooth_fiveg_plr
        ]
        for i, a in enumerate(actions):
            rewards[i], _ = RLEnv.get_reward_and_metrics(smoothed_obs, a, self.last_action, **self.reward_params)
        return actions[rewards.argmax()]

    def _lb_random_uniform(self) -> float:
        return np.random.uniform(0, 1)

    def shutdown(self):
        pass
