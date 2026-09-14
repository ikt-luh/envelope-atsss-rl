# PPO Agent

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import ray
import signal
import time
import uuid
import torch
from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from rl_agent.lstm_agent import _get_results_dir
from rl_agent.ray_utils import ensure_ray_initialized
from rl_agent.rl_env import MPQUICPPOEnv


logger = logging.getLogger(__name__)

# Global registry for pmf_client and session_id to avoid serialization issues
_env_registry = {}


def _create_ppo_env(env_config):
    key = env_config.get("env_key", "MPQUICPPOEnv_default")
    registry_data = _env_registry.get(key, {})

    # print(f"ENV CREATION: Registry content {_env_registry}")
    # print(f"ENV CREATION: Got config {env_config}")
    # print(f"ENV CREATION: Got registry data {registry_data}")

    decision_interval = registry_data.get("decision_interval")
    pmf = registry_data.get("pmf_client")
    sid = registry_data.get("session_id", "default")
    records = registry_data.get("training_records")
    start_time = registry_data.get("training_start_time", 0.0)
    early_term_threshold = registry_data.get("early_termination_reward_threshold")
    early_term_window = registry_data.get("early_termination_window", 50)
    reward_params = registry_data.get("reward_params", {})
    max_steps = registry_data.get("max_steps", 1000)
    scenario_manager = registry_data.get("scenario_manager")
    episodes_per_config = registry_data.get("episodes_per_config", 1)
    aue_client = registry_data.get("aue_client")

    # print(f"ENV CREATION: PMF exists: {pmf is not None}")

    window_size = registry_data.get("window_size", 10)
    env = MPQUICPPOEnv(
        decision_interval, pmf, sid,
        reward_params=reward_params,
        early_termination_reward_threshold=early_term_threshold,
        early_termination_window=early_term_window,
        max_steps=max_steps,
        scenario_manager=scenario_manager,
        episodes_per_config=episodes_per_config,
        window_size=window_size,
        aue_client=aue_client,
    )
    if records is not None:
        from rl_agent.recording_env import RecordingEnv
        env = RecordingEnv(env, records, start_time)
    return env


class MPQUICSACAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None, window_size: int = 10):
        self.config = config or self._default_config()
        self.agent = None
        self.env = None
        self._window_size = window_size
        self._obs_buffer: list = []  # rolling window for inference stacking

    def _default_config(self) -> Dict[str, Any]:
        return {
            "disable_env_runner_and_connector_v2": True,
            "disable_rl_module_and_learner": True,
            "env": "MPQUICPPOEnv",
            "framework": "torch",
            "num_workers": 0,
            "train_batch_size_per_learner": 256,
            "train_batch_size": 256,
            "num_steps_sampled_before_learning_starts": 200 * 10, # TODO: adjust based on episode length
            "min_time_s_per_iteration": 0,  # iterations are only based on samples, not seconds
            "min_sample_timesteps_per_iteration": 200,  # always one entire episode TODO adjust based on episode length
            "actor_lr": 5e-5, # TODO: not sure if lr is actually applied
            "gamma": 0.9,
        }

    def initialize(
        self,
        decision_interval,
        pmf_client,
        session_id: str = "default",
        training_records=None,
        training_start_time=0.0,
        early_termination_reward_threshold: Optional[float] = None,
        early_termination_window: int = 50,
        reward_params: Optional[Dict[str, Any]] = None,
        max_steps: int = 1000,
        scenario_manager=None,
        episodes_per_config: int = 1,
        window_size: int = 10,
        aue_client=None,
        **kwargs,
    ):
        # validate num_workers: PMF client and scenario manager are stored in a
        # process-local registry and cannot be serialized to Ray workers.
        num_workers = self.config.get("num_workers", 0)
        if num_workers > 0:
            raise ValueError(
                f"num_workers={num_workers} is not supported. PMF client and "
                "scenario manager use process-local state that cannot be "
                "serialized across Ray workers. Set num_workers: 0."
            )

        # store for later use
        self.pmf_client = pmf_client
        self.session_id = session_id
        self.scenario_manager = scenario_manager
        self.window_size = window_size

        # store in global registry to avoid serialization issues
        env_key = f"MPQUICPPOEnv_{uuid.uuid4().hex[:8]}"
        _env_registry[env_key] = {
            "decision_interval": decision_interval,
            "pmf_client": pmf_client,
            "session_id": session_id,
            "training_records": training_records,
            "training_start_time": training_start_time or time.time(),
            "early_termination_reward_threshold": early_termination_reward_threshold,
            "early_termination_window": early_termination_window,
            "reward_params": reward_params or {},
            "max_steps": max_steps,
            "scenario_manager": scenario_manager,
            "episodes_per_config": episodes_per_config,
            "window_size": window_size,
            "aue_client": aue_client,
        }

        print(f"[train] DEBUG: Creating environment with pmf {pmf_client is not None}")

        # store key as instance variable for later use
        self._env_key = env_key

        register_env("MPQUICPPOEnv", _create_ppo_env)

        ensure_ray_initialized()

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print("SAC Config", self.config)
        config = SACConfig.from_dict(self.config)  # self.config)
        print("[train] SAC: LOADING DEFAULT CONFIG")
        config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        config.training(replay_buffer_config={
            'type': 'MultiAgentReplayBuffer',
            "capacity": int(10_000),
        })
        config.environment(env_config={
            "env_key": self._env_key
        })

        def _logger_creator(cfg):
            logdir = _get_results_dir("sac")
            return UnifiedLogger(cfg, logdir, trial=None)

        if torch.cuda.is_available():
            config.resources(num_gpus=1)
            print("[train] Info: Using GPU")

        self.agent = SAC(config=config, logger_creator=_logger_creator)

        logger.info("SAC RL Agent initialized")

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        # maintain rolling window matching training env (window_size stacked obs)
        self._obs_buffer.append(observation.copy())
        if len(self._obs_buffer) > self._window_size:
            self._obs_buffer.pop(0)
        # pad with copies of the first obs if buffer not full yet (same as env reset)
        while len(self._obs_buffer) < self._window_size:
            self._obs_buffer.insert(0, self._obs_buffer[0].copy())
        stacked = np.concatenate(self._obs_buffer[-self._window_size:])

        action = self.agent.compute_single_action(stacked)
        return np.array(action, dtype=np.float32)

    def train(self, num_iterations: int = 100, checkpoint_freq: int = 10):
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        logger.info(f"Starting SAC training for {num_iterations} iterations")

        for i in range(num_iterations):
            result = self.agent.train()

            if (i + 1) % checkpoint_freq == 0:
                checkpoint_path = self.agent.save()
                logger.info(f"Iteration {i+1}, checkpoint saved: {checkpoint_path}")
                logger.info(f"Mean reward: {result['episode_reward_mean']:.2f}")

        logger.info("SAC training completed")

    def load_checkpoint(self, checkpoint_path: str):
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        self.agent.restore(checkpoint_path)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def save_checkpoint(self, path: str) -> str:
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        checkpoint_path = self.agent.save(path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def shutdown(self):
        """Shutdown the agent and cleanup."""
        if self.agent:
            self.agent.stop()
        if ray.is_initialized():
            ray.shutdown()
        sm = getattr(self, "scenario_manager", None)
        if sm is not None and hasattr(sm, "shutdown"):
            try:
                sm.shutdown()
            except Exception:
                pass

        logger.info("SAC RL Agent shutdown")
