# PPO+LSTM Agent

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import signal
import ray
import time
import uuid
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from rl_agent.ray_utils import ensure_ray_initialized
from rl_agent.rl_env import LogRLEnvMetricsCallback, MPQUICLSTMEnv

from ray.rllib.policy.sample_batch import SampleBatch

logger = logging.getLogger(__name__)


# Global registry for pmf_client and session_id to avoid serialization issues
_env_registry = {}


def _get_results_dir(agent_name: str) -> str:
    override_dir = os.getenv("RL_TRAIN_RESULTS_DIR")
    if override_dir:
        results_dir = Path(override_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        return str(results_dir)
    base_dir = Path(__file__).resolve().parents[1]
    results_dir = base_dir / "results" / "training" / agent_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


def _create_lstm_env(env_config):
    key = env_config.get("env_key", "MPQUICLSTMEnv_default")
    registry_data = _env_registry.get(key, {})
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
    env = MPQUICLSTMEnv(
        decision_interval, pmf, sid,
        reward_params=reward_params,
        early_termination_reward_threshold=early_term_threshold,
        early_termination_window=early_term_window,
        max_steps=max_steps,
        scenario_manager=scenario_manager,
        episodes_per_config=episodes_per_config,
    )
    if records is not None:
        from rl_agent.recording_env import RecordingEnv
        env = RecordingEnv(env, records, start_time)
    return env


class MPQUICLSTMAgent:
    def __init__(self, sequence_len: int = 10, config: Optional[Dict[str, Any]] = None):
        self.sequence_len = sequence_len  # BPTT window for training (max_seq_len)
        self.config = config or self._default_config()
        self.agent = None
        self.env = None
        self._lstm_state: Optional[list] = None  # [h, c] carried across inference steps

    def _default_config(self) -> Dict[str, Any]:
        return {
            "disable_env_runner_and_connector_v2": True,
            "disable_rl_module_and_learner": True,
            "env": "MPQUICLSTMEnv",
            "framework": "torch",
            "num_workers": 0,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "lstm_cell_size": 256,
                "lstm_use_prev_action": False,
                "lstm_use_prev_reward": False,
                "max_seq_len": self.sequence_len,
            },
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 30,
            "lr": 5e-5,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "vf_clip_param": 0.2,
            "entropy_coeff": 0.01,
        }

    def initialize(
        self,
        decision_interval,
        pmf_client=None,
        session_id: str = "default",
        training_records=None,
        training_start_time=0.0,
        early_termination_reward_threshold: Optional[float] = None,
        early_termination_window: int = 50,
        reward_params: Optional[Dict[str, Any]] = None,
        max_steps: int = 1000,
        scenario_manager=None,
        episodes_per_config: int = 1,
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
        # store in global registry to avoid serialization issues
        env_key = f"MPQUICLSTMEnv_{uuid.uuid4().hex[:8]}"
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
        }

        # store key as instance variable for later use
        self._env_key = env_key

        register_env("MPQUICLSTMEnv", _create_lstm_env)

        ensure_ray_initialized()

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        config = PPOConfig.from_dict(self.config)
        config.model["use_lstm"] = True  # lstm agent must use lstm
        config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        config.environment(env_config={
            "env_key": self._env_key
        })
        config.callbacks(callbacks_class=LogRLEnvMetricsCallback)

        def _logger_creator(cfg):
            logdir = _get_results_dir("lstm")
            return UnifiedLogger(cfg, logdir, trial=None)

        if torch.cuda.is_available():
            config.resources(num_gpus=1)
            print("[train] Info: Using GPU")

        self.agent = PPO(config=config, logger_creator=_logger_creator)
        # HOTFIX: pop infos as RLLib training crashes for RNN with object type infos
        self.agent.get_policy().view_requirements.pop(SampleBatch.INFOS, None)

        self._lstm_state = None  # reset on re-initialize
        logger.info(f"PPO+LSTM RL Agent initialized (sequence_len={self.sequence_len})")

    def _get_initial_state(self) -> list:
        #return zero LSTM state [h, c] in the format RLlib expects.
        return self.agent.get_policy().get_initial_state()

    def reset_state(self) -> None:
        #reset LSTM hidden state to zeros. Call at episode boundaries.
        if self.agent is not None:
            self._lstm_state = self._get_initial_state()

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        if self._lstm_state is None:
            self._lstm_state = self._get_initial_state()
        action, self._lstm_state, _ = self.agent.compute_single_action(
            observation, state=self._lstm_state, full_fetch=True
        )
        return np.array([action], dtype=np.float32)

    def train(self, num_iterations: int = 100, checkpoint_freq: int = 10):
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        logger.info(f"Starting PPO+LSTM training for {num_iterations} iterations")

        for i in range(num_iterations):
            result = self.agent.train()

            if (i + 1) % checkpoint_freq == 0:
                checkpoint_path = self.agent.save()
                logger.info(f"Iteration {i+1}, checkpoint saved: {checkpoint_path}")
                logger.info(f"Mean reward: {result['episode_reward_mean']:.2f}")

        logger.info("PPO+LSTM training completed")

    def load_checkpoint(self, checkpoint_path: str):
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        self.agent.restore(checkpoint_path)
        self._lstm_state = None  # reset state after loading new weights
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def save_checkpoint(self, path: str) -> str:
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        checkpoint_path = self.agent.save(path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def shutdown(self):
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

        logger.info("PPO+LSTM RL Agent shutdown")
