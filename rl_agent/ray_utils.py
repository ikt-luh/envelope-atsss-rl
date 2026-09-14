# shared ray setup for training agents (log noise control).

from __future__ import annotations

import logging
import os

import ray
import torch


def ensure_ray_initialized() -> None:
    if ray.is_initialized():
        return
    # ERROR hides Ray core INFO/WARNING (object store / shm hints, gputil, etc.).
    level = logging.ERROR
    if os.environ.get("RL_VERBOSE_RAY", "").lower() in ("1", "true", "yes"):
        level = logging.INFO
    kwargs = dict(ignore_reinit_error=True, log_to_driver=False, logging_level=level)
    if torch.cuda.is_available():
        kwargs["num_gpus"] = 1
    try:
        ray.init(**kwargs)
    except TypeError:
        kwargs.pop("logging_level", None)
        ray.init(**kwargs)
