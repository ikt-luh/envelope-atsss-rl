# centralized config for the rl agent system.

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_optional_float(key: str, default: float) -> Optional[float]:
    val = os.getenv(key, str(default))
    if str.lower(val) == "none":
        return None
    return float(val)

def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_optional_int(key: str) -> Optional[int]:
    v = os.getenv(key, "").strip()
    if not v:
        return None
    return int(v)


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# rl agent

AGENT_TYPE: str = (_env("AGENT_TYPE", "ppo") or "ppo").strip().lower()
REWARD_TYPE: str = _env("REWARD_TYPE", "logarithmic")
REWARD_R_MAX: float = _env_float("REWARD_R_MAX", 1.0)
REWARD_EPSILON: float = _env_float("REWARD_EPSILON", 1e-6)
REWARD_ALPHA: float = _env_float("REWARD_ALPHA", 25)
SESSION_ID: str = _env("SESSION_ID", "cam-session-001")
CHECKPOINT_PATH: str = _env("CHECKPOINT_PATH", "")
SEQUENCE_LEN: int = _env_int("SEQUENCE_LEN", 10)
# PPO inference: must match MPQUICPPOEnv window (stacked obs dim = OBS_DIM * window).
PPO_OBS_WINDOW_SIZE: int = _env_int("PPO_OBS_WINDOW_SIZE", 10)

# heuristic agent (when AGENT_TYPE=heuristic)
HEURISTIC_TYPE: str = _env("HEURISTIC_TYPE", "lb-rtt-min")
HEURISTIC_ALPHA: float = _env_float("HEURISTIC_ALPHA", 0.95)

# pmf

PMF_UE_URL: str = _env("PMF_UE_URL", "")
PMF_UPF_URL: str = _env("PMF_UPF_URL", "")

PMF_UPF_IP: str = _env("PMF_UPF_IP", "")
PMF_UE_WIFI_IP: str = _env("PMF_UE_WIFI_IP", "")
PMF_UE_5G_IP: str = _env("PMF_UE_5G_IP", "")

_PMF_UE_WIFI_IFACE_DEFAULT: str = os.getenv("TESTBED_SCENARIO_WIFI_IFACE", "")
_PMF_UE_5G_IFACE_DEFAULT: str = os.getenv("TESTBED_SCENARIO_5G_IFACE", "")
PMF_UE_WIFI_IFACE: str = _env("PMF_UE_WIFI_IFACE", _PMF_UE_WIFI_IFACE_DEFAULT)
PMF_UE_5G_IFACE: str = _env("PMF_UE_5G_IFACE", _PMF_UE_5G_IFACE_DEFAULT)

PMF_USE_UPLINK: bool = _env_bool("PMF_USE_UPLINK", True)
USE_PMF_POLLING: bool = _env_bool("USE_PMF_POLLING", True)
PMF_POLL_INTERVAL: float = _env_float("PMF_POLL_INTERVAL", 0.1)
PMF_ASYNCHRONOUS: bool = _env_bool("PMF_ASYNCHRONOUS", False)

# pmf measurement parameters
PMF_PROBE_COUNT: int = _env_int("PMF_PROBE_COUNT", 5)
PMF_PROBE_TIMEOUT_MS: int = _env_int("PMF_PROBE_TIMEOUT_MS", 1000)
PMF_REQUEST_TIMEOUT: float = _env_float("PMF_REQUEST_TIMEOUT", 5.0)
PMF_JOB_TIMEOUT: float = _env_float("PMF_JOB_TIMEOUT", 30.0)

# cache ttl for pmf metrics (seconds). cached metrics older than this
# are considered stale and will not be served to the RL agent.
PMF_CACHE_MAX_AGE: float = _env_float("PMF_CACHE_MAX_AGE", 30.0)

# aue / decision sender
# the decision sender posts ratio updates to the aue

AUE_URL: str = _env("AUE_URL", "")
AUE_TIMEOUT: float = _env_float("AUE_TIMEOUT", 5.0)
DECISION_INTERVAL: float = _env_optional_float("DECISION_INTERVAL", 1.0)

# logging

LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")

# network scenarios
_SCENARIO_ANE_IP_DEFAULT: str = os.getenv("TESTBED_SCENARIO_ANE_IP", "")
SCENARIO_ANE_IP: str = os.getenv("SCENARIO_ANE_IP", _SCENARIO_ANE_IP_DEFAULT)
SCENARIO_DRY_RUN: bool = _env_bool("SCENARIO_DRY_RUN", False)
SCENARIO_WIFI_BIND_IP: str = _env("SCENARIO_WIFI_BIND_IP", "")
SCENARIO_FIVEG_BIND_IP: str = _env("SCENARIO_FIVEG_BIND_IP", "")

# iperf3 ports for scenarios: same env vars as testbed.sh (cmd_scenarios ANE servers).
TESTBED_SCENARIO_IPERF_CROSS_PORT: Optional[int] = _env_optional_int("TESTBED_SCENARIO_IPERF_CROSS_PORT")


def scenario_iperf_port_overrides() -> Dict[str, Any]:
    o: Dict[str, Any] = {}
    if TESTBED_SCENARIO_IPERF_CROSS_PORT is not None:
        o["cross_traffic_port"] = TESTBED_SCENARIO_IPERF_CROSS_PORT
    return o
