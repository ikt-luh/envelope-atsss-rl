# structured scenario dimensions: traffic classes, RTT presets, loss, capacity, cross-traffic.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TrafficClass(str, Enum):
    CONTROL = "control"
    BULK = "bulk"
    BURSTY = "bursty"


class CapacityRatio(str, Enum):
    R1_1 = "1:1"
    R1_2 = "1:2"
    R1_5 = "1:5"


class CrossTrafficIfaceMode(str, Enum):
    WIFI = "wifi"
    FIVEG = "fiveg"
    RANDOM = "random"


@dataclass(frozen=True)
class ValueOrRange:
    value: Optional[float] = None
    range_from: Optional[float] = None
    range_to: Optional[float] = None

    def __post_init__(self):
        self.check()

    def check(self):
        assert (self.value is not None) or (self.range_from is not None and self.range_to is not None)
        if self.value is None:
            assert self.range_to >= self.range_from

    def __call__(self, *args, **kwds):
        if self.value is not None:
            return self.value
        return np.random.rand() * (self.range_to - self.range_from) + self.range_from


@dataclass
class WiFiFiveGPair:
    wifi: ValueOrRange
    fiveg: ValueOrRange

DEFAULT_LOSS_PAIRS: Tuple[WiFiFiveGPair, ...] = (
    WiFiFiveGPair(wifi=ValueOrRange(value=0.0), fiveg=ValueOrRange(value=0.0)),
    WiFiFiveGPair(wifi=ValueOrRange(value=0.1), fiveg=ValueOrRange(value=1.0)),
    WiFiFiveGPair(wifi=ValueOrRange(value=0.5), fiveg=ValueOrRange(value=3.0)),
)

DEFAULT_RTT_PAIRS: Tuple[WiFiFiveGPair, ...] = (
    WiFiFiveGPair(wifi=ValueOrRange(value=0.0), fiveg=ValueOrRange(value=0.0)),
    WiFiFiveGPair(wifi=ValueOrRange(value=30.0), fiveg=ValueOrRange(value=0.0)),
    WiFiFiveGPair(wifi=ValueOrRange(value=0.0), fiveg=ValueOrRange(value=30.0)),
)

DEFAULT_CROSS_TRAFFIC_LEVELS: Optional[Tuple[float, ...]] = tuple()  # (0.2, 0.6, 0.9)

DEFAULT_CAPACITY_RATIOS: Optional[Tuple[CapacityRatio, ...]] = tuple()  # (CapacityRatio.R1_1, CapacityRatio.R1_2, CapacityRatio.R1_5)

@dataclass
class ScenarioSettings:
    # loaded from YAML + env overrides.
    mpquic_fifo_input_path: str = os.getenv("TESTBED_UE_FIFO_INPUT_PATH", "/tmp/fifo_input")
    wifi_iface: str = os.getenv("PMF_UE_WIFI_IFACE", os.getenv("TESTBED_SCENARIO_WIFI_IFACE", ""))
    fiveg_iface: str = os.getenv("PMF_UE_5G_IFACE", os.getenv("TESTBED_SCENARIO_5G_IFACE", ""))
    ane_ip: str = os.getenv("SCENARIO_ANE_IP", os.getenv("TESTBED_SCENARIO_ANE_IP", ""))
    wifi_cap_mbps: float = 200.0
    fiveg_cap_mbps: float = 60.0
    cross_traffic_bottleneck_mbps: Optional[float] = None
    bulk_mbps: float = 10.0
    bursty_mbit_per_period: float = 10.0
    bursty_period_s: float = 5.0
    control_bandwidth_mbps: float = 0.5
    cross_traffic_port: int = 5201
    cross_traffic_iface: CrossTrafficIfaceMode = CrossTrafficIfaceMode.RANDOM
    # number of RL agent episodes per config
    episodes_per_config: int = 1
    dry_run: bool = False
    # RTT: three levels as extra delay (ms) on WiFi / 5G (from ping + design).
    rtt_pairs: Tuple[WiFiFiveGPair, ...] = DEFAULT_RTT_PAIRS
    loss_pairs: Tuple[WiFiFiveGPair, ...] = DEFAULT_LOSS_PAIRS
    cross_traffic_levels: Tuple[float, ...] = DEFAULT_CROSS_TRAFFIC_LEVELS
    capacity_ratios: Tuple[CapacityRatio, ...] = DEFAULT_CAPACITY_RATIOS
    # if set, used for iperf3 -B; otherwise resolved via `ip` on the UE.
    wifi_bind_ip: Optional[str] = None
    fiveg_bind_ip: Optional[str] = None


def wifi_fiveg_mbps_for_ratio(
    ratio: CapacityRatio,
    wifi_cap: float,
    fiveg_cap: float,
) -> Tuple[Optional[float], Optional[float]]:
    if ratio == CapacityRatio.R1_1:
        return (min(fiveg_cap, wifi_cap), None)
    if ratio == CapacityRatio.R1_2:
        return (2.0 * fiveg_cap, None)
    if ratio == CapacityRatio.R1_5:
        return (None, wifi_cap / 5.0)
    return (None, None)


def _parse_value_or_range(raw: Any) -> ValueOrRange:
    if isinstance(raw, (list, tuple)):
        assert len(raw) == 2, f"Provided value or range {raw} could not be parsed. Please provide either a value '4' or a range '[0, 20]'."
        return ValueOrRange(range_from=float(raw[0]), range_to=float(raw[1]))

    return ValueOrRange(value=float(raw))


def _parse_wifi_fiveg_pairs(raw: Any) -> Tuple[WiFiFiveGPair, ...]:
    if not raw:
        return None
    out: List[WiFiFiveGPair] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(WiFiFiveGPair(_parse_value_or_range(item["wifi"]), _parse_value_or_range(item["fiveg"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append(WiFiFiveGPair(_parse_value_or_range(item[0]), _parse_value_or_range(item[1])))
        else:
            raise ValueError(f"Could not parse WifiFiveGPair {raw}")
    return tuple(out)


def _parse_cross_iface(s: Any) -> CrossTrafficIfaceMode:
    if s is None:
        return CrossTrafficIfaceMode.RANDOM
    v = str(s).lower().strip()
    if v == "wifi":
        return CrossTrafficIfaceMode.WIFI
    if v == "fiveg":
        return CrossTrafficIfaceMode.FIVEG
    return CrossTrafficIfaceMode.RANDOM


def load_scenario_settings(data: Dict[str, Any]) -> ScenarioSettings:
    cb = data.get("cross_traffic_bottleneck_mbps")

    return ScenarioSettings(
        wifi_iface=str(data.get("wifi_iface", ScenarioSettings.wifi_iface)),
        fiveg_iface=str(data.get("fiveg_iface", ScenarioSettings.fiveg_iface)),
        ane_ip=str(data.get("ane_ip", ScenarioSettings.ane_ip)),
        wifi_cap_mbps=float(data.get("wifi_cap_mbps", 200)),
        fiveg_cap_mbps=float(data.get("fiveg_cap_mbps", 60)),
        cross_traffic_bottleneck_mbps=float(cb) if cb is not None else None,
        bursty_mbit_per_period=float(data.get("bursty_mbit_per_period", 10)),
        bursty_period_s=float(data.get("bursty_period_s", 5)),
        control_bandwidth_mbps=float(data.get("control_bandwidth_mbps", 0.5)),
        cross_traffic_port=int(data.get("cross_traffic_port", 5201)),
        cross_traffic_iface=_parse_cross_iface(data.get("cross_traffic_iface")),
        episodes_per_config=int(data.get("episodes_per_config", 1)),
        dry_run=bool(data.get("dry_run", False)),
        rtt_pairs=_parse_wifi_fiveg_pairs(data.get("rtt_pairs")),
        loss_pairs=_parse_wifi_fiveg_pairs(data.get("loss_pairs")),
        cross_traffic_levels=tuple(
            float(x) for x in data.get("cross_traffic_levels", DEFAULT_CROSS_TRAFFIC_LEVELS)
        ),
        capacity_ratios=tuple(
            CapacityRatio(str(x)) for x in data.get("capacity_ratios", ["1:1", "1:2", "1:5"])
        ) if data.get("capacity_ratios") else None,
        wifi_bind_ip=(str(data["wifi_bind_ip"]) if data.get("wifi_bind_ip") else None),
        fiveg_bind_ip=(str(data["fiveg_bind_ip"]) if data.get("fiveg_bind_ip") else None),
    )


def cross_traffic_bottleneck_mbps(settings: ScenarioSettings) -> float:
    if settings.cross_traffic_bottleneck_mbps is not None:
        return float(settings.cross_traffic_bottleneck_mbps)
    return min(settings.wifi_cap_mbps, settings.fiveg_cap_mbps)
