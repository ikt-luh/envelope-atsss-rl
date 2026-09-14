# random scenario sampling for RL training (per-class 3x3 grid).
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from scenarios.config_space import (
    CapacityRatio,
    CrossTrafficIfaceMode,
    ScenarioSettings,
    TrafficClass,
    cross_traffic_bottleneck_mbps,
    wifi_fiveg_mbps_for_ratio,
)


@dataclass
class ScenarioConfig:

    traffic_class: TrafficClass
    rtt_preset_idx: int = 0
    loss_pair_idx: int = 0
    capacity_ratio_idx: int = 0
    cross_traffic_level_idx: int = 0
    wifi_extra_delay_ms: float = 0.0
    fiveg_extra_delay_ms: float = 0.0
    wifi_loss_pct: float = 0.0
    fiveg_loss_pct: float = 0.0
    wifi_rate_limit_mbps: Optional[float] = None
    fiveg_rate_limit_mbps: Optional[float] = None

    cross_traffic_mbps: float = 0.0
    cross_traffic_on_wifi: bool = True

    user_data_mode: str = "continuous"
    user_data_mbps: float = 0.5
    bursty_mbit_per_period: float = 10.0
    bursty_period_s: float = 5.0

    label: str = ""

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "scenario_label": self.label,
            "traffic_class": self.traffic_class.value,
            "rtt_preset_idx": self.rtt_preset_idx,
            "loss_pair_idx": self.loss_pair_idx,
            "capacity_ratio_idx": self.capacity_ratio_idx,
            "cross_traffic_level_idx": self.cross_traffic_level_idx,
            "wifi_extra_delay_ms": self.wifi_extra_delay_ms,
            "fiveg_extra_delay_ms": self.fiveg_extra_delay_ms,
            "wifi_loss_pct": self.wifi_loss_pct,
            "fiveg_loss_pct": self.fiveg_loss_pct,
            "wifi_rate_limit_mbps": self.wifi_rate_limit_mbps,
            "fiveg_rate_limit_mbps": self.fiveg_rate_limit_mbps,
            "cross_traffic_mbps": self.cross_traffic_mbps,
            "cross_traffic_on_wifi": self.cross_traffic_on_wifi,
            "user_data_mode": self.user_data_mode,
            "user_data_mbps": self.user_data_mbps,
            "bursty_mbit_per_period": self.bursty_mbit_per_period,
            "bursty_period_s": self.bursty_period_s,
        }


def _resolve_cross_iface(settings: ScenarioSettings, rng: random.Random) -> bool:
    #return True if cross-traffic should be on WiFi (else 5G).
    mode = settings.cross_traffic_iface
    if mode == CrossTrafficIfaceMode.WIFI:
        return True
    if mode == CrossTrafficIfaceMode.FIVEG:
        return False
    return rng.random() < 0.5

def sample_limits(settings: ScenarioSettings, rng: random.Random):
    if settings.capacity_ratios is None:
        return None, None, None
    cap_i = rng.randrange(len(settings.capacity_ratios))
    ratio = settings.capacity_ratios[cap_i]
    w_lim, f_lim = wifi_fiveg_mbps_for_ratio(
        ratio, settings.wifi_cap_mbps, settings.fiveg_cap_mbps
    )
    return cap_i, w_lim, f_lim


def sample_random_config(settings: ScenarioSettings, rng: Optional[random.Random] = None) -> ScenarioConfig:
    rng = rng or random.Random()
    traffic_class = rng.choice([TrafficClass.CONTROL, TrafficClass.BULK, TrafficClass.BURSTY])

    # NOTE we are temporarily only considering the configuration space of traffic class x rtt x loss
    rtt_i = rng.randrange(len(settings.rtt_pairs))
    loss_i = rng.randrange(len(settings.loss_pairs))

    user_data_mode = None
    user_data_mbps = None
    bursty_mbit_per_period = None
    bursty_period_s = None

    if traffic_class == TrafficClass.CONTROL:
        user_data_mode = "continuous"
        user_data_mbps = settings.control_bandwidth_mbps
    elif traffic_class == TrafficClass.BULK:
        user_data_mode = "continuous"
        user_data_mbps = settings.bulk_mbps
    elif traffic_class == TrafficClass.BURSTY:
        user_data_mode = "burst"
        user_data_mbps = 0.0
        bursty_mbit_per_period = settings.bursty_mbit_per_period
        bursty_period_s = settings.bursty_period_s
    else:
        raise NotImplementedError()

    rtt = settings.rtt_pairs[rtt_i]
    loss = settings.loss_pairs[loss_i]

    cfg = ScenarioConfig(
        traffic_class=traffic_class,
        rtt_preset_idx=rtt_i,
        loss_pair_idx=loss_i,
        wifi_extra_delay_ms=rtt.wifi(),
        fiveg_extra_delay_ms=rtt.fiveg(),
        wifi_loss_pct=loss.wifi(),
        fiveg_loss_pct=loss.fiveg(),
        user_data_mode=user_data_mode,
        user_data_mbps=user_data_mbps,
        bursty_mbit_per_period=bursty_mbit_per_period,
        bursty_period_s=bursty_period_s,
        label=f"{traffic_class.value}_rtt{rtt_i}_loss{loss_i}",
    )
    return cfg

    bottleneck = cross_traffic_bottleneck_mbps(settings)
    #sample one scenario uniformly from the experiment grid (3 classes, each 3x3 sub-grid).
    if t == TrafficClass.CONTROL:
        rtt_i = rng.randrange(len(settings.rtt_pairs))
        loss_i = rng.randrange(len(settings.loss_pairs))
        rtt_preset = settings.rtt_pairs[rtt_i]
        w_loss, f_loss = _apply_loss_pair(settings, loss_i, wifi_is_good)
        cfg = ScenarioConfig(
            traffic_class=t,
            rtt_preset_idx=rtt_i,
            loss_pair_idx=loss_i,
            wifi_is_good_path=wifi_is_good,
            wifi_extra_delay_ms=float(rtt_preset.get("wifi_extra_ms", 0.0)),
            fiveg_extra_delay_ms=float(rtt_preset.get("fiveg_extra_ms", 0.0)),
            wifi_loss_pct=w_loss,
            fiveg_loss_pct=f_loss,
            user_data_mode="continuous",
            user_data_mbps=settings.control_bandwidth_mbps,
            label=f"control_rtt{rtt_i}_loss{loss_i}_{'wg' if wifi_is_good else 'fg'}",
        )
        return cfg

    if t == TrafficClass.BULK:
        cross_i = rng.randrange(len(settings.cross_traffic_levels))
        cap_i, w_lim, f_lim = sample_limits(settings, rng)
        cross_on_wifi = _resolve_cross_iface(settings, rng)
        level = settings.cross_traffic_levels[cross_i]
        cross_mbps = max(0.1, level * bottleneck)
        bulk_mbps = max(0.1, settings.bulk_measured_max_mbps * settings.bulk_fraction)
        cfg = ScenarioConfig(
            traffic_class=t,
            capacity_ratio_idx=cap_i,
            cross_traffic_level_idx=cross_i,
            wifi_rate_limit_mbps=w_lim,
            fiveg_rate_limit_mbps=f_lim,
            cross_traffic_mbps=cross_mbps,
            cross_traffic_on_wifi=cross_on_wifi,
            user_data_mode="continuous",
            user_data_mbps=bulk_mbps,
            label=f"bulk{f'_cap{cap_i}' if cap_i is not None else ''}_x{cross_i}_{'wifi' if cross_on_wifi else '5g'}",
        )
        return cfg

    # bursty: 3 capacity x 3 RTT
    rtt_i = rng.randrange(len(settings.rtt_pairs))
    cap_i, w_lim, f_lim = sample_limits(settings, rng)
    rtt_preset = settings.rtt_pairs[rtt_i]
    cfg = ScenarioConfig(
        traffic_class=t,
        rtt_preset_idx=rtt_i,
        capacity_ratio_idx=cap_i,
        wifi_extra_delay_ms=float(rtt_preset.get("wifi_extra_ms", 0.0)),
        fiveg_extra_delay_ms=float(rtt_preset.get("fiveg_extra_ms", 0.0)),
        wifi_rate_limit_mbps=w_lim,
        fiveg_rate_limit_mbps=f_lim,
        user_data_mode="burst",
        user_data_mbps=0.0,
        bursty_mbit_per_period=settings.bursty_mbit_per_period,
        bursty_period_s=settings.bursty_period_s,
        label=f"bursty{f'_cap{cap_i}' if cap_i is not None else ''}_rtt{rtt_i}",
    )
    return cfg
