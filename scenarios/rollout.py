# deterministic enumeration: 27 experiment-grid scenarios (3 traffic classes x 9 each)
# plus 3 baseline configs (one per class, no artificial impairment).

from __future__ import annotations

from typing import Iterator, List

from scenarios.config_space import (
    ScenarioSettings,
    TrafficClass,
    cross_traffic_bottleneck_mbps,
    wifi_fiveg_mbps_for_ratio,
)
from scenarios.sampler import ScenarioConfig


def baseline_configs(settings: ScenarioSettings) -> Iterator[ScenarioConfig]:
    #one scenario per traffic class: natural path, no tc delay/loss/caps, no cross-traffic.
    preset0 = settings.rtt_pairs[0]
    lp0 = settings.loss_pairs[0]
    w_loss, f_loss = lp0.wifi, lp0.fiveg
    bulk_mbps = settings.bulk_mbps
    yield ScenarioConfig(
        traffic_class=TrafficClass.CONTROL,
        rtt_preset_idx=0,
        loss_pair_idx=0,
        wifi_extra_delay_ms=float(preset0.get("wifi_extra_ms", 0.0)),
        fiveg_extra_delay_ms=float(preset0.get("fiveg_extra_ms", 0.0)),
        wifi_loss_pct=w_loss,
        fiveg_loss_pct=f_loss,
        user_data_mode="continuous",
        user_data_mbps=settings.control_bandwidth_mbps,
        label="control_baseline",
    )
    yield ScenarioConfig(
        traffic_class=TrafficClass.BULK,
        cross_traffic_level_idx=0,
        wifi_rate_limit_mbps=None,
        fiveg_rate_limit_mbps=None,
        cross_traffic_mbps=0.0,
        cross_traffic_on_wifi=True,
        user_data_mode="continuous",
        user_data_mbps=bulk_mbps,
        label="bulk_baseline",
    )
    yield ScenarioConfig(
        traffic_class=TrafficClass.BURSTY,
        rtt_preset_idx=0,
        wifi_extra_delay_ms=float(preset0.get("wifi_extra_ms", 0.0)),
        fiveg_extra_delay_ms=float(preset0.get("fiveg_extra_ms", 0.0)),
        wifi_rate_limit_mbps=None,
        fiveg_rate_limit_mbps=None,
        user_data_mode="burst",
        user_data_mbps=0.0,
        bursty_mbit_per_period=settings.bursty_mbit_per_period,
        bursty_period_s=settings.bursty_period_s,
        label="bursty_baseline",
    )


def iterate_experiment_configs(settings: ScenarioSettings) -> Iterator[ScenarioConfig]:
    yield from _traffic_class_rtt_loss_configs(settings)
    # yield from _control_configs(settings)
    # yield from _bulk_configs(settings)
    # yield from _bursty_configs(settings)


def experiment_config_list(settings: ScenarioSettings) -> List[ScenarioConfig]:
    return list(iterate_experiment_configs(settings))
    # baseline configs are already included in config
    # return list(baseline_configs(settings)) + list(iterate_experiment_configs(settings))


def _traffic_class_rtt_loss_configs(settings: ScenarioSettings) -> Iterator[ScenarioConfig]:
    for t in [TrafficClass.CONTROL, TrafficClass.BULK, TrafficClass.BURSTY]:

        user_data_mode = None
        user_data_mbps = None
        bursty_mbit_per_period = None
        bursty_period_s = None

        if t == TrafficClass.CONTROL:
            user_data_mode = "continuous"
            user_data_mbps = settings.control_bandwidth_mbps
        elif t == TrafficClass.BULK:
            user_data_mode = "continuous"
            user_data_mbps = settings.bulk_mbps
        elif t == TrafficClass.BURSTY:
            user_data_mode = "burst"
            user_data_mbps = 0.0
            bursty_mbit_per_period = settings.bursty_mbit_per_period
            bursty_period_s = settings.bursty_period_s
        else:
            raise NotImplementedError()
        
        for r in range(len(settings.rtt_pairs)):
            for l in range(len(settings.loss_pairs)):
                rtt = settings.rtt_pairs[r]
                loss = settings.loss_pairs[l]
                yield ScenarioConfig(
                    traffic_class=t,
                    rtt_preset_idx=r,
                    loss_pair_idx=l,
                    wifi_extra_delay_ms=rtt.wifi(),
                    fiveg_extra_delay_ms=rtt.fiveg(),
                    wifi_loss_pct=loss.wifi(),
                    fiveg_loss_pct=loss.fiveg(),
                    user_data_mode=user_data_mode,
                    user_data_mbps=user_data_mbps,
                    bursty_mbit_per_period=bursty_mbit_per_period,
                    bursty_period_s=bursty_period_s,
                    label=f"{t.value}_rtt{r}_loss{l}",
                )

def _control_configs(settings: ScenarioSettings) -> Iterator[ScenarioConfig]:
    for r in range(len(settings.rtt_pairs)):
        for l in range(len(settings.loss_pairs)):
            preset = settings.rtt_pairs[r]
            lp = settings.loss_pairs[l]
            w_loss, f_loss = lp.wifi, lp.fiveg
            yield ScenarioConfig(
                traffic_class=TrafficClass.CONTROL,
                rtt_preset_idx=r,
                loss_pair_idx=l,
                wifi_extra_delay_ms=float(preset.get("wifi_extra_ms", 0.0)),
                fiveg_extra_delay_ms=float(preset.get("fiveg_extra_ms", 0.0)),
                wifi_loss_pct=w_loss,
                fiveg_loss_pct=f_loss,
                user_data_mode="continuous",
                user_data_mbps=settings.control_bandwidth_mbps,
                label=f"control_rtt{r}_loss{l}",
            )


def _bulk_configs(settings: ScenarioSettings) -> Iterator[ScenarioConfig]:
    from scenarios.config_space import CrossTrafficIfaceMode

    bottleneck = cross_traffic_bottleneck_mbps(settings)
    bulk_mbps = max(0.1, settings.bulk_measured_max_mbps * settings.bulk_fraction)

    # determine cross-traffic interface placement for experiment mode.
    # if set to "random", experiment mode defaults to WiFi for reproducibility
    # (training sampler handles randomization separately).
    ct_mode = settings.cross_traffic_iface
    if ct_mode == CrossTrafficIfaceMode.WIFI:
        on_wifi = True
        suffix = "wifi"
    elif ct_mode == CrossTrafficIfaceMode.FIVEG:
        on_wifi = False
        suffix = "5g"
    else:
        on_wifi = True  
        suffix = "wifi"

    capacity_rations = settings.capacity_ratios or [None]
    for ci in range(len(capacity_rations)):
        for xi in range(len(settings.cross_traffic_levels)):
            ratio = capacity_rations[ci]
            if ratio is None:
                w_lim, f_lim = None, None
            else:
                w_lim, f_lim = wifi_fiveg_mbps_for_ratio(
                    ratio, settings.wifi_cap_mbps, settings.fiveg_cap_mbps
                )
            level = settings.cross_traffic_levels[xi]
            cross_mbps = max(0.1, level * bottleneck)
            yield ScenarioConfig(
                traffic_class=TrafficClass.BULK,
                cross_traffic_level_idx=xi,
                wifi_rate_limit_mbps=w_lim,
                fiveg_rate_limit_mbps=f_lim,
                cross_traffic_mbps=cross_mbps,
                cross_traffic_on_wifi=on_wifi,
                user_data_mode="continuous",
                user_data_mbps=bulk_mbps,
                label=f"bulk{f'_cap{ratio.value}' if ratio is not None else ''}_x{xi}_{suffix}",
            )


def _bursty_configs(settings: ScenarioSettings) -> Iterator[ScenarioConfig]:
    capacity_rations = settings.capacity_ratios or [None]
    for ci in range(len(capacity_rations)):
        for ri in range(len(settings.rtt_pairs)):
            ratio = capacity_rations[ci]
            if ratio is None:
                w_lim, f_lim = None, None
            else:
                w_lim, f_lim = wifi_fiveg_mbps_for_ratio(
                    ratio, settings.wifi_cap_mbps, settings.fiveg_cap_mbps
                )
            preset = settings.rtt_pairs[ri]
            yield ScenarioConfig(
                traffic_class=TrafficClass.BURSTY,
                rtt_preset_idx=ri,
                wifi_extra_delay_ms=float(preset.get("wifi_extra_ms", 0.0)),
                fiveg_extra_delay_ms=float(preset.get("fiveg_extra_ms", 0.0)),
                wifi_rate_limit_mbps=w_lim,
                fiveg_rate_limit_mbps=f_lim,
                user_data_mode="burst",
                user_data_mbps=0.0,
                bursty_mbit_per_period=settings.bursty_mbit_per_period,
                bursty_period_s=settings.bursty_period_s,
                label=f"bursty{f'_cap{ratio.value}' if ratio is not None else ''}_rtt{ri}",
            )
