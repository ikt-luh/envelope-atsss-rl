# pmf client

import time
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging
from dataclasses import dataclass

if TYPE_CHECKING:
    from .polling_service import PMFPollingService

logger = logging.getLogger(__name__)

import numpy as np

@dataclass
class PMFMetrics:
    wifi_rtt_ms: float
    wifi_valid: bool
    fiveg_rtt_ms: float
    fiveg_valid: bool
    wifi_plr: float
    fiveg_plr: float
    timestamp: str
    session_id: str
    source: str
    wifi_rtt_min_ms: float = 0.0
    wifi_rtt_max_ms: float = 0.0
    wifi_rtt_std_ms: float = 0.0
    fiveg_rtt_min_ms: float = 0.0
    fiveg_rtt_max_ms: float = 0.0
    fiveg_rtt_std_ms: float = 0.0
    observation_age_s: float = 0.0


def random_uniform_range(ranges, p):
    range_idx = np.random.choice(np.arange(len(ranges)), p=p)
    range = ranges[range_idx]
    return np.random.uniform(range[0], range[1])


class PMFClientManager:

    def __init__(self,
                 dry_run: bool,
                 pmf_ue_url: str,
                 pmf_upf_url: str,
                 use_polling: bool = True,
                 poll_interval: float = 2.0,
                 job_timeout: float = 30.0,
                 request_timeout: float = 5.0,
                 probe_count: int = 10,
                 probe_timeout_ms: int = 1000,
                 cache_max_age: float = 30.0,
                 upf_ip: str = "",
                 ue_wifi_ip: str = "",
                 ue_5g_ip: str = "",
                 ue_wifi_iface: str = "",
                 ue_5g_iface: str = "",
                 use_uplink: bool = True,
                 asynchronous: bool = True,
                 inter_job_wait_time_s: Optional[int] = None):

        self.dry_run = dry_run
        # TODO: Dynamic run should be replaced and controlled by scenarios applier
        self.dry_run_dynamic = True
        if self.dry_run:
            self.dry_run_sample_random_conditions()
        self.probe_count = probe_count
        self.use_polling = use_polling
        self.polling_service: Optional['PMFPollingService'] = None
        self.asynchronous = asynchronous

        # TODO: As polling is the only implemented option, it could be better to merge this code with PMFPollingService
        if not self.dry_run and use_polling:
            from .polling_service import PMFPollingService
            self.polling_service = PMFPollingService(
                pmf_ue_url=pmf_ue_url,
                pmf_upf_url=pmf_upf_url,
                poll_interval=poll_interval,
                job_timeout=job_timeout,
                request_timeout=request_timeout,
                probe_count=probe_count,
                probe_timeout_ms=probe_timeout_ms,
                cache_max_age=cache_max_age,
                upf_ip=upf_ip,
                ue_wifi_ip=ue_wifi_ip,
                ue_5g_ip=ue_5g_ip,
                ue_wifi_iface=ue_wifi_iface,
                ue_5g_iface=ue_5g_iface,
                use_uplink=use_uplink,
                asynchronous=asynchronous,
                inter_job_wait_time_s=inter_job_wait_time_s
            )
            self.polling_service.start()
            logger.info("PMF Polling Service enabled")

    def get_dry_run_metrics_mean(self):
        if self.dry_run_dynamic:
            self.dr_wifi_rtt_state += (
                0.02 * (self.dr_wifi_rtt_base - self.dr_wifi_rtt_state)
                + np.random.normal(0, 0.4)
            )
            self.dr_wifi_rtt_state = max(self.dr_wifi_rtt_state, 0)
            self.dr_fiveg_rtt_state += (
                0.02 * (self.dr_fiveg_rtt_base - self.dr_fiveg_rtt_state)
                + np.random.normal(0, 0.4)
            )
            self.dr_fiveg_rtt_state = max(self.dr_fiveg_rtt_state, 0)
            self.dr_wifi_plr_state += (
                0.2 * (self.dr_wifi_plr_base - self.dr_wifi_plr_state)
                + np.random.normal(0, 0.001)
            )
            self.dr_wifi_plr_state = np.clip(self.dr_wifi_plr_state, 0, 1)
            self.dr_fiveg_plr_state += (
                0.2 * (self.dr_fiveg_plr_base - self.dr_fiveg_plr_state)
                + np.random.normal(0, 0.001)
            )
            self.dr_fiveg_plr_state = np.clip(self.dr_fiveg_plr_state, 0, 1)

            return self.dr_wifi_rtt_state, self.dr_fiveg_rtt_state, self.dr_wifi_plr_state, self.dr_fiveg_plr_state
        else:
            wifi_rtt_mean = 15
            fiveg_rtt_mean = 25
            wifi_plr_mean = 0.03
            fiveg_plr_mean = 0.06

            return wifi_rtt_mean, fiveg_rtt_mean, wifi_plr_mean, fiveg_plr_mean

    def dry_run_set_wifi(self, delay_ms, packet_loss):
        self.dr_wifi_rtt_base = delay_ms
        self.dr_wifi_rtt_state = self.dr_wifi_rtt_base
        self.dr_wifi_plr_base = packet_loss
        self.dr_wifi_plr_state = self.dr_wifi_plr_base

    def dry_run_set_fiveg(self, delay_ms, packet_loss):
        self.dr_fiveg_rtt_base = delay_ms
        self.dr_fiveg_rtt_state = self.dr_fiveg_rtt_base
        self.dr_fiveg_plr_base = packet_loss
        self.dr_fiveg_plr_state = self.dr_fiveg_plr_base

    def dry_run_sample_random_conditions(self):
        self.dry_run_set_wifi(np.random.uniform(0, 100), np.random.uniform(0, 0.1))
        self.dry_run_set_fiveg(np.random.uniform(0, 100), np.random.uniform(0, 0.1))

    def dry_run_reset_metrics_to_base(self):
        self.dr_fiveg_rtt_state = self.dr_fiveg_rtt_base
        self.dr_fiveg_plr_state = self.dr_fiveg_plr_base

    def get_metrics(self, session_id: str) -> Optional[PMFMetrics]:
        if self.dry_run:
            wifi_rtt_mean, fiveg_rtt_mean, wifi_plr_mean, fiveg_plr_mean = self.get_dry_run_metrics_mean()

            # create dummy probes and return dummy metrics
            wifi_rtt_probes = np.clip(np.random.normal(wifi_rtt_mean, 2, size=(self.probe_count)), a_min=0, a_max=100_000_000)
            fiveg_rtt_probes = np.clip(np.random.normal(fiveg_rtt_mean, 2, size=(self.probe_count)), a_min=0, a_max=100_000_000)
            wifi_plr = np.clip(np.random.normal(wifi_plr_mean, 0.003, size=(self.probe_count)).mean(), a_min=0, a_max=1)
            fiveg_plr = np.clip(np.random.normal(fiveg_plr_mean, 0.003, size=(self.probe_count)).mean(), a_min=0, a_max=1)
            return PMFMetrics(
                wifi_rtt_ms=wifi_rtt_probes.mean(),
                fiveg_rtt_ms=fiveg_rtt_probes.mean(),
                wifi_valid=True,
                fiveg_valid=True,
                wifi_plr=wifi_plr,
                fiveg_plr=fiveg_plr,
                timestamp="dummy_time",
                session_id=session_id,
                source="source",
                wifi_rtt_min_ms=wifi_rtt_probes.min(),
                wifi_rtt_max_ms=wifi_rtt_probes.max(),
                wifi_rtt_std_ms=wifi_rtt_probes.std(),
                fiveg_rtt_min_ms=fiveg_rtt_probes.min(),
                fiveg_rtt_max_ms=fiveg_rtt_probes.max(),
                fiveg_rtt_std_ms=fiveg_rtt_probes.std(),
                observation_age_s=0
            )

        if not self.asynchronous:
            self.flush_cache()
            self.trigger_new_measurements()
            if not self.polling_service.wait_for_polling_completion(session_id, 120):
                raise RuntimeError("Critical error: Failed to get PMF metrics within 120 seconds")

        if self.use_polling and self.polling_service:
            cached_metrics = self.polling_service.get_cached_metrics(session_id)
            if cached_metrics is not None:
                logger.debug("Using cached metrics from polling service")
                return cached_metrics
            else:
                logger.debug("Cache empty or stale, no metrics available")

        return None

    def trigger_new_measurements(self) -> None:
        """Submit new PMF jobs immediately, synchronized with an RL step."""
        if self.polling_service:
            self.polling_service.trigger_new_measurements()

    def wait_for_fresh_metrics(self, since_t: float, max_wait_s: float = 5.0) -> bool:
        """Block until a measurement newer than since_t is cached, or timeout.

        Call immediately after trigger_new_measurements() to ensure the next
        get_metrics() returns an observation that post-dates the action.
        """
        if self.polling_service:
            return self.polling_service.wait_for_fresh_metrics(since_t, max_wait_s)
        return False

    def flush_cache(self) -> None:
        # invalidate cached metrics so get_cached_metrics() returns None until fresh measurements arrive.
        if self.polling_service:
            self.polling_service.flush_cache()

    def shutdown(self):
        if self.polling_service:
            self.polling_service.stop()
            logger.info("PMF Polling Service stopped")

    def get_polling_status(self) -> Optional[Dict[str, Any]]:
        if self.polling_service:
            return self.polling_service.get_cache_status()
        return None
