# pmf polling service

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from threading import Thread, Event, Lock
import requests
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


@dataclass
class PMFJob:
    """Represents a PMF measurement job."""
    job_id: str
    job_type: str  # "rtt_wifi", "rtt_5g", "plr_wifi", "plr_5g"
    submitted_at: float
    access_type: str  # "3gpp" or "non3gpp"
    status: str = "pending"  # "pending", "completed", "failed"
    result: Optional[Dict[str, Any]] = None


@dataclass
class CachedMetrics:
    wifi_rtt_ms: Optional[float] = None
    fiveg_rtt_ms: Optional[float] = None
    wifi_plr: Optional[float] = None
    fiveg_plr: Optional[float] = None
    wifi_rtt_min_ms: Optional[float] = None
    wifi_rtt_max_ms: Optional[float] = None
    wifi_rtt_std_ms: Optional[float] = None
    fiveg_rtt_min_ms: Optional[float] = None
    fiveg_rtt_max_ms: Optional[float] = None
    fiveg_rtt_std_ms: Optional[float] = None
    last_update: Optional[float] = None
    source: str = "pmf-polling-service"
    wifi_rtt_measured_at: Optional[float] = None
    wifi_rtt_job_completion_time: Optional[float] = None
    # whether the last job request resulted in a valid measurement (True) or whether it timed out (False)
    wifi_rtt_valid: Optional[bool] = None
    fiveg_rtt_measured_at: Optional[float] = None
    fiveg_rtt_job_completion_time: Optional[float] = None
    fiveg_rtt_valid: Optional[bool] = None
    wifi_plr_measured_at: Optional[float] = None
    wifi_plr_valid: Optional[bool] = None
    wifi_plr_job_completion_time: Optional[float] = None
    fiveg_plr_measured_at: Optional[float] = None
    fiveg_plr_valid: Optional[bool] = None
    fiveg_plr_job_completion_time: Optional[float] = None

    def is_stale(self, max_age_seconds: float) -> bool:
        if self.last_update is None:
            return True
        return (time.time() - self.last_update) > max_age_seconds

    def oldest_measurement_time(self) -> Optional[float]:
        """Return the oldest measurement timestamp across all 4 metrics."""
        times = [t for t in (
            self.wifi_rtt_measured_at, self.fiveg_rtt_measured_at,
            self.wifi_plr_measured_at, self.fiveg_plr_measured_at,
        ) if t is not None]
        return min(times) if times else self.last_update

    def to_pmf_metrics(self, session_id: str, max_age_seconds: float = 30.0) -> Optional['PMFMetrics']:
        from .client import PMFMetrics

        if self.is_stale(max_age_seconds):
            return None

        if (self.wifi_rtt_ms is None or self.fiveg_rtt_ms is None or
                self.wifi_plr is None or self.fiveg_plr is None):
            return None

        oldest = self.oldest_measurement_time()
        obs_age = (time.time() - oldest) if oldest is not None else 0.0

        return PMFMetrics(
            wifi_rtt_ms=self.wifi_rtt_ms,
            wifi_valid=self.wifi_plr_valid and self.wifi_rtt_valid,
            fiveg_rtt_ms=self.fiveg_rtt_ms,
            fiveg_valid=self.fiveg_plr_valid and self.fiveg_rtt_valid,
            wifi_plr=self.wifi_plr,
            fiveg_plr=self.fiveg_plr,
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ") if self.last_update else "",
            session_id=session_id,
            source=self.source,
            wifi_rtt_min_ms=self.wifi_rtt_min_ms if self.wifi_rtt_min_ms is not None else self.wifi_rtt_ms,
            wifi_rtt_max_ms=self.wifi_rtt_max_ms if self.wifi_rtt_max_ms is not None else self.wifi_rtt_ms,
            wifi_rtt_std_ms=self.wifi_rtt_std_ms if self.wifi_rtt_std_ms is not None else 0.0,
            fiveg_rtt_min_ms=self.fiveg_rtt_min_ms if self.fiveg_rtt_min_ms is not None else self.fiveg_rtt_ms,
            fiveg_rtt_max_ms=self.fiveg_rtt_max_ms if self.fiveg_rtt_max_ms is not None else self.fiveg_rtt_ms,
            fiveg_rtt_std_ms=self.fiveg_rtt_std_ms if self.fiveg_rtt_std_ms is not None else 0.0,
            observation_age_s=max(obs_age, 0.0),
        )


class PMFPollingService:
    def __init__(self,
                 pmf_ue_url: str,
                 pmf_upf_url: str,
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
                 inter_job_wait_time_s: Optional[int] = None,
                 get_plr_from_rtt_job = True):

        self.pmf_ue_url = pmf_ue_url.rstrip('/')
        self.pmf_upf_url = pmf_upf_url.rstrip('/')
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout
        self.request_timeout = request_timeout
        self.probe_count = probe_count
        self.probe_timeout_ms = probe_timeout_ms
        self.cache_max_age = cache_max_age
        self.upf_ip = upf_ip
        self.ue_wifi_ip = ue_wifi_ip
        self.ue_5g_ip = ue_5g_ip
        self.ue_wifi_iface = ue_wifi_iface
        self.ue_5g_iface = ue_5g_iface
        self.use_uplink = use_uplink
        self.asynchronous = asynchronous
        self.inter_job_wait_time_s = inter_job_wait_time_s
        self.get_plr_from_rtt_job = get_plr_from_rtt_job

        self._cache = CachedMetrics()
        self._cache_lock = Lock()
        self.active_jobs: Dict[str, PMFJob] = {}
        self.polling_thread: Optional[Thread] = None
        self.stop_event = Event()
        self._trigger_event = Event()
        self.running = False
        self.session = requests.Session()
        self._pmf_unreachable_logged = False

        logger.info(
            f"PMF Polling Service initialized "
            f"(interval={poll_interval}s, job_timeout={job_timeout}s, "
            f"request_timeout={request_timeout}s, "
            f"probe_count={probe_count}, probe_timeout_ms={probe_timeout_ms}ms, "
            f"cache_max_age={cache_max_age}s, asynchronous={asynchronous})"
        )

    def start(self):
        if self.running:
            logger.warning("Polling service already running")
            return

        self.running = True
        self.stop_event.clear()
        self.polling_thread = Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        logger.info("PMF Polling Service started")

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        if self.polling_thread:
            self.polling_thread.join(timeout=5.0)

        logger.info("PMF Polling Service stopped")

    def get_cached_metrics(self, session_id: str) -> Optional['PMFMetrics']:
        with self._cache_lock:
            return self._cache.to_pmf_metrics(session_id, self.cache_max_age)

    def trigger_new_measurements(self) -> None:
        """Wake the background polling thread so it submits and polls sooner.

        Called by the RL env after each action.  Only the background thread
        may touch ``active_jobs``; this method never calls
        ``_submit_measurement_jobs`` directly to avoid the race condition where
        both threads see "no pending job" and both submit a duplicate.
        Setting ``_trigger_event`` causes the sleeping background thread to
        wake immediately, submit fresh jobs (if none are pending), and start
        polling — without waiting for the full ``poll_interval``.
        """
        self._trigger_event.set()

    def wait_for_fresh_metrics(self, since_t: float, max_wait_s: float = 5.0) -> bool:

        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            with self._cache_lock:
                if self._cache.last_update is not None and self._cache.last_update > since_t:
                    return True
            time.sleep(0.05)
        logger.warning("wait_for_fresh_metrics timed out after %.1fs — using stale cache", max_wait_s)
        return False
    
    def wait_for_polling_completion(self, session_id, max_wait_s: Optional[float]) -> bool:
        deadline = None if max_wait_s is None else time.time() + max_wait_s
        while (len(self.active_jobs) > 0 or self.get_cached_metrics(session_id) is None) and (deadline is None or time.time() < deadline):
            time.sleep(0.01)

        if len(self.active_jobs) == 0:
            return True
        
        return False

    def flush_cache(self) -> None:
        #invalidate cached metrics so get_cached_metrics() returns None until fresh measurements arrive.
        
        with self._cache_lock:
            self._cache = CachedMetrics()
        # clear completed jobs so the next loop iteration submits fresh ones
        self.active_jobs = {
            k: v for k, v in self.active_jobs.items() if v.status == "pending"
        }
        # self._trigger_event.set()
        logger.info("PMF cache flushed")

    def _polling_loop(self):
        logger.info("PMF polling loop started")

        while self.running and not self.stop_event.is_set():
            if self.asynchronous:
                try:
                    self._submit_measurement_jobs()
                    self._poll_job_results()
                    self._cleanup_old_jobs()
                    # sleep for poll_interval but wake early if the RL env
                    # calls trigger_new_measurements() (sets _trigger_event).
                    self._trigger_event.wait(timeout=self.poll_interval)
                    self._trigger_event.clear()
                except Exception as e:
                    logger.error(f"Error in polling loop: {e}", exc_info=True)
                    time.sleep(1.0)
            else:
                # wait until polling is triggered
                self._trigger_event.wait()
                self._trigger_event.clear()
                self._submit_measurement_jobs() # submit all jobs
                while self.running and len(self.active_jobs) > 0 and not self.stop_event.is_set():
                    # continuously check if results are ready 
                    try:
                        self._poll_job_results()
                        self._cleanup_old_jobs()
                        time.sleep(self.poll_interval)
                    except Exception as e:
                        logger.error(f"Error in polling loop: {e}", exc_info=True)

        logger.info("PMF polling loop stopped")

    def _inter_job_wait(self):
        if self.inter_job_wait_time_s is not None:
            time.sleep(self.inter_job_wait_time_s)

    def _submit_measurement_jobs(self):
        needs_wifi_rtt = not any(j.job_type == "rtt_wifi" and j.status == "pending"
                                 for j in self.active_jobs.values())
        needs_5g_rtt = not any(j.job_type == "rtt_5g" and j.status == "pending"
                              for j in self.active_jobs.values())

        if self.get_plr_from_rtt_job:
            needs_wifi_plr = False
            needs_5g_plr = False
        else:
            needs_wifi_plr = not any(j.job_type == "plr_wifi" and j.status == "pending"
                                    for j in self.active_jobs.values())
            needs_5g_plr = not any(j.job_type == "plr_5g" and j.status == "pending"
                                for j in self.active_jobs.values())

        if needs_wifi_rtt:
            try:
                if self.use_uplink:
                    job = self._submit_rtt_job_uplink("non3gpp", self.ue_wifi_iface, self.ue_wifi_ip, "rtt_wifi")
                else:
                    job = self._submit_rtt_job_downlink("non3gpp", "eth0", self.ue_wifi_ip, "rtt_wifi")

                self._inter_job_wait()
                if job:
                    self.active_jobs[job.job_id] = job
            except Exception as e:
                logger.warning(f"WiFi RTT job submission failed: {e}")

        if needs_5g_rtt:
            try:
                if self.use_uplink:
                    job = self._submit_rtt_job_uplink("3gpp", self.ue_5g_iface, self.ue_5g_ip, "rtt_5g")
                else:
                    job = self._submit_rtt_job_downlink("3gpp", "eth0", self.ue_5g_ip, "rtt_5g")

                self._inter_job_wait()
                if job:
                    self.active_jobs[job.job_id] = job
            except Exception as e:
                logger.warning(f"5G PLR job submission failed: {e}")

        if needs_wifi_plr:
            try:
                if self.use_uplink:
                    job = self._submit_plr_job_uplink("non3gpp", self.ue_wifi_iface, self.ue_wifi_ip, "plr_wifi")
                else:
                    job = self._submit_plr_job_downlink("non3gpp", "eth0", self.ue_wifi_ip, "plr_wifi")

                self._inter_job_wait()
                if job:
                    self.active_jobs[job.job_id] = job
            except Exception as e:
                logger.warning(f"WiFi PLR job submission failed: {e}")

        if needs_5g_plr:
            try:
                if self.use_uplink:
                    job = self._submit_plr_job_uplink("3gpp", self.ue_5g_iface, self.ue_5g_ip, "plr_5g")
                else:
                    job = self._submit_plr_job_downlink("3gpp", "eth0", self.ue_5g_ip, "plr_5g")

                self._inter_job_wait()
                if job:
                    self.active_jobs[job.job_id] = job
            except Exception as e:
                logger.warning(f"5G PLR job submission failed: {e}")

    def _submit_rtt_job_uplink(self, access_type: str, iface: str, source_ip: str, job_type: str) -> Optional[PMFJob]:
        try:
            url = f"{self.pmf_ue_url}/rtt_udp"
            payload = {
                "target": self.upf_ip,
                "count": self.probe_count,
                "timeout_ms": self.probe_timeout_ms,
                "access_type": access_type,
                "iface": iface,
                "source_ip": source_ip
            }

            logger.info(f"POST {url} payload {payload}")

            response = self._http_post_with_retry(url, payload)
            # PMF API returns 202 (Accepted) for job submission
            if response.status_code not in (200, 202):
                logger.warning(
                    f"PMF UE RTT job submission returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                response.raise_for_status()

            data = response.json()
            job_id = data.get("job_id")

            if job_id:
                now = time.time()
                job = PMFJob(
                    job_id=job_id,
                    job_type=job_type,
                    submitted_at=now,
                    access_type=access_type,
                    status="pending"
                )
                if self._pmf_unreachable_logged:
                    logger.info("PMF connectivity recovered")
                    self._pmf_unreachable_logged = False
                logger.info(f"Submitted uplink {job_type} job: {job_id} at {now}")
                return job
            else:
                logger.warning(f"No job_id in RTT response from PMF UE: {data}")
                return None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if not self._pmf_unreachable_logged:
                logger.warning(f"PMF UE unavailable for RTT submission ({job_type}): {e}")
                self._pmf_unreachable_logged = True
            return None
        except Exception as e:
            logger.warning(f"Failed to submit RTT job to PMF UE ({job_type}): {e}")
            return None

    def _submit_rtt_job_downlink(self, access_type: str, iface: str, target_ip: str, job_type: str) -> Optional[PMFJob]:
        try:
            url = f"{self.pmf_upf_url}/rtt_udp"
            payload = {
                "target": target_ip,
                "count": self.probe_count,
                "timeout_ms": self.probe_timeout_ms,
                "access_type": access_type,
                "iface": iface,
                "source_ip": self.upf_ip
            }

            logger.info(f"POST {url} payload {payload}")

            response = self._http_post_with_retry(url, payload)
            if response.status_code not in (200, 202):
                logger.warning(
                    f"PMF UPF RTT job submission returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                response.raise_for_status()

            data = response.json()
            job_id = data.get("job_id")

            if job_id:
                job = PMFJob(
                    job_id=job_id,
                    job_type=job_type,
                    submitted_at=time.time(),
                    access_type=access_type,
                    status="pending"
                )
                logger.info(f"Submitted downlink {job_type} job: {job_id}")
                return job
            else:
                logger.warning(f"No job_id in RTT response from PMF UPF: {data}")
                return None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if not self._pmf_unreachable_logged:
                logger.warning(f"PMF UPF unavailable for RTT submission ({job_type}): {e}")
                self._pmf_unreachable_logged = True
            return None
        except Exception as e:
            logger.warning(f"Failed to submit RTT job to PMF UPF ({job_type}): {e}")
            return None

    def _submit_plr_job_uplink(self, access_type: str, iface: str, source_ip: str, job_type: str) -> Optional[PMFJob]:
        try:
            url = f"{self.pmf_ue_url}/plr_uplink"
            payload = {
                "target": self.upf_ip,
                "count": self.probe_count,
                "timeout_ms": self.probe_timeout_ms,
                "access_type": access_type,
                "iface": iface,
                "source_ip": source_ip
            }

            logger.info(f"POST {url} payload {payload}")

            response = self._http_post_with_retry(url, payload)
            if response.status_code not in (200, 202):
                logger.warning(
                    f"PMF UE PLR job submission returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                response.raise_for_status()

            data = response.json()
            job_id = data.get("job_id")

            if job_id:
                job = PMFJob(
                    job_id=job_id,
                    job_type=job_type,
                    submitted_at=time.time(),
                    access_type=access_type,
                    status="pending"
                )
                logger.info(f"Submitted uplink {job_type} job: {job_id}")
                return job
            else:
                logger.warning(f"No job_id in PLR response from PMF UE: {data}")
                return None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if not self._pmf_unreachable_logged:
                logger.warning(f"PMF UE unavailable for PLR submission ({job_type}): {e}")
                self._pmf_unreachable_logged = True
            return None
        except Exception as e:
            logger.warning(f"Failed to submit PLR job to PMF UE ({job_type}): {e}")
            return None

    def _submit_plr_job_downlink(self, access_type: str, iface: str, target_ip: str, job_type: str) -> Optional[PMFJob]:
        try:
            url = f"{self.pmf_upf_url}/plr_downlink"
            payload = {
                "target": target_ip,
                "count": self.probe_count,
                "timeout_ms": self.probe_timeout_ms,
                "access_type": access_type,
                "iface": iface,
                "source_ip": self.upf_ip
            }

            logger.info(f"POST {url} payload {payload}")

            response = self._http_post_with_retry(url, payload)
            if response.status_code not in (200, 202):
                logger.warning(
                    f"PMF UPF PLR job submission returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                response.raise_for_status()

            data = response.json()
            job_id = data.get("job_id")

            if job_id:
                job = PMFJob(
                    job_id=job_id,
                    job_type=job_type,
                    submitted_at=time.time(),
                    access_type=access_type,
                    status="pending"
                )
                logger.info(f"Submitted downlink {job_type} job: {job_id}")
                return job
            else:
                logger.warning(f"No job_id in PLR response from PMF UPF: {data}")
                return None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if not self._pmf_unreachable_logged:
                logger.warning(f"PMF UPF unavailable for PLR submission ({job_type}): {e}")
                self._pmf_unreachable_logged = True
            return None
        except Exception as e:
            logger.warning(f"Failed to submit PLR job to PMF UPF ({job_type}): {e}")
            return None


    def _update_cache_for_timeout(self, job, now):
        age = now - job.submitted_at
        if job.job_type == "rtt_wifi":
            sentinel = float(self.probe_timeout_ms)
            with self._cache_lock:
                self._cache.wifi_rtt_ms = sentinel
                self._cache.wifi_rtt_min_ms = sentinel
                self._cache.wifi_rtt_max_ms = sentinel
                self._cache.wifi_rtt_std_ms = 0.0
                self._cache.wifi_rtt_measured_at = now
                self._cache.wifi_rtt_job_completion_time = self.job_timeout
                self._cache.wifi_rtt_valid = False
                self._cache.last_update = now
                if self.get_plr_from_rtt_job:
                    self._cache.wifi_plr = 1.0
                    self._cache.wifi_plr_measured_at = now
                    self._cache.wifi_plr_valid = False
                    self._cache.wifi_plr_job_completion_time = self.job_timeout
            logger.warning(
                "WiFi RTT probe timed out after %.0fs (sentinel %.0fms used)",
                age, sentinel,
            )
        elif job.job_type == "rtt_5g":
            sentinel = float(self.probe_timeout_ms)
            with self._cache_lock:
                self._cache.fiveg_rtt_ms = sentinel
                self._cache.fiveg_rtt_min_ms = sentinel
                self._cache.fiveg_rtt_max_ms = sentinel
                self._cache.fiveg_rtt_std_ms = 0.0
                self._cache.fiveg_rtt_measured_at = now
                self._cache.fiveg_rtt_job_completion_time = self.job_timeout
                self._cache.fiveg_rtt_valid = False
                self._cache.last_update = now
                if self.get_plr_from_rtt_job:
                    self._cache.fiveg_plr = 1.0
                    self._cache.fiveg_plr_measured_at = now
                    self._cache.fiveg_plr_valid = False
                    self._cache.fiveg_plr_job_completion_time = self.job_timeout
            logger.warning(
                "5G RTT probe timed out after %.0fs (sentinel %.0fms used)",
                age, sentinel,
            )
        elif job.job_type == "plr_wifi":
            with self._cache_lock:
                self._cache.wifi_plr = 1.0
                self._cache.wifi_plr_measured_at = now
                self._cache.wifi_plr_job_completion_time = self.job_timeout
                self._cache.wifi_plr_valid = False
                self._cache.last_update = now
            logger.warning(
                "WiFi path NOT ACCESSIBLE — PLR probe timed out after %.0fs "
                "(sentinel 100%% loss used)",
                age,
            )
        elif job.job_type == "plr_5g":
            with self._cache_lock:
                self._cache.fiveg_plr = 1.0
                self._cache.fiveg_plr_measured_at = now
                self._cache.fiveg_plr_job_completion_time = self.job_timeout
                self._cache.fiveg_plr_valid = False
                self._cache.last_update = now
            logger.warning(
                "5G path NOT ACCESSIBLE — PLR probe timed out after %.0fs "
                "(sentinel 100%% loss used)",
                age,
            )

    def _poll_job_results(self):
        for job_id, job in list(self.active_jobs.items()):
            # completed / failed jobs do not need to be considered
            if job.status != "pending":
                continue

            now = time.time()
            age = now - job.submitted_at 
            if age > self.job_timeout:
                logger.warning(f"Job {job_id} ({job.job_type}) timed out after {age:.2f}s")
                job.status = "failed"
                self._update_cache_for_timeout(job, now)                
                continue

            try:
                # try to poll result from PMF
                if job.job_type.startswith("rtt_"):
                    result = self._poll_rtt_result(job_id)
                elif job.job_type.startswith("plr_"):
                    result = self._poll_plr_result(job_id)
                else:
                    continue

                if result is None:
                    logger.info(
                        f"Job {job_id} ({job.job_type}) still pending "
                        f"(age={age:.0f}s, result=None/404)"
                    )
                    continue

                if isinstance(result, dict):
                    job_status = result.get("status")
                    if job_status == "pending":
                        logger.info(
                            f"Job {job_id} ({job.job_type}) still pending "
                            f"(age={age:.0f}s, PMF status={job_status})"
                        )
                        continue

                    if job_status == "timeout":
                        logger.warning(
                            f"Job {job_id} ({job.job_type}) timed out internally "
                            f"(age={age:.0f}s, PMF status={job_status})"
                        )

                    # Check if we have actual measurement data (completed or timed out job)
                    has_rtt_data = "avg_rtt_ms" in result or "mean_rtt_ms" in result or "rtt_ms" in result
                    has_plr_data = "plr_pct" in result or "plr" in result or "packet_loss_rate" in result
                    if not (has_rtt_data or has_plr_data):
                        logger.info(
                            f"Job {job_id} ({job.job_type}) returned unrecognised payload "
                            f"(age={age:.0f}s): {result}"
                        )
                        job.status = "failed"
                        continue

                job.status = "completed"
                job.result = result
                self._update_cache_from_result(job)
                logger.info(f"Job {job_id} ({job.job_type}) completed after {age}s: {result}")

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"PMF unavailable while polling job {job_id}: {e}")
                job.status = "failed"
            except Exception as e:
                logger.warning(f"Error polling job {job_id}: {e}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
        reraise=True,
    )
    def _http_get_with_retry(self, url: str) -> requests.Response:
        """HTTP GET with retry for transient failures."""
        return self.session.get(url, timeout=self.request_timeout)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
        reraise=True,
    )
    def _http_post_with_retry(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        """HTTP POST with retry for transient failures."""
        return self.session.post(url, json=payload, timeout=self.request_timeout)

    def _poll_rtt_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Poll for RTT job result (tries UE first, then UPF)."""
        if self.use_uplink:
            result = self._poll_rtt_result_ue(job_id)
            if result is not None:
                return result
            return self._poll_rtt_result_upf(job_id)
        else:
            return self._poll_rtt_result_upf(job_id)

    def _poll_rtt_result_ue(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.pmf_ue_url}/rtt_results/{job_id}"
            response = self._http_get_with_retry(url)

            if response.status_code == 404:
                logger.debug(f"PMF UE RTT result 404 for {job_id}")
                return None

            response.raise_for_status()
            data = response.json()
            logger.debug(f"PMF UE RTT result for {job_id}: HTTP {response.status_code} -> {data}")
            return data

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"PMF UE unavailable while polling RTT result {job_id}: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logger.debug(f"PMF UE timeout while polling RTT result {job_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error polling RTT result from PMF UE {job_id}: {e}")
            return None

    def _poll_rtt_result_upf(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.pmf_upf_url}/rtt_results/{job_id}"
            response = self._http_get_with_retry(url)

            if response.status_code == 404:
                logger.debug(f"PMF UPF RTT result 404 for {job_id}")
                return None

            response.raise_for_status()
            data = response.json()
            logger.debug(f"PMF UPF RTT result for {job_id}: HTTP {response.status_code} -> {data}")
            return data

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"PMF UPF unavailable while polling RTT result {job_id}: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logger.debug(f"PMF UPF timeout while polling RTT result {job_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error polling RTT result from PMF UPF {job_id}: {e}")
            return None

    def _poll_plr_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        if self.use_uplink:
            result = self._poll_plr_result_ue(job_id)
            if result is not None:
                return result
            return self._poll_plr_result_upf(job_id)
        else:
            return self._poll_plr_result_upf(job_id)

    def _poll_plr_result_ue(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.pmf_ue_url}/plr_results/{job_id}"
            response = self._http_get_with_retry(url)

            if response.status_code == 404:
                logger.debug(f"PMF UE PLR result 404 for {job_id}")
                return None

            response.raise_for_status()
            data = response.json()
            logger.debug(f"PMF UE PLR result for {job_id}: HTTP {response.status_code} -> {data}")
            return data

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"PMF UE unavailable while polling PLR result {job_id}: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logger.debug(f"PMF UE timeout while polling PLR result {job_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error polling PLR result from PMF UE {job_id}: {e}")
            return None

    def _poll_plr_result_upf(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.pmf_upf_url}/plr_results/{job_id}"
            response = self._http_get_with_retry(url)

            if response.status_code == 404:
                logger.debug(f"PMF UPF PLR result 404 for {job_id}")
                return None

            response.raise_for_status()
            data = response.json()
            logger.debug(f"PMF UPF PLR result for {job_id}: HTTP {response.status_code} -> {data}")
            return data

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"PMF UPF unavailable while polling PLR result {job_id}: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logger.debug(f"PMF UPF timeout while polling PLR result {job_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error polling PLR result from PMF UPF {job_id}: {e}")
            return None

    def _update_cache_from_result(self, job: PMFJob):
        if not job.result:
            return

        now = time.time()
        age = now - job.submitted_at
        try:
            if job.job_type == "rtt_wifi":
                rtt_ms = self._extract_rtt_from_result(job.result)
                loss_pct = self._safe_float(job.result.get("loss_pct"))
                with self._cache_lock:
                    self._cache.last_update = now
                    self._cache.wifi_rtt_measured_at = now
                    self._cache.wifi_rtt_job_completion_time = age
                    if rtt_ms is not None:
                        rtt_min, rtt_max, rtt_std = self._extract_rtt_stats_from_result(job.result)
                        self._cache.wifi_rtt_ms = rtt_ms
                        self._cache.wifi_rtt_min_ms = rtt_min
                        self._cache.wifi_rtt_max_ms = rtt_max
                        self._cache.wifi_rtt_std_ms = rtt_std
                        logger.info(f"Updated WiFi RTT: {rtt_ms:.2f}ms")
                    else:
                        sentinel = float(self.probe_timeout_ms)
                        self._cache.wifi_rtt_ms = sentinel
                        self._cache.wifi_rtt_min_ms = sentinel
                        self._cache.wifi_rtt_max_ms = sentinel
                        self._cache.wifi_rtt_std_ms = 0.0
                        logger.info(f"Updated WiFi RTT: {sentinel:.2f}ms (RTT was None)")

                    if self.get_plr_from_rtt_job:
                        self._cache.wifi_plr = loss_pct / 100
                        self._cache.wifi_plr_measured_at = now
                        self._cache.wifi_plr_valid = True
                        self._cache.wifi_plr_job_completion_time = age
                        logger.info(f"Updated WiFi PLR from rtt job: {loss_pct:.2f}%")

            if job.job_type == "rtt_5g":
                rtt_ms = self._extract_rtt_from_result(job.result)
                loss_pct = self._safe_float(job.result.get("loss_pct"))
                with self._cache_lock:
                    self._cache.last_update = now
                    self._cache.fiveg_rtt_measured_at = now
                    self._cache.fiveg_rtt_job_completion_time = age
                    if rtt_ms is not None:
                        rtt_min, rtt_max, rtt_std = self._extract_rtt_stats_from_result(job.result)
                        self._cache.fiveg_rtt_ms = rtt_ms
                        self._cache.fiveg_rtt_min_ms = rtt_min
                        self._cache.fiveg_rtt_max_ms = rtt_max
                        self._cache.fiveg_rtt_std_ms = rtt_std
                        logger.info(f"Updated 5G RTT: {rtt_ms:.2f}ms")
                    else:
                        sentinel = float(self.probe_timeout_ms)
                        self._cache.fiveg_rtt_ms = sentinel
                        self._cache.fiveg_rtt_min_ms = sentinel
                        self._cache.fiveg_rtt_max_ms = sentinel
                        self._cache.fiveg_rtt_std_ms = 0.0
                        logger.info(f"Updated 5G RTT: {sentinel:.2f}ms (RTT was None)")

                    if self.get_plr_from_rtt_job:
                        self._cache.fiveg_plr = loss_pct / 100
                        self._cache.fiveg_plr_measured_at = now
                        self._cache.fiveg_plr_valid = True
                        self._cache.fiveg_plr_job_completion_time = age
                        logger.info(f"Updated 5G PLR from rtt job: {loss_pct:.2f}%")

            elif job.job_type == "rtt_5g":
                rtt_ms = self._extract_rtt_from_result(job.result)
                if rtt_ms is not None:
                    rtt_min, rtt_max, rtt_std = self._extract_rtt_stats_from_result(job.result)
                    meas_time = self._extract_measurement_time(job.result) or now
                    with self._cache_lock:
                        self._cache.fiveg_rtt_ms = rtt_ms
                        self._cache.fiveg_rtt_min_ms = rtt_min
                        self._cache.fiveg_rtt_max_ms = rtt_max
                        self._cache.fiveg_rtt_std_ms = rtt_std
                        self._cache.fiveg_rtt_measured_at = meas_time
                        self._cache.last_update = now
                    logger.info(f"Updated 5G RTT: {rtt_ms:.2f}ms")

            elif job.job_type == "plr_wifi":
                plr = self._extract_plr_from_result(job.result)
                if plr is not None:
                    with self._cache_lock:
                        self._cache.wifi_plr = plr
                        self._cache.wifi_plr_measured_at = now
                        self._cache.last_update = now
                        self._cache.wifi_plr_job_completion_time = age
                        self._cache.wifi_plr_valid = True
                    logger.info(f"Updated WiFi PLR: {plr:.4f}")

            elif job.job_type == "plr_5g":
                plr = self._extract_plr_from_result(job.result)
                if plr is not None:
                    with self._cache_lock:
                        self._cache.fiveg_plr = plr
                        self._cache.fiveg_plr_measured_at = now
                        self._cache.last_update = now
                        self._cache.fiveg_plr_job_completion_time = age
                        self._cache.fiveg_plr_valid = True
                    logger.info(f"Updated 5G PLR: {plr:.4f}")

        except Exception as e:
            logger.error(f"Error updating cache from job {job.job_id}: {e}")

    def _safe_float(self, val: Any) -> Optional[float]:
        """Convert to float; return None if val is None or invalid."""
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _extract_rtt_from_result(self, result: Dict[str, Any]) -> Optional[float]:
        """Extract average RTT in ms from a PMF result dict."""
        if "avg_rtt_ms" in result:
            v = self._safe_float(result["avg_rtt_ms"])
            if v is not None:
                return v
        if "mean_rtt_ms" in result:
            v = self._safe_float(result["mean_rtt_ms"])
            if v is not None:
                return v
        if "rtt_ms" in result:
            v = self._safe_float(result["rtt_ms"])
            if v is not None:
                return v
        if "rtt_us" in result:
            v = self._safe_float(result["rtt_us"])
            if v is not None:
                return v / 1000.0
        if any(k in result for k in ("avg_rtt_ms", "mean_rtt_ms", "rtt_ms", "rtt_us")):
            logger.debug(f"RTT result has null values: {result.get('loss_pct')}% loss")
        else:
            logger.warning(f"Unknown RTT result format: {result}")
        return None

    def _extract_rtt_stats_from_result(
        self, result: Dict[str, Any],
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Extract (min_rtt_ms, max_rtt_ms, stddev_rtt_ms) from a PMF result dict."""
        return (
            self._safe_float(result.get("min_rtt_ms")),
            self._safe_float(result.get("max_rtt_ms")),
            self._safe_float(result.get("stddev_rtt_ms")),
        )

    def _extract_measurement_time(self, result: Dict[str, Any]) -> Optional[float]:
        """Extract the earliest sent_ts from per_packet data (RTT results only)."""
        per_packet = result.get("per_packet")
        if not per_packet or not isinstance(per_packet, list):
            return None
        sent_times = []
        for pkt in per_packet:
            ts = self._safe_float(pkt.get("sent_ts") if isinstance(pkt, dict) else None)
            if ts is not None:
                sent_times.append(ts)
        return min(sent_times) if sent_times else None

    def _extract_plr_from_result(self, result: Dict[str, Any]) -> Optional[float]:
        if "plr" in result:
            v = self._safe_float(result["plr"])
            if v is not None:
                return v
        if "plr_pct" in result:
            v = self._safe_float(result["plr_pct"])
            if v is not None:
                return v / 100.0
        if "packet_loss_rate" in result:
            v = self._safe_float(result["packet_loss_rate"])
            if v is not None:
                return v
        if "loss_rate" in result:
            v = self._safe_float(result["loss_rate"])
            if v is not None:
                return v
        if "plr_percent" in result:
            v = self._safe_float(result["plr_percent"])
            if v is not None:
                return v / 100.0
        if any(k in result for k in ("plr", "plr_pct", "packet_loss_rate", "loss_rate", "plr_percent", "loss_pct")):
            logger.debug(f"PLR result has null values")
        else:
            logger.warning(f"Unknown PLR result format: {result}")
        return None

    def _cleanup_old_jobs(self):
        jobs_to_remove = []
        for job_id, job in self.active_jobs.items():
            if job.status != "pending":
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]

    def get_cache_status(self) -> Dict[str, Any]:
        with self._cache_lock:
            return {
                "running": self.running,
                "cache_age_seconds": time.time() - self._cache.last_update if self._cache.last_update else None,
                "cache_stale": self._cache.is_stale(self.cache_max_age),
                "active_jobs": len(self.active_jobs),
                "cache": {
                    "wifi_rtt_ms": self._cache.wifi_rtt_ms,
                    "fiveg_rtt_ms": self._cache.fiveg_rtt_ms,
                    "wifi_plr": self._cache.wifi_plr,
                    "fiveg_plr": self._cache.fiveg_plr,
                    "last_update": self._cache.last_update
                }
            }
