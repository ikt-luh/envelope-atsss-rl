# apply ScenarioConfig via tc (HTB + netem / tbf) and iperf3 user + cross traffic.

from __future__ import annotations

import logging
import os
import random
import re
import signal
import subprocess
import threading
import time
import traceback
from typing import List, Optional

from pmf_client.client import PMFClientManager
from scenarios.config_space import ScenarioSettings, TrafficClass
from scenarios.sampler import ScenarioConfig, sample_random_config
from scripts.traffic_gen import get_bytes_sent, reset_bytes_sent, run_fifo_writer

logger = logging.getLogger(__name__)


def _has_sudo() -> bool:
    #  check if sudo is available on this system.
    import shutil
    return shutil.which("sudo") is not None


def _run(cmd: List[str], dry_run: bool, *, allow_fail: bool = False) -> subprocess.CompletedProcess[str]:
    prefix = "<CMD NOT EXECUTED>" if dry_run else "$"
    logger.info(f"[scenarios] {prefix} %s", " ".join(cmd))
    if dry_run:
        return None
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (
            f"Command failed (rc={result.returncode}): {' '.join(cmd)}"
            f"\n  stderr: {result.stderr.strip()}"
        )
        if allow_fail:
            logger.debug(msg)
            return result
        else:
            logger.error(msg)
            raise RuntimeError(msg)
    return result


def _iface_ipv4(iface: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["ip", "-4", "addr", "show", "dev", iface],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    m = re.search(r"inet\s+([0-9.]+)/", out)
    return m.group(1) if m else None


def _tc_clean(iface: str, sudo: bool, dry_run: bool) -> None:
    cmd = (["sudo"] if sudo else []) + ["tc", "qdisc", "del", "dev", iface, "root"]
    _run(cmd, dry_run, allow_fail=True) 


def _apply_qdisc(
    iface: str,
    delay_ms: float,
    loss_pct: float,
    rate_mbps: Optional[float],
    sudo: bool,
    dry_run: bool,
) -> None:
    _tc_clean(iface, sudo, dry_run)

    has_netem = delay_ms > 0.0 or loss_pct > 0.0
    rate = float(rate_mbps) if rate_mbps is not None and rate_mbps > 0 else 0.0

    if rate > 0 and not has_netem:
        cmd = (["sudo"] if sudo else []) + [
            "tc", "qdisc", "replace", "dev", iface, "root", "tbf",
            "rate", f"{rate}mbit", "burst", "32k", "limit", "3000",
        ]
        _run(cmd, dry_run)
        return

    if has_netem and rate <= 0:
        args = (["sudo"] if sudo else []) + ["tc", "qdisc", "replace", "dev", iface, "root", "netem"]
        if delay_ms > 0:
            args.extend(["delay", f"{delay_ms}ms"])
        if loss_pct > 0:
            args.extend(["loss", f"{loss_pct}%"])
        _run(args, dry_run)
        return

    if has_netem and rate > 0:
        r = max(0.1, rate)
        pre = ["sudo"] if sudo else []
        _run(pre + [
            "tc", "qdisc", "replace", "dev", iface, "root", "handle", "1:", "htb", "default", "1",
        ], dry_run)
        _run(pre + [
            "tc", "class", "add", "dev", iface, "parent", "1:", "classid", "1:1", "htb",
            "rate", f"{r}mbit", "ceil", f"{r}mbit",
        ], dry_run)
        leaf = pre + ["tc", "qdisc", "add", "dev", iface, "parent", "1:1", "handle", "10:", "netem"]
        if delay_ms > 0:
            leaf.extend(["delay", f"{delay_ms}ms"])
        if loss_pct > 0:
            leaf.extend(["loss", f"{loss_pct}%"])
        _run(leaf, dry_run)
        return


class ScenarioApplier:
   #applies tc rules and manages iperf3 processes for one UE host.

    def __init__(
        self,
        settings: ScenarioSettings,
        *,
        use_sudo: bool = True,
        pmf_client: Optional[PMFClientManager] = None
    ):
        self.settings = settings
        # auto-detect: if sudo requested but not available (e.g., Docker with
        # NET_ADMIN capability), fall back to running tc directly.
        if use_sudo and not _has_sudo():
            logger.info("sudo not found; running tc commands directly")
            use_sudo = False
        self.use_sudo = use_sudo
        self.dry_run = settings.dry_run
        self._iperf_procs: List[subprocess.Popen] = []
        self._mpquic_traffic_gen_thread: Optional[threading.Thread] = None
        self._mpquic_traffic_gen_thread_event = threading.Event()

        def _handle_signal(sig: int, _frame) -> None:
            logger.info("APL: Signal %d received — stopping.", sig)
            if self._mpquic_traffic_gen_thread is not None:
                self._mpquic_traffic_gen_thread_event.set()
                self._mpquic_traffic_gen_thread.join()
            import sys
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        self._traffic_checked = False
        self._pmf_client: PMFClientManager = pmf_client

        if self.dry_run:
            # during a dry run, we apply the scenarios via the PMF client
            assert self._pmf_client is not None
            # TODO: replace dynamic logic through scenarios..
            self._pmf_client.dry_run_dynamic = True

    def _bind_ip_for_iface(
        self,
        bind_ip: Optional[str],
        iface: str,
        *,
        path_label: str,
        env_hint: str,
    ) -> str:
        iface = (iface or "").strip()
        bind = (bind_ip or "").strip() or None
        if not iface:
            if bind:
                return bind
            raise RuntimeError(
                f"{path_label}_iface is empty. Set {env_hint} in .env or export those vars."
            )
        resolved = _iface_ipv4(iface)
        if not resolved:
            if bind:
                logger.warning(
                    "No IPv4 on %s interface %r; using configured bind_ip %s (iperf -B may fail)",
                    path_label,
                    iface,
                    bind,
                )
                return bind
            raise RuntimeError(f"No IPv4 address on interface {iface!r}")
        if not bind or bind == resolved:
            return resolved
        logger.warning(
            "%s bind IP %s does not match kernel %s on %r; using %s for iperf -B "
            "(update PMF_UE_*_IP or SCENARIO_*_BIND_IP in .env)",
            path_label,
            bind,
            resolved,
            iface,
            resolved,
        )
        return resolved

    def _wifi_ip(self) -> str:
        return self._bind_ip_for_iface(
            self.settings.wifi_bind_ip,
            self.settings.wifi_iface,
            path_label="wifi",
            env_hint="PMF_UE_WIFI_IFACE or TESTBED_SCENARIO_WIFI_IFACE",
        )

    def _fiveg_ip(self) -> str:
        return self._bind_ip_for_iface(
            self.settings.fiveg_bind_ip,
            self.settings.fiveg_iface,
            path_label="5g",
            env_hint="PMF_UE_5G_IFACE or TESTBED_SCENARIO_5G_IFACE",
        )

    def cleanup(self) -> None:
        self._kill_iperf()
        self._stop_user_traffic_gen()
        if self.dry_run:
            return
        w = (self.settings.wifi_iface or "").strip()
        f = (self.settings.fiveg_iface or "").strip()
        if w:
            _tc_clean(w, self.use_sudo, self.dry_run)
        if f:
            _tc_clean(f, self.use_sudo, self.dry_run)

    def _kill_iperf(self) -> None:
        for p in self._iperf_procs:
            if p.poll() is not None:
                continue  
            try:
                p.terminate()
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    # kill the entire process group (iperf3 uses start_new_session)
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=5)
                except Exception:
                    logger.warning("Could not kill iperf3 process pid=%d", p.pid)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self._iperf_procs.clear()

    def _start_iperf_client(
        self,
        bind_ip: str,
        port: int,
        duration_s: int,
        bitrate_mbps: Optional[float],
        extra_args: Optional[List[str]] = None,
    ) -> None:
        cmd = ["iperf3", "-c", self.settings.ane_ip, "-B", bind_ip, "-p", str(port), "-t", str(duration_s)]
        if bitrate_mbps is not None and bitrate_mbps > 0:
            cmd.extend(["-b", f"{bitrate_mbps}M"])
        if extra_args:
            cmd.extend(extra_args)
        if self.dry_run:
            logger.info("[dry-run] %s", " ".join(cmd))
            return
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # brief pause to catch immediate failures (e.g. server unreachable)
        time.sleep(0.5)
        if p.poll() is not None:
            stderr = p.stderr.read() if p.stderr else ""
            hint = ""
            if "Cannot assign requested address" in (stderr or ""):
                hint = (
                    " (bind/source: check PMF_UE_WIFI_IP / PMF_UE_5G_IP and iface names vs "
                    "`ip -4 addr` on the UE; ANE must run `iperf3 -s` on this port)"
                )
            logger.debug(
                "iperf3 client exited immediately (rc=%d, port=%d, -B %s -c %s): %s%s",
                p.returncode,
                port,
                bind_ip,
                self.settings.ane_ip,
                stderr.strip() if stderr else "(no stderr)",
                hint,
            )
        else:
            self._iperf_procs.append(p)

    def _check_traffic_gen(self) -> None:
        import socket as _socket

        tport = int(os.getenv("TESTBED_TRAFFIC_PORT", "5204"))

        # 1. is the traffic_gen sink reachable on ANE?
        if self.settings.ane_ip:
            try:
                s = _socket.create_connection((self.settings.ane_ip, tport), timeout=2)
                s.close()
                logger.info("[traffic_gen] server reachable at %s:%d", self.settings.ane_ip, tport)
            except OSError as e:
                logger.warning(
                    "[traffic_gen] server NOT reachable at %s:%d: %s "
                    "(expected if TESTBED_TRAFFIC_ENABLE=false — cross-traffic background generator is disabled)",
                    self.settings.ane_ip, tport, e,
                )

        # 2. interface TX bytes — non-zero means traffic has flowed through the tunnel
        for label, iface in [("wifi", (self.settings.wifi_iface or "").strip()),
                              ("5g",   (self.settings.fiveg_iface or "").strip())]:
            if not iface:
                continue
            try:
                out = subprocess.run(
                    ["ip", "-s", "link", "show", "dev", iface],
                    capture_output=True, text=True, check=False,
                ).stdout
                tx_bytes = None
                lines = out.splitlines()
                for i, line in enumerate(lines):
                    if "TX:" in line and i + 1 < len(lines):
                        parts = lines[i + 1].split()
                        if parts:
                            try:
                                tx_bytes = int(parts[0])
                            except ValueError:
                                pass
                        break
                if tx_bytes is None:
                    logger.warning("[traffic_gen] %s (%s): could not read TX stats", label, iface)
                elif tx_bytes > 0:
                    logger.info("[traffic_gen] %s (%s) TX bytes: %d — traffic flowing", label, iface, tx_bytes)
                else:
                    logger.warning("[traffic_gen] %s (%s) TX bytes: 0 — no traffic detected on this path", label, iface)
            except Exception as e:
                logger.warning("[traffic_gen] %s (%s) check failed: %s", label, iface, e)

    def _stop_user_traffic_gen(self):
        if self._mpquic_traffic_gen_thread is not None:
            self._mpquic_traffic_gen_thread_event.set()
            self._mpquic_traffic_gen_thread.join()
            self._mpquic_traffic_gen_thread_event.clear()

    def apply(self, cfg: ScenarioConfig) -> None:
        if not self._traffic_checked and not self.dry_run:
            self._check_traffic_gen()
            self._traffic_checked = True
        self.cleanup()

        w_delay = cfg.wifi_extra_delay_ms
        f_delay = cfg.fiveg_extra_delay_ms
        w_loss = cfg.wifi_loss_pct
        f_loss = cfg.fiveg_loss_pct
        w_rate_limit = cfg.wifi_rate_limit_mbps
        f_rate_limit = cfg.fiveg_rate_limit_mbps

        wifi_iface = (self.settings.wifi_iface or "").strip()
        logger.info(f"[scenarios] WiFi: {w_delay} ms delay, {w_loss} loss, {w_rate_limit} rate limit")
        if self.dry_run:
            logger.info(f"[scenarios] DRY RUN: Applying delay & loss via dummy PMF")
            self._pmf_client.dry_run_set_wifi(w_delay, w_loss / 100)

        if wifi_iface:
            _apply_qdisc(
                wifi_iface, w_delay, w_loss, w_rate_limit,
                self.use_sudo, self.dry_run
            )
        elif not self.dry_run:
            raise RuntimeError(
                "wifi_iface is empty (PMF_UE_WIFI_IFACE / TESTBED_SCENARIO_WIFI_IFACE). "
                "Cannot apply WiFi tc rules."
            )

        fiveg_iface = (self.settings.fiveg_iface or "").strip()
        logger.info(f"[scenarios] 5G: {f_delay} ms delay, {f_loss} loss, {f_rate_limit} rate limit")
        if self.dry_run:
            logger.info(f"[scenarios] DRY RUN: Applying delay & loss via dummy PMF")
            self._pmf_client.dry_run_set_fiveg(f_delay, f_loss / 100)

        if fiveg_iface:
            _apply_qdisc(
                fiveg_iface, f_delay, f_loss, f_rate_limit,
                self.use_sudo, self.dry_run,
            )
        elif not self.dry_run:
            raise RuntimeError(
                "fiveg_iface is empty (PMF_UE_5G_IFACE / TESTBED_SCENARIO_5G_IFACE). "
                "Cannot apply 5G tc rules."
            )

        # user traffic gen

        # start traffic gen
        # if self.dry_run:
        #     logger.info("[dry-run] %s; using placeholder 5G bind %s", e, fiveg_ip)
        # else:
        is_burst_mode = cfg.user_data_mode == "burst"
        traffic_gen_mode = "bursty" if is_burst_mode else "periodic"
        info_mbps = cfg.bursty_mbit_per_period if is_burst_mode else cfg.user_data_mbps
        info_extra = f'{cfg.bursty_period_s} s' if is_burst_mode else ''
        if cfg.traffic_class == TrafficClass.CONTROL:
            # small chunk size for fine-grained control
            # Example: 0.5 Mbps => 62.5 kBs => 125 chunks of size 500 B per second
            chunk_size = 500
        else:
            # big chunk size for bulky traffic
            # as we want to avoid having to spend too many cycles on traffic gen
            chunk_size = 16_000

        # NOTE: user traffic can be enabled here even in dry run for debugging purposes
        user_traffic_enabled = not self.dry_run  # or True
        user_traffic_info = "" if user_traffic_enabled else "<USER TRAFFIC IGNORED> "
        logger.info(f"[scenarios] {user_traffic_info}User traffic: mode {traffic_gen_mode}, {info_mbps} Mbps {info_extra}")

        if user_traffic_enabled:
            self._stop_user_traffic_gen()
            self._mpquic_traffic_gen_thread = threading.Thread(
                target=run_fifo_writer,
                args=(self.settings.mpquic_fifo_input_path, traffic_gen_mode, cfg.user_data_mbps, cfg.bursty_mbit_per_period, cfg.bursty_period_s, chunk_size, self._mpquic_traffic_gen_thread_event),
                daemon=False,
            )
            self._mpquic_traffic_gen_thread.start()

        # cross-traffic

        # if self.dry_run and not (self.settings.wifi_bind_ip or wifi_iface):
        #     wifi_ip = "127.0.0.1"
        #     logger.info("[dry-run] using placeholder wifi bind IP %s (no iface / bind_ip)", wifi_ip)
        # elif self.dry_run:
        #     try:
        #         wifi_ip = self._wifi_ip()
        #     except RuntimeError as e:
        #         wifi_ip = "127.0.0.1"
        #         logger.info("[dry-run] %s; using placeholder wifi bind %s", e, wifi_ip)
        # else:
        #     wifi_ip = self._wifi_ip()

        # if self.dry_run and not (self.settings.fiveg_bind_ip or fiveg_iface):
        #     fiveg_ip = "127.0.0.1"
        #     logger.info("[dry-run] using placeholder 5G bind IP %s (no iface / bind_ip)", fiveg_ip)
        # elif self.dry_run:
        #     try:
        #         fiveg_ip = self._fiveg_ip()
        #     except RuntimeError as e:
        #         fiveg_ip = "127.0.0.1"
        #         logger.info("[dry-run] %s; using placeholder 5G bind %s", e, fiveg_ip)
        # else:
        #     fiveg_ip = self._fiveg_ip()

        if cfg.cross_traffic_mbps > 0:
            # TODO: Binding does not work on testbed (?)
            logger.warning("[scenarios] CROSS TRAFFIC IS NOT IMPLEMENTED")
            # cross_port = self.settings.cross_traffic_port
            # duration = 86400
            # cross_bind = wifi_ip if cfg.cross_traffic_on_wifi else fiveg_ip
            # self._start_iperf_client(cross_bind, cross_port, duration, cfg.cross_traffic_mbps, None)


class InterfaceStatistics:
    def __init__(self, iname, dry_run, statistics_key="tx_bytes"):
        self.iname = iname
        self.dry_run = dry_run
        self.key = statistics_key
        self.base = 0
        self.value = 0
        self.reset()

    def reset(self):
        self._update()
        self.base = self.value

    def _update(self):
        cmd = ["cat", f"/sys/class/net/{self.iname}/statistics/{self.key}"]
        result = _run(cmd, self.dry_run, allow_fail=True)
        if self.dry_run:
            return
        try:
            self.value = float(result.stdout)
        except Exception:
            logging.error(traceback.format_exc())

    def get(self):
        self._update()
        return self.value - self.base


class TrafficWatcher:
    def __init__(self, settings: ScenarioSettings):
        self.bytes_sent_wifi = InterfaceStatistics(settings.wifi_iface, settings.dry_run, "tx_bytes")
        self.bytes_sent_fiveg = InterfaceStatistics(settings.fiveg_iface, settings.dry_run, "tx_bytes")

    def reset_bytes_sent(self):
        self.bytes_sent_fiveg.reset()
        self.bytes_sent_wifi.reset()
        # user traffic automatically resets when applying new user traffic
        # but we still reset it here in case we perform multiple episodes
        reset_bytes_sent()

    def get_bytes_sent_fifo(self):
        return get_bytes_sent()

    def get_bytes_sent_wifi(self):
        return self.bytes_sent_wifi.get()

    def get_bytes_sent_fiveg(self):
        return self.bytes_sent_fiveg.get()

    @staticmethod
    def get_bytes_sent_dict(watcher: Optional[TrafficWatcher]) -> dict:
        if watcher is None:
            return dict(
                bytes_sent_fifo=0,
                bytes_sent_wifi=0,
                bytes_sent_fiveg=0
            )
        return dict(
            bytes_sent_fifo=watcher.get_bytes_sent_fifo(),
            bytes_sent_wifi=watcher.get_bytes_sent_wifi(),
            bytes_sent_fiveg=watcher.get_bytes_sent_fiveg()
        )

    @staticmethod
    def delta(bytes_sent_past: dict, bytes_sent_now: dict) -> dict:
        delta = {}
        for k in bytes_sent_now:
            if k in bytes_sent_past:
                delta[k] = bytes_sent_now[k] - bytes_sent_past[k]
        return delta


class ScenarioManager:
    # sampler + applier; call `sample_and_apply` from the RL env on episode reset.
    # TODO: Manager and applier could be merged..
    def __init__(
        self,
        settings: ScenarioSettings,
        *,
        rng: Optional[random.Random] = None,
        use_sudo: bool = True,
        pmf_client: Optional[object] = None,
    ):
        self.settings = settings
        self.rng = rng or random.Random()
        self.applier = ScenarioApplier(settings, use_sudo=use_sudo, pmf_client=pmf_client)
        self.watcher = TrafficWatcher(settings)

        self.current_config: Optional[ScenarioConfig] = None
        self._pmf_client = pmf_client

    def sample_and_apply(self) -> ScenarioConfig:
        cfg = sample_random_config(self.settings, self.rng)
        self.applier.apply(cfg)
        self.current_config = cfg
        if self._pmf_client is not None and hasattr(self._pmf_client, "flush_cache"):
            self._pmf_client.flush_cache()

        self.watcher.reset_bytes_sent()
        return cfg

    def shutdown(self) -> None:
        self.applier.cleanup()
