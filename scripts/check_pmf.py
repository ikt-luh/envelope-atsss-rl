#!/usr/bin/env python3
# a script to check pmf connectivity. use before training to verify pmf is reachable. make sure to connect with the testbed before running this script on your machine.
"""
Usage:
  python scripts/check_pmf.py
  python scripts/check_pmf.py --ue-url http://192.XXX.XXX.XXX:XXXX --upf-url http://192.XXX.XXX.XXX:XXXX
  python scripts/check_pmf.py --wait-s 60
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_DOTENV_LOADED: bool = False
try:

    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)
        _DOTENV_LOADED = True
except Exception:
    _DOTENV_LOADED = False

import config 
import requests


def check_url(name: str, url: str, timeout: float = 5.0) -> bool:
    """Try to reach a URL. Returns True if we get any HTTP response.

    A 404 on GET / only means there is no route for the root path; the host:port
    still answered, which is what this step checks. PMF endpoints live under paths
    like /rtt_udp (verified in steps 2-3).
    """
    try:
        r = requests.get(url.rstrip("/") + "/", timeout=timeout)
        if r.status_code == 404:
            print(
                f"  {name}: OK (reachable; HTTP 404 on GET / — no root route, "
                "this is normal for PMF)"
            )
        else:
            print(f"  {name}: OK (HTTP {r.status_code})")
        return True
    except requests.exceptions.ConnectionError as e:
        print(f"  {name}: FAILED (connection refused or unreachable)")
        print(f"    -> {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"  {name}: FAILED (timeout after {timeout}s)")
        return False
    except Exception as e:
        print(f"  {name}: FAILED ({e})")
        return False


def check_pmf_job_submit(
    ue_url: str,
    upf_url: str,
    use_uplink: bool,
    timeout: float = 5.0,
) -> bool:

    try:
        if use_uplink:
            url = ue_url
            payload = {
                "target": config.PMF_UPF_IP,
                "count": 100,
                "timeout_ms": 1000,
                "access_type": "non3gpp",
                "iface": config.PMF_UE_WIFI_IFACE,
                "source_ip": config.PMF_UE_WIFI_IP,
            }
        else:
            url = upf_url
            payload = {
                "target": config.PMF_UE_WIFI_IP,
                "count": 100,
                "timeout_ms": 1000,
                "access_type": "non3gpp",
                "iface": "eth0",
                "source_ip": config.PMF_UPF_IP,
            }

        r = requests.post(url.rstrip("/") + "/rtt_udp", json=payload, timeout=timeout)
        if r.status_code in (200, 202):
            data = r.json()
            job_id = data.get("job_id", "?")
            print(f"  RTT job submit: OK (job_id={job_id})")
            return True
        else:
            print(f"  RTT job submit: HTTP {r.status_code} - {r.text[:100]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  RTT job submit: FAILED (connection refused)")
        return False
    except requests.exceptions.Timeout:
        print(f"  RTT job submit: FAILED (timeout)")
        return False
    except Exception as e:
        print(f"  RTT job submit: FAILED ({e})")
        return False


def check_pmf_client_metrics(
    session_id: str,
    pmf_ue_url: str,
    pmf_upf_url: str,
    wait_s: float = 45.0,
    job_timeout: float | None = None,
) -> bool:
    """Create PMFClientManager, wait for cache, try get_metrics."""
    try:
        from pmf_client.client import PMFClientManager
    except ImportError:
        print("  PMF client metrics: SKIP (pmf_client not importable)")
        return False

    timeout = job_timeout if job_timeout is not None else max(60.0, config.PMF_JOB_TIMEOUT)
    effective_wait = max(wait_s, timeout + 15)
    print(f"  Creating PMF client (poll_interval=2s, job_timeout={timeout}s, waiting up to {effective_wait}s for cache)...")
    client = PMFClientManager(
        dry_run=False,
        pmf_ue_url=pmf_ue_url,
        pmf_upf_url=pmf_upf_url,
        use_polling=True,
        poll_interval=2.0,
        job_timeout=timeout,
        request_timeout=config.PMF_REQUEST_TIMEOUT,
        probe_count=5,
        probe_timeout_ms=500,
        cache_max_age=config.PMF_CACHE_MAX_AGE,
        upf_ip=config.PMF_UPF_IP,
        ue_wifi_ip=config.PMF_UE_WIFI_IP,
        ue_5g_ip=config.PMF_UE_5G_IP,
        ue_wifi_iface=config.PMF_UE_WIFI_IFACE,
        ue_5g_iface=config.PMF_UE_5G_IFACE,
        use_uplink=config.PMF_USE_UPLINK,
        asynchronous=False
    )
    # PMFClientManager.__init__ already starts the polling service.

    for i in range(int(effective_wait / 2) + 1):
        metrics = client.get_metrics(session_id)
        if metrics is not None:
            client.shutdown()
            wifi_note = " [NOT ACCESSIBLE — sentinel value]" if metrics.wifi_rtt_ms >= 500.0 else ""
            print(
                f"  PMF metrics: OK "
                f"(wifi_rtt={metrics.wifi_rtt_ms:.1f}ms{wifi_note}, "
                f"5g_rtt={metrics.fiveg_rtt_ms:.1f}ms, "
                f"wifi_plr={metrics.wifi_plr:.1f}, "
                f"5g_plr={metrics.fiveg_plr:.1f})"
            )
            return True
        time.sleep(2)

    try:
        polling_status = client.get_polling_status()
    except Exception:
        polling_status = None

    cache = (polling_status or {}).get("cache", {}) if polling_status else {}
    active_jobs = (polling_status or {}).get("active_jobs") if polling_status else None

    client.shutdown()
    print("  PMF metrics: FAILED (no cached metrics after wait - PMF may be slow or unreachable)")
    if cache:
        print(
            "  Cache snapshot: "
            f"wifi_rtt_ms={cache.get('wifi_rtt_ms')}, fiveg_rtt_ms={cache.get('fiveg_rtt_ms')}, "
            f"wifi_plr={cache.get('wifi_plr')}, fiveg_plr={cache.get('fiveg_plr')}, "
            f"active_jobs={active_jobs}"
        )
        if (cache.get("wifi_plr") is not None or cache.get("fiveg_plr") is not None) and (
            cache.get("wifi_rtt_ms") is None and cache.get("fiveg_rtt_ms") is None
        ):
            print("  Hint: PLR arrived but RTT did not; PMF UE-side RTT responder may be unavailable.")
    return False


def main() -> int:
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Hello from the PMF check script")

    parser = argparse.ArgumentParser(description="Check PMF connectivity")
    parser.add_argument("--ue-url", default=None, help="PMF UE URL (default: from config/env)")
    parser.add_argument("--upf-url", default=None, help="PMF UPF URL (default: from config/env)")
    parser.add_argument("--wait-s", type=float, default=45.0, help="Seconds to wait for PMF cache (default: 45)")
    parser.add_argument("--job-timeout", type=float, default=None, help="PMF job timeout in seconds (default: max(60, PMF_JOB_TIMEOUT))")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip full metrics check (faster)")
    args = parser.parse_args()

    ue_url = args.ue_url or config.PMF_UE_URL
    upf_url = args.upf_url or config.PMF_UPF_URL
    use_uplink = config.PMF_USE_UPLINK

    print("PMF Connectivity Check")
    print("=" * 50)
    print(f"PMF_UE_URL:  {ue_url}")
    print(f"PMF_UPF_URL: {upf_url}")
    print(f"PMF_USE_UPLINK: {use_uplink}")
    print()

    ok = True

    print("1. Basic HTTP reachability:")
    if use_uplink:
        if not check_url("PMF UE", ue_url):
            ok = False
    if not check_url("PMF UPF", upf_url):
        ok = False
    print()

    job_label = "PMF UE (uplink)" if use_uplink else "PMF UPF (downlink)"
    print(f"2. RTT job submission ({job_label}):")
    if not check_pmf_job_submit(ue_url, upf_url, use_uplink):
        ok = False
    print()

    if not args.skip_metrics:
        print("3. Full PMF client (polling + cached metrics):")
        if not check_pmf_client_metrics(
            config.SESSION_ID,
            ue_url,
            upf_url,
            wait_s=args.wait_s,
            job_timeout=args.job_timeout,
        ):
            ok = False
        print()

    if ok:
        print("Result: PMF is accessible. Training should work with PMF client.")
        return 0
    else:
        print("Result: PMF check failed. Fix connectivity before training.")
        print("  - Ensure PMF services are running on UE and UPF")
        print("  - Check PMF_UE_URL and PMF_UPF_URL in .env")
        print("  - Check PMF_USE_UPLINK setting (uplink vs downlink mode)")
        print("  - Ensure network allows connections to PMF ports")
        return 1


if __name__ == "__main__":
    sys.exit(main())
