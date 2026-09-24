# standalone traffic generator / sink for the RL MPQUIC testbed.
# replaces iperf3 for paths where VRF prevents direct socket bind.


from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
import time
from typing import Optional

CHUNK = 8_000 # 8 kB chunks

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)-7s %(message)s",
#     datefmt="%H:%M:%S",
# )
log = logging.getLogger("tgen")

_stop = threading.Event()

_progress_bytes_sent = 0
_progress_bytes_sent_lock = threading.Lock()


def get_bytes_sent():
    with _progress_bytes_sent_lock:
        return _progress_bytes_sent


def _increment_bytes_sent(value):
    global _progress_bytes_sent
    with _progress_bytes_sent_lock:
        _progress_bytes_sent += value


def reset_bytes_sent():
    global _progress_bytes_sent
    with _progress_bytes_sent_lock:
        _progress_bytes_sent = 0


def _handle_signal(sig: int, _frame) -> None:
    log.info("TGEN: Signal %d received — stopping.", sig)
    _stop.set()


# server (TCP sink)

def _drain(conn: socket.socket, _addr: tuple) -> None:
    try:
        conn.settimeout(1.0)
        while not _stop.is_set():
            try:
                if not conn.recv(CHUNK):
                    break
            except socket.timeout:
                continue
    except OSError:
        pass
    finally:
        conn.close()


def run_server(port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(16)
    srv.settimeout(1.0)
    log.info("Traffic sink listening on 0.0.0.0:%d", port)
    while not _stop.is_set():
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        log.info("Client connected from %s:%d", addr[0], addr[1])
        threading.Thread(target=_drain, args=(conn, addr), daemon=True).start()
    srv.close()
    log.info("Server stopped.")


# client helpers

def _connect(target: str, bind_ip: str, port: int,
             bind_device: str = "", retry_s: float = 3.0) -> socket.socket:
    while not _stop.is_set():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if bind_device:
            # SO_BINDTODEVICE: force traffic out through a specific interface
            # without entering its VRF routing context. Requires CAP_NET_RAW
            # (run with: sudo python3 traffic_gen.py --bind-device IFACE ...).
            _SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
            try:
                sock.setsockopt(socket.SOL_SOCKET, _SO_BINDTODEVICE,
                                bind_device.encode() + b"\0")
            except OSError as e:
                sock.close()
                log.error("SO_BINDTODEVICE(%s) failed: %s — try: sudo python3 traffic_gen.py ...", bind_device, e)
                _stop.wait(retry_s)
                continue
        elif bind_ip:
            try:
                sock.bind((bind_ip, 0))
            except OSError as e:
                sock.close()
                log.error("bind(%s) failed: %s", bind_ip, e)
                _stop.wait(retry_s)
                continue
        try:
            sock.connect((target, port))
            log.info("Connected to %s:%d (iface=%s bind=%s)",
                     target, port, bind_device or "auto", bind_ip or "auto")
            return sock
        except OSError as e:
            sock.close()
            log.warning("Connect failed: %s — retry in %.0fs", e, retry_s)
            _stop.wait(retry_s)
    sys.exit(0)


def _send_paced(sock: socket.socket, rate_mbps: float) -> None:
    # continuous send paced to rate_mbps 
    bps = rate_mbps * 1e6 / 8.0
    payload = b"X" * CHUNK
    t0 = time.monotonic()
    sent = 0
    while not _stop.is_set():
        budget = bps * (time.monotonic() - t0) - sent
        if budget < CHUNK:
            _stop.wait(min((CHUNK - budget) / bps, 0.05))
            continue
        try:
            sent += sock.send(payload)
        except OSError:
            return


def _send_bursty(sock: socket.socket, burst_bytes: int) -> None:
    # send exactly burst_bytes then return 
    payload = b"X" * CHUNK
    sent = 0
    while sent < burst_bytes and not _stop.is_set():
        chunk = min(CHUNK, burst_bytes - sent)
        try:
            sent += sock.send(payload[:chunk])
        except OSError:
            return


# FIFO writer (tunnel application traffic)

import gc
import sys

def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

def _write_paced_fifo(fd, rate_mbps: float, stop_event: threading.Event, chunk_size: int) -> None:
    assert chunk_size > 0
    bps = rate_mbps * 1e6 / 8.0
    payload = b'X' * chunk_size
    t0 = time.monotonic()
    sent = 0
    while not stop_event.is_set():
        budget = bps * (time.monotonic() - t0) - sent
        if budget < chunk_size:
            stop_event.wait(min((chunk_size - budget) / bps, 0.05))
            continue
        try:
            n = fd.write(payload)
            if n:
                sent += n
                _increment_bytes_sent(n)
        except Exception as e:
            log.error("FIFO writer: %s", e)


def _write_bursty_fifo(fd, burst_bytes: int, stop_event: threading.Event, chunk_size: int) -> None:
    assert chunk_size > 0
    payload = b'X' * chunk_size
    sent = 0
    while sent < burst_bytes and not stop_event.is_set():
        chunk = min(chunk_size, burst_bytes - sent)
        try:
            n = fd.write(payload[:chunk])
            if n:
                sent += n
                _increment_bytes_sent(n)
        except Exception as e:
            log.error("FIFO writer: %s", e)


def run_fifo_writer(
    path: str,
    mode: str,
    rate_mbps: float,
    burst_mb: float,
    burst_period_s: float,
    chunk_size: Optional[int] = None,
    stop_event: Optional[threading.Event] = None
) -> None:
    log.info("FIFO writer starting — path=%s mode=%s", path, mode)

    reset_bytes_sent()

    if chunk_size is None:
        chunk_size = CHUNK

    if stop_event is None:
        stop_event = _stop

    while not stop_event.is_set():
        try:
            log.info("Opening FIFO '%s' (waiting for reader)...", path)
            fd = open(path, "wb", buffering=0)  # noqa: WPS515
        except FileNotFoundError:
            log.error("FIFO %s not found — retry in 2s", path)
            stop_event.wait(2.0)
            continue
        except OSError as e:
            log.error("Cannot open FIFO %s: %s — retry in 2s", path, e)
            stop_event.wait(2.0)
            continue

        log.info("FIFO %s open, writing %s traffic", path, mode)
        try:
            if mode == "bursty":
                burst_bytes = max(1, int(burst_mb * 1e6 / 8.0))
                log.info(f"Bursty traffic with {burst_bytes:.2f} bytes, {burst_period_s} s burst period")
                while not stop_event.is_set():
                    t0 = time.monotonic()
                    _write_bursty_fifo(fd, burst_bytes, stop_event, chunk_size)
                    rest = max(0.0, burst_period_s - (time.monotonic() - t0))
                    if rest > 0:
                        stop_event.wait(rest)
            else:
                log.info(f"Periodic traffic with {rate_mbps:.2f} Mbps")
                _write_paced_fifo(fd, rate_mbps, stop_event, chunk_size)
        except OSError as e:
            log.error("FIFO writer: %s — reopening in 2s", e)
        finally:
            try:
                log.info("Trying to close FIFO...")
                fd.close()
            except OSError as e:
                log.error("FIFO writer: %s", e)
                pass

        if not stop_event.is_set():
            stop_event.wait(2.0)

    log.info("FIFO writer stopped.")


# client entry point

def run_client(
    target: str,
    bind_ip: str,
    port: int,
    mode: str,
    rate_mbps: float,
    burst_mb: float,
    burst_period_s: float,
    bind_device: str = "",
) -> None:
    log.info("Starting — mode=%s target=%s:%d iface=%s bind=%s",
             mode, target, port, bind_device or "auto", bind_ip or "auto")
    burst_bytes = max(1, int(burst_mb * 1e6 / 8.0))

    if mode == "bursty":
        while not _stop.is_set():
            sock = _connect(target, bind_ip, port, bind_device)
            t0 = time.monotonic()
            try:
                _send_bursty(sock, burst_bytes)
            except OSError:
                pass
            finally:
                sock.close()
            rest = max(0.0, burst_period_s - (time.monotonic() - t0))
            if rest > 0:
                _stop.wait(rest)
    else:
        while not _stop.is_set():
            sock = _connect(target, bind_ip, port, bind_device)
            try:
                _send_paced(sock, rate_mbps)
            except OSError:
                pass
            finally:
                sock.close()
            if not _stop.is_set():
                log.warning("Connection lost — reconnecting...")

    log.info("Client stopped.")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Traffic generator / sink for the RL MPQUIC testbed",
    )
    ap.add_argument("--server", action="store_true",
                    help="Run as TCP sink server (ANE side)")
    ap.add_argument("--target", default="",
                    help="[client] target IP")
    ap.add_argument("--bind", default="",
                    help="[client] source IP to bind (5G path)")
    ap.add_argument("--bind-device", default="",
                    help="[client] bind to interface by name via SO_BINDTODEVICE, bypassing VRF routing "
                         "(use for WiFi path instead of ip vrf exec); requires CAP_NET_RAW: sudo python3 ...")
    ap.add_argument("--port", type=int, default=5204,
                    help="TCP port (default 5204)")
    ap.add_argument("--mode", choices=["periodic", "bursty"], default="periodic",
                    help="periodic | bursty (default periodic)")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="[periodic] Mbps (default 10)")
    ap.add_argument("--burst-mb", type=float, default=10.0,
                    help="[bursty] Mbits per burst (default 10)")
    ap.add_argument("--burst-period", type=float, default=5.0,
                    help="[bursty] seconds between bursts (default 5)")
    ap.add_argument("--output-fifo", default="",
                    help="Write traffic into a named FIFO instead of a TCP socket "
                         "(tunnel application traffic mode; replaces --server/--target)")
    return ap


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    args = _build_parser().parse_args()
    if args.output_fifo:
        run_fifo_writer(
            path=args.output_fifo,
            mode=args.mode,
            rate_mbps=args.rate,
            burst_mb=args.burst_mb,
            burst_period_s=args.burst_period,
        )
    elif args.server:
        run_server(args.port)
    else:
        if not args.target:
            _build_parser().error("--target is required in client mode")
        run_client(
            target=args.target,
            bind_ip=args.bind,
            port=args.port,
            mode=args.mode,
            rate_mbps=args.rate,
            burst_mb=args.burst_mb,
            burst_period_s=args.burst_period,
            bind_device=args.bind_device,
        )


if __name__ == "__main__":
    main()
