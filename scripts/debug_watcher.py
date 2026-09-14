from scenarios.applier import InterfaceStatistics
import time


def format_num_bytes(b) -> str:
    return f"{b / 1000:.2f}"


if __name__ == "__main__":
    iface = "dummy"
    stats_sent = InterfaceStatistics(iface, dry_run=False, statistics_key="tx_bytes")
    stats_received = InterfaceStatistics(iface, dry_run=False, statistics_key="rx_bytes")

    previous_sent = 0
    previous_received = 0
    for a in range(100):
        sent = stats_sent.get()
        received = stats_received.get()
        print(f"{a}s: Sent {format_num_bytes(sent)}kB {format_num_bytes(sent - previous_sent)}kBps, received {format_num_bytes(received)}kB {format_num_bytes(received - previous_received)}kBps")
        previous_sent = sent
        previous_received = received
        time.sleep(1)
