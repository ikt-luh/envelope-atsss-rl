# shared plotting utilities for training and evaluation.


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# consistent colors for wifi and 5g across all visualizations
WIFI_COLOR = "#3498db"
FIVEG_COLOR = "#e74c3c"


def plot_latency(records: List[Dict[str, Any]], out_path: Path, title: str = "Latency Over Time") -> None:
    xs = [r["elapsed_s"] for r in records]
    ys = [r["latency_ms"] for r in records]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Elapsed (s)")
    plt.ylabel("Latency (ms)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_decision_time_over_time(
    records: List[Dict[str, Any]],
    out_path: Path,
    title: str = "Training Decision Time Over Time",
    reset_threshold_ms: float = 1000.0,
) -> None:
    rows = [
        r for r in records
        if r.get("decision_time_ms") is not None and r.get("elapsed_s") is not None
    ]
    if not rows:
        return

    xs = [float(r["elapsed_s"]) for r in rows]
    ys = [float(r["decision_time_ms"]) for r in rows]

    inference_x = [x for x, y in zip(xs, ys) if y < reset_threshold_ms]
    inference_y = [y for y in ys if y < reset_threshold_ms]
    reset_x     = [x for x, y in zip(xs, ys) if y >= reset_threshold_ms]
    reset_y     = [y for y in ys if y >= reset_threshold_ms]

    import statistics
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [1, 2]})

    # Top: episode reset waits
    ax_top = axes[0]
    if reset_x:
        ax_top.bar(reset_x, reset_y, width=8, color="#E57373", alpha=0.8,
                   label="Episode reset (PMF cache wait)")
        ax_top.axhline(statistics.mean(reset_y), color="#B71C1C", linestyle="--",
                       linewidth=1.0, label="mean %.0fms" % statistics.mean(reset_y))
    ax_top.set_ylabel("Reset wait (ms)")
    ax_top.set_title("Episode Reset Wait  (scenario change + PMF cache refill)")
    ax_top.legend(fontsize=8)
    ax_top.grid(alpha=0.3)

    # Bottom: inference decision time
    ax_bot = axes[1]
    if inference_x:
        ax_bot.scatter(inference_x, inference_y, s=1.5, alpha=0.4,
                       color="#1565C0", label="Inference step")
        p50 = statistics.median(inference_y)
        p95 = sorted(inference_y)[int(len(inference_y) * 0.95)]
        ax_bot.axhline(p50, color="#0D47A1", linestyle="--", linewidth=1.0,
                       label="median %.1fms" % p50)
        ax_bot.axhline(p95, color="#1976D2", linestyle=":", linewidth=1.0,
                       label="p95 %.1fms" % p95)
    ax_bot.set_ylabel("Inference time (ms)")
    ax_bot.set_xlabel("Elapsed (s)")
    ax_bot.set_title("Agent Inference Time  (pure neural network decision)")
    ax_bot.legend(fontsize=8)
    ax_bot.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decision_time_histogram(
    records: List[Dict[str, Any]],
    out_path: Path,
    title: str = "Decision Time Distribution",
    reset_threshold_ms: float = 1000.0,
) -> None:
    vals = [
        float(r["decision_time_ms"])
        for r in records
        if r.get("decision_time_ms") is not None
        and float(r["decision_time_ms"]) < reset_threshold_ms
    ]
    if not vals:
        return

    import statistics
    p50 = statistics.median(vals)
    p95 = sorted(vals)[int(len(vals) * 0.95)]
    p99 = sorted(vals)[int(len(vals) * 0.99)]

    # clip x-axis to p99.9 so the distribution is readable
    x_max = min(sorted(vals)[int(len(vals) * 0.999)], reset_threshold_ms - 1)
    display_vals = [v for v in vals if v <= x_max]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(display_vals, bins=min(60, max(10, len(display_vals) // 20)),
            color="#1565C0", edgecolor="white", alpha=0.85, range=(0, x_max))
    ax.axvline(p50, color="#E53935", linestyle="--", linewidth=1.2,
               label="median %.1fms" % p50)
    ax.axvline(p95, color="#FB8C00", linestyle="--", linewidth=1.2,
               label="p95 %.1fms" % p95)
    ax.axvline(p99, color="#43A047", linestyle="--", linewidth=1.2,
               label="p99 %.1fms" % p99)
    ax.set_xlim(0, x_max)
    ax.set_title("%s  (episode resets excluded)" % title)
    ax.set_xlabel("Inference time (ms)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ratio(records: List[Dict[str, Any]], out_path: Path) -> None:
    ok_records = [r for r in records if r.get("ok", True)]
    if not ok_records:
        return
    xs = [r["elapsed_s"] for r in ok_records]
    wifi = [r["wifi_ratio"] for r in ok_records]
    fiveg = [r["fiveg_ratio"] for r in ok_records]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, wifi, label="WiFi ratio", linewidth=1.0, color=WIFI_COLOR)
    plt.plot(xs, fiveg, label="5G ratio", linewidth=1.0, color=FIVEG_COLOR)
    plt.ylim(-0.05, 1.05)
    plt.title("Decision Ratios Over Time")
    plt.xlabel("Elapsed (s)")
    plt.ylabel("Ratio")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_eval_timeline_reward_action(records: List[Dict[str, Any]], out_path: Path) -> None:

    ok = [r for r in records if r.get("ok", True) and r.get("reward") is not None]
    if not ok:
        return

    steps  = list(range(1, len(ok) + 1))
    reward = [float(r["reward"]) for r in ok]
    ratio  = [float(r["wifi_ratio"]) for r in ok if r.get("wifi_ratio") is not None]

    # detect scenario boundaries from scenario_label changes
    boundaries = []
    labels = []
    prev_label = None
    for i, r in enumerate(ok):
        lbl = r.get("scenario_label")
        if lbl and lbl != prev_label:
            boundaries.append(i + 1)
            labels.append(lbl)
            prev_label = lbl

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Panel 1: reward
    ax = axes[0]
    ax.plot(steps, reward, color="#1565C0", linewidth=0.8, alpha=0.7)
    ax.axhline(0, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="Optimal")
    for b in boundaries:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("Reward (per step)")
    ax.set_title("Reward Over Evaluation Steps")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # Panel 2: WiFi ratio
    ax = axes[1]
    if ratio:
        ax.plot(steps[:len(ratio)], ratio, color="#9C27B0", linewidth=0.8, alpha=0.8)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5, label="50/50")
    for i, (b, lbl) in enumerate(zip(boundaries, labels)):
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(b + 0.5, 0.02, lbl, fontsize=5, rotation=90, color="grey",
                va="bottom", ha="left", clip_on=True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("WiFi ratio")
    ax.set_xlabel("Step")
    ax.set_title("Agent Action (WiFi Ratio) Over Evaluation Steps")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Evaluation Timeline", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eval_timeline_conditions_action(records: List[Dict[str, Any]], out_path: Path) -> None:

    ok = [r for r in records if r.get("ok", True) and r.get("reward") is not None]
    if not ok:
        return

    steps  = list(range(1, len(ok) + 1))
    ratio  = [float(r["wifi_ratio"]) for r in ok if r.get("wifi_ratio") is not None]
    wifi_rtt  = [float(r["wifi_rtt_ms"])  if r.get("wifi_rtt_ms")  is not None else None for r in ok]
    fiveg_rtt = [float(r["fiveg_rtt_ms"]) if r.get("fiveg_rtt_ms") is not None else None for r in ok]
    wifi_plr  = [float(r["wifi_plr"]) * 100  if r.get("wifi_plr")  is not None else None for r in ok]
    fiveg_plr = [float(r["fiveg_plr"]) * 100 if r.get("fiveg_plr") is not None else None for r in ok]

    # detect scenario boundaries from scenario_label changes
    boundaries = []
    labels = []
    prev_label = None
    for i, r in enumerate(ok):
        lbl = r.get("scenario_label")
        if lbl and lbl != prev_label:
            boundaries.append(i + 1)
            labels.append(lbl)
            prev_label = lbl

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

    def _plot_series(ax, xs, ys, color, label):
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pairs:
            px, py = zip(*pairs)
            ax.plot(px, py, color=color, linewidth=0.9, alpha=0.8, label=label)

    WIFI = "#2196F3"
    G5   = "#FF5722"

    ax = axes[0]
    _plot_series(ax, steps, wifi_rtt,  WIFI, "WiFi RTT")
    _plot_series(ax, steps, fiveg_rtt, G5,   "5G RTT")
    for b in boundaries:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("RTT (ms)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    _plot_series(ax, steps, wifi_plr,  WIFI, "WiFi PLR")
    _plot_series(ax, steps, fiveg_plr, G5,   "5G PLR")
    for b in boundaries:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("PLR (%)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.0f%%" % v))

    ax = axes[2]
    if ratio:
        ax.plot(steps[:len(ratio)], ratio, color="#9C27B0", linewidth=0.8, alpha=0.8)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5, label="50/50")
    for i, (b, lbl) in enumerate(zip(boundaries, labels)):
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(b + 0.5, 0.02, lbl, fontsize=5, rotation=90, color="grey",
                va="bottom", ha="left", clip_on=True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("WiFi ratio")
    ax.set_xlabel("Step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bytes_sent(records: List[Dict[str, Any]], out_path: Path) -> None:
    ok = [r for r in records if r.get("ok", True)]
    if not ok:
        return

    steps     = list(range(1, len(ok) + 1))
    bytes_sent_fifo  = [float(r["bytes_sent_fifo"])  if r.get("bytes_sent_fifo")  is not None else None for r in ok]
    bytes_sent_wifi = [float(r["bytes_sent_wifi"]) if r.get("bytes_sent_wifi") is not None else None for r in ok]
    bytes_sent_fiveg  = [float(r["bytes_sent_fiveg"])  if r.get("bytes_sent_fiveg")  is not None else None for r in ok]

    boundaries = []
    labels = []
    prev_label = None
    for i, r in enumerate(ok):
        lbl = r.get("scenario_label")
        if lbl and lbl != prev_label:
            boundaries.append(i + 1)
            labels.append(lbl)
            prev_label = lbl

    FIFO_COLOR = "#29CF42"

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

    def _plot_series(ax, xs, ys, color, label):
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pairs:
            px, py = zip(*pairs)
            ax.plot(px, py, color=color, linewidth=0.9, alpha=0.8, label=label)

    ax = axes[0]
    _plot_series(ax, steps, bytes_sent_fifo,  FIFO_COLOR, "MPQUIC input FIFO")
    for b in boundaries:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("Bytes sent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    _plot_series(ax, steps, bytes_sent_wifi,  WIFI_COLOR, "WiFi interface")
    for b in boundaries:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("Bytes sent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    _plot_series(ax, steps, bytes_sent_fiveg,  FIVEG_COLOR, "5G interface")
    for i, (b, lbl) in enumerate(zip(boundaries, labels)):
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(b + 0.5, 0.1, lbl, fontsize=5, rotation=90, color="grey",
                va="bottom", ha="left", clip_on=True, transform=ax.get_xaxis_transform())
    ax.set_ylabel("Bytes sent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax.set_xlabel("Step")

    fig.suptitle("Total bytes sent per episode", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eval_network_conditions(records: List[Dict[str, Any]], out_path: Path) -> None:
    ok = [r for r in records if r.get("ok", True)]
    if not ok:
        return

    steps     = list(range(1, len(ok) + 1))
    wifi_rtt  = [float(r["wifi_rtt_ms"])  if r.get("wifi_rtt_ms")  is not None else None for r in ok]
    fiveg_rtt = [float(r["fiveg_rtt_ms"]) if r.get("fiveg_rtt_ms") is not None else None for r in ok]
    wifi_plr  = [float(r["wifi_plr"]) * 100  if r.get("wifi_plr")  is not None else None for r in ok]
    fiveg_plr = [float(r["fiveg_plr"]) * 100 if r.get("fiveg_plr") is not None else None for r in ok]

    boundaries = []
    prev_label = None
    for i, r in enumerate(ok):
        lbl = r.get("scenario_label")
        if lbl and lbl != prev_label:
            boundaries.append(i + 1)
            prev_label = lbl

    WIFI = "#2196F3"
    G5   = "#FF5722"

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    def _plot_series(ax, xs, ys, color, label):
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pairs:
            px, py = zip(*pairs)
            ax.plot(px, py, color=color, linewidth=0.9, alpha=0.8, label=label)

    ax = axes[0]
    _plot_series(ax, steps, wifi_rtt,  WIFI, "WiFi RTT")
    _plot_series(ax, steps, fiveg_rtt, G5,   "5G RTT")
    for b in boundaries:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("RTT (ms)")
    ax.set_title("Actual Path RTT During Evaluation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    _plot_series(ax, steps, wifi_plr,  WIFI, "WiFi PLR")
    _plot_series(ax, steps, fiveg_plr, G5,   "5G PLR")
    for b in boundaries:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("PLR (%)")
    ax.set_xlabel("Step")
    ax.set_title("Actual Packet Loss Rate During Evaluation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.0f%%" % v))

    fig.suptitle("Network Conditions During Evaluation", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_network_distribution(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Pie chart: WiFi vs 5G traffic split (from agent decisions)."""
    ok = [r for r in records if r.get("ok", True)]
    wifi_ratios = [r["wifi_ratio"] for r in ok if r.get("wifi_ratio") is not None]
    fiveg_ratios = [r["fiveg_ratio"] for r in ok if r.get("fiveg_ratio") is not None]
    if not wifi_ratios and not fiveg_ratios:
        return

    wifi_pct = sum(wifi_ratios) / len(wifi_ratios) * 100 if wifi_ratios else 0
    fiveg_pct = sum(fiveg_ratios) / len(fiveg_ratios) * 100 if fiveg_ratios else 0
    total = wifi_pct + fiveg_pct
    if total == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sizes = [wifi_pct, fiveg_pct]
    labels = ["WiFi", "5G"]
    colors = [WIFI_COLOR, FIVEG_COLOR]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1),
    )
    ax.set_title("Network Distribution (WiFi vs 5G)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


