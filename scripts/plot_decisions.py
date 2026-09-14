# rl decision visualizer

# Usage:
#   python scripts/plot_decisions.py                     # latest log in logs/
#   python scripts/plot_decisions.py logs/decisions_20260209.csv
#   python scripts/plot_decisions.py logs/decisions_*.csv --output plots/

import sys
import os
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"])
    df["elapsed_s"] = (df["timestamp_us"] - df["timestamp_us"].iloc[0]) / 1e6
    return df


def find_latest_log(log_dir: str = "logs") -> str:
    files = sorted(glob.glob(os.path.join(log_dir, "decisions_*.csv")))
    if not files:
        print(f"No decision logs found in {log_dir}/")
        sys.exit(1)
    return files[-1]


# plot 1: ratio over time

def plot_ratio_over_time(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df["elapsed_s"], df["wifi_ratio"], label="WiFi ratio", color="#2196F3", linewidth=1.2)
    ax.plot(df["elapsed_s"], df["fiveg_ratio"], label="5G ratio", color="#FF5722", linewidth=1.2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="50/50 baseline")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Traffic ratio")
    ax.set_title("Traffic Split Ratio Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "ratio_over_time.png", dpi=150)
    plt.close(fig)
    print(f"  [1/5] ratio_over_time.png")


# plot 2: network metrics over time

def plot_network_metrics(df: pd.DataFrame, output_dir: Path):
    fig, (ax_rtt, ax_plr) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # rtt
    ax_rtt.plot(df["elapsed_s"], df["wifi_rtt_ms"], label="WiFi RTT", color="#2196F3", linewidth=1.0, alpha=0.8)
    ax_rtt.plot(df["elapsed_s"], df["fiveg_rtt_ms"], label="5G RTT", color="#FF5722", linewidth=1.0, alpha=0.8)
    ax_rtt.set_ylabel("RTT (ms)")
    ax_rtt.set_title("Network Metrics Over Time")
    ax_rtt.legend(loc="upper right")
    ax_rtt.grid(True, alpha=0.3)

    # plr
    ax_plr.plot(df["elapsed_s"], df["wifi_plr"], label="WiFi PLR", color="#2196F3", linewidth=1.0, alpha=0.8)
    ax_plr.plot(df["elapsed_s"], df["fiveg_plr"], label="5G PLR", color="#FF5722", linewidth=1.0, alpha=0.8)
    ax_plr.set_xlabel("Time (s)")
    ax_plr.set_ylabel("Packet Loss Rate")
    ax_plr.legend(loc="upper right")
    ax_plr.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "network_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  [2/5] network_metrics.png")


# plot 3: ratio vs network state

def plot_ratio_vs_network(df: pd.DataFrame, output_dir: Path):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_ratio = "#4CAF50"
    ax1.fill_between(df["elapsed_s"], 0, df["wifi_ratio"], alpha=0.3, color="#2196F3", label="WiFi ratio")
    ax1.fill_between(df["elapsed_s"], df["wifi_ratio"], 1, alpha=0.3, color="#FF5722", label="5G ratio")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Traffic ratio")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(df["elapsed_s"], df["wifi_rtt_ms"], label="WiFi RTT", color="#1565C0",
             linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.plot(df["elapsed_s"], df["fiveg_rtt_ms"], label="5G RTT", color="#BF360C",
             linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.set_ylabel("RTT (ms)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Agent Decision vs Network Conditions")
    ax1.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "ratio_vs_network.png", dpi=150)
    plt.close(fig)
    print(f"  [3/5] ratio_vs_network.png")


# plot 4: action distribution histogram

def plot_action_distribution(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["wifi_ratio"], bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(x=df["wifi_ratio"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {df['wifi_ratio'].mean():.3f}")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="50/50 baseline")

    ax.set_xlabel("WiFi ratio")
    ax.set_ylabel("Frequency")
    ax.set_title("Action Distribution (WiFi Ratio)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "action_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  [4/5] action_distribution.png")


# plot 5: decision stability (ratio change per step)

def plot_stability(df: pd.DataFrame, output_dir: Path):
    ratio_change = df["wifi_ratio"].diff().abs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # time series of changes
    ax1.plot(df["elapsed_s"].iloc[1:], ratio_change.iloc[1:], color="#9C27B0", linewidth=0.8, alpha=0.7)
    ax1.axhline(y=ratio_change.iloc[1:].mean(), color="red", linestyle="--",
                label=f"Mean change: {ratio_change.iloc[1:].mean():.4f}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("|ratio change|")
    ax1.set_title("Decision Stability Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # histogram of changes
    ax2.hist(ratio_change.iloc[1:], bins=50, color="#9C27B0", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("|ratio change| per step")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Stability Distribution")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "decision_stability.png", dpi=150)
    plt.close(fig)
    print(f"  [5/5] decision_stability.png")


# main

def main():
    parser = argparse.ArgumentParser(description="Plot RL decision logs")
    parser.add_argument("csv_file", nargs="?", default=None, help="Path to decision CSV (default: latest in logs/)")
    parser.add_argument("--output", "-o", default="plots", help="Output directory for plots (default: plots/)")
    args = parser.parse_args()

    csv_path = args.csv_file or find_latest_log()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path}")
    df = load_data(csv_path)
    print(f"  {len(df)} decisions, {df['elapsed_s'].iloc[-1]:.0f}s duration")
    print(f"Generating plots in {output_dir}/:")

    plot_ratio_over_time(df, output_dir)
    plot_network_metrics(df, output_dir)
    plot_ratio_vs_network(df, output_dir)
    plot_action_distribution(df, output_dir)
    plot_stability(df, output_dir)

    print(f"\nDone. All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
