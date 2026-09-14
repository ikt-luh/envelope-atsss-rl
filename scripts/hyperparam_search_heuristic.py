# hyperparameter search for heuristic baselines.

#
# usage:
#   python scripts/hyperparam_search_heuristic.py --steps 100
#   python scripts/hyperparam_search_heuristic.py --steps 50 --alphas 0.9 0.95 0.99

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_agent.heuristic_agent import HEURISTIC_TYPES


def run_eval(
    heuristic: str,
    alpha: float,
    steps: int,
    output_dir: Path,
) -> dict:

    cmd = [
        sys.executable,
        "scripts/eval.py",
        "--agent", "heuristic",
        "--heuristic", heuristic,
        "--heuristic-alpha", str(alpha),
        "--steps", str(steps),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    if result.returncode != 0:
        return {"error": result.stderr or result.stdout, "reward_mean": float("-inf")}

    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {"error": "summary.json not found", "reward_mean": float("-inf")}

    with summary_path.open() as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for heuristic baselines. Reports top 3 alpha per heuristic."
    )
    parser.add_argument("--steps", type=int, default=100, help="Evaluation steps per run (default: 100)")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.9, 0.95, 0.99],
        help="EMA alpha values (default: 0.9 0.95 0.99)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (default: results/hyperparam_search/heuristic)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Report top K configurations per heuristic (default: 3)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else repo_root / "results" / "hyperparam_search" / "heuristic"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hyperparameter search: steps={args.steps}, alphas={args.alphas}")
    print(f"Output base: {base_dir}")
    print()

    all_results: dict[str, list[dict]] = {h: [] for h in HEURISTIC_TYPES}

    for heuristic in HEURISTIC_TYPES:
        for alpha in args.alphas:
            run_dir = base_dir / heuristic / f"alpha_{alpha}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{heuristic}] alpha={alpha} ... ", end="", flush=True)
            summary = run_eval(heuristic, alpha, args.steps, run_dir)

            if "error" in summary:
                print(f"FAILED: {summary.get('error', '')[:80]}")
                all_results[heuristic].append({
                    "alpha": alpha,
                    "reward_mean": float("-inf"),
                    "error": summary.get("error", ""),
                })
            else:
                reward = summary.get("reward_mean")
                if reward is None:
                    reward = float("-inf")
                print(f"reward_mean={reward:.4f}")
                all_results[heuristic].append({
                    "alpha": alpha,
                    "reward_mean": reward,
                    "steps_ok": summary.get("steps_ok"),
                })

    # report top K per heuristic
    print()
    print("=" * 60)
    print("Top {} per heuristic (by reward_mean, higher is better)".format(args.top_k))
    print("=" * 60)

    for heuristic in HEURISTIC_TYPES:
        rows = [r for r in all_results[heuristic] if r.get("reward_mean", float("-inf")) > float("-inf")]
        rows.sort(key=lambda r: r["reward_mean"], reverse=True)
        top = rows[: args.top_k]

        print(f"\n{heuristic}:")
        if not top:
            print("  (no successful runs)")
        else:
            for i, r in enumerate(top, 1):
                print(f"  {i}. alpha={r['alpha']}: reward_mean={r['reward_mean']:.4f}")

    results_path = base_dir / "search_results.json"
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
