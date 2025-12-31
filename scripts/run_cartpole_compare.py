#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 这个脚本就是：把一堆 `python -m rl_algo.train ...` 跑完，然后把日志汇总成 summary + 图。

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--total-steps", type=int, default=80_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 43])
    parser.add_argument("--algos", type=str, nargs="+", default=["ppo", "grpo", "grpo_scaling", "dqn", "sarsa"])
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="训练使用的设备：'cpu' / 'cuda' / 'auto'（自动选择）。会透传给 `python -m rl_algo.train --device ...`。",
    )
    parser.add_argument("--eval-frequency", type=int, default=0, help="可选：每隔多少 step 做一次评估（0 关闭）。")
    parser.add_argument("--eval-episodes", type=int, default=5, help="每次评估跑多少回合。")
    return parser.parse_args()


def list_model_dirs(models_root: Path) -> set[Path]:
    if not models_root.exists():
        return set()
    return {p for p in models_root.iterdir() if p.is_dir()}


def find_new_run_dir(
    before: set[Path],
    after: set[Path],
    env_id: str,
    algo: str,
    seed: int,
) -> Path:
    prefix = f"{env_id}__{algo}__{seed}__"
    candidates = [p for p in (after - before) if p.name.startswith(prefix)]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # 如果差集没找到（比如你手动动过 models/），那就按前缀在 models 里挑最新的一个
    all_candidates = [p for p in after if p.name.startswith(prefix)]
    if not all_candidates:
        raise FileNotFoundError(f"未找到 run_dir：prefix={prefix}（请检查训练是否成功生成 models/ 目录）")
    return max(all_candidates, key=lambda p: p.stat().st_mtime)


def find_train_log_csv(run_dir: Path) -> Path:
    """
    返回本次 run 的训练日志路径。

    约定：训练脚本默认写 `train_log.csv`。
    但考虑到你可能手动改过文件名，这里做个简单兼容：
    - 优先使用 `train_log__*.csv`（带 env/algo/seed/类别，更清晰）
    - 否则使用 `train_log.csv`
    - 否则在目录下寻找唯一的 `*_train_log.csv`
    """
    named = sorted(run_dir.glob("train_log__*.csv"))
    if len(named) == 1:
        return named[0]

    canonical = run_dir / "train_log.csv"
    if canonical.exists():
        return canonical

    candidates = sorted(run_dir.glob("*_train_log.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"未找到训练日志：{canonical}（且目录下无 *_train_log.csv）")
    raise FileNotFoundError(f"未找到训练日志：{canonical}（目录下存在多个 *_train_log.csv，无法唯一确定）")


def read_train_log(csv_path: Path) -> dict[str, np.ndarray]:
    episode_idx: list[int] = []
    episodic_return: list[float] = []
    episodic_length: list[int] = []
    elapsed_sec: list[float] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_idx.append(int(row["episode_idx"]))
            episodic_return.append(float(row["episodic_return"]))
            episodic_length.append(int(row["episodic_length"]))
            elapsed_sec.append(float(row["elapsed_sec"]))

    return {
        "episode_idx": np.asarray(episode_idx, dtype=np.int32),
        "episodic_return": np.asarray(episodic_return, dtype=np.float32),
        "episodic_length": np.asarray(episodic_length, dtype=np.int32),
        "elapsed_sec": np.asarray(elapsed_sec, dtype=np.float32),
    }


def first_reach_500(returns: np.ndarray) -> int | None:
    hit = np.where(returns >= 500.0)[0]
    if hit.size == 0:
        return None
    return int(hit[0] + 1)  # episode_idx 从 1 开始


def first_reach_ma10_500(returns: np.ndarray) -> int | None:
    if returns.size < 10:
        return None
    # 最近10回合平均（MA10）
    kernel = np.ones(10, dtype=np.float32) / 10.0
    ma10 = np.convolve(returns, kernel, mode="valid")  # 长度 = N-9，对应 episode 10..N
    hit = np.where(ma10 >= 500.0)[0]
    if hit.size == 0:
        return None
    return int(hit[0] + 10)  # valid[0] 对应 episode 10


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models_root = repo_root / "models"
    figs_root = repo_root / "figs"
    results_root = repo_root / "results"
    figs_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    algo_to_seed_to_log: dict[str, dict[int, Path]] = {}
    algo_to_seed_to_run_dir: dict[str, dict[int, Path]] = {}

    for algo in args.algos:
        algo_to_seed_to_log[algo] = {}
        algo_to_seed_to_run_dir[algo] = {}
        for seed in args.seeds:
            print(f"\n=== Running: algo={algo}, seed={seed} ===")
            before = list_model_dirs(models_root)

            cmd = [
                sys.executable,
                "-m",
                "rl_algo.train",
                "--algo",
                algo,
                "--env-id",
                args.env_id,
                "--seed",
                str(seed),
                "--total-steps",
                str(args.total_steps),
                "--device",
                args.device,
            ]
            if int(args.eval_frequency) > 0:
                cmd += ["--eval-frequency", str(int(args.eval_frequency)), "--eval-episodes", str(int(args.eval_episodes))]
            subprocess.run(cmd, check=True, cwd=str(repo_root))

            after = list_model_dirs(models_root)
            run_dir = find_new_run_dir(before, after, args.env_id, algo, seed)
            log_path = find_train_log_csv(run_dir)

            algo_to_seed_to_log[algo][seed] = log_path
            algo_to_seed_to_run_dir[algo][seed] = run_dir
            print(f"[OK] run_dir={run_dir}")

    # ================= 聚合与画图 =================
    import matplotlib.pyplot as plt

    summary: dict[str, Any] = {
        "env_id": args.env_id,
        "total_steps": int(args.total_steps),
        "seeds": [int(s) for s in args.seeds],
        "runs": {},
        "aggregate": {},
        "notes": {
            "reward_500_definition": "CartPole-v1 每步 reward=1，单回合最大 500；这里的 episodic_return 是未折扣 sum(reward)。",
            "first_ep_return_500": "首次出现单回合 episodic_return>=500 的 episode_idx。",
            "first_ep_ma10_500": "首次出现最近10回合平均 episodic_return>=500 的 episode_idx。",
            "alignment": "三种子按 episode_idx 对齐，取最短长度截断。",
        },
    }

    # 图 1：回报曲线
    plt.figure(figsize=(8, 4.5))
    for algo in args.algos:
        runs = [read_train_log(algo_to_seed_to_log[algo][seed]) for seed in args.seeds]
        min_len = min(r["episodic_return"].shape[0] for r in runs)
        returns = np.stack([r["episodic_return"][:min_len] for r in runs], axis=0)
        x = np.arange(1, min_len + 1, dtype=np.int32)
        mean = returns.mean(axis=0)
        std = returns.std(axis=0)

        (line,) = plt.plot(x, mean, label=f"{algo} mean")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())

    plt.title(f"{args.env_id} - Episodic Return vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return (undiscounted)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return_fig_path = figs_root / "cartpole_return_curve.png"
    plt.tight_layout()
    plt.savefig(return_fig_path, dpi=200)
    plt.close()

    # 图 2：耗时曲线
    plt.figure(figsize=(8, 4.5))
    for algo in args.algos:
        runs = [read_train_log(algo_to_seed_to_log[algo][seed]) for seed in args.seeds]
        min_len = min(r["elapsed_sec"].shape[0] for r in runs)
        elapsed = np.stack([r["elapsed_sec"][:min_len] for r in runs], axis=0)
        x = np.arange(1, min_len + 1, dtype=np.int32)
        mean = elapsed.mean(axis=0)
        std = elapsed.std(axis=0)

        (line,) = plt.plot(x, mean, label=f"{algo} mean")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())

    plt.title(f"{args.env_id} - Elapsed Time vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Elapsed Seconds (wall-clock)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    time_fig_path = figs_root / "cartpole_time_curve.png"
    plt.tight_layout()
    plt.savefig(time_fig_path, dpi=200)
    plt.close()

    # ================= 统计指标（每次 run + 三种子聚合）=================
    probe_episodes = [100, 200, 300, 400, 500]
    for algo in args.algos:
        summary["runs"][algo] = {}
        per_seed_first_500: list[int | None] = []
        per_seed_first_ma10_500: list[int | None] = []
        per_seed_elapsed_probe: dict[int, list[float | None]] = {ep: [] for ep in probe_episodes}

        for seed in args.seeds:
            log_path = algo_to_seed_to_log[algo][seed]
            run_dir = algo_to_seed_to_run_dir[algo][seed]
            log = read_train_log(log_path)

            first_500 = first_reach_500(log["episodic_return"])
            first_ma10_500 = first_reach_ma10_500(log["episodic_return"])
            per_seed_first_500.append(first_500)
            per_seed_first_ma10_500.append(first_ma10_500)

            elapsed_at = {}
            for ep in probe_episodes:
                if log["elapsed_sec"].shape[0] >= ep:
                    v = float(log["elapsed_sec"][ep - 1])
                else:
                    v = None
                elapsed_at[str(ep)] = v
                per_seed_elapsed_probe[ep].append(v)

            summary["runs"][algo][str(seed)] = {
                "run_dir": str(run_dir),
                "train_log": str(log_path),
                "first_ep_return_500": first_500,
                "first_ep_ma10_500": first_ma10_500,
                "elapsed_at_episodes": elapsed_at,
            }

        def mean_ignore_none(xs: list[float | None]) -> float | None:
            vals = [x for x in xs if x is not None]
            if not vals:
                return None
            return float(np.mean(vals))

        summary["aggregate"][algo] = {
            "first_ep_return_500_per_seed": {str(s): per_seed_first_500[i] for i, s in enumerate(args.seeds)},
            "first_ep_ma10_500_per_seed": {str(s): per_seed_first_ma10_500[i] for i, s in enumerate(args.seeds)},
            "elapsed_at_episodes_mean": {
                str(ep): mean_ignore_none(per_seed_elapsed_probe[ep]) for ep in probe_episodes
            },
        }

    summary_path = results_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"- return curve: {return_fig_path}")
    print(f"- time curve:   {time_fig_path}")
    print(f"- summary:      {summary_path}")

    # 额外：生成报告所需的“全套图表（>=10，PNG+PDF）”与 `fig_list.md`
    # 说明：这一步只读 results/summary.json 与 models/*/train_log.csv，不会改动训练结果。
    plot_script = repo_root / "scripts" / "plot_report_figures.py"
    if plot_script.exists():
        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--summary",
                str(summary_path),
            ],
            check=True,
            cwd=str(repo_root),
        )


if __name__ == "__main__":
    main()
