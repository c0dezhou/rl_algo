#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 出图脚本：读 results/summary.json + models/*/train_log.csv，然后把报告里要用的图都画出来。

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/summary.json")
    parser.add_argument("--figs-dir", type=str, default="figs")
    parser.add_argument("--fig-list-out", type=str, default="fig_list.md")
    parser.add_argument("--formats", type=str, default="png,pdf", help="逗号分隔，例如 png,pdf")
    return parser.parse_args()


def set_matplotlib_chinese_style() -> None:
    import matplotlib as mpl
    from matplotlib import font_manager

    # 尽量选择常见中文字体；若系统缺失会自动 fallback（但中文可能显示为方框）。
    preferred = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "AR PL UMing CN",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in preferred if name in installed]
    mpl.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False


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


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    if x.size < window:
        return np.full_like(x, np.nan, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid").astype(np.float32)


def first_reach_threshold(returns: np.ndarray, threshold: float = 500.0) -> int | None:
    hit = np.where(returns >= threshold)[0]
    if hit.size == 0:
        return None
    return int(hit[0] + 1)


def first_reach_ma_threshold(returns: np.ndarray, window: int, threshold: float = 500.0) -> int | None:
    ma = moving_average(returns, window=window)
    if ma.size == 0 or np.all(np.isnan(ma)):
        return None
    hit = np.where(ma >= threshold)[0]
    if hit.size == 0:
        return None
    # valid[0] 对应 episode=window
    return int(hit[0] + window)


def elapsed_at_episode(elapsed_sec: np.ndarray, episode_idx: int) -> float | None:
    if episode_idx <= 0:
        return None
    if elapsed_sec.size < episode_idx:
        return None
    return float(elapsed_sec[episode_idx - 1])


def cumulative_steps(episodic_length: np.ndarray) -> np.ndarray:
    """把逐回合长度累加成“累计交互步数”（cum_steps），用于 steps 口径的曲线/指标。"""
    if episodic_length.size == 0:
        return episodic_length.astype(np.int32)
    return np.cumsum(episodic_length, dtype=np.int64).astype(np.int64)


def episode_delta_elapsed(elapsed_sec: np.ndarray) -> np.ndarray:
    """每个回合的耗时增量 dt（秒）。dt[i] = elapsed[i] - elapsed[i-1]。"""
    if elapsed_sec.size == 0:
        return elapsed_sec.astype(np.float32)
    dt = np.empty_like(elapsed_sec, dtype=np.float32)
    dt[0] = float(elapsed_sec[0])
    if elapsed_sec.size > 1:
        dt[1:] = np.diff(elapsed_sec).astype(np.float32)
    # 避免出现 0/负数导致吞吐计算异常（日志时间应该单调递增；不递增就置为 NaN）
    dt[dt <= 0] = np.nan
    return dt


def interp_to_grid(x: np.ndarray, y: np.ndarray, grid_x: np.ndarray) -> np.ndarray:
    """把 (x,y) 线性插值到统一的 grid_x 上（用于跨 seeds 对齐并求均值±标准差）。"""
    if x.size == 0 or y.size == 0 or grid_x.size == 0:
        return np.full_like(grid_x, np.nan, dtype=np.float32)
    # np.interp 需要 x 单调递增；cum_steps 天然递增
    return np.interp(grid_x, x.astype(np.float64), y.astype(np.float64)).astype(np.float32)


def read_target_episodes(repo_root: Path, algo: str) -> int:
    """
    从默认配置里读取“一次 update 收集多少条完整轨迹”：
    - PPO：batch_episodes
    - GRPO：group_size（若缺失则退化为 batch_episodes）
    """
    cfg_path = repo_root / "rl_algo" / "config" / f"{algo}.yaml"
    if not cfg_path.exists():
        return 8
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if algo in ("grpo", "grpo_scaling"):
        return int(cfg.get("group_size") or cfg.get("batch_episodes") or 8)
    return int(cfg.get("batch_episodes") or 8)


def save_figure(fig: Any, out_stem: Path, formats: list[str], dpi: int = 200) -> list[Path]:
    out_paths: list[Path] = []
    for ext in formats:
        ext = ext.strip().lower()
        if not ext:
            continue
        out_path = out_stem.with_suffix(f".{ext}")
        if ext in {"png"}:
            fig.savefig(out_path, dpi=dpi)
        else:
            fig.savefig(out_path)
        out_paths.append(out_path)
    return out_paths


@dataclass
class FigureItem:
    fig_no: str
    title_zh: str
    x_label: str
    y_label: str
    data_sources: list[str]
    command: str
    outputs: list[str]


def write_fig_list(path: Path, items: list[FigureItem]) -> None:
    lines: list[str] = []
    lines.append("# 图表清单（自动生成）")
    lines.append("")
    lines.append("说明：每张图均保存为 PNG+PDF（便于 Word/矢量图要求）。")
    lines.append("")
    for it in items:
        lines.append(f"## {it.fig_no} {it.title_zh}")
        lines.append(f"- 横坐标：{it.x_label}")
        lines.append(f"- 纵坐标：{it.y_label}")
        lines.append(f"- 数据来源：")
        for s in it.data_sources:
            lines.append(f"  - {s}")
        lines.append(f"- 生成命令：`{it.command}`")
        lines.append(f"- 输出文件：")
        for o in it.outputs:
            lines.append(f"  - `{o}`")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    summary_path = (repo_root / args.summary).resolve()
    figs_dir = (repo_root / args.figs_dir).resolve()
    fig_list_out = (repo_root / args.fig_list_out).resolve()
    figs_dir.mkdir(parents=True, exist_ok=True)

    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    set_matplotlib_chinese_style()

    fig_items: list[FigureItem] = []
    cmd_str = f"python scripts/plot_report_figures.py --summary {summary_path.relative_to(repo_root)}"

    # ========== 实验数据图（依赖 results/summary.json） ==========
    if not summary_path.exists():
        raise FileNotFoundError(
            f"未找到 {summary_path}。请先运行：python scripts/run_cartpole_compare.py --total-steps 200000"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    env_id = summary.get("env_id", "CartPole-v1")
    seeds = [int(x) for x in (summary.get("seeds") or [])]
    runs = summary.get("runs") or {}
    algos = list(runs.keys())

    algo_seed_logs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for algo in algos:
        algo_seed_logs[algo] = {}
        for seed_str, run_info in (runs.get(algo) or {}).items():
            seed = int(seed_str)
            log_path = Path(run_info["train_log"])
            if not log_path.exists():
                # 不直接崩溃：允许用户先生成结构图；实验图会基于“能找到的日志”尽量输出。
                # 同时在报告中会提示：要满足 3 seeds，需要先重新跑对比脚本。
                print(f"[WARN] 未找到 train_log，跳过该 run：{log_path}")
                continue
            algo_seed_logs[algo][seed] = read_train_log(log_path)

    import matplotlib.pyplot as plt

    cmap_algos = plt.get_cmap("tab10")
    algo_list = sorted(algos)
    algo_color = {a: cmap_algos(i % 10) for i, a in enumerate(algo_list)}

    # 图10：平均回报曲线（mean±std）
    plt.figure(figsize=(8, 4.5))
    for algo in algos:
        logs = [algo_seed_logs[algo][s] for s in seeds if s in algo_seed_logs.get(algo, {})]
        if not logs:
            continue
        min_len = min(l["episodic_return"].shape[0] for l in logs)
        ret = np.stack([l["episodic_return"][:min_len] for l in logs], axis=0)
        x = np.arange(1, min_len + 1, dtype=np.int32)
        mean = ret.mean(axis=0)
        std = ret.std(axis=0)
        plt.plot(x, mean, label=f"{algo.upper()} 平均", color=algo_color.get(algo, None))
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=algo_color.get(algo, None))

    plt.title(f"{env_id}：平均回合回报曲线（3 种子，均值±标准差）")
    plt.xlabel("回合（Episode）")
    plt.ylabel("回合回报（未折扣）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig10 = plt.gcf()
    fig10.tight_layout()
    out10 = save_figure(fig10, figs_dir / "fig10_cartpole_return_mean_std", formats=formats)
    plt.close(fig10)
    fig_items.append(
        FigureItem(
            fig_no="图10",
            title_zh="CartPole-v1 平均回合回报曲线（3 种子，均值±标准差）",
            x_label="回合（Episode）",
            y_label="回合回报（未折扣）",
            data_sources=["results/summary.json", "models/*/train_log.csv"],
            command=cmd_str,
            outputs=[str(p.relative_to(repo_root)) for p in out10],
        )
    )

    # 图11：平均耗时曲线（mean±std）
    plt.figure(figsize=(8, 4.5))
    for algo in algos:
        logs = [algo_seed_logs[algo][s] for s in seeds if s in algo_seed_logs.get(algo, {})]
        if not logs:
            continue
        min_len = min(l["elapsed_sec"].shape[0] for l in logs)
        t = np.stack([l["elapsed_sec"][:min_len] for l in logs], axis=0)
        x = np.arange(1, min_len + 1, dtype=np.int32)
        mean = t.mean(axis=0)
        std = t.std(axis=0)
        plt.plot(x, mean, label=f"{algo.upper()} 平均", color=algo_color.get(algo, None))
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=algo_color.get(algo, None))

    plt.title(f"{env_id}：平均墙钟时间曲线（3 种子，均值±标准差）")
    plt.xlabel("回合（Episode）")
    plt.ylabel("累计墙钟时间（秒）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig11 = plt.gcf()
    fig11.tight_layout()
    out11 = save_figure(fig11, figs_dir / "fig11_cartpole_time_mean_std", formats=formats)
    plt.close(fig11)
    fig_items.append(
        FigureItem(
            fig_no="图11",
            title_zh="CartPole-v1 平均墙钟时间曲线（3 种子，均值±标准差）",
            x_label="回合（Episode）",
            y_label="累计墙钟时间（秒）",
            data_sources=["results/summary.json", "models/*/train_log.csv"],
            command=cmd_str,
            outputs=[str(p.relative_to(repo_root)) for p in out11],
        )
    )

    # 图12：单种子回报曲线（拆成两张：PPO 3 seeds / GRPO 3 seeds）
    # 原来 6 条线叠在一起较乱，这里拆开后：
    # - 每个 seed 使用不同颜色（更直观看随机性）
    # - 每张图只展示一种算法（避免视觉拥挤，便于写报告分析）
    import matplotlib.pyplot as plt  # 已在其他图中使用，这里显式导入便于阅读

    seed_list = sorted([int(s) for s in seeds])
    cmap = plt.get_cmap("tab10")
    seed_color = {s: cmap(i % 10) for i, s in enumerate(seed_list)}

    def plot_single_algo_per_seed(*, algo: str, title_suffix: str, out_name: str, fig_no: str) -> None:
        if algo not in algos:
            return
        if not algo_seed_logs.get(algo, {}):
            return

        plt.figure(figsize=(8, 4.5))
        for seed in seed_list:
            if seed not in algo_seed_logs.get(algo, {}):
                continue
            log = algo_seed_logs[algo][seed]
            plt.plot(
                log["episode_idx"],
                log["episodic_return"],
                linewidth=1.2,
                label=f"seed={seed}",
                color=seed_color.get(seed, None),
                alpha=0.9,
            )
        plt.title(f"{env_id}：{title_suffix}（3 条）")
        plt.xlabel("回合（Episode）")
        plt.ylabel("回合回报（未折扣）")
        plt.legend(ncol=3, fontsize=9)
        plt.grid(True, alpha=0.3)
        fig = plt.gcf()
        fig.tight_layout()
        out_paths = save_figure(fig, figs_dir / out_name, formats=formats)
        plt.close(fig)
        fig_items.append(
            FigureItem(
                fig_no=fig_no,
                title_zh=f"{title_suffix}（3 seeds）",
                x_label="回合（Episode）",
                y_label="回合回报（未折扣）",
                data_sources=["models/*/train_log.csv"],
                command=cmd_str,
                outputs=[str(p.relative_to(repo_root)) for p in out_paths],
            )
        )

    plot_single_algo_per_seed(algo="ppo", title_suffix="PPO 单种子回合回报曲线", out_name="fig12a_return_per_seed_ppo", fig_no="图12(a)")
    plot_single_algo_per_seed(algo="grpo", title_suffix="GRPO 单种子回合回报曲线", out_name="fig12b_return_per_seed_grpo", fig_no="图12(b)")
    plot_single_algo_per_seed(
        algo="grpo_scaling",
        title_suffix="GRPO-Scaling 单种子回合回报曲线",
        out_name="fig12c_return_per_seed_grpo_scaling",
        fig_no="图12(c)",
    )

    # 图13：回报 MA10/MA20（按“3 种子平均回报”再做滑动平均）
    plt.figure(figsize=(8, 4.5))
    for algo in algos:
        logs = [algo_seed_logs[algo][s] for s in seeds if s in algo_seed_logs.get(algo, {})]
        if not logs:
            continue
        min_len = min(l["episodic_return"].shape[0] for l in logs)
        ret = np.stack([l["episodic_return"][:min_len] for l in logs], axis=0).mean(axis=0)
        ma10 = moving_average(ret, window=10)
        ma20 = moving_average(ret, window=20)
        x10 = np.arange(10, 10 + ma10.size, dtype=np.int32)
        x20 = np.arange(20, 20 + ma20.size, dtype=np.int32)
        plt.plot(x10, ma10, label=f"{algo.upper()} MA10", color=algo_color.get(algo, None), linewidth=2)
        plt.plot(x20, ma20, label=f"{algo.upper()} MA20", color=algo_color.get(algo, None), linewidth=2, linestyle="--")

    plt.title(f"{env_id}：平均回报的滑动平均曲线（MA10 / MA20）")
    plt.xlabel("回合（Episode）")
    plt.ylabel("回合回报滑动平均（未折扣）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig13 = plt.gcf()
    fig13.tight_layout()
    out13 = save_figure(fig13, figs_dir / "fig13_return_moving_average", formats=formats)
    plt.close(fig13)
    fig_items.append(
        FigureItem(
            fig_no="图13",
            title_zh="平均回报的滑动平均曲线（MA10 / MA20）",
            x_label="回合（Episode）",
            y_label="回合回报滑动平均（未折扣）",
            data_sources=["models/*/train_log.csv"],
            command=cmd_str,
            outputs=[str(p.relative_to(repo_root)) for p in out13],
        )
    )

    # 图14：回报标准差随回合变化（按 3 seeds）
    plt.figure(figsize=(8, 4.5))
    for algo in algos:
        logs = [algo_seed_logs[algo][s] for s in seeds if s in algo_seed_logs.get(algo, {})]
        if not logs:
            continue
        min_len = min(l["episodic_return"].shape[0] for l in logs)
        ret = np.stack([l["episodic_return"][:min_len] for l in logs], axis=0)
        x = np.arange(1, min_len + 1, dtype=np.int32)
        std = ret.std(axis=0)
        plt.plot(x, std, label=f"{algo.upper()} 标准差", color=algo_color.get(algo, None))

    plt.title(f"{env_id}：回报标准差随回合变化（3 种子）")
    plt.xlabel("回合（Episode）")
    plt.ylabel("回报标准差")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig14 = plt.gcf()
    fig14.tight_layout()
    out14 = save_figure(fig14, figs_dir / "fig14_return_std_over_episode", formats=formats)
    plt.close(fig14)
    fig_items.append(
        FigureItem(
            fig_no="图14",
            title_zh="回报标准差随回合变化（3 种子）",
            x_label="回合（Episode）",
            y_label="回报标准差",
            data_sources=["models/*/train_log.csv"],
            command=cmd_str,
            outputs=[str(p.relative_to(repo_root)) for p in out14],
        )
    )

    # 图15：episodes-to-500（柱状图；未达到用灰色并标注）
    plt.figure(figsize=(8, 4.5))
    bar_w = 0.35
    x = np.arange(len(seeds), dtype=np.float32)
    for i, algo in enumerate(algos):
        vals: list[float] = []
        labels: list[str] = []
        for seed in seeds:
            log = algo_seed_logs.get(algo, {}).get(seed)
            if log is None:
                vals.append(0.0)
                labels.append("缺失")
                continue
            ep = first_reach_threshold(log["episodic_return"], threshold=500.0)
            if ep is None:
                vals.append(0.0)
                labels.append("未达到")
            else:
                vals.append(float(ep))
                labels.append("")
        bars = plt.bar(x + (i - 0.5) * bar_w, vals, width=bar_w, label=algo.upper(), color=algo_color.get(algo, None), alpha=0.85)
        for j, b in enumerate(bars):
            if labels[j]:
                b.set_color("lightgray")
                plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, labels[j], ha="center", va="bottom", fontsize=8)

    plt.title(f"{env_id}：达到回报=500 的回合数（首次单回合）")
    plt.xlabel("随机种子")
    plt.ylabel("首次达到 500 的回合号（Episode）")
    plt.xticks(x, [str(s) for s in seeds])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    fig15 = plt.gcf()
    fig15.tight_layout()
    out15 = save_figure(fig15, figs_dir / "fig15_episodes_to_500_bar", formats=formats)
    plt.close(fig15)
    fig_items.append(
        FigureItem(
            fig_no="图15",
            title_zh="达到回报=500 的回合数（首次单回合）",
            x_label="随机种子",
            y_label="首次达到 500 的回合号（Episode）",
            data_sources=["models/*/train_log.csv"],
            command=cmd_str,
            outputs=[str(p.relative_to(repo_root)) for p in out15],
        )
    )

    # 图16：time-to-500（柱状图；未达到用灰色并标注）
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(seeds), dtype=np.float32)
    for i, algo in enumerate(algos):
        vals: list[float] = []
        labels: list[str] = []
        for seed in seeds:
            log = algo_seed_logs.get(algo, {}).get(seed)
            if log is None:
                vals.append(0.0)
                labels.append("缺失")
                continue
            ep = first_reach_threshold(log["episodic_return"], threshold=500.0)
            if ep is None:
                vals.append(0.0)
                labels.append("未达到")
            else:
                t = elapsed_at_episode(log["elapsed_sec"], ep)
                vals.append(float(t) if t is not None else 0.0)
                labels.append("")
        bars = plt.bar(x + (i - 0.5) * bar_w, vals, width=bar_w, label=algo.upper(), color=algo_color.get(algo, None), alpha=0.85)
        for j, b in enumerate(bars):
            if labels[j]:
                b.set_color("lightgray")
                plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, labels[j], ha="center", va="bottom", fontsize=8)

    plt.title(f"{env_id}：达到回报=500 的墙钟时间（首次单回合）")
    plt.xlabel("随机种子")
    plt.ylabel("首次达到 500 时累计墙钟时间（秒）")
    plt.xticks(x, [str(s) for s in seeds])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    fig16 = plt.gcf()
    fig16.tight_layout()
    out16 = save_figure(fig16, figs_dir / "fig16_time_to_500_bar", formats=formats)
    plt.close(fig16)
    fig_items.append(
        FigureItem(
            fig_no="图16",
            title_zh="达到回报=500 的墙钟时间（首次单回合）",
            x_label="随机种子",
            y_label="首次达到 500 时累计墙钟时间（秒）",
            data_sources=["models/*/train_log.csv"],
            command=cmd_str,
            outputs=[str(p.relative_to(repo_root)) for p in out16],
        )
    )

    # 图17：以 steps 为横轴的平均回报曲线（3 seeds，均值±标准差）
    # 说明：按 episode 对齐会受到“回合越长→同样 episode 数消耗更多 steps”的影响；
    # steps 口径能更公平地体现“同等交互预算下”的样本效率。
    plt.figure(figsize=(8, 4.5))
    # 选取一个统一的 step 网格（跨算法+跨种子取最短，避免插值超界）
    all_total_steps: list[int] = []
    for algo in algos:
        for seed in seeds:
            log = algo_seed_logs.get(algo, {}).get(seed)
            if log is None:
                continue
            cs = cumulative_steps(log["episodic_length"])
            if cs.size:
                all_total_steps.append(int(cs[-1]))
    min_steps = int(min(all_total_steps)) if all_total_steps else 0
    step_interval = 1000
    grid_steps = np.arange(step_interval, min_steps + 1, step_interval, dtype=np.int64)

    for algo in algos:
        seed_series: list[np.ndarray] = []
        for seed in seeds:
            log = algo_seed_logs.get(algo, {}).get(seed)
            if log is None:
                continue
            cs = cumulative_steps(log["episodic_length"])
            y = log["episodic_return"]
            seed_series.append(interp_to_grid(cs, y, grid_steps))
        if not seed_series:
            continue
        mat = np.stack(seed_series, axis=0)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        plt.plot(grid_steps, mean, label=f"{algo.upper()} 均值", color=algo_color.get(algo, None), linewidth=2)
        plt.fill_between(grid_steps, mean - std, mean + std, alpha=0.2, color=algo_color.get(algo, None))

    plt.title(f"{env_id}：平均回合回报曲线（按 steps 对齐，3 种子，均值±标准差）")
    plt.xlabel("累计环境交互步数（Steps）")
    plt.ylabel("回合回报（未折扣）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig17 = plt.gcf()
    fig17.tight_layout()
    out17 = save_figure(fig17, figs_dir / "fig17_return_mean_std_vs_steps", formats=formats)
    plt.close(fig17)
    fig_items.append(
        FigureItem(
            fig_no="图17",
            title_zh="平均回合回报曲线（steps 对齐，3 种子，均值±标准差）",
            x_label="累计环境交互步数（Steps）",
            y_label="回合回报（未折扣）",
            data_sources=["models/*/train_log.csv"],
            command=cmd_str,
            outputs=[str(p.relative_to(repo_root)) for p in out17],
        )
    )

    # 图18(a)/(b)：按 steps 展示单种子曲线（每个 seed 一条）
    def plot_seed_curves_vs_steps(*, algo: str, fig_no: str, title_zh: str, out_name: str) -> None:
        if algo not in algos:
            return
        plt.figure(figsize=(8, 4.5))
        for seed in seeds:
            log = algo_seed_logs.get(algo, {}).get(seed)
            if log is None:
                continue
            cs = cumulative_steps(log["episodic_length"])
            plt.plot(cs, log["episodic_return"], linewidth=1.0, label=f"seed={seed}")
        plt.title(f"{env_id}：{title_zh}")
        plt.xlabel("累计环境交互步数（Steps）")
        plt.ylabel("回合回报（未折扣）")
        plt.legend(ncol=3, fontsize=9)
        plt.grid(True, alpha=0.3)
        fig = plt.gcf()
        fig.tight_layout()
        out_paths = save_figure(fig, figs_dir / out_name, formats=formats)
        plt.close(fig)
        fig_items.append(
            FigureItem(
                fig_no=fig_no,
                title_zh=title_zh,
                x_label="累计环境交互步数（Steps）",
                y_label="回合回报（未折扣）",
                data_sources=["models/*/train_log.csv"],
                command=cmd_str,
                outputs=[str(p.relative_to(repo_root)) for p in out_paths],
            )
        )

    plot_seed_curves_vs_steps(algo="ppo", fig_no="图18(a)", title_zh="PPO 单种子回合回报曲线（steps 横轴，3 seeds）", out_name="fig18a_ppo_seed_curves_vs_steps")
    plot_seed_curves_vs_steps(algo="grpo", fig_no="图18(b)", title_zh="GRPO 单种子回合回报曲线（steps 横轴，3 seeds）", out_name="fig18b_grpo_seed_curves_vs_steps")
    plot_seed_curves_vs_steps(
        algo="grpo_scaling",
        fig_no="图18(c)",
        title_zh="GRPO-Scaling 单种子回合回报曲线（steps 横轴，3 seeds）",
        out_name="fig18c_grpo_scaling_seed_curves_vs_steps",
    )

    # 图19：后期“变慢”的量化（update 触发回合的额外停顿，按训练阶段分段）
    # 估计方法：dt_update - dt_non_update，其中 dt 为每回合 elapsed_sec 的增量。
    # 分段以 cum_steps 为基准（更贴近“训练到多少步以后变慢”）。
    total_steps_cfg = int(summary.get("total_steps") or 0)
    stage_edges = [0, min(20000, total_steps_cfg), min(50000, total_steps_cfg), total_steps_cfg]
    stage_edges = [x for i, x in enumerate(stage_edges) if i == 0 or x > stage_edges[i - 1]]
    stage_names = []
    for i in range(len(stage_edges) - 1):
        stage_names.append(f"{stage_edges[i]//1000}-{stage_edges[i+1]//1000}k")

    def update_extra_time_by_stage(*, algo: str) -> tuple[np.ndarray, np.ndarray]:
        target_ep = read_target_episodes(repo_root, algo)
        per_seed: list[np.ndarray] = []
        for seed in seeds:
            log = algo_seed_logs.get(algo, {}).get(seed)
            if log is None:
                continue
            cs = cumulative_steps(log["episodic_length"]).astype(np.int64)
            dt = episode_delta_elapsed(log["elapsed_sec"])
            ep_idx = log["episode_idx"].astype(np.int32)
            is_update_ep = (ep_idx % target_ep) == 0

            stage_vals: list[float] = []
            for i in range(len(stage_edges) - 1):
                lo, hi = stage_edges[i], stage_edges[i + 1]
                in_stage = (cs >= lo) & (cs < hi)
                upd = dt[in_stage & is_update_ep]
                non = dt[in_stage & (~is_update_ep)]
                if upd.size == 0 or non.size == 0:
                    stage_vals.append(np.nan)
                else:
                    stage_vals.append(float(np.nanmean(upd) - np.nanmean(non)))
            per_seed.append(np.asarray(stage_vals, dtype=np.float32))
        if not per_seed:
            return np.full((len(stage_edges) - 1,), np.nan, dtype=np.float32), np.full((len(stage_edges) - 1,), np.nan, dtype=np.float32)
        mat = np.stack(per_seed, axis=0)
        return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

    if total_steps_cfg > 0 and len(stage_names) >= 1:
        plt.figure(figsize=(8, 4.5))
        x = np.arange(len(stage_names), dtype=np.float32)
        bar_w = 0.35
        for i, algo in enumerate(algos):
            mean, std = update_extra_time_by_stage(algo=algo)
            plt.bar(
                x + (i - 0.5) * bar_w,
                mean,
                width=bar_w,
                yerr=std,
                capsize=3,
                label=algo.upper(),
                color=algo_color.get(algo, None),
                alpha=0.85,
            )
        plt.title(f"{env_id}：update 额外停顿随训练阶段变化（dt_update - dt_non_update）")
        plt.xlabel("训练阶段（按累计 steps 分段）")
        plt.ylabel("每次 update 的额外停顿（秒/回合，均值±标准差）")
        plt.xticks(x, stage_names)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        fig19 = plt.gcf()
        fig19.tight_layout()
        out19 = save_figure(fig19, figs_dir / "fig19_update_extra_pause_by_stage", formats=formats)
        plt.close(fig19)
        fig_items.append(
            FigureItem(
                fig_no="图19",
                title_zh="update 额外停顿随训练阶段变化（dt_update - dt_non_update）",
                x_label="训练阶段（按累计 steps 分段）",
                y_label="每次 update 的额外停顿（秒/回合，均值±标准差）",
                data_sources=["models/*/train_log.csv", "rl_algo/config/*.yaml（batch_episodes/group_size）"],
                command=cmd_str,
                outputs=[str(p.relative_to(repo_root)) for p in out19],
            )
        )

    # 图20：吞吐（steps/sec）随训练进程变化（按 steps 对齐，均值±标准差）
    plt.figure(figsize=(8, 4.5))
    for algo in algos:
        seed_series: list[np.ndarray] = []
        for seed in seeds:
            log = algo_seed_logs.get(algo, {}).get(seed)
            if log is None:
                continue
            cs = cumulative_steps(log["episodic_length"]).astype(np.int64)
            dt = episode_delta_elapsed(log["elapsed_sec"])
            sps = (log["episodic_length"].astype(np.float32) / dt).astype(np.float32)
            # 回合级吞吐做个轻量平滑，避免尖峰影响读图
            ma = moving_average(sps, window=10)
            if ma.size == 0 or np.all(np.isnan(ma)):
                continue
            # moving_average(valid) 对应 episode 10..N，所以 cum_steps 也要对齐到同样的末端索引
            cs_ma = cs[10 - 1 :]
            seed_series.append(interp_to_grid(cs_ma, ma, grid_steps))
        if not seed_series:
            continue
        mat = np.stack(seed_series, axis=0)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        plt.plot(grid_steps, mean, label=f"{algo.upper()} 均值", color=algo_color.get(algo, None), linewidth=2)
        plt.fill_between(grid_steps, mean - std, mean + std, alpha=0.2, color=algo_color.get(algo, None))

    plt.title(f"{env_id}：训练吞吐（steps/sec，MA10，steps 对齐，均值±标准差）")
    plt.xlabel("累计环境交互步数（Steps）")
    plt.ylabel("吞吐（steps/秒）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig20 = plt.gcf()
    fig20.tight_layout()
    out20 = save_figure(fig20, figs_dir / "fig20_throughput_steps_per_sec", formats=formats)
    plt.close(fig20)
    fig_items.append(
        FigureItem(
            fig_no="图20",
            title_zh="训练吞吐曲线（steps/sec，MA10，steps 对齐，均值±标准差）",
            x_label="累计环境交互步数（Steps）",
            y_label="吞吐（steps/秒）",
            data_sources=["models/*/train_log.csv"],
            command=cmd_str,
            outputs=[str(p.relative_to(repo_root)) for p in out20],
        )
    )

    # 同时写一份“从日志计算的指标表”，便于报告复现（不算图，但很适合做表格数据源）
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = results_dir / "analysis_metrics_table.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "seed",
                "episodes",
                "total_steps",
                "first_ep_return_500",
                "first_steps_return_500",
                "first_time_return_500_sec",
                "first_ep_ma10_500",
                "first_steps_ma10_500",
                "first_time_ma10_500_sec",
                "last50_mean_return",
                "last50_std_return",
                "last50_rate_return_500",
                "last50_rate_return_lt400",
                "final_elapsed_sec",
            ],
        )
        w.writeheader()
        for algo in algos:
            for seed in seeds:
                log = algo_seed_logs.get(algo, {}).get(seed)
                if log is None:
                    continue
                ep = log["episode_idx"].astype(np.int32)
                ret = log["episodic_return"].astype(np.float32)
                leng = log["episodic_length"].astype(np.int32)
                el = log["elapsed_sec"].astype(np.float32)
                cs = cumulative_steps(leng)

                ep500 = first_reach_threshold(ret, threshold=500.0)
                epma = first_reach_ma_threshold(ret, window=10, threshold=500.0)
                idx500 = (ep500 - 1) if ep500 is not None else None
                idxma = (epma - 1) if epma is not None else None

                tail = ret[-50:] if ret.size >= 50 else ret
                row = {
                    "algo": algo,
                    "seed": int(seed),
                    "episodes": int(ep[-1]) if ep.size else 0,
                    "total_steps": int(cs[-1]) if cs.size else 0,
                    "first_ep_return_500": int(ep500) if ep500 is not None else "",
                    "first_steps_return_500": int(cs[idx500]) if idx500 is not None else "",
                    "first_time_return_500_sec": float(el[idx500]) if idx500 is not None else "",
                    "first_ep_ma10_500": int(epma) if epma is not None else "",
                    "first_steps_ma10_500": int(cs[idxma]) if idxma is not None else "",
                    "first_time_ma10_500_sec": float(el[idxma]) if idxma is not None else "",
                    "last50_mean_return": float(tail.mean()) if tail.size else "",
                    "last50_std_return": float(tail.std()) if tail.size else "",
                    "last50_rate_return_500": float((tail >= 500.0).mean()) if tail.size else "",
                    "last50_rate_return_lt400": float((tail < 400.0).mean()) if tail.size else "",
                    "final_elapsed_sec": float(el[-1]) if el.size else "",
                }
                w.writerow(row)

    # 写出图表清单
    write_fig_list(fig_list_out, fig_items)
    print(f"[OK] 写出图表清单：{fig_list_out}")


if __name__ == "__main__":
    main()
