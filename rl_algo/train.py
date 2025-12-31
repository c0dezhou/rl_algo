from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from rl_algo.algos import create_agent, list_algos, load_config
from rl_algo.core.buffers import ReplayBuffer
from rl_algo.core.utils import set_seed
from rl_algo.envs.make_env import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="rl_algo 统一训练入口（精简版）")
    parser.add_argument("--algo", type=str, required=True, help="算法名称，例如 ppo / dqn / qlearning / sarsa / mc ...")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="可选：yaml 配置路径；不传则默认读取 `rl_algo/config/{algo}.yaml`（若存在）。",
    )
    parser.add_argument("--total-steps", type=int, default=None, help="覆盖配置中的 total_steps（可选）。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=0,
        help="可选：每隔多少 step 做一次贪心评估（0 表示关闭；评估不参与训练）。",
    )
    parser.add_argument("--eval-episodes", type=int, default=5, help="每次评估跑多少回合。")
    parser.add_argument(
        "--batch-episodes",
        type=int,
        default=None,
        help="trajectory-on-policy 算法每次 update 收集多少条完整轨迹；不传则用配置里的 batch_episodes（默认 1）。",
    )
    return parser.parse_args()


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_config_snapshot(path: Path, config: Any):
    # 把这次跑的配置存一份，后面写报告/复现实验用
    if hasattr(config, "to_dict"):
        cfg_dict = config.to_dict()
    else:
        cfg_dict = dict(getattr(config, "__dict__", {}))
    path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")


def build_flat_batch(batch_episodes: List[Dict[str, Any]], device: torch.device):
    # 把多条 episode 拼成一个大 batch（按 step 拼接），update 的时候更省事
    batch: Dict[str, torch.Tensor] = {
        "observations": torch.cat([ep["observations"] for ep in batch_episodes], dim=0),
        "actions": torch.cat([ep["actions"] for ep in batch_episodes], dim=0),
        "rewards": torch.cat([ep["rewards"] for ep in batch_episodes], dim=0).to(device),
        "dones": torch.cat([ep["dones"] for ep in batch_episodes], dim=0).to(device),
    }
    # 每个 step 属于哪条 episode（GRPO-Scaling 的 episode mask 会用到）
    ep_ids: list[torch.Tensor] = []
    for ep_i, ep in enumerate(batch_episodes):
        T = int(ep["rewards"].shape[0])
        ep_ids.append(torch.full((T,), int(ep_i), device=device, dtype=torch.long))
    batch["episode_ids"] = torch.cat(ep_ids, dim=0)
    if all("log_probs" in ep for ep in batch_episodes):
        batch["log_probs"] = torch.cat([ep["log_probs"] for ep in batch_episodes], dim=0)
    if all("values" in ep for ep in batch_episodes):
        batch["values"] = torch.cat([ep["values"] for ep in batch_episodes], dim=0)
    return batch


def add_gae_advantages_and_returns(batch: Dict[str, torch.Tensor], config: Any, device: torch.device):
    # GAE 我这里就按最常见那套写（只需要 rewards/dones/values）
    #
    # 公式记三行就够了：
    # delta：δ_t = r_t + γ * V(s_{t+1}) * (1-done_t) - V(s_t)
    # GAE：  A_t = δ_t + γ * λ * (1-done_t) * A_{t+1}
    # return：R_t = A_t + V(s_t)
    rewards = batch["rewards"].to(device).view(-1)
    dones = batch["dones"].to(device).view(-1)
    values = batch.get("values")
    if values is None:
        raise ValueError("GAE 需要 batch['values']（agent.select_action 必须返回 value）。")
    values = values.to(device).view(-1)

    gamma = float(getattr(config, "gamma", 0.99))
    lam = float(getattr(config, "gae_lambda", 0.95))

    # 从后往前推 A（遇到 done 就断开）
    advantages = torch.zeros_like(rewards, device=device)
    last_gae = torch.tensor(0.0, dtype=torch.float32, device=device)
    for t in reversed(range(rewards.shape[0])):
        next_value = values[t + 1] if (t + 1) < rewards.shape[0] else torch.tensor(0.0, device=device)
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae

    # PPO 里一般直接用：return = A + V
    batch["advantages"] = advantages.unsqueeze(1)
    batch["returns"] = (advantages + values).unsqueeze(1)


def add_grpo_group_advantages(batch: Dict[str, torch.Tensor], batch_episodes: List[Dict[str, Any]], device: torch.device):
    # GRPO：没有 critic，所以优势我就用“组内回报归一化”
    # R_i = sum_t r_{i,t}
    # A_i = (R_i - mean(R)) / (std(R) + eps)
    # 然后把 A_i 直接广播到这条轨迹里的每一步
    episode_returns = np.asarray([float(ep["episode_return"]) for ep in batch_episodes], dtype=np.float32)
    mean = float(episode_returns.mean())
    std = float(episode_returns.std())
    eps = 1e-8

    adv_list: list[torch.Tensor] = []
    for ep in batch_episodes:
        A = (float(ep["episode_return"]) - mean) / (std + eps)
        T = int(ep["rewards"].shape[0])
        adv_list.append(torch.full((T, 1), float(A), device=device, dtype=torch.float32))
    batch["advantages"] = torch.cat(adv_list, dim=0)


def _init_csv(path: Path, fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv(path: Path, fieldnames: list[str], row: Dict[str, Any]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

    # 如果没法做软链接，我就多写一份到 train_log.csv / eval_log.csv，保证老脚本还能读
    if path.name.startswith("train_log__"):
        compat = path.with_name("train_log.csv")
        if compat.exists() and compat != path:
            with open(compat, "a", newline="", encoding="utf-8") as f2:
                writer2 = csv.DictWriter(f2, fieldnames=fieldnames)
                writer2.writerow(row)
    if path.name.startswith("eval_log__"):
        compat = path.with_name("eval_log.csv")
        if compat.exists() and compat != path:
            with open(compat, "a", newline="", encoding="utf-8") as f2:
                writer2 = csv.DictWriter(f2, fieldnames=fieldnames)
                writer2.writerow(row)


def _action_to_int(action: Any):
    if isinstance(action, torch.Tensor):
        return int(action.detach().cpu().numpy()[0])
    return int(action)


def _evaluate(eval_env, agent, seed: int, episodes: int, device: torch.device, global_step: int):
    returns: list[float] = []
    for ep_i in range(int(episodes)):
        obs_eval, _ = eval_env.reset(seed=int(seed) + 20_000 + ep_i)
        done_eval = False
        ep_return = 0.0
        while not done_eval:
            obs_t_eval = torch.tensor(obs_eval, dtype=torch.float32, device=device).unsqueeze(0)
            action_eval, _info = agent.select_action(obs_t_eval, explore=False, global_step=global_step)
            obs_eval, r, terminated, truncated, _ = eval_env.step(_action_to_int(action_eval))
            done_eval = bool(terminated or truncated)
            ep_return += float(r)
        returns.append(ep_return)
    return float(sum(returns) / max(1, len(returns)))


def _log_episode(
    train_log_path: Path,
    episode_idx: int,
    ep_return: float,
    ep_len: int,
    start_time: float,
    recent_returns: list[float],
    global_step: int,
):
    elapsed_sec = float(time.time() - start_time)
    _append_csv(
        train_log_path,
        ["episode_idx", "episodic_return", "episodic_length", "elapsed_sec"],
        {
            "episode_idx": int(episode_idx),
            "episodic_return": float(ep_return),
            "episodic_length": int(ep_len),
            "elapsed_sec": float(elapsed_sec),
        },
    )

    recent_returns.append(float(ep_return))
    if len(recent_returns) > 10:
        recent_returns.pop(0)
    ma10 = float(sum(recent_returns) / max(1, len(recent_returns)))
    print(f"step={global_step:>7} | ep={episode_idx:>4} | return={ep_return:>6.1f} | ma10={ma10:>6.1f} | len={ep_len:>3}")


def _maybe_eval(eval_env, eval_log_path: Path, args, agent, device: torch.device, global_step: int):
    if eval_env is None:
        return
    if int(args.eval_frequency) <= 0:
        return
    if global_step % int(args.eval_frequency) != 0:
        return

    avg = _evaluate(eval_env, agent, int(args.seed), int(args.eval_episodes), device, global_step)
    _append_csv(
        eval_log_path,
        ["global_step", "eval_return_mean"],
        {"global_step": int(global_step), "eval_return_mean": float(avg)},
    )
    print(f"[eval] step={global_step} | avg_return={avg:.1f}")


def train_off_policy(env, eval_env, agent, config: Any, args, device: torch.device, train_log_path: Path, eval_log_path: Path):
    buffer = ReplayBuffer(int(getattr(config, "buffer_size", 100_000)), env.observation_space, env.action_space, device)

    obs, _ = env.reset(seed=int(args.seed))
    global_step = 0
    episode_idx = 0
    episode_return_acc = 0.0
    episode_len_acc = 0
    start_time = time.time()
    recent_returns: list[float] = []

    while global_step < int(getattr(config, "total_steps", 0)):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, _info = agent.select_action(obs_t, explore=True, global_step=global_step)
        action_env = _action_to_int(action)

        next_obs, reward, terminated, truncated, info = env.step(action_env)
        episode_done = bool(terminated or truncated)

        # DQN 用 bootstrap：TimeLimit 截断别当 done（不然满分附近学得很奇怪）
        done_for_learning = bool(terminated)

        global_step += 1
        episode_return_acc += float(reward)
        episode_len_acc += 1

        buffer.add(obs, action_env, float(reward), next_obs, done_for_learning)

        batch_size = int(getattr(config, "batch_size", 64))
        train_frequency = int(getattr(config, "train_frequency", 1))
        gradient_steps = int(getattr(config, "gradient_steps", 1))
        learning_starts = int(getattr(config, "learning_starts", batch_size))

        if len(buffer) >= max(batch_size, learning_starts) and global_step % train_frequency == 0:
            for _ in range(max(1, gradient_steps)):
                batch = buffer.sample(batch_size)
                agent.update(batch, global_step)

        obs = next_obs
        if episode_done:
            episode_idx += 1
            if "episode" in info:
                ep_return = float(info["episode"]["r"])
                ep_len = int(info["episode"]["l"])
            else:
                ep_return = float(episode_return_acc)
                ep_len = int(episode_len_acc)
            _log_episode(train_log_path, episode_idx, ep_return, ep_len, start_time, recent_returns, global_step)

            episode_return_acc = 0.0
            episode_len_acc = 0
            obs, _ = env.reset()

        _maybe_eval(eval_env, eval_log_path, args, agent, device, global_step)

    return global_step


def train_step_on_policy(env, eval_env, agent, config: Any, args, device: torch.device, train_log_path: Path, eval_log_path: Path):
    obs, _ = env.reset(seed=int(args.seed))
    global_step = 0
    episode_idx = 0
    episode_return_acc = 0.0
    episode_len_acc = 0
    start_time = time.time()
    recent_returns: list[float] = []

    while global_step < int(getattr(config, "total_steps", 0)):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, agent_info = agent.select_action(obs_t, explore=True, global_step=global_step)
        action_env = _action_to_int(action)

        next_obs, reward, terminated, truncated, info = env.step(action_env)
        episode_done = bool(terminated or truncated)

        # Q-learning / SARSA 也是 bootstrap：TimeLimit 截断别当 done
        done_for_learning = bool(terminated)

        global_step += 1
        episode_return_acc += float(reward)
        episode_len_acc += 1

        transition = {
            "obs": obs_t,
            "action": torch.as_tensor(action, device=device),
            "reward": float(reward),
            "next_obs": torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0),
            "done": done_for_learning,
            **(agent_info or {}),
        }

        if args.algo == "sarsa":
            next_action, _ = agent.select_action(transition["next_obs"], explore=True, global_step=global_step)
            transition["next_action"] = next_action

        agent.train_step(transition)

        obs = next_obs
        if episode_done:
            episode_idx += 1
            if "episode" in info:
                ep_return = float(info["episode"]["r"])
                ep_len = int(info["episode"]["l"])
            else:
                ep_return = float(episode_return_acc)
                ep_len = int(episode_len_acc)
            _log_episode(train_log_path, episode_idx, ep_return, ep_len, start_time, recent_returns, global_step)

            episode_return_acc = 0.0
            episode_len_acc = 0
            obs, _ = env.reset()

        _maybe_eval(eval_env, eval_log_path, args, agent, device, global_step)

    return global_step


def train_trajectory_on_policy(
    env,
    eval_env,
    agent,
    config: Any,
    args,
    device: torch.device,
    train_log_path: Path,
    eval_log_path: Path,
):
    # 轨迹类算法：攒够几条完整 episode 再 update
    batch_episodes_target = int(args.batch_episodes or int(getattr(config, "batch_episodes", 1)))
    if args.algo in ("grpo", "grpo_scaling"):
        batch_episodes_target = int(getattr(config, "group_size", batch_episodes_target))

    episode_obs: list[torch.Tensor] = []
    episode_actions: list[torch.Tensor] = []
    episode_log_probs: list[torch.Tensor] = []
    episode_values: list[torch.Tensor] = []
    episode_rewards: list[float] = []
    episode_dones: list[bool] = []
    batch_episodes: list[Dict[str, Any]] = []

    obs, _ = env.reset(seed=int(args.seed))
    global_step = 0
    episode_idx = 0
    episode_return_acc = 0.0
    episode_len_acc = 0
    start_time = time.time()
    recent_returns: list[float] = []

    while global_step < int(getattr(config, "total_steps", 0)):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, agent_info = agent.select_action(obs_t, explore=True, global_step=global_step)
        action_env = _action_to_int(action)

        next_obs, reward, terminated, truncated, info = env.step(action_env)
        episode_done = bool(terminated or truncated)

        global_step += 1
        episode_return_acc += float(reward)
        episode_len_acc += 1

        # 先把这一条 episode 的数据攒起来
        episode_obs.append(obs_t)
        episode_actions.append(torch.as_tensor(action, device=device))
        episode_rewards.append(float(reward))
        episode_dones.append(bool(episode_done))
        if agent_info and "log_prob" in agent_info:
            episode_log_probs.append(agent_info["log_prob"])
        if agent_info and "value" in agent_info:
            episode_values.append(agent_info["value"])

        if episode_done:
            ep_obs = torch.cat(episode_obs, dim=0)
            ep_actions = torch.stack(episode_actions, dim=0)
            ep_rewards = torch.tensor(episode_rewards, dtype=torch.float32, device=device).view(-1)
            ep_dones = torch.tensor(episode_dones, dtype=torch.float32, device=device).view(-1)

            ep: Dict[str, Any] = {
                "observations": ep_obs,
                "actions": ep_actions,
                "rewards": ep_rewards,
                "dones": ep_dones,
                "episode_return": float(sum(episode_rewards)),
                "episode_length": int(len(episode_rewards)),
            }
            if episode_log_probs:
                ep["log_probs"] = torch.stack(episode_log_probs, dim=0).view(-1)
            if episode_values:
                ep["values"] = torch.stack(episode_values, dim=0).view(-1)

            batch_episodes.append(ep)

            episode_obs.clear()
            episode_actions.clear()
            episode_log_probs.clear()
            episode_values.clear()
            episode_rewards.clear()
            episode_dones.clear()

            if len(batch_episodes) >= batch_episodes_target:
                batch_like = build_flat_batch(batch_episodes, device=device)

                if args.algo in ("grpo", "grpo_scaling"):
                    add_grpo_group_advantages(batch_like, batch_episodes, device=device)
                if bool(getattr(config, "use_gae", False)) and "values" in batch_like:
                    add_gae_advantages_and_returns(batch_like, config, device=device)

                agent.update(batch_like, global_step)
                batch_episodes.clear()

        obs = next_obs
        if episode_done:
            episode_idx += 1
            if "episode" in info:
                ep_return = float(info["episode"]["r"])
                ep_len = int(info["episode"]["l"])
            else:
                ep_return = float(episode_return_acc)
                ep_len = int(episode_len_acc)
            _log_episode(train_log_path, episode_idx, ep_return, ep_len, start_time, recent_returns, global_step)

            episode_return_acc = 0.0
            episode_len_acc = 0
            obs, _ = env.reset()

        _maybe_eval(eval_env, eval_log_path, args, agent, device, global_step)

    return global_step


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(int(args.seed))

    available_algos = list_algos()
    if args.algo not in available_algos:
        raise ValueError(f"未知算法：{args.algo}，可用算法：{available_algos}")

    config = load_config(args.algo, args.config)
    if args.total_steps is not None:
        config.total_steps = int(args.total_steps)

    render_mode = "human" if args.render else None
    env = make_env(args.env_id, seed=int(args.seed), render_mode=render_mode)
    eval_env = None
    if int(args.eval_frequency) > 0:
        eval_env = make_env(args.env_id, seed=int(args.seed) + 10_000, render_mode=None)

    agent = create_agent(args.algo, env.observation_space, env.action_space, config, device)

    run_name = f"{args.env_id}__{args.algo}__{args.seed}__{int(time.time())}"
    run_dir = Path("models") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(run_dir / "config.yaml", config)

    print(f"run_dir={run_dir}")
    print(f"algo={args.algo}, env={args.env_id}, device={device}, total_steps={getattr(config, 'total_steps', 0)}")

    if args.algo == "dqn":
        algo_kind = "offpolicy"
    elif args.algo in ("qlearning", "sarsa"):
        algo_kind = "step_onpolicy"
    else:
        algo_kind = "traj_onpolicy"

    # 日志名我写得长一点，后面整理论文表格省时间
    train_log_named = run_dir / f"train_log__{args.env_id}__{args.algo}__{algo_kind}__seed{int(args.seed)}.csv"
    train_log_path = run_dir / "train_log.csv"

    _init_csv(train_log_named, ["episode_idx", "episodic_return", "episodic_length", "elapsed_sec"])
    try:
        if not train_log_path.exists():
            train_log_path.symlink_to(train_log_named.name)
    except Exception:
        # 如果系统不支持软链接，就再写一份同样的表头
        _init_csv(train_log_path, ["episode_idx", "episodic_return", "episodic_length", "elapsed_sec"])

    eval_log_path = run_dir / "eval_log.csv"
    if eval_env is not None:
        eval_log_named = run_dir / f"eval_log__{args.env_id}__{args.algo}__{algo_kind}__seed{int(args.seed)}.csv"
        _init_csv(eval_log_named, ["global_step", "eval_return_mean"])
        try:
            if not eval_log_path.exists():
                eval_log_path.symlink_to(eval_log_named.name)
        except Exception:
            _init_csv(eval_log_path, ["global_step", "eval_return_mean"])

    if args.algo == "dqn":
        global_step = train_off_policy(env, eval_env, agent, config, args, device, train_log_named, eval_log_path)
    elif args.algo in ("qlearning", "sarsa"):
        global_step = train_step_on_policy(env, eval_env, agent, config, args, device, train_log_named, eval_log_path)
    else:
        global_step = train_trajectory_on_policy(
            env, eval_env, agent, config, args, device, train_log_named, eval_log_path
        )

    final_path = run_dir / "final.pt"
    torch.save(
        {
            "algo": args.algo,
            "env_id": args.env_id,
            "seed": int(args.seed),
            "global_step": int(global_step),
            "agent_state": agent.save(),
            "config_path": str(args.config) if args.config is not None else None,
        },
        final_path,
    )
    env.close()
    if eval_env is not None:
        eval_env.close()
    print(f"saved: {final_path}")


if __name__ == "__main__":
    main()
