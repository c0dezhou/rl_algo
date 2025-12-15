# rlx/train.py 文件
"""
一个统一的训练脚本,用于启动、管理和监控任何已注册的强化学习算法的训练过程。

该脚本是整个框架的执行入口。它负责处理以下核心任务：
1.  **命令行参数解析**: 使用 `argparse` 解析用户输入的参数,如算法名称、环境 ID、
    配置文件路径、总训练步数、随机种子等。

2.  **配置加载与覆盖**: 从指定的 YAML 文件加载算法配置,并允许用户通过命令行
    参数 `--override-cfg` 动态覆盖个别超参数,便于快速实验和调参。

3.  **环境和智能体创建**: 根据参数创建指定的 Gym 环境和 RL 智能体实例。

4.  **主训练循环**: 实现了通用的训练循环,能够适应不同类型的 RL 算法：
    - **异策略 (Off-Policy)** 算法 (如 DQN): 从经验回放池 (Replay Buffer) 中采样数据进行训练。
    - **同策略 (On-Policy) 且基于轨迹**的算法 (如 PPO): 收集完整的回合 (episode) 数据后进行训练。
    - **同策略 (On-Policy) 且基于单步**的算法 (如 SARSA): 在每个时间步进行训练。

5.  **模型评估与保存**:
    - 在训练过程中,定期评估模型性能,并保存表现最佳的模型 (`best.pt`)。
    - 训练结束后,保存最终模型 (`final.pt`)。
    - 支持从检查点 (`checkpoint`) 恢复训练,继续之前的进度。
"""

import argparse
import os
import random
import time
from pathlib import Path

# 这是一个用于 argparse 的辅助函数, 能够将命令行中输入的 "true", "yes", "1" 等字符串智能地解析为布尔值 True, 否则为 False。
def str_to_bool(val: str) -> bool:
    if isinstance(val, bool):
        return val
    val_l = str(val).lower()
    return val_l in ("yes", "true", "t", "1")

from typing import Any, Dict, Type

import gymnasium as gym
import numpy as np
import torch
import yaml
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from rl_algo.core.buffers import ReplayBuffer
from rl_algo.core.registry import registry
from rl_algo.core.utils import set_seed, dynamic_import_agents
from rl_algo.core.types import Batch, Transition
from rl_algo.envs.make_env import make_env

# 动态导入所有在 `rlx/algos` 目录下的算法, 这一步是实现算法自动注册的关键。
# 执行后, `registry` 对象中将包含所有已发现的算法及其配置。
dynamic_import_agents()

def parse_args():
    """解析命令行参数,为训练提供所有必要的配置。"""
    parser = argparse.ArgumentParser(description="rl-gym 统一训练脚本")
    parser.add_argument("--algo", type=str, required=True, help="要训练的算法名称 (例如: 'qlearning', 'ppo')。")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="要使用的 Gymnasium 环境 ID。")
    parser.add_argument("--config", type=str, required=True, help="指向算法特定配置的 YAML 文件的路径。")
    parser.add_argument("--total-steps", type=int, default=50000, help="训练过程的总时间步数。")
    parser.add_argument("--seed", type=int, default=42, help="用于 PyTorch, NumPy 和环境的随机种子,以确保实验的可复现性。")
    parser.add_argument("--device", type=str, default="auto", help="指定计算设备 ('cpu', 'cuda', 或 'auto' 自动选择)。")
    parser.add_argument("--render", type=str_to_bool, default=False, nargs="?", const=True, help="如果设置,则在训练期间实时渲染环境画面。")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="提供一个检查点文件的路径 (.pt), 从中恢复训练。")
    parser.add_argument("--eval-frequency", type=int, default=10000, help="模型评估的频率 (以总步数为单位)。")
    parser.add_argument("--eval-episodes", type=int, default=10, help="每次评估时运行的回合数,以获得更稳定的性能度量。")
    # 允许通过命令行动态覆盖 YAML 文件中的配置参数,这对于快速实验和超参数搜索非常有用。
    # 示例: --override-cfg lr=1e-3 gamma=0.98
    parser.add_argument('--override-cfg', nargs='*', help="覆盖配置参数。示例: --override-cfg lr=1e-3 actor.lr=0.001")

    return parser.parse_args()

def load_config(config_path: str, algo: str, overrides: list[str] | None) -> BaseModel:
    """
    加载、解析并覆盖算法的配置。
    1. 从注册表中获取算法对应的 Pydantic 配置类。
    2. 加载基础的 YAML 配置文件。
    3. 如果有命令行覆盖项,则解析并更新配置。
    4. 使用最终的配置字典实例化 Pydantic 配置类,进行类型验证。
    """
    config_class = registry.get_config_class(algo)
    if not config_class:
        raise ValueError(f"算法 '{algo}' 在注册表中没有找到对应的配置类。")

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # 处理命令行覆盖
    if overrides:
        for override in overrides:
            k, v = override.split("=")
            try:
                # 尝试将值解析为 Python 对象 (例如, '1e-3' -> 0.001, 'True' -> True)
                v = eval(v)
            except (NameError, SyntaxError):
                # 如果解析失败,则保持其为字符串 (例如, 'my_string')
                pass 

            # 更新配置字典,支持点分隔的嵌套键 (e.g., 'actor.lr=0.1')
            keys = k.split('.')
            d = yaml_config # d 是指向 yaml_config 的指针,直接修改 d 会改变 yaml_config
            for key_part in keys[:-1]:
                # 如果嵌套的字典不存在,则当场创建一个
                d = d.setdefault(key_part, {})
            d[keys[-1]] = v
    
    # 使用最终的配置字典实例化 Pydantic 模型,这会进行数据验证和类型转换
    return config_class(**yaml_config)

def main():
    args = parse_args()

    # ================== 1. 初始化设置 ==================
    # 自动选择设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"正在使用设备: {device}")

    # 加载配置,并允许命令行总步数覆盖文件中的设置
    config = load_config(args.config, args.algo, args.override_cfg)
    if args.total_steps:
        setattr(config, "total_steps", args.total_steps)
    
    # 创建一个唯一的运行名称,用于日志和模型保存
    run_name = f"{args.env_id}__{args.algo}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # 设置全局随机种子以保证实验的可复现性
    set_seed(args.seed)

    # ================== 2. 创建环境和智能体 ==================
    render_mode = "human" if args.render else None
    env = make_env(
        args.env_id,
        args.seed,
        render_mode=render_mode,
    )
    agent = registry.create_agent(args.algo, env.observation_space, env.action_space, config, device)

    # 准备模型保存目录
    save_path = f"models/{run_name}"
    os.makedirs(save_path, exist_ok=True)

    # 如果是异策略 (Off-Policy) 算法,则初始化经验回放池 (Replay Buffer)
    is_off_policy = hasattr(config, "buffer_size") 
    buffer = None 
    if is_off_policy:
        buffer = ReplayBuffer(config.buffer_size, env.observation_space, env.action_space, device)

    # ================== 3. 训练循环设置 ==================
    # 根据算法类型设置不同的数据收集和更新策略标志
    algo = args.algo
    # 基于完整轨迹更新的同策略算法
    trajectory_on_policy = algo in {"ppo", "a2c", "reinforce", "mc"}
    # 基于单步更新的同策略算法（逐步更新）
    step_on_policy = algo in {"sarsa", "td0", "qlearning"}

    # 为基于轨迹的同策略算法准备临时缓冲区
    traj_obs, traj_actions, traj_log_probs, traj_rewards, traj_values, traj_dones = [], [], [], [], [], []
    
    # 初始化环境和训练状态
    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    best_return = float("-inf") # 用于追踪最佳模型

    # ================== 4. 检查点 (Checkpoint) 功能 ==================
    def save_checkpoint(path: str, step: int):
        """保存检查点,包括训练步数、智能体状态和当前记录的最佳回报。"""
        agent_state = agent.save()
        torch.save({
            'global_step': step,
            'agent_state_dict': agent_state,
            'best_return': best_return,
            }, path)
        
    # 如果提供了检查点路径,则从中恢复训练
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"从检查点恢复训练: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device, weights_only=False)
        agent.load(checkpoint['agent_state_dict'])
        global_step = checkpoint.get('global_step', 0)
        best_return = checkpoint.get('best_return', float("-inf"))
        print(f"已从第 {global_step} 步恢复, 记录的最佳回报为 {best_return:.2f}")

    
    # ================== 5. 评估函数 ==================
    def evaluate_agent(agent_state_dict: Dict, episodes: int = 5) -> float:
        """使用给定的智能体状态在单独的环境中进行评估。"""
        print("\n--- 开始评估 ---")
        
        # 创建一个独立的评估环境,以确保评估的纯粹性
        eval_env = make_env(
            env_id=args.env_id,
            seed=args.seed + 1,  # 使用不同的种子以避免与训练环境产生偏差
            render_mode=None,    # 评估时通常不渲染
        )
        
        # 创建一个新的智能体实例并加载要评估的状态
        eval_agent = registry.create_agent(
            args.algo, 
            eval_env.observation_space, 
            eval_env.action_space, 
            config, 
            device
        )
        eval_agent.load(agent_state_dict)
        
        total_rewards = []
        for ep in range(episodes):
            obs_ep, _ = eval_env.reset(seed=(args.seed + ep + 1))
            done_ep = False
            frames = 0
            
            while not done_ep:
                # 将单个观测数据扩展为一个批次 (BatchSize=1) 以匹配网络输入
                obs_t = torch.tensor(obs_ep, dtype=torch.float32, device=device).unsqueeze(0)
                
                # 关键: 在评估时关闭探索 (explore=False), 让智能体使用其学到的最佳策略
                action, _ = eval_agent.select_action(
                    obs_t, 
                    explore=False, 
                    global_step=global_step
                )
                action_np = action.cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                
                obs_ep, reward, terminated, truncated, info = eval_env.step(action_np)
                done_ep = terminated or truncated
                frames += 1
            
            # `RecordEpisodeStatistics` 包装器会在回合结束时将回报和长度记录在 info 字典中
            if "episode" in info:
                ep_r = info["episode"]["r"]
                total_rewards.append(ep_r)
                print(f"[评估] 回合={ep+1}/{episodes}, 帧数={frames}, 回报={ep_r:.2f}")
            else:
                print(f"[评估] 回合={ep+1}/{episodes} 完成 (警告: 未在 info 中找到 'episode' 数据)")

        eval_env.close()

        if total_rewards:
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"[评估] {episodes} 个回合的平均回报: {avg_reward:.2f}")
            print("--- 评估结束 ---\n")
            return avg_reward
        
        print("[评估] 未能收集到任何回报数据。")
        print("--- 评估结束 ---\n")
        return 0.0
    
    # ================== 6. 主训练循环 ==================
    print(f"开始训练 {args.algo} 算法, 共 {config.total_steps} 步...")
    while global_step < config.total_steps:
        # a. 与环境交互
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, agent_info = agent.select_action(obs_tensor, global_step=global_step)
        action_np = action.cpu().numpy()[0] if isinstance(action, torch.Tensor) else action

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        global_step += 1

        transition: Transition = {
            "obs": obs_tensor, "action": torch.as_tensor(action, device=device),
            "reward": reward, "next_obs": torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0),
            "done": done, **(agent_info or {}),
        }

        # 特殊处理: SARSA 算法需要在更新时知道下一个状态的下一个动作
        if algo == "sarsa":
            next_action, _ = agent.select_action(transition["next_obs"], explore=True, global_step=global_step)
            transition["next_action"] = next_action

        # b. 根据算法类型进行数据存储和模型更新
        if is_off_policy:
            buffer.add(obs, action_np, reward, next_obs, done)
            if global_step > config.batch_size and global_step % config.train_frequency == 0:
                batch = buffer.sample(config.batch_size)
                metrics = agent.update(batch, global_step)
                if global_step % 100 == 0 and metrics:
                    for key, value in metrics.items():
                        writer.add_scalar(f"train/{key}", value, global_step)

        elif trajectory_on_policy:
            traj_obs.append(obs_tensor)
            traj_actions.append(transition["action"])
            traj_rewards.append(reward)
            traj_dones.append(done)
            if "log_prob" in agent_info: traj_log_probs.append(agent_info["log_prob"])
            if "value" in agent_info: traj_values.append(agent_info["value"])

            if done: # 当一个回合结束时, 处理收集到的整个轨迹
                b_obs = torch.cat(traj_obs, dim=0)
                b_actions = torch.stack(traj_actions, dim=0)
                
                batch_like: Batch = {
                    "observations": b_obs,
                    "actions": b_actions,
                    "rewards": torch.tensor(traj_rewards, dtype=torch.float32, device=device),
                    "dones": torch.tensor(traj_dones, dtype=torch.float32, device=device),
                }
                if traj_log_probs: batch_like["log_probs"] = torch.stack(traj_log_probs, dim=0)
                
                # 为 PPO 等使用 GAE 的算法计算优势 (Advantage) 和回报 (Return)
                if getattr(config, "use_gae", False) and traj_values:
                    values = torch.stack(traj_values, dim=0).squeeze(-1)
                    rewards = torch.tensor(traj_rewards, dtype=torch.float32, device=device)
                    dones_mask = torch.tensor(traj_dones, dtype=torch.float32, device=device)
                    gamma, lam = config.gamma, config.gae_lambda
                    advantages = torch.zeros_like(rewards)
                    last_gae = 0.0
                    for t in reversed(range(len(rewards))):
                        next_val = values[t + 1] if t + 1 < len(values) else 0.0
                        delta = rewards[t] + gamma * next_val * (1.0 - dones_mask[t]) - values[t]
                        last_gae = delta + gamma * lam * (1.0 - dones_mask[t]) * last_gae
                        advantages[t] = last_gae
                    batch_like["advantages"] = advantages.unsqueeze(1)
                    batch_like["returns"] = (advantages + values).unsqueeze(1)

                metrics = agent.update(batch_like, global_step)
                if metrics and global_step % 100 == 0:
                    for key, value in metrics.items(): writer.add_scalar(f"train/{key}", value, global_step)

                # 清空轨迹缓冲区,为下一个回合做准备
                traj_obs.clear(); traj_actions.clear(); traj_log_probs.clear(); traj_rewards.clear(); traj_values.clear(); traj_dones.clear()

        elif step_on_policy:
            metrics = agent.train_step(transition)
            if metrics and global_step % 100 == 0:
                for key, value in metrics.items(): writer.add_scalar(f"train/{key}", value, global_step)

        # c. 处理回合结束逻辑
        obs = next_obs
        if done:
            if "episode" in info:
                ep_r, ep_l = info["episode"]["r"], info["episode"]["l"]
                print(f"步数={global_step}, 回合回报={ep_r:.2f}, 回合长度={ep_l}")
                writer.add_scalar("charts/episodic_return", ep_r, global_step)
                writer.add_scalar("charts/episodic_length", ep_l, global_step)
                
                # 如果获得更高回报,则保存为最佳模型
                if ep_r > best_return:
                    best_return = ep_r
                    best_path = f"{save_path}/best.pt"
                    save_checkpoint(best_path, global_step)
                    print(f"发现新的最佳回报={best_return:.2f}, 模型已保存至 {best_path}")
            obs, _ = env.reset()

        # d. 定期评估
        if args.eval_frequency > 0 and global_step % args.eval_frequency == 0:
            current_agent_state = agent.save()
            avg_reward = evaluate_agent(current_agent_state, episodes=args.eval_episodes)
            writer.add_scalar("charts/eval_return", avg_reward, global_step)

    # ================== 7. 训练结束 ==================
    final_path = f"{save_path}/final.pt"
    save_checkpoint(final_path, global_step)
    print(f"训练完成。最终模型已保存至 {final_path}")

    # 训练结束后,对最终模型进行一次最终评估
    final_checkpoint = torch.load(final_path, map_location=device, weights_only=False)
    evaluate_agent(final_checkpoint['agent_state_dict'], episodes=args.eval_episodes)

    env.close()
    writer.close()
    if hasattr(args, "wandb_project") and args.wandb_project:
        try:
            import wandb
            wandb.finish()
        except Exception: pass

if __name__ == "__main__":
    main()
