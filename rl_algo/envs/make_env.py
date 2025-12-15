# rlx/envs/make_env.py 模块
"""
该模块提供一个工厂函数 `make_env`, 用于创建和配置 Gymnasium 环境。

这个函数是环境管理的核心, 它将环境的创建、包装和种子设置等标准化流程
封装起来, 使得训练脚本 (`train.py`) 的逻辑更清晰。

核心功能:
- **环境创建**: 根据指定的 `env_id` 创建一个 Gym 环境实例。
- **渲染模式**: 支持传入 `render_mode` 参数, 以便在需要时 (例如, 在本地
  调试或演示时) 实时渲染环境画面。
- **统计数据记录**: 自动使用 `RecordEpisodeStatistics` 包装器来包裹环境。
  这个包装器是至关重要的, 它会自动追踪每个回合 (episode) 的累计回报 (return)
  和长度 (length), 并在回合结束时将这些信息存入 `info` 字典中, 方便
  日志记录和性能监控。
- **种子设置**: 通过调用 `env.reset(seed=seed)` 来正确地为环境设置随机种子,
  确保实验的可复现性。
"""

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

def make_env(env_id: str, seed: int, render_mode: str | None = None) -> gym.Env:
    """
    一个用于创建和包装 Gym 环境的工厂函数。

    Args:
        env_id (str): 要创建的环境的 ID (例如, "CartPole-v1")。
        seed (int): 用于可复现性的随机种子。
        render_mode (str | None): 渲染模式。如果设置为 "human", 则会创建
                                  一个窗口来实时显示环境画面。如果为 None,
                                  则不进行渲染。

    Returns:
        gym.Env: 一个经过配置和包装的 Gym 环境实例。
    """
    # 创建基础的 Gym 环境, 并根据需要传入渲染模式。
    env = gym.make(env_id, render_mode=render_mode)
    
    # 使用 RecordEpisodeStatistics 包装器, 这是进行性能追踪的关键。
    # 它会在每个回合结束后, 自动在返回的 info 字典中添加一个 'episode' 键,
    # 其中包含了该回合的 'r' (回报) 和 'l' (长度)。
    env = RecordEpisodeStatistics(env)
    
    # 根据 Gymnasium 的最佳实践, 设置环境 (包括观测随机性) 的正确且唯一的
    # 方法是在开始时调用 env.reset(seed=seed)。
    # 这个调用会同时为环境的主随机数生成器 (RNG) 和动作空间设置种子。
    env.reset(seed=seed)
    
    return env
