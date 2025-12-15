# rlx/core/types.py 模块
"""
该模块定义了在整个框架中使用的、标准化的数据结构类型。
使用明确的类型定义 (如 `Batch`, `Transition`, `Config`) 可以极大地提高代码的
可读性、可维护性, 并能利用静态类型检查工具 (如 MyPy) 提前发现潜在的错误。

主要定义:
- `Config`: 所有算法配置类的 Pydantic 基类, 提供自动的数据验证和类型转换。
- `Batch`: 一个 `NamedTuple`, 用于封装从经验回放缓冲区中采样出的一批数据,
           主要用于异策略 (Off-Policy) 算法的 `update` 方法。
- `Transition`: 一个类型别名 (`Dict[str, Any]`), 用于表示单个时间步的经验,
                主要用于在线数据收集和单步更新的算法 (如 SARSA)。
"""

from typing import Any, Dict, NamedTuple

import torch
from pydantic import BaseModel

class Config(BaseModel):
    """
    所有算法配置类的 Pydantic 基类。

    通过继承这个类, 所有的算法配置都将自动获得 Pydantic 提供的强大功能,
    例如:
    - **类型强制**: 从 YAML 文件加载的配置项 (通常是字符串或数字) 会被自动
      转换为 Python 中正确的类型 (例如, `lr: float = 0.1`)。
    - **数据验证**: 可以为字段添加验证器, 确保超参数在合理的范围内。
    - **默认值**: 可以为超参数提供默认值, 简化配置文件的编写。
    """
    class Config:
        # Pydantic 的内部配置, `extra = 'allow'` 允许在 YAML 文件中出现
        # 未在模型中明确定义的额外字段。这增加了灵活性, 允许在配置文件中
        # 添加注释或临时参数而不会导致程序崩溃。
        extra = 'allow'

class Batch(NamedTuple):
    """
    一个具名元组 (NamedTuple), 用于封装一批 (batch) 用于模型更新的数据。
    它主要由异策略算法的经验回放缓冲区 (`ReplayBuffer.sample()`) 生成,
    并作为 `agent.update()` 方法的输入。

    使用 `NamedTuple` 的好处是, 它既像元组一样轻量、不可变, 又能像类实例
    一样通过字段名 (如 `batch.observations`) 访问数据, 提高了代码的可读性。

    字段说明 (shape 中的 N 代表批次大小 `batch_size`):
      - observations: torch.Tensor, 形状为 (N, *obs_shape)
      - actions: torch.Tensor, 形状为 (N, *action_shape)
      - rewards: torch.Tensor, 形状为 (N, 1)
      - next_observations: torch.Tensor, 形状为 (N, *obs_shape)
      - dones: torch.Tensor, 形状为 (N, 1), 标记一个回合是否在此步之后结束。
    """
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor

# `Transition` 是一个字典的类型别名, 用于表示单个时间步的完整经验数据。
# 它在训练循环中被实时创建, 并传递给需要逐-步处理数据的组件, 例如
# 同策略算法的 `train_step` 方法或经验回放缓冲区的 `add` 方法。
#
# 与 `Batch` 相比, `Transition` 的字段更灵活, 可以包含算法特定的额外信息,
# 这对于需要传递额外上下文 (如动作概率) 的算法至关重要。
#
# 典型字段包括:
# {
#   'obs': torch.Tensor,       # 当前观测, shape (1, *obs_shape)
#   'action': torch.Tensor,    # 采取的动作
#   'reward': float,           # 获得的奖励
#   'next_obs': torch.Tensor,  # 下一时刻的观测, shape (1, *obs_shape)
#   'done': bool,              # 回合是否结束
#
#   # --- 可选字段, 取决于算法需求 ---
#   'log_prob': torch.Tensor,  # 动作的对数概率 (用于 PPO, A2C 等策略梯度算法)
#   'value': torch.Tensor,     # 状态的价值估计 (用于 GAE 计算)
#   'next_action': torch.Tensor, # SARSA 算法需要的下一个状态的下一个动作
# }
Transition = Dict[str, Any]
