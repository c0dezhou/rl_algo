# rlx/core/buffers.py
"""
该模块包含用于强化学习算法的各种经验回放缓冲区 (Experience Replay Buffers)。

经验回放是异策略 (Off-Policy) 算法（如 DQN, DDPG, SAC) 中的一个关键组件。
它通过存储智能体与环境交互的大量经验（即“转移” (transitions)）, 并从中随机采样
小批量数据来训练模型。这种做法有两大好处：
1.  **打破数据相关性**: 连续的经验是高度相关的, 随机采样可以打破这种时间上的
    关联, 使训练过程更稳定, 更符合机器学习中独立同分布 (i.i.d) 的假设。
2.  **提高数据利用率**: 每一次的经验都有可能被多次采样和学习, 提高了宝贵的
    交互数据的使用效率。

目前实现的缓冲区:
- `ReplayBuffer`: 一个标准的、高效的、基于 PyTorch 张量的循环经验回放缓冲区。
"""

import gymnasium as gym
import numpy as np
import torch
from rlx.core.types import Batch

class ReplayBuffer:
    """
    一个高效的、基于 PyTorch 的循环经验回放缓冲区。

    它使用预先分配的 PyTorch 张量来存储所有经验数据, 从而最小化 Python 的开销,
    并能直接在指定的设备 (如 GPU) 上进行操作, 避免了数据在 CPU 和 GPU 之间
    的频繁拷贝。

    其工作方式像一个环形队列：当缓冲区被填满后, 新加入的数据会从头开始覆盖
    最旧的数据。
    """
    def __init__(self, 
                 buffer_size: int, 
                 obs_space: gym.Space, 
                 act_space: gym.Space, 
                 device: torch.device
                ):
        """
        初始化经验回放缓冲区。

        Args:
            buffer_size (int): 缓冲区的最大容量。
            obs_space (gym.Space): 环境的观测空间, 用于确定观测数据的形状。
            act_space (gym.Space): 环境的动作空间, 用于确定动作数据的形状和类型。
            device (torch.device): 存储张量的设备 (例如, 'cpu' 或 'cuda')。
        """
        self.buffer_size = buffer_size
        self.device = device

        # 从 Gym 空间中提取观测和动作的形状
        obs_shape = obs_space.shape
        act_shape = act_space.shape

        # 根据动作空间类型确定动作的数据类型
        obs_dtype = torch.float32
        act_dtype = (torch.int64 if isinstance(act_space, gym.spaces.Discrete)
                     else torch.float32)
        
        # 预分配内存, 创建零张量来存储未来的数据。这比动态追加列表更高效。
        # 示例: 若 buffer_size=10000, obs_shape=(4,), 则张量形状为 (10000, 4)。
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=obs_dtype, device=device)
        self.actions = torch.zeros((buffer_size, *act_shape), dtype=act_dtype, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=obs_dtype, device=device)
        # 将终止标记存储为 float32 (True -> 1.0, False -> 0.0), 以便在计算折扣回报时直接进行乘法运算。
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        
        self.pos = 0      # 当前要插入数据的位置指针
        self.full = False # 标记缓冲区是否已满

    def add(self, 
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_obs: np.ndarray,
            done: bool) -> None:
        """
        向缓冲区中添加一个时间步的经验 (一个 transition)。
        输入的数据通常是 NumPy 数组或标量, 在函数内部会被转换为 PyTorch 张量。
        """
        # 将输入的 numpy 数组或标量转换为张量, 并存储在指针 `pos` 指向的位置
        self.observations[self.pos] = torch.as_tensor(obs, device=self.device)
        self.actions[self.pos] = torch.as_tensor(action, device=self.device)
        self.rewards[self.pos] = torch.as_tensor([reward], device=self.device)
        self.next_observations[self.pos] = torch.as_tensor(next_obs, device=self.device)
        self.dones[self.pos] = torch.as_tensor([done], device=self.device)
        
        # 移动指针
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True  # 标记缓冲区已满
            self.pos = 0      # 将指针重置到开头, 实现循环覆盖
    
    def sample(self, batch_size: int) -> Batch:
        """
        从缓冲区中随机采样一个批次 (batch) 的数据。

        Args:
            batch_size (int): 要采样的批次大小。

        Returns:
            Batch: 一个包含观测、动作、奖励等张量的数据类对象。

        Raises:
            ValueError: 如果缓冲区中的样本数量不足以采样一个批次。
        """
        current_size = self.buffer_size if self.full else self.pos
        if current_size < batch_size:
            raise ValueError(
                f"缓冲区中的样本数量 ({current_size}) 不足以采样一个大小为 ({batch_size}) 的批次。"
            )
        
        # 在有效的数据范围内 [0, current_size) 随机生成不重复的索引
        batch_inds = torch.randint(0, current_size, (batch_size,), device=self.device)

        # 根据索引提取数据
        sampled_actions = self.actions[batch_inds]
        
        # 特殊处理离散动作: 确保其类型为 long 且形状正确 (batch_size,) 而不是 (batch_size, 1)
        if sampled_actions.dtype in (torch.int64, torch.long):
            if sampled_actions.dim() == 2 and sampled_actions.shape[1] == 1:
                sampled_actions = sampled_actions.squeeze(1).long()
            else:
                sampled_actions = sampled_actions.long()

        return Batch(
            observations=self.observations[batch_inds],
            actions=sampled_actions,
            rewards=self.rewards[batch_inds],
            next_observations=self.next_observations[batch_inds],
            dones=self.dones[batch_inds],
        )
    
    def __len__(self) -> int:
        """使 `len(buffer)` 能够返回缓冲区当前存储的样本数量。"""
        return self.buffer_size if self.full else self.pos
