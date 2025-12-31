import gymnasium as gym
import numpy as np
import torch
from rl_algo.core.types import Batch


class ReplayBuffer:
    # 最普通的 replay buffer：存 (s,a,r,s',done)，采样给 DQN 用
    def __init__(self, 
                 buffer_size: int, 
                 obs_space: gym.Space, 
                 act_space: gym.Space, 
                 device: torch.device
                ):
        self.buffer_size = buffer_size
        self.device = device

        # 从 Gym 空间中提取观测和动作的形状
        obs_shape = obs_space.shape
        act_shape = act_space.shape

        # 根据动作空间类型确定动作的数据类型
        obs_dtype = torch.float32
        act_dtype = (torch.int64 if isinstance(act_space, gym.spaces.Discrete)
                     else torch.float32)
        
        # 直接预分配张量，写起来简单也够快
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=obs_dtype, device=device)
        self.actions = torch.zeros((buffer_size, *act_shape), dtype=act_dtype, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=obs_dtype, device=device)
        # done 直接存成 0/1，算 target 的时候好乘
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        
        self.pos = 0      # 当前要插入数据的位置指针
        self.full = False # 标记缓冲区是否已满

    def add(self, 
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_obs: np.ndarray,
            done: bool):
        # 存一条 transition
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
    
    def sample(self, batch_size: int):
        # 随机采样一个 batch 给 DQN 更新
        current_size = self.buffer_size if self.full else self.pos
        if current_size < batch_size:
            raise ValueError(
                f"缓冲区中的样本数量 ({current_size}) 不足以采样一个大小为 ({batch_size}) 的批次。"
            )
        
        # 在有效的数据范围内 [0, current_size) 随机生成不重复的索引
        batch_inds = torch.randint(0, current_size, (batch_size,), device=self.device)

        # 根据索引提取数据
        sampled_actions = self.actions[batch_inds]
        
        # 离散动作：确保是 long 且 shape 是 (B,)，别带一个多余的维度
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
    
    def __len__(self):
        # len(buffer) 返回当前有多少条数据
        return self.buffer_size if self.full else self.pos
