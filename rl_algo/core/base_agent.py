import abc
from typing import Dict

import gymnasium as gym
import torch
from rl_algo.core.types import Config


class BaseAgent(abc.ABC):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device = torch.device("cpu"),
    ):
        # 所有 agent 都会用到的 4 个最基本东西
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self.device = device

    @abc.abstractmethod
    def select_action(self, obs: torch.Tensor, explore: bool = True, global_step: int = 0):
        # 训练时 explore=True（带探索），评估时 explore=False（尽量贪心/确定性）
        # 返回值约定：(action, info_dict)，其中 info_dict 可选放 log_prob/value 等
        raise NotImplementedError

    def update(self, batch, global_step: int = 0):
        # “批量更新”接口：DQN / PPO / GRPO / A2C / REINFORCE / MC 会用到
        return {}
    
    def train_step(self, transition):
        # “逐步更新”接口：Q-learning / SARSA 会用到
        return {}
    
    @abc.abstractmethod
    def save(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, data: Dict):
        raise NotImplementedError
