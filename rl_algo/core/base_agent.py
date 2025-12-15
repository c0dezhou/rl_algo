# rlx/core/base_agent.py 模块

"""
定义了所有强化学习智能体 (Agent) 的抽象基类 (Abstract Base Class, ABC)。
这个文件是整个框架的核心契约, 任何新的算法实现都必须继承自 `BaseAgent` 类,
并实现其所有抽象方法, 以确保与训练框架的兼容性。
"""

import abc
from typing import Any, Dict, Tuple

import gymnasium as gym
import torch
from rl_algo.core.types import Batch, Transition, Config

class BaseAgent(abc.ABC):
    """
    所有强化学习智能体的抽象基类。

    它定义了智能体必须具备的核心接口：
    - `__init__`: 初始化智能体, 设置环境空间、配置和设备。
    - `select_action`: 根据当前观测选择一个动作。
    - `update`: (可选) 使用一批数据更新模型, 主要用于异策略和基于轨迹的同策略算法。
    - `train_step`: (可选) 使用单步转移数据更新模型, 主要用于传统的表格型算法。
    - `save`: 保存智能体的内部状态 (如模型权重、Q-table)。
    - `load`: 从一个状态字典加载智能体的状态。
    """

    def __init__(
            self,
            obs_space: gym.Space,
            act_space: gym.Space,
            config: Config,
            device: torch.device = torch.device("cuda"),
    ):
        """
        初始化智能体。

        Args:
            obs_space (gym.Space): 环境的观测空间。
            act_space (gym.Space): 环境的动作空间。
            config (Config): 包含所有超参数的 Pydantic 配置对象。
            device (torch.device): 计算设备 (例如, 'cpu' 或 'cuda')。
        """
        self.obs_space = obs_space  # 环境的观测空间
        self.act_space = act_space  # 环境的动作空间
        self.config = config        # 算法的超参数配置
        self.device = device        # PyTorch 计算设备

    @abc.abstractmethod
    def select_action(self, obs: torch.Tensor, explore: bool = True, global_step: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        根据当前观测选择一个动作。这是一个必须被子类实现的抽象方法。

        Args:
            obs (torch.Tensor): 当前的环境观测, 形状通常为 (N, *obs_shape), 其中 N 是批次大小。
            explore (bool): 一个标志, 指示是否应进行探索。
                          - `True` (训练时): 智能体可以采取随机或探索性动作 (例如, epsilon-greedy)。
                          - `False` (评估时): 智能体应采取其学到的最优策略 (贪心动作)。
            global_step (int): 当前训练的总步数, 可用于实现随时间变化的探索策略 (例如, epsilon 衰减)。
        
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
            - 一个动作张量, 形状为 (N, *action_shape)。
            - 一个包含辅助信息的字典, 例如动作的对数概率 (`log_prob`) 或价值函数估计 (`value`), 这对于某些算法 (如 PPO) 是必需的。
        """
        raise NotImplementedError

    def update(self, batch: Batch | Dict[str, torch.Tensor], global_step: int = 0) -> Dict[str, float]:
        """
        使用一批 (batch) 经验数据来更新智能体的参数 (例如, 神经网络的权重)。
        这是一个可选方法, 默认实现为空。主要用于异策略算法 (如 DQN) 和基于轨迹的同策略算法 (如 PPO)。

        Args:
            batch (Batch | Dict[str, torch.Tensor]): 一批经验数据, 通常是一个包含 "observations", "actions", "rewards" 等键的字典。
            global_step (int): 当前训练的总步数, 可用于学习率调度等。
        
        Returns:
            Dict[str, float]: 一个包含训练指标 (metrics) 的字典, 用于日志记录 (例如, `{"loss": 0.5, "q_value": 1.2}`)。
        """
        return {}
    
    def train_step(self, transition: Transition) -> Dict[str, float]:
        """
        使用单个时间步的转移 (transition) 数据来更新模型。
        这是一个可选方法, 默认实现为空。主要用于传统的、每步更新的同策略算法 (如 Q-learning, SARSA)。

        Args: 
            transition (Transition): 一个包含单个时间步数据的字典, 至少应包含 'obs', 'action', 'reward', 'next_obs', 'done'。
                                     某些算法 (如 SARSA) 可能还要求包含 'next_action'。
        
        Returns:
            Dict[str, float]: 一个包含训练指标 (metrics) 的字典, 用于日志记录。
        """
        return {}
    
    @abc.abstractmethod
    def save(self) -> Dict:
        """
        保存智能体的内部状态 (例如, Q-table 或神经网络权重) 并返回一个可序列化的字典。
        这是一个必须被子类实现的抽象方法。
        
        Returns:
            Dict: 包含智能体状态的字典。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, data: Dict) -> None:
        """
        从一个字典加载智能体的状态。
        这是一个必须被子类实现的抽象方法。

        Args:
            data (Dict): 包含智能体状态的字典, 通常由 `save` 方法生成。
        """
        raise NotImplementedError
