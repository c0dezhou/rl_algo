from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.types import Config, Transition


@dataclass
class SarsaConfig(Config):
    """SARSA 智能体的配置结构定义。"""

    lr: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    total_steps: int = 50000
    num_bins: int = 10
    clip_obs: float = 10.0


class SarsaAgent(BaseAgent):
    """支持离散与连续观测的表格型 SARSA 智能体。"""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("SARSA 目前只支持离散动作空间。")

        self.state_bins: List[np.ndarray] | None = None
        self._bin_low: np.ndarray | None = None
        self._bin_high: np.ndarray | None = None

        if isinstance(obs_space, gym.spaces.Box):
            num_bins = getattr(config, "num_bins", 10)
            clip_obs = abs(getattr(config, "clip_obs", 10.0))
            obs_low = np.asarray(obs_space.low, dtype=np.float32)
            obs_high = np.asarray(obs_space.high, dtype=np.float32)

            if (
                obs_space.shape == (4,)
                and np.isfinite(obs_low[0])
                and np.isfinite(obs_high[0])
                and np.isfinite(obs_low[2])
                and np.isfinite(obs_high[2])
                and (not np.isfinite(obs_low[1]))
                and (not np.isfinite(obs_high[1]))
                and (not np.isfinite(obs_low[3]))
                and (not np.isfinite(obs_high[3]))
            ):
                obs_low = np.asarray([-2.4, -3.0, -0.2095, -3.5], dtype=np.float32)
                obs_high = np.asarray([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)
            else:
                obs_low = np.clip(obs_low, -clip_obs, clip_obs)
                obs_high = np.clip(obs_high, -clip_obs, clip_obs)

            bin_lows, bin_highs = [], []
            state_bins: List[np.ndarray] = []
            for i in range(obs_space.shape[0]):
                low = float(obs_low[i])
                high = float(obs_high[i])
                if low >= high:
                    span = clip_obs if clip_obs > 0 else 1.0
                    low, high = -span, span

                bins = np.linspace(low, high, num_bins)
                state_bins.append(bins)
                bin_lows.append(bins[0])
                bin_highs.append(bins[-1])

            self.state_bins = state_bins
            self._bin_low = np.asarray(bin_lows, dtype=np.float32)
            self._bin_high = np.asarray(bin_highs, dtype=np.float32)
        elif not isinstance(obs_space, gym.spaces.Discrete):
            raise TypeError("SARSA 目前只支持离散观测空间或 Box 观测空间。")

        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = config.epsilon

    def discretize_state(self, obs: np.ndarray):
        """将(可能是连续的)观测映射为可哈希的离散状态。"""
        if self.state_bins is None or self._bin_low is None or self._bin_high is None:
            return tuple(obs) if isinstance(obs, (list, np.ndarray)) else (obs,)

        obs_arr = np.asarray(obs, dtype=np.float32)
        clipped = np.clip(obs_arr, self._bin_low, self._bin_high)
        state = [
            int(np.digitize(val, self.state_bins[i]))
            for i, val in enumerate(clipped)
        ]
        return tuple(state)

    def select_action(
        self,
        obs: torch.Tensor,
        global_step: int = 0,
        explore: bool = True,
    ):
        obs_np = obs.cpu().numpy().flatten()
        state = self.discretize_state(obs_np)

        if explore and np.random.rand() < self.epsilon:
            action = self.act_space.sample()
        else:
            action = int(np.argmax(self.q_table[state]))

        return action, {}

    def train_step(self, transition: Transition):
        # SARSA 更新也很简单：
        # y = r + γ*(1-done)*Q(s', a')   (a' 是下一步真的选到的动作)
        # Q(s,a) <- Q(s,a) + α*(y - Q(s,a))
        obs = transition["obs"].cpu().numpy().flatten()
        next_obs = transition["next_obs"].cpu().numpy().flatten()

        action = transition["action"]
        if isinstance(action, torch.Tensor):
            action = int(action.item())

        next_action = transition["next_action"]
        if isinstance(next_action, torch.Tensor):
            next_action = int(next_action.item())

        reward = float(transition["reward"])
        done = bool(transition["done"])

        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        old_value = self.q_table[state][action]
        next_value = self.q_table[next_state][next_action]
        # TD target: y = r + γ*(1-done)*Q(s',a')
        target = reward + self.gamma * next_value * (1 - float(done))
        # 表格更新：Q <- Q + α(y - Q)
        new_value = old_value + self.lr * (target - old_value)
        self.q_table[state][action] = new_value

        # epsilon 衰减：逐渐减少探索
        if self.epsilon > 0.01:
            self.epsilon *= 0.9999

        return {"q_value": float(new_value), "epsilon": float(self.epsilon)}

    def save(self):
        return {"q_table": dict(self.q_table)}

    def load(self, state: Dict[str, Dict]):
        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        self.q_table.update(state["q_table"])
