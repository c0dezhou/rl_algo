from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.types import Config


@dataclass
class MCConfig(Config):
    """蒙特卡洛控制智能体的配置结构定义。"""

    gamma: float = 0.99
    epsilon: float = 1.0
    total_steps: int = 50000
    num_bins: int = 10
    clip_obs: float = 10.0


class MCAgent(BaseAgent):
    """同策略、首访蒙特卡洛控制智能体。"""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("MC（蒙特卡洛控制）目前只支持离散动作空间。")

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
            raise TypeError("MC（蒙特卡洛控制）目前只支持离散观测空间或 Box 观测空间。")

        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.gamma = config.gamma
        self.epsilon = config.epsilon

    def discretize_state(self, obs: np.ndarray):
        # 连续观测要先离散化，不然没法当字典 key
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

        # epsilon-greedy：随机探索 / 否则选 Q 最大的动作
        if explore and np.random.rand() < self.epsilon:
            action = self.act_space.sample()
        else:
            action = int(np.argmax(self.q_table[state]))

        return action, {}

    def update(self, batch: Dict[str, torch.Tensor], global_step: int):
        # MC（first-visit）就做两件事：
        # 1) 先把整条轨迹的 return 从后往前算出来：G <- γG + r_t
        # 2) 同一条轨迹里每个 (s,a) 只更新第一次出现那次，然后用平均回报当 Q
        rewards = batch["rewards"].cpu().numpy()
        observations = batch["observations"].cpu().numpy()
        actions = batch["actions"].cpu().numpy()

        G = 0.0
        visited_sa_pairs = set()

        for t in reversed(range(len(rewards))):
            G = self.gamma * G + rewards[t]

            obs_t = observations[t].flatten()
            state_t = self.discretize_state(obs_t)
            action_t = int(actions[t])
            sa_pair = (state_t, action_t)

            if sa_pair not in visited_sa_pairs:
                # Q(s,a) 就用“这个 (s,a) 见过的 return 的平均值”
                self.returns_sum[sa_pair] += G
                self.returns_count[sa_pair] += 1.0
                self.q_table[state_t][action_t] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]
                visited_sa_pairs.add(sa_pair)

        # epsilon 慢慢变小
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

        return {"q_table_size": float(len(self.q_table)), "epsilon": float(self.epsilon)}

    def save(self):
        return {
            "q_table": dict(self.q_table),
            "returns_sum": dict(self.returns_sum),
            "returns_count": dict(self.returns_count),
        }

    def load(self, state: Dict[str, Dict]):
        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.q_table.update(state["q_table"])
        self.returns_sum.update(state["returns_sum"])
        self.returns_count.update(state["returns_count"])
