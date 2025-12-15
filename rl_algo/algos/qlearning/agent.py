# rlx/algos/qlearning/agent.py 模块
"""在连续空间使用简单离散化的表格型 Q-Learning 智能体。"""

from collections import defaultdict
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.registry import registry
from rl_algo.core.types import Config, Transition


class QLearningConfig(Config):
    """Q-Learning 智能体的配置结构定义。"""

    lr: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    total_steps: int = 50000
    num_bins: int = 10
    clip_obs: float = 10.0


@registry.register_agent("qlearning", QLearningConfig)
class QLearningAgent(BaseAgent):
    """经典表格型 Q-Learning, 使用 epsilon-贪婪探索。"""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: str,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("Q-Learning agent currently only supports discrete action spaces.")

        self.state_bins: List[np.ndarray] | None = None
        self._bin_low: np.ndarray | None = None
        self._bin_high: np.ndarray | None = None

        if isinstance(obs_space, gym.spaces.Box):
            num_bins = getattr(config, "num_bins", 10)
            clip_obs = abs(getattr(config, "clip_obs", 10.0))
            bin_lows, bin_highs = [], []
            state_bins: List[np.ndarray] = []
            for i in range(obs_space.shape[0]):
                low = obs_space.low[i]
                high = obs_space.high[i]

                if not np.isfinite(low) or low < -clip_obs:
                    low = -clip_obs
                if not np.isfinite(high) or high > clip_obs:
                    high = clip_obs
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
            raise TypeError("Q-Learning agent only supports discrete or box observation spaces.")

        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = config.epsilon

    def discretize_state(self, obs: np.ndarray) -> Tuple[int, ...]:
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
    ) -> Tuple[int, Dict]:
        obs_np = obs.cpu().numpy().flatten()
        state = self.discretize_state(obs_np)

        if explore and np.random.rand() < self.epsilon:
            action = self.act_space.sample()
        else:
            action = int(np.argmax(self.q_table[state]))

        return action, {}

    def train_step(self, transition: Transition) -> Dict[str, float]:
        obs = transition["obs"].cpu().numpy().flatten()
        next_obs = transition["next_obs"].cpu().numpy().flatten()

        action = transition["action"]
        if isinstance(action, torch.Tensor):
            action = int(action.item())

        reward = float(transition["reward"])
        done = bool(transition["done"])

        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        target = reward + self.gamma * next_max * (1 - float(done))
        new_value = (1 - self.lr) * old_value + self.lr * target
        self.q_table[state][action] = new_value

        if self.epsilon > 0.01:
            self.epsilon *= 0.999

        return {"q_value": float(new_value), "epsilon": float(self.epsilon)}

    def save(self) -> Dict[str, Dict]:
        return {"q_table": dict(self.q_table)}

    def load(self, state: Dict[str, Dict]) -> None:
        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        self.q_table.update(state["q_table"])
