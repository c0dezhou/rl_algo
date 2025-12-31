from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.types import Config
from rl_algo.core.utils import get_schedule_fn


class QNetwork(nn.Module):
    """简单的 MLP：给定状态，输出每个动作的 Q 值。"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        return self.model(x)


@dataclass
class DQNConfig(Config):
    """DQN的超参数配置"""

    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 100_000
    batch_size: int = 64
    train_frequency: int = 4
    gradient_steps: int = 1
    learning_starts: int = 1000
    target_update_frequency: int = 1000
    double_dqn: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    # `--total-steps` 经常会被覆盖；用 fraction 更稳（默认 10% 的交互步数用于衰减）。
    epsilon_decay_fraction: float = 0.1
    epsilon_decay_steps: Optional[int] = None
    total_steps: int = 200_000
    max_grad_norm: Optional[float] = 10.0


class DQNAgent(BaseAgent):
    """DQN 智能体。"""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("DQN 只支持离散动作空间。")

        if isinstance(obs_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(obs_space.shape))
        else:
            raise ValueError("DQN 需要连续观测空间（Box）。")

        # policy_net：当前 Q 网络 Q(s,a;θ)
        # target_net：目标网络 Q(s,a;θ-)（计算 TD 目标时更稳定）
        # 初始化时把 policy 的参数复制给 target（两者一开始完全一致）

        action_dim = act_space.n
        self.policy_net = QNetwork(self.obs_dim, action_dim).to(device)
        self.target_net = QNetwork(self.obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
        # Huber loss（SmoothL1）比 MSE 对大 TD 误差更不敏感，训练更稳一点
        self.loss_fn = nn.SmoothL1Loss()

        self.update_steps = 0
        # epsilon 衰减：默认按 total_steps 的一个比例来决定衰减步数（这样你覆盖 --total-steps 时更稳）
        decay_steps = getattr(config, "epsilon_decay_steps", None)
        if decay_steps is None:
            frac = float(getattr(config, "epsilon_decay_fraction", 0.1))
            decay_steps = max(1, int(frac * int(getattr(config, "total_steps", 1))))
        self.epsilon_schedule = get_schedule_fn(config.epsilon_start, config.epsilon_end, float(decay_steps))
        self.epsilon = config.epsilon_start

    def select_action(
        self,
        obs: torch.Tensor,
        explore: bool = True,
        global_step: int = 0,
    ):
        obs = obs.to(self.device)

        if explore:
            # 训练时：epsilon-greedy（有概率随机探索）
            self.epsilon = self.epsilon_schedule(int(global_step))
            if np.random.rand() < self.epsilon:
                action = self.act_space.sample()
                return torch.tensor([action], device=self.device, dtype=torch.int64), {"epsilon": self.epsilon}
        else:
            # 评估时：纯贪心
            self.epsilon = 0.0

        with torch.no_grad():
            q_values = self.policy_net(obs)
            action = int(torch.argmax(q_values, dim=1).item())
        return torch.tensor([action], device=self.device, dtype=torch.int64), {"epsilon": self.epsilon}

    def update(self, batch, global_step: int = 0):
        # DQN 的更新我就按这一句来写：
        # y = r + γ*(1-done)*max_a' Q_target(s',a')
        # loss = huber(Q_policy(s,a), y)
        # （double_dqn 开的话：argmax 用 policy_net，取值用 target_net）
        observations = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        rewards = batch.rewards.to(self.device)
        next_observations = batch.next_observations.to(self.device)
        dones = batch.dones.to(self.device)

        # Q(s,a)：从网络输出的所有动作 Q 值里，取出实际执行的动作对应的那一列
        q_values = self.policy_net(observations).gather(1, actions.long())

        with torch.no_grad():
            if bool(getattr(self.config, "double_dqn", True)):
                # 双重 DQN（Double DQN）：用 policy_net 选动作，用 target_net 评估这个动作的 Q 值
                next_actions = self.policy_net(next_observations).argmax(dim=1, keepdim=True)
                max_next_q = self.target_net(next_observations).gather(1, next_actions)
            else:
                # 普通 DQN：直接在 target_net 上取 max_a' Q(s',a')
                next_q_values = self.target_net(next_observations)
                max_next_q = next_q_values.max(dim=1, keepdim=True).values
            # TD target: y = r + γ*(1-done)*max_next_q
            targets = rewards + self.config.gamma * (1.0 - dones) * max_next_q

        # loss = Huber(Q(s,a), y)
        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        max_grad_norm = getattr(self.config, "max_grad_norm", None)
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.config.target_update_frequency == 0:
            # 周期性同步目标网络
            self.target_net.load_state_dict(self.policy_net.state_dict())

        metrics = {
            "loss": loss.item(),
            "avg_q": q_values.mean().item(),
            "epsilon": self.epsilon,
        }
        return metrics

    def save(self):
        return {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_steps": self.update_steps,
        }

    def load(self, data: Dict[str, Any]):
        self.policy_net.load_state_dict(data["policy_state_dict"])
        self.target_net.load_state_dict(data["target_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.epsilon = data.get("epsilon", self.config.epsilon_end)
        self.update_steps = data.get("update_steps", 0)
