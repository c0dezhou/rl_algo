# rlx/algos/dqn/agent.py
"""
Deep Q-Network (DQN) agent implementation.

This agent relies on an experience replay buffer (handled by the training loop)
and maintains a target network for stabilising Q-value updates. Epsilon-greedy
exploration is controlled via a linear schedule.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rlx.core.base_agent import BaseAgent
from rlx.core.registry import registry
from rlx.core.types import Batch, Config
from rlx.core.utils import get_schedule_fn


class QNetwork(nn.Module):
    """Simple MLP used by both policy and target networks."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.model(x)


class DQNConfig(Config):
    """Configuration schema for the DQN agent."""

    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 100_000
    batch_size: int = 64
    train_frequency: int = 4
    target_update_frequency: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    total_steps: int = 200_000
    max_grad_norm: Optional[float] = 10.0


@registry.register_agent("dqn", DQNConfig)
class DQNAgent(BaseAgent):
    """Classic DQN agent with target network and epsilon schedule."""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("DQN agent currently supports discrete action spaces only.")

        if isinstance(obs_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(obs_space.shape))
        else:
            raise ValueError("DQN agent expects a Box observation space.")

        action_dim = act_space.n
        self.policy_net = QNetwork(self.obs_dim, action_dim).to(device)
        self.target_net = QNetwork(self.obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.update_steps = 0
        self.epsilon_schedule = get_schedule_fn(
            config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps
        )
        self.epsilon = config.epsilon_start

    def select_action(
        self,
        obs: torch.Tensor,
        explore: bool = True,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = obs.to(self.device)

        if explore:
            self.epsilon = self.epsilon_schedule(global_step)
            if np.random.rand() < self.epsilon:
                action = self.act_space.sample()
                return torch.tensor([action], device=self.device, dtype=torch.int64), {"epsilon": self.epsilon}
        else:
            self.epsilon = 0.0

        with torch.no_grad():
            q_values = self.policy_net(obs)
            action = int(torch.argmax(q_values, dim=1).item())
        return torch.tensor([action], device=self.device, dtype=torch.int64), {"epsilon": self.epsilon}

    def update(self, batch: Batch, global_step: int = 0) -> Dict[str, float]:
        observations = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        rewards = batch.rewards.to(self.device)
        next_observations = batch.next_observations.to(self.device)
        dones = batch.dones.to(self.device)

        q_values = self.policy_net(observations).gather(1, actions.long())

        with torch.no_grad():
            next_q_values = self.target_net(next_observations)
            max_next_q = next_q_values.max(dim=1, keepdim=True).values
            targets = rewards + self.config.gamma * (1.0 - dones) * max_next_q

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        max_grad_norm = getattr(self.config, "max_grad_norm", None)
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.config.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        metrics = {
            "loss": loss.item(),
            "avg_q": q_values.mean().item(),
            "epsilon": self.epsilon,
        }
        return metrics

    def save(self) -> Dict[str, Any]:
        return {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_steps": self.update_steps,
        }

    def load(self, data: Dict[str, Any]) -> None:
        self.policy_net.load_state_dict(data["policy_state_dict"])
        self.target_net.load_state_dict(data["target_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.epsilon = data.get("epsilon", self.config.epsilon_end)
        self.update_steps = data.get("update_steps", 0)
