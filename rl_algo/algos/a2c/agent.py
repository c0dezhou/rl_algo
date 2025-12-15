# rlx/algos/a2c/agent.py 模块
"""Advantage Actor-Critic (A2C) 智能体实现。"""

from typing import Dict, Tuple

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.registry import registry
from rl_algo.core.types import Batch, Config


class A2CConfig(Config):
    """A2C 智能体的配置结构定义。"""

    gamma: float = 0.99
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    total_steps: int = 200_000


class ActorCritic(nn.Module):
    """共享特征提取的策略-价值联合网络。"""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs.view(obs.size(0), -1)
        hidden = self.body(x)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return logits, value


@registry.register_agent("a2c", A2CConfig)
class A2CAgent(BaseAgent):
    """优势 actor-critic，同策略单步更新。"""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("A2C 仅支持离散动作空间。")
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("A2C 需要连续观测空间。")

        obs_dim = int(torch.tensor(obs_space.shape).prod().item())
        action_dim = act_space.n

        self.ac = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.lr)

    def select_action(
        self,
        obs: torch.Tensor,
        explore: bool = True,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        obs = obs.to(self.device)
        logits, value = self.ac(obs)
        dist = Categorical(logits=logits)

        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=1)

        log_prob = dist.log_prob(action)
        return action, {"log_prob": log_prob.detach(), "value": value.detach()}

    def update(self, batch: Batch | Dict[str, torch.Tensor], global_step: int = 0) -> Dict[str, float]:
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).view(-1, 1)
        dones = batch["dones"].to(self.device).view(-1, 1)
        old_log_probs = batch.get("log_probs")

        logits, values = self.ac(observations)
        dist = Categorical(logits=logits)
        actions = actions.squeeze(-1).long() if actions.dim() > 1 else actions.long()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 计算折扣回报
        returns = torch.zeros_like(rewards, device=self.device)
        running_return = torch.zeros(1, device=self.device)
        for t in reversed(range(rewards.size(0))):
            running_return = rewards[t] + self.config.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        advantages = (returns - values.detach())

        policy_loss = -(advantages.squeeze(-1) * log_probs).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
        if old_log_probs is not None:
            with torch.no_grad():
                kl = (old_log_probs.to(self.device).view(-1) - log_probs).mean().abs()
            metrics["kl"] = kl.item()
        return metrics

    def save(self) -> Dict:
        return {"model": self.ac.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load(self, data: Dict) -> None:
        self.ac.load_state_dict(data["model"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
