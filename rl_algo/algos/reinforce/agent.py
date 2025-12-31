from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.types import Config


@dataclass
class ReinforceConfig(Config):
    """REINFORCE 智能体的配置结构定义。"""

    gamma: float = 0.99
    lr: float = 3e-4
    total_steps: int = 100_000


class PolicyNet(nn.Module):
    """简单的策略网络。"""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor):
        x = obs.view(obs.size(0), -1)
        return self.model(x)


class ReinforceAgent(BaseAgent):
    """无基线的 REINFORCE 策略梯度。"""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("REINFORCE 仅支持离散动作空间。")
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("REINFORCE 需要连续观测空间。")

        obs_dim = int(torch.tensor(obs_space.shape).prod().item())
        action_dim = act_space.n

        self.policy = PolicyNet(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

    def select_action(
        self,
        obs: torch.Tensor,
        explore: bool = True,
        global_step: int = 0,
    ):
        obs = obs.to(self.device)
        logits = self.policy(obs)
        dist = Categorical(logits=logits)

        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=1)

        log_prob = dist.log_prob(action)
        return action, {"log_prob": log_prob.detach()}

    def update(self, batch, global_step: int = 0):
        # REINFORCE 就一句话：
        # loss = -E[ G_t * logπ(a_t|s_t) ]
        # G_t 是折扣回报，从后往前算；我这里顺手把 returns 标准化一下，训练会稳一点
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).view(-1, 1)
        dones = batch["dones"].to(self.device).view(-1, 1)

        # 1) 先算当前策略对每一步动作的 log_prob
        logits = self.policy(observations)
        dist = Categorical(logits=logits)
        actions = actions.squeeze(-1).long() if actions.dim() > 1 else actions.long()
        log_probs = dist.log_prob(actions)

        # 2) 从后往前算折扣回报 G_t（dones=1 时把 running_return 清零）
        returns = torch.zeros_like(rewards, device=self.device)
        running_return = torch.zeros(1, device=self.device)
        for t in reversed(range(rewards.size(0))):
            running_return = rewards[t] + self.config.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        # 3) 做个简单标准化：减少尺度问题，让训练更稳（作业里常用的小技巧）
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        # 4) 最大化 E[G_t * logπ] <=> 最小化 -(G_t * logπ)
        policy_loss = -(returns.squeeze(-1) * log_probs).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return {"policy_loss": policy_loss.item()}

    def save(self):
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load(self, data: Dict):
        self.policy.load_state_dict(data["policy"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
