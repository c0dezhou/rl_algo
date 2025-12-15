# rlx/algos/ppo/agent.py 模块
"""PPO (Proximal Policy Optimization) 智能体实现。"""

from typing import Dict, Tuple

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.registry import registry
from rl_algo.core.types import Batch, Config


class PPOConfig(Config):
    """PPO 智能体的配置结构定义。"""

    gamma: float = 0.99
    lr: float = 3e-4
    total_steps: int = 1_000_000
    update_epochs: int = 4
    minibatch_size: int = 64
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    use_gae: bool = True
    gae_lambda: float = 0.95


class ActorCritic(nn.Module):
    """共享干路的策略-价值网络。"""

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


@registry.register_agent("ppo", PPOConfig)
class PPOAgent(BaseAgent):
    """剪切策略梯度的离线批更新版本。"""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("PPO 目前仅支持离散动作空间。")
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("PPO 需要连续观测空间。")

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
        advantages = batch.get("advantages")
        returns = batch.get("returns")

        # 重新计算当前策略的输出
        logits, values = self.ac(observations)
        dist = Categorical(logits=logits)
        actions = actions.squeeze(-1).long() if actions.dim() > 1 else actions.long()
        current_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 如果没有预先计算优势/回报，则在线计算
        if returns is None:
            returns = torch.zeros_like(rewards, device=self.device)
            running_return = torch.zeros(1, device=self.device)
            for t in reversed(range(rewards.size(0))):
                running_return = rewards[t] + self.config.gamma * running_return * (1.0 - dones[t])
                returns[t] = running_return
        else:
            returns = returns.to(self.device)

        if advantages is None:
            advantages = returns - values.detach()
        else:
            advantages = advantages.to(self.device)

        advantages = advantages.squeeze(-1)
        returns = returns.view(-1, 1)
        values = values.view(-1, 1)

        # 归一化优势有助于稳定训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if old_log_probs is None:
            old_log_probs = current_log_probs.detach()
        else:
            old_log_probs = old_log_probs.to(self.device).view(-1).detach()

        batch_size = observations.size(0)
        inds = torch.randperm(batch_size, device=self.device)

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0

        for _ in range(self.config.update_epochs):
            for start in range(0, batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = inds[start:end]

                mb_logits, mb_values = self.ac(observations[mb_inds])
                mb_dist = Categorical(logits=mb_logits)
                mb_log_probs = mb_dist.log_prob(actions[mb_inds])
                mb_entropy = mb_dist.entropy().mean()

                ratio = (mb_log_probs - old_log_probs[mb_inds]).exp()
                mb_adv = advantages[mb_inds]

                surrogate1 = ratio * mb_adv
                surrogate2 = torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * mb_adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                mb_returns = returns[mb_inds]
                value_loss = 0.5 * F.mse_loss(mb_values, mb_returns)

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * mb_entropy

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_loss_acc += policy_loss.item()
                value_loss_acc += value_loss.item()
                entropy_acc += mb_entropy.item()

        num_updates = self.config.update_epochs * max(1, (batch_size + self.config.minibatch_size - 1) // self.config.minibatch_size)
        return {
            "policy_loss": policy_loss_acc / num_updates,
            "value_loss": value_loss_acc / num_updates,
            "entropy": entropy_acc / num_updates,
        }

    def save(self) -> Dict:
        return {
            "model": self.ac.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load(self, data: Dict) -> None:
        self.ac.load_state_dict(data["model"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
