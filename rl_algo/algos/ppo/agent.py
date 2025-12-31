from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.types import Config


@dataclass
class PPOConfig(Config):
    gamma: float = 0.99
    total_steps: int = 200_000

    # 同策略（按轨迹更新）：一次 update 收集多少条完整轨迹
    batch_episodes: int = 8

    hidden_dim: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    use_gae: bool = True
    gae_lambda: float = 0.95


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor):
        x = obs.view(obs.size(0), -1)
        return self.net(x)


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor):
        x = obs.view(obs.size(0), -1)
        return self.net(x).squeeze(-1)


class PPOAgent(BaseAgent):
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
            raise ValueError("PPO 需要连续观测空间（Box）。")

        obs_dim = int(torch.tensor(obs_space.shape).prod().item())
        action_dim = act_space.n
        hidden_dim = int(getattr(config, "hidden_dim", 128))

        self.actor = ActorNet(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNet(obs_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(getattr(config, "actor_lr", 3e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(getattr(config, "critic_lr", 1e-3)))

    def select_action(
        self,
        obs: torch.Tensor,
        explore: bool = True,
        global_step: int = 0,
    ):
        obs = obs.to(self.device)
        logits = self.actor(obs)
        value = self.critic(obs)
        dist = Categorical(logits=logits)
        action = dist.sample() if explore else torch.argmax(logits, dim=1)
        log_prob = dist.log_prob(action)
        return action, {"log_prob": log_prob.detach(), "value": value.detach()}

    def update(self, batch: Dict[str, torch.Tensor], global_step: int = 0):
        # PPO-Clip 我这里就按最常见写法来：
        # ratio = exp(logp_new - logp_old)
        # policy_loss = -mean(min(ratio*A, clip(ratio)*A))
        # value_loss  = MSE(V(s), return)
        # loss = policy_loss + value_coef*value_loss - entropy_coef*entropy
        #
        # A/return 一般用 GAE（train.py 里算好传进来）。
        # 如果你没传 advantages/returns，我这里就用 MC return 粗算一个（能跑就行）。
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch.get("log_probs")
        advantages = batch.get("advantages")
        returns = batch.get("returns")

        actions = actions.squeeze(-1).long() if actions.dim() > 1 else actions.long()

        if old_log_probs is None:
            raise ValueError("PPO update 需要 `log_probs`（采样时的 old log_prob）。")
        old_log_probs = old_log_probs.to(self.device).view(-1).detach()

        if returns is None or advantages is None:
            rewards = batch["rewards"].to(self.device).view(-1)
            dones = batch["dones"].to(self.device).view(-1)
            with torch.no_grad():
                values_old = self.critic(observations).view(-1)

            # 没传 advantages/returns：就用 MC return 粗算一下
            returns_mc = torch.zeros_like(rewards, device=self.device)
            running = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            for t in reversed(range(rewards.shape[0])):
                running = rewards[t] + float(getattr(self.config, "gamma", 0.99)) * running * (1.0 - dones[t])
                returns_mc[t] = running
            returns = returns_mc.unsqueeze(1)
            advantages = (returns_mc - values_old).unsqueeze(1)
        else:
            returns = returns.to(self.device).view(-1)
            advantages = advantages.to(self.device).view(-1)

        # 标准化优势（PPO 更稳、对学习率不那么敏感）
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = int(observations.size(0))
        minibatch_size = int(getattr(self.config, "minibatch_size", 256))
        num_minibatches = max(1, (batch_size + minibatch_size - 1) // minibatch_size)

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        approx_kl_acc = 0.0

        for _ in range(int(getattr(self.config, "update_epochs", 4))):
            inds = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                mb_obs = observations[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_log_probs = old_log_probs[mb_inds]
                mb_adv = advantages[mb_inds]
                mb_returns = returns[mb_inds]

                logits = self.actor(mb_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # r_t = exp(logπ_new - logπ_old)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                clip_coef = float(getattr(self.config, "clip_coef", 0.2))
                clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)

                # L_clip = min(r_t*A, clip(r_t)*A)，取负号变成 loss
                surrogate1 = ratio * mb_adv
                surrogate2 = clipped * mb_adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # actor: policy_loss - entropy_coef * entropy
                actor_loss = policy_loss - float(getattr(self.config, "entropy_coef", 0.01)) * entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                max_grad_norm = float(getattr(self.config, "max_grad_norm", 0.0))
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                self.actor_optimizer.step()

                # critic: 让 V(s) 拟合 return
                values = self.critic(mb_obs)
                value_loss = F.mse_loss(values, mb_returns)
                critic_loss = float(getattr(self.config, "value_coef", 0.5)) * value_loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().abs()

                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                entropy_acc += float(entropy.item())
                approx_kl_acc += float(approx_kl.item())

        denom = int(getattr(self.config, "update_epochs", 4)) * num_minibatches
        return {
            "policy_loss": policy_loss_acc / denom,
            "value_loss": value_loss_acc / denom,
            "entropy": entropy_acc / denom,
            "approx_kl": approx_kl_acc / denom,
        }

    def save(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load(self, data: Dict):
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        if "actor_optimizer" in data:
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        if "critic_optimizer" in data:
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
