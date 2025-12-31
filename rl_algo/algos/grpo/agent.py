from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.types import Config


@dataclass
class GRPOConfig(Config):
    gamma: float = 0.99
    total_steps: int = 200_000

    # trajectory batch/group 相关（由 train.py 使用；agent 本身只关心 update）
    batch_episodes: int = 8
    group_size: int = 8

    hidden_dim: int = 128
    actor_lr: float = 3e-4

    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


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


class GRPOAgent(BaseAgent):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("GRPO 仅支持离散动作空间。")
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("GRPO 需要连续观测空间（Box）。")

        obs_dim = int(torch.tensor(obs_space.shape).prod().item())
        action_dim = act_space.n
        hidden_dim = int(getattr(config, "hidden_dim", 128))

        self.actor = ActorNet(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(getattr(config, "actor_lr", 3e-4)))

    def select_action(
        self,
        obs: torch.Tensor,
        explore: bool = True,
        global_step: int = 0,
    ):
        obs = obs.to(self.device)
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample() if explore else torch.argmax(logits, dim=1)
        log_prob = dist.log_prob(action)
        return action, {"log_prob": log_prob.detach()}

    def update(self, batch: Dict[str, torch.Tensor], global_step: int = 0):
        # GRPO 可以当成“没有 critic 的 PPO-Clip”
        # - 优势 A 不在 agent 里算，训练入口已经给了（组内回报归一化 + 广播到每一步）
        # - update 里就是标准 PPO-Clip 那一套：
        #   ratio = exp(logp_new - logp_old)
        #   L = -mean(min(ratio*A, clip(ratio)*A)) - entropy_coef * entropy
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device).view(-1).detach()
        advantages = batch["advantages"].to(self.device).view(-1)

        actions = actions.squeeze(-1).long() if actions.dim() > 1 else actions.long()
        # 优势再标准化一下，训练会稳一点
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = int(observations.size(0))
        minibatch_size = int(getattr(self.config, "minibatch_size", 256))
        num_minibatches = max(1, (batch_size + minibatch_size - 1) // minibatch_size)

        policy_loss_acc = 0.0
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

                logits = self.actor(mb_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # r_t = exp(logπ_new - logπ_old)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                clip_ratio = torch.clamp(
                    ratio,
                    1.0 - float(getattr(self.config, "clip_coef", 0.2)),
                    1.0 + float(getattr(self.config, "clip_coef", 0.2)),
                )

                # L_clip = min(r_t*A, clip(r_t)*A)，取负号变成 loss
                surrogate1 = ratio * mb_adv
                surrogate2 = clip_ratio * mb_adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # 只有 actor：policy_loss - entropy_coef * entropy
                loss = policy_loss - float(getattr(self.config, "entropy_coef", 0.01)) * entropy

                self.optimizer.zero_grad()
                loss.backward()
                max_grad_norm = float(getattr(self.config, "max_grad_norm", 0.0))
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().abs()

                policy_loss_acc += float(policy_loss.item())
                entropy_acc += float(entropy.item())
                approx_kl_acc += float(approx_kl.item())

        denom = int(getattr(self.config, "update_epochs", 4)) * num_minibatches
        return {
            "policy_loss": policy_loss_acc / denom,
            "entropy": entropy_acc / denom,
            "approx_kl": approx_kl_acc / denom,
        }

    def save(self):
        return {"actor": self.actor.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load(self, data: Dict):
        self.actor.load_state_dict(data["actor"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
