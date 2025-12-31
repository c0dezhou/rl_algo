from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.types import Config


@dataclass
class ScalingGRPOConfig(Config):
    # 基本沿用 GRPOConfig，只加了“Scaling GRPO”需要的几个开关
    gamma: float = 0.99
    total_steps: int = 200_000

    batch_episodes: int = 8
    group_size: int = 8

    hidden_dim: int = 128
    actor_lr: float = 3e-4

    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # (1) Unbiased KL estimate + 可选 KL penalty
    kl_coef: float = 0.0  # 默认 0：不改变原 GRPO 行为

    # (2) Off-policy episode masking（默认关）
    offpolicy_mask_enabled: bool = False
    offpolicy_kl_threshold: float = 0.1
    offpolicy_adv_threshold: float = 0.0

    # (3) Keep sampling mask（动作无效 mask）
    keep_action_mask: bool = True
    action_mask_value: float = -1e9


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

    def forward(self, obs: torch.Tensor, routing_state: Any = None):
        # routing_state：先留着，后面想做路由/门控再用
        x = obs.view(obs.size(0), -1)
        return self.net(x)


class ScalingGRPOAgent(BaseAgent):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: Config,
        device: torch.device,
    ):
        super().__init__(obs_space, act_space, config, device)

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("ScalingGRPO 仅支持离散动作空间。")
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("ScalingGRPO 需要连续观测空间（Box）。")

        obs_dim = int(torch.tensor(obs_space.shape).prod().item())
        action_dim = int(act_space.n)
        hidden_dim = int(getattr(config, "hidden_dim", 128))

        self.actor = ActorNet(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(getattr(config, "actor_lr", 3e-4)))

        self._action_dim = action_dim

    def select_action(
        self,
        obs: torch.Tensor,
        explore: bool = True,
        global_step: int = 0,
    ):
        obs = obs.to(self.device)
        logits = self.actor(obs, routing_state=None)
        dist = Categorical(logits=logits)
        action = dist.sample() if explore else torch.argmax(logits, dim=1)
        log_prob = dist.log_prob(action)
        return action, {"log_prob": log_prob.detach()}

    @staticmethod
    def _masked_mean(x: torch.Tensor, keep: Optional[torch.Tensor]):
        if keep is None:
            return x.mean()
        w = keep.to(torch.float32)
        return (x * w).sum() / w.sum().clamp_min(1.0)

    def _prepare_action_mask(self, batch: Dict[str, torch.Tensor], batch_size: int):
        if not bool(getattr(self.config, "keep_action_mask", True)):
            return None
        action_mask = batch.get("action_mask", None)
        if action_mask is None:
            return None

        action_mask = action_mask.to(self.device)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask != 0

        if (
            action_mask.dim() != 2
            or int(action_mask.shape[0]) != int(batch_size)
            or int(action_mask.shape[1]) != int(self._action_dim)
        ):
            raise ValueError(
                f"action_mask 需要是 [B, action_dim]，但拿到 shape={tuple(action_mask.shape)}，"
                f"期望=({batch_size},{self._action_dim})"
            )

        if bool((action_mask.sum(dim=1) == 0).any().item()):
            raise ValueError("action_mask 存在某些 step 全为无效动作（请检查数据管线）")

        return action_mask

    def _mask_logits(self, logits: torch.Tensor, action_mask: Optional[torch.Tensor]):
        if action_mask is None:
            return logits
        fill = float(getattr(self.config, "action_mask_value", -1e9))
        return logits.masked_fill(~action_mask, fill)

    def update(self, batch: Dict[str, torch.Tensor], global_step: int = 0):
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device).view(-1).detach()
        advantages = batch["advantages"].to(self.device).view(-1)

        # actions 允许 [B] 或 [B,1]
        actions = actions.squeeze(-1).long() if actions.dim() > 1 else actions.long()

        # 标准化优势：避免尺度过大/过小导致学习不稳定
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = int(observations.size(0))
        minibatch_size = int(getattr(self.config, "minibatch_size", 256))
        num_minibatches = max(1, (batch_size + minibatch_size - 1) // minibatch_size)

        action_mask = self._prepare_action_mask(batch, batch_size)

        episode_ids = batch.get("episode_ids", None)
        if episode_ids is not None:
            episode_ids = episode_ids.to(self.device).view(-1)

        routing_state = batch.get("routing_state", None)  # 现在不用，先放这儿

        policy_loss_acc = 0.0
        entropy_acc = 0.0
        approx_kl_acc = 0.0
        kl_est_acc = 0.0
        kl_penalty_acc = 0.0

        masked_frac_acc = 0.0
        episode_kl_mean_acc = 0.0
        episode_kl_max_acc = 0.0

        update_epochs = int(getattr(self.config, "update_epochs", 4))
        clip_coef = float(getattr(self.config, "clip_coef", 0.2))
        entropy_coef = float(getattr(self.config, "entropy_coef", 0.01))
        kl_coef = float(getattr(self.config, "kl_coef", 0.0))

        for _ in range(update_epochs):
            # ====== Off-policy episode mask（可选）=====
            keep_all: Optional[torch.Tensor] = None
            masked_frac = 0.0
            episode_kl_mean = 0.0
            episode_kl_max = 0.0

            if bool(getattr(self.config, "offpolicy_mask_enabled", False)) and episode_ids is not None:
                with torch.no_grad():
                    logits_all = self._mask_logits(self.actor(observations, routing_state=routing_state), action_mask)
                    new_logp_all = Categorical(logits=logits_all).log_prob(actions)
                    kl_sample_all = old_log_probs - new_logp_all  # 采样 KL：logp_old - logp_new

                    uniq, inv = torch.unique(episode_ids, return_inverse=True)
                    num_eps = int(uniq.numel())
                    counts = torch.bincount(inv, minlength=num_eps).to(torch.float32).clamp_min(1.0)

                    kl_sum = torch.zeros(num_eps, device=self.device).scatter_add_(0, inv, kl_sample_all.to(torch.float32))
                    adv_sum = torch.zeros(num_eps, device=self.device).scatter_add_(0, inv, advantages.to(torch.float32))
                    ep_kl = kl_sum / counts
                    ep_adv = adv_sum / counts

                    kl_th = float(getattr(self.config, "offpolicy_kl_threshold", 0.1))
                    adv_th = float(getattr(self.config, "offpolicy_adv_threshold", 0.0))
                    drop_ep = (ep_kl > kl_th) & (ep_adv < adv_th)

                    drop_step = drop_ep[inv]
                    keep_all = ~drop_step

                    masked_frac = float(drop_step.to(torch.float32).mean().item())
                    episode_kl_mean = float(ep_kl.mean().item()) if num_eps > 0 else 0.0
                    episode_kl_max = float(ep_kl.max().item()) if num_eps > 0 else 0.0

            masked_frac_acc += masked_frac
            episode_kl_mean_acc += episode_kl_mean
            episode_kl_max_acc += episode_kl_max

            # ====== PPO-Clip 更新（再加 KL penalty + episode mask + action_mask）=====
            inds = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                mb_obs = observations[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_log_probs = old_log_probs[mb_inds]
                mb_adv = advantages[mb_inds]

                mb_keep = keep_all[mb_inds] if keep_all is not None else None
                mb_action_mask = action_mask[mb_inds] if action_mask is not None else None

                logits = self._mask_logits(self.actor(mb_obs, routing_state=routing_state), mb_action_mask)
                dist = Categorical(logits=logits)

                if mb_action_mask is not None:
                    # 数据一致性检查：动作必须在 mask 的有效子空间里
                    valid_taken = mb_action_mask.gather(1, mb_actions.view(-1, 1)).view(-1)
                    if bool((~valid_taken).any().item()):
                        raise ValueError("发现 invalid action：action_mask=False 但 actions 选了它（请检查数据管线）")

                new_log_probs = dist.log_prob(mb_actions)
                entropy_per_step = dist.entropy()

                # ratio = exp(logπ_new - logπ_old)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                clip_ratio = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)

                # PPO-Clip surrogate（先算 per-step，再用 mask 做平均）
                surrogate1 = ratio * mb_adv
                surrogate2 = clip_ratio * mb_adv
                policy_loss = self._masked_mean(-torch.min(surrogate1, surrogate2), mb_keep)

                # entropy bonus（同样做 mask 平均）
                entropy = self._masked_mean(entropy_per_step, mb_keep)

                # 采样 KL：kl_sample = logπ_old(a) - logπ_new(a)，a 是 old policy 采出来的动作
                kl_sample = mb_old_log_probs - new_log_probs
                kl_est_sample = self._masked_mean(kl_sample, mb_keep)

                # KL penalty：默认 kl_coef=0 就等于没开
                # 为了不让采样噪声把 KL 弄成负数，这里简单截断一下
                kl_for_penalty = torch.clamp(kl_est_sample, min=0.0)
                kl_penalty = kl_coef * kl_for_penalty

                loss = policy_loss - entropy_coef * entropy + kl_penalty

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
                kl_est_acc += float(kl_est_sample.item())
                kl_penalty_acc += float(kl_penalty.item())

        denom = max(1, update_epochs * num_minibatches)
        return {
            "policy_loss": policy_loss_acc / denom,
            "entropy": entropy_acc / denom,
            "approx_kl": approx_kl_acc / denom,
            "kl_est": kl_est_acc / denom,
            "kl_penalty": kl_penalty_acc / denom,
            "masked_frac": masked_frac_acc / max(1, update_epochs),
            "episode_kl_mean": episode_kl_mean_acc / max(1, update_epochs),
            "episode_kl_max": episode_kl_max_acc / max(1, update_epochs),
            "invalid_action_count": 0.0,
        }

    def save(self):
        return {"actor": self.actor.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load(self, data: Dict):
        self.actor.load_state_dict(data["actor"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
