from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from rl_algo.core.base_agent import BaseAgent
from rl_algo.core.types import Config


@dataclass
class A2CConfig(Config):
    """A2C 智能体的配置结构定义。"""

    gamma: float = 0.99
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
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

    def forward(self, obs: torch.Tensor):
        x = obs.view(obs.size(0), -1)
        hidden = self.body(x)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return logits, value


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
    ):
        obs = obs.to(self.device)
        logits, value = self.ac(obs)
        dist = Categorical(logits=logits)

        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=1)

        log_prob = dist.log_prob(action)
        return action, {"log_prob": log_prob.detach(), "value": value.detach()}

    def update(self, batch, global_step: int = 0):
        # A2C 我这里就按最基础那套：
        # return R_t 从后往前算（遇到 done 清零）
        # advantage A_t = R_t - V(s_t)
        # policy_loss = -E[logπ(a|s) * A]
        # value_loss  = MSE(V(s), R)
        # loss = policy_loss + value_coef*value_loss - entropy_coef*entropy
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).view(-1, 1)
        dones = batch["dones"].to(self.device).view(-1, 1)
        old_log_probs = batch.get("log_probs")
        actions = actions.squeeze(-1).long() if actions.dim() > 1 else actions.long()

        # 1) 先把一串 step-level 数据还原成“多段 episode”的折扣回报
        #    dones=1 的地方表示 episode 结束，需要把 running_return 清零
        returns = torch.zeros_like(rewards, device=self.device)
        running_return = torch.zeros(1, device=self.device)
        gamma = float(getattr(self.config, "gamma", 0.99))
        for t in reversed(range(rewards.size(0))):
            running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

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
                mb_returns = returns[mb_inds]

                mb_logits, mb_values = self.ac(mb_obs)
                mb_dist = Categorical(logits=mb_logits)
                mb_log_probs = mb_dist.log_prob(mb_actions)
                mb_entropy = mb_dist.entropy().mean()

                # 2) 优势：A = return - V(s)
                #    这里用 detach 避免把 critic 的梯度错误地传到优势里
                mb_adv = mb_returns - mb_values.detach()
                # 3) 优势标准化：更稳、更不挑学习率（作业里很常用）
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)

                # 4) policy loss：最大化 E[logπ(a|s) * A]，实现时写成最小化负号
                policy_loss = -(mb_adv.squeeze(-1) * mb_log_probs).mean()
                # 5) value loss：让 V(s) 拟合 return
                value_loss = F.mse_loss(mb_values, mb_returns)
                # 6) entropy：鼓励探索（防止过早塌缩成确定性策略）
                loss = policy_loss + float(getattr(self.config, "value_coef", 0.5)) * value_loss - float(
                    getattr(self.config, "entropy_coef", 0.01)
                ) * mb_entropy

                # 7) 反向传播 + 梯度裁剪（防止梯度爆炸）
                self.optimizer.zero_grad()
                loss.backward()
                max_grad_norm = float(getattr(self.config, "max_grad_norm", 0.0))
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.ac.parameters(), max_grad_norm)
                self.optimizer.step()

                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                entropy_acc += float(mb_entropy.item())

                if old_log_probs is not None:
                    with torch.no_grad():
                        mb_old = old_log_probs.to(self.device).view(-1)[mb_inds]
                        approx_kl_acc += float((mb_old - mb_log_probs).mean().abs().item())

        denom = int(getattr(self.config, "update_epochs", 4)) * num_minibatches
        metrics = {
            "policy_loss": policy_loss_acc / denom,
            "value_loss": value_loss_acc / denom,
            "entropy": entropy_acc / denom,
        }
        if old_log_probs is not None:
            metrics["kl"] = approx_kl_acc / denom
        return metrics

    def save(self):
        return {"model": self.ac.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load(self, data: Dict):
        self.ac.load_state_dict(data["model"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
