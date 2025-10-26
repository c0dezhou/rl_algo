# rlx/algos/qlearning/agent.py
"""
实现了经典的表格型 Q-Learning 算法。

Q-Learning 是一种无需模型的 (model-free)、异策略的 (off-policy) 时序差分 (TD)
控制算法。它是强化学习领域的基石之一, 目标是学习一个最优的动作价值函数 Q*(s, a),
该函数表示在状态 s 下采取动作 a 后, 遵循最优策略所能得到的期望回报。

核心特点:
1.  **异策略 (Off-Policy)**: 这是 Q-Learning 最关键的特性。它在更新 Q-table 时,
    使用的是对下一个状态 `s'` 所有可能动作的 Q 值取最大值的操作 (`max_a' Q(s', a')`)。
    这意味着它学习的是一个贪心策略 (最优策略), 而不管实际用于探索的策略是什么
    (例如, epsilon-greedy)。这使得学习过程非常稳定和直接。

2.  **时序差分 (Temporal-Difference, TD) 学习**: Q-Learning 在每一步之后都
    会立即使用观察到的奖励 `r` 和对下一个状态价值的估计 `max Q(s', a')` 来
    更新当前状态-动作对的 Q 值。它不需要等待一个回合结束, 这使得它比蒙特卡洛
    方法有更快的学习速度和更低的方差。

3.  **表格化 (Tabular)**: 该实现使用字典 (defaultdict) 作为 Q-table, 将状态
    (离散化后) 和动作映射到对应的 Q 值。适用于状态和动作空间较小的环境。

4.  **状态离散化**: 为了处理连续状态空间 (如 CartPole), 该智能体实现了一个
    简单的分箱 (binning) 方法, 将连续的观测值映射到离散的状态元组中。
"""
import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict

from rlx.core.base_agent import BaseAgent
from rlx.core.registry import registry
from rlx.core.types import Transition, Config

class QLearningConfig(Config):
    """Q-Learning 算法的 Pydantic 配置类。"""
    lr: float = 0.1             # 学习率 (alpha): 控制每次更新的步长
    gamma: float = 0.99         # 折扣因子: 用于计算未来奖励的当前价值
    epsilon: float = 1.0        # epsilon-greedy 策略的初始探索率
    total_steps: int = 50000    # 总训练步数 (主要用于 epsilon 衰减的调度)

@registry.register_agent("qlearning", QLearningConfig)
class QLearningAgent(BaseAgent):
    """表格型 Q-Learning 智能体。"""

    def __init__(self, obs_space: gym.Space, act_space: gym.Space, config: Config, device: str):
        super().__init__(obs_space, act_space, config, device)
        
        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("Q-Learning 智能体目前仅支持离散动作空间。")

        # 支持离散和连续 (Box) 观测空间
        if isinstance(obs_space, gym.spaces.Box):
            # 为连续状态空间创建分箱 (bins) 以进行离散化
            self.state_bins = [
                np.linspace(obs_space.low[i], obs_space.high[i], 10)
                for i in range(obs_space.shape[0])
            ]
        elif not isinstance(obs_space, gym.spaces.Discrete):
            raise TypeError("Q-Learning 智能体仅支持离散 (Discrete) 和连续 (Box) 观测空间。")

        # 使用 defaultdict 可以方便地处理未见过的状态, 自动为其创建默认的 Q 值数组
        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = config.epsilon

    def discretize_state(self, obs: np.ndarray) -> tuple:
        """将连续的观测值离散化为一个可哈希的元组。"""
        # 如果环境的观测空间本身就是离散的, 则无需处理
        if not hasattr(self, "state_bins"):
            return tuple(obs) if isinstance(obs, (list, np.ndarray)) else (obs,)
        
        # 使用 np.digitize 将每个维度的观测值分配到对应的箱子索引中
        state = [np.digitize(val, self.state_bins[i]) for i, val in enumerate(obs)]
        return tuple(state)

    def select_action(self, obs: torch.Tensor, global_step: int = 0, explore: bool = True) -> tuple[int, dict]:
        """使用 epsilon-greedy 策略选择一个动作。"""
        obs_np = obs.cpu().numpy().flatten()
        state = self.discretize_state(obs_np)

        # 探索: 以 epsilon 的概率随机选择一个动作
        if explore and np.random.rand() < self.epsilon:
            action = self.act_space.sample()
        # 利用: 以 1-epsilon 的概率选择当前估计的最优动作
        else:
            action = np.argmax(self.q_table[state])
            
        return action, {}

    def train_step(self, transition: Transition) -> dict:
        """执行单步 Q-Learning 更新。"""
        obs = transition["obs"].cpu().numpy().flatten()
        next_obs = transition["next_obs"].cpu().numpy().flatten()
        action = transition["action"]
        reward = transition["reward"]
        done = transition["done"]

        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        # Q-Learning 更新规则:
        # Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        # `max_a' Q(s', a')` 是关键，它代表了在下一个状态 `s'` 所有可能的动作中，能带来的最大 Q 值
        # TD error: 减去 Q (s,a) 的核心目的是计算 “当前估计值” 与 “目标估计值” 之间的误差，从而通过这个误差来修正 Q 值
        old_value = self.q_table[state][action]
        # 核心: 无论下一步实际采取什么动作, 都使用下一个状态的最大 Q 值来更新
        next_max = np.max(self.q_table[next_state])
        
        # 计算新的 Q 值
        # 用1 - done清零这部分未来价值, 如果 done 为 True, 则 (1 - done) 为 0, 未来回报的价值为 0
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max * (1 - done))
        self.q_table[state][action] = new_value
        
        # 探索率衰减: 随着训练的进行, 逐渐减少探索的概率
        if self.epsilon > 0.01:
            self.epsilon *= 0.999

        return {"q_value": new_value, "epsilon": self.epsilon}

    def save(self) -> dict:
        """将 Q-table 保存到一个可序列化的字典中。"""
        # 将 defaultdict 转换为普通 dict, 以便能够被 pickle/torch.save 保存
        return {"q_table": dict(self.q_table)}

    def load(self, state: dict):
        """从一个字典加载 Q-table。"""
        # 重新创建空的 defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        # 使用加载的数据更新它
        self.q_table.update(state["q_table"])
