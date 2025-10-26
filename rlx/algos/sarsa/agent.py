# rlx/algos/sarsa/agent.py
"""
实现了经典的表格型 SARSA 算法。

SARSA (State-Action-Reward-State-Action) 是一种无需模型的 (model-free)、
同策略的 (on-policy) 时序差分 (TD) 控制算法。它与 Q-Learning 非常相似,
但有一个关键的区别, 这个区别也体现在它的名字中。

核心特点:
1.  **同策略 (On-Policy)**: 这是 SARSA 与 Q-Learning 最本质的区别。在更新
    Q-table 时, SARSA 使用的是在下一个状态 `s'` *实际将要采取* 的动作 `a'`
    所对应的 Q 值, 即 `Q(s', a')`。由于 `a'` 是由当前策略 (例如, epsilon-greedy)
    生成的, 这意味着 SARSA 评估和改进的是它正在执行的同一个策略。相比之下,
    Q-Learning (异策略) 总是使用最优的 `max Q(s', a')` 来更新, 学习的是
    一个与行为策略分离的贪心策略。

2.  **时序差分 (Temporal-Difference, TD) 学习**: 与 Q-Learning 一样, SARSA
    在每一步之后都会立即使用观察到的奖励和对下一步的价值估计来更新 Q 值,
    无需等待回合结束。

3.  **保守性**: 由于 SARSA 的更新考虑了探索性动作可能带来的后果 (因为 `a'`
    可能是随机选择的), 它在学习过程中通常比 Q-Learning 更“保守”。例如,
    在悬崖行走问题中, SARSA 会倾向于选择一条远离悬崖的安全路径, 即使
    那条路径的回报稍低, 而 Q-Learning 则会紧贴悬崖行走以寻求最高回报,
    更容易因为探索而掉下悬崖。

4.  **表格化 (Tabular)** 和 **状态离散化**: 与 Q-Learning 实现相同, 使用
    字典作为 Q-table, 并通过分箱 (binning) 方法处理连续状态空间。
"""
import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict

from rlx.core.base_agent import BaseAgent
from rlx.core.registry import registry
from rlx.core.types import Transition, Config

class SarsaConfig(Config):
    """SARSA 算法的 Pydantic 配置类。"""
    lr: float = 0.1             # 学习率 (alpha): 控制每次更新的步长
    gamma: float = 0.99         # 折扣因子: 用于计算未来奖励的当前价值
    epsilon: float = 1.0        # epsilon-greedy 策略的初始探索率
    total_steps: int = 50000    # 总训练步数 (主要用于 epsilon 衰减的调度)

@registry.register_agent("sarsa", SarsaConfig)
class SarsaAgent(BaseAgent):
    """表格型 SARSA 智能体。"""

    def __init__(self, obs_space: gym.Space, act_space: gym.Space, config: Config, device: str):
        super().__init__(obs_space, act_space, config, device)
        
        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("SARSA 智能体目前仅支持离散动作空间。")

        # 支持离散和连续 (Box) 观测空间
        if isinstance(obs_space, gym.spaces.Box):
            # 为连续状态空间创建分箱 (bins) 以进行离散化
            self.state_bins = [
                # 在 “最小值 - 最大值” 之间均匀划分出 10 个 “分界点”，生成一个数组（比如[-4.8, -3.73, -2.67, ..., 4.8]），这个数组就是 “分箱（bins）”
                np.linspace(obs_space.low[i], obs_space.high[i], 10)
                for i in range(obs_space.shape[0]) # 对每个维度都分箱
            ]
        elif not isinstance(obs_space, gym.spaces.Discrete):
            raise TypeError("SARSA 智能体仅支持离散 (Discrete) 和连续 (Box) 观测空间。")

        # 使用 defaultdict 可以方便地处理未见过的状态, 自动为其创建默认的 Q 值数组
        # 当遇到新状态 s_new 时，self.q_table[s_new] 会直接报错；
        # 而 defaultdict 会自动为 s_new 创建初始的动作价值数组，无需手动判断 “状态是否存在”。
        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n)) # self.act_space.n：获取离散动作空间的「动作总数」
        # 不能直接用 np.zeros(self.act_space.n) 作为默认值：
        # 错误写法：defaultdict(np.zeros(self.act_space.n)) —— 这会立即执行 np.zeros，生成一个固定数组，所有新状态都会共享这个数组（修改一个状态的动作价值，会影响所有状态），完全错误；
        # 必须用 “无参数可调用对象”：
        # 当访问新状态时，defaultdict 会每次都调用这个 lambda 函数，为该状态重新生成一个全新的数组,使每组Q值数组独立
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = config.epsilon

    def discretize_state(self, obs: np.ndarray) -> tuple:
        """将连续的观测值离散化为一个可哈希的元组。"""
        if not hasattr(self, "state_bins"): # 如果是离散观测空间, 则无需处理
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
        """
        执行单步 SARSA 更新。
        `transition` 字典必须包含 'next_action' 键。
        """
        obs = transition["obs"].cpu().numpy().flatten()
        next_obs = transition["next_obs"].cpu().numpy().flatten()
        action = transition["action"]
        next_action = transition["next_action"] # 这是 SARSA 的关键
        reward = transition["reward"]
        done = transition["done"]

        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        # SARSA 更新规则:
        # Q(s, a) <- Q(s, a) + lr * [r + gamma * Q(s', a') - Q(s, a)]
        # TD error: 减去 Q (s,a) 的核心目的是计算 “当前估计值” 与 “目标估计值” 之间的误差，从而通过这个误差来修正 Q 值
        old_value = self.q_table[state][action]
        # 核心: 使用下一个状态 `s'` 和在该状态下实际将要执行的动作 `a'` 的 Q 值
        next_value = self.q_table[next_state][next_action]
        
        # 计算新的 Q 值
        # 为了给 train_step 提供 next_action，训练脚本 train.py 为 SARSA 做了特殊处理：
        # 在得到 next_obs 后，会再调用一次 agent.select_action() 来获得 next_action，然后才进行更新。
        new_value = old_value + self.lr * (reward + self.gamma * next_value * (1 - done) - old_value)
        self.q_table[state][action] = new_value
        
        # 探索率衰减
        if self.epsilon > 0.01:
            self.epsilon *= 0.9999

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
