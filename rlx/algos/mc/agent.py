# rlx/algos/mc/agent.py
"""
实现了同策略、首次访问的蒙特卡洛 (On-Policy, First-Visit Monte Carlo, MC) 控制算法。

蒙特卡洛方法是一类经典的、无需模型的 (model-free) 强化学习算法。它直接从
完整的经验回合 (episodes) 中学习价值函数, 而无需了解环境的动态模型 (即状态
转移概率和奖励函数)。

核心特点:
1.  **基于完整回合**: MC 方法必须等到一个完整的回合结束后, 才能根据该回合中
    所有后续的奖励来计算每个状态-动作对的价值。这与 TD 方法 (如 Q-learning)
    在每一步都进行更新形成对比。

2.  **同策略 (On-Policy)**: 它评估和改进的是用于生成数据的同一个策略。具体来说,
    它使用当前的 epsilon-greedy 策略来与环境交互, 并用收集到的数据来更新
    这个策略本身。

3.  **首次访问 (First-Visit)**: 在一个回合中, 某个状态-动作对 (s, a) 可能被
    访问多次。首次访问 MC 仅考虑该回合中第一次访问 (s, a) 之后所获得的
    回报 (return) 来更新其 Q 值。这有助于减少回报计算的方差。

4.  **表格化 (Tabular)**: 该实现使用字典 (defaultdict) 作为 Q-table, 将状态
    (离散化后) 和动作映射到对应的 Q 值。适用于状态和动作空间较小的环境。

5.  **状态离散化**: 为了处理连续状态空间 (如 CartPole), 该智能体实现了一个
    简单的分箱 (binning) 方法, 将连续的观测值映射到离散的状态元组中。
"""
import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict

from rlx.core.base_agent import BaseAgent
from rlx.core.registry import registry
from rlx.core.types import Config

class MCConfig(Config):
    """蒙特卡洛算法的 Pydantic 配置类。"""
    gamma: float = 0.99       # 折扣因子, 用于计算未来奖励的当前价值
    epsilon: float = 1.0      # epsilon-greedy 策略的初始探索率
    total_steps: int = 50000  # 总训练步数 (主要用于 epsilon 衰减的调度)

@registry.register_agent("mc", MCConfig)
class MCAgent(BaseAgent):
    """同策略、首次访问的蒙特卡洛控制智能体。"""

    def __init__(self, obs_space: gym.Space, act_space: gym.Space, config: Config, device: str):
        super().__init__(obs_space, act_space, config, device)
        
        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("蒙特卡洛智能体目前仅支持离散动作空间。")

        # 支持离散和连续 (Box) 观测空间
        if isinstance(obs_space, gym.spaces.Box):
            # 为连续状态空间创建分箱 (bins) 以进行离散化
            self.state_bins = [
                np.linspace(obs_space.low[i], obs_space.high[i], 10)
                for i in range(obs_space.shape[0])
            ]
        elif not isinstance(obs_space, gym.spaces.Discrete):
            raise TypeError("蒙特卡洛智能体仅支持离散 (Discrete) 和连续 (Box) 观测空间。")

        # 使用 defaultdict 可以方便地处理未见过的状态, 自动为其创建默认的 Q 值数组
        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        # 用于计算平均回报的辅助字典
        self.returns_sum = defaultdict(float)   # 存储每个 (状态, 动作) 对累积的回报总和
        self.returns_count = defaultdict(float) # 存储每个 (状态, 动作) 对被访问的次数
        
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

    def update(self, batch: dict, global_step: int) -> dict:
        """
        在一个完整的回合结束后, 使用收集到的轨迹数据来更新 Q-table。
        `batch` 字典包含了该回合的所有观测、动作和奖励。
        """
        rewards = batch["rewards"].cpu().numpy()
        observations = batch["observations"].cpu().numpy()
        actions = batch["actions"].cpu().numpy()
        
        G = 0  # 初始化回报 (Return)
        visited_sa_pairs = set() # 用于追踪本回合中已访问过的 (状态, 动作) 对
        
        # 从回合的最后一步开始, 反向遍历整个轨迹
        for t in reversed(range(len(rewards))):
            # 计算从时间步 t 开始的折扣回报 G
            G = self.gamma * G + rewards[t]
            
            obs_t = observations[t].flatten()
            state_t = self.discretize_state(obs_t)
            action_t = actions[t]

            sa_pair = (state_t, action_t)
            
            # 首次访问 MC 的核心逻辑: 仅当这是本回合第一次遇到该 (状态, 动作) 对时才进行更新
            if sa_pair not in visited_sa_pairs:
                self.returns_sum[sa_pair] += G
                self.returns_count[sa_pair] += 1.0
                
                # 通过平均回报来更新 Q 值
                self.q_table[state_t][action_t] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]
                
                visited_sa_pairs.add(sa_pair)
        
        # 探索率衰减: 随着训练的进行, 逐渐减少探索的概率
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

        return {"q_table_size": len(self.q_table), "epsilon": self.epsilon}

    def save(self) -> dict:
        """将 Q-table 和回报计数器保存到一个可序列化的字典中。"""
        # 将 defaultdict 转换为普通 dict, 以便能够被 pickle/torch.save 保存
        return {
            "q_table": dict(self.q_table),
            "returns_sum": dict(self.returns_sum),
            "returns_count": dict(self.returns_count),
        }

    def load(self, state: dict):
        """从一个字典加载 Q-table 和回报计数器。"""
        # 重新创建空的 defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.act_space.n))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        
        # 使用加载的数据更新它们
        self.q_table.update(state["q_table"])
        self.returns_sum.update(state["returns_sum"])
        self.returns_count.update(state["returns_count"])
